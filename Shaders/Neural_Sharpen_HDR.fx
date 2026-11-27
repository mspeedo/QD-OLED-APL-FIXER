/*
    Neural Sharpen HDR
    HDR-aware and GPU-optimized standalone adaptation of Barbatos Neural Sharpening (12-Channel CNN v2.8).

    Credits & License:
    - Original Neural Sharpening 12-channel CNN architecture and pretrained Model A / Model B weights:
      Barbatos (https://github.com/BarbatosAWLS/Reshade-Shaders), MIT License.
    - HDR color pipeline (ST.2084 PQ / scRGB Rec.2020 luminance-ratio sharpening, OLED black-level protection)
      and GPU optimizations (tex2DgatherR 3x3 loading, column-major FP16 vector FMAs, static PQ LUT,
      pre-CNN warp early-out, vertex culling): MIT License.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    File layout:
      1. Compile-Time Preprocessor Switches
      2. UI Controls & Constants
      3. GPU Resources
      4. Color Science & ST.2084 PQ Math
      5. 12-Channel Neural Network (Model A / Model B)
      6. Passes & Technique
*/

#include "ReShade.fxh"


// ============================================================================
// 1. COMPILE-TIME PREPROCESSOR SWITCHES
// ============================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 0
#endif

// 0 = Runtime UI selector (Model A / Model B)
// 1 = Compile-time Model A only (strips Model B for smaller shader binary)
// 2 = Compile-time Model B only (strips Model A for smaller shader binary)
#ifndef NEURAL_SHARPEN_MODEL
    #define NEURAL_SHARPEN_MODEL 0
#endif

// 0 strips the HDR pipeline and PQ LUT at compile time in confirmed SDR (BUFFER_COLOR_SPACE == 1).
// 1 keeps manual HDR override compiled even in confirmed SDR.
#ifndef ALLOW_SDR_MANUAL_OVERRIDE
    #define ALLOW_SDR_MANUAL_OVERRIDE 0
#endif

#if (BUFFER_COLOR_SPACE != 1) || (ALLOW_SDR_MANUAL_OVERRIDE != 0)
    #define NSHDR_PIPELINE_COMPILED 1
#else
    #define NSHDR_PIPELINE_COMPILED 0
#endif

// Use a 1024-entry static R32F compute-initialized PQ decode LUT on D3D11/D3D12/Vulkan/GL4.3+.
#if NSHDR_PIPELINE_COMPILED && defined(__RENDERER__) && (__RENDERER__ >= 0x0B000)
    #define NSHDR_USE_PQ_LUT 1
#else
    #define NSHDR_USE_PQ_LUT 0
#endif

#ifndef ENABLE_UI_TOOLTIPS
    #define ENABLE_UI_TOOLTIPS 1
#endif

#if ENABLE_UI_TOOLTIPS
    #define NSHDR_TOOLTIP(text) ui_tooltip = text;
#else
    #define NSHDR_TOOLTIP(text)
#endif


namespace NeuralSharpenHDR
{

// ============================================================================
// 2. UI CONTROLS & CONSTANTS
// ============================================================================

#if NSHDR_PIPELINE_COMPILED
uniform int ColorSpaceMode <
    ui_type = "combo";
    ui_items = "Auto\0SDR (sRGB)\0HDR scRGB (Linear)\0HDR10 (PQ / Rec.2020)\0";
    ui_label = "Color Space";
    NSHDR_TOOLTIP("Auto detects sRGB, scRGB, or HDR10 PQ from ReShade's buffer color space (defaults to SDR if Unknown). Override manually if needed.")
> = 0;
#endif

#if NEURAL_SHARPEN_MODEL == 0
uniform int ModelSelector <
    ui_type = "combo";
    ui_items = "Model A (Balanced)\0Model B (Edge-Enhanced)\0";
    ui_label = "Neural Model";
    NSHDR_TOOLTIP("Model A: Balanced detail sharpening. Model B: Stronger edge-enhanced sharpening. Set preprocessor NEURAL_SHARPEN_MODEL=1 or 2 to compile only one model.")
> = 0;
#endif

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Intensity";
    NSHDR_TOOLTIP("Overall neural sharpening strength. At 0.0, all passes are vertex-culled for zero shader cost.")
> = 1.0;

uniform float EdgeResponse <
    ui_type = "drag";
    ui_min = -1.0;
    ui_max = 1.0;
    ui_step = 0.05;
    ui_label = "Edge Response";
    NSHDR_TOOLTIP("Controls how sharpening adapts to detected edges. Negative = softer strong edges, 0 = uniform, Positive = stronger edges.")
> = 0.0;

uniform float ArtifactProtection <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.05;
    ui_label = "Artifact Protection";
    NSHDR_TOOLTIP("Suppresses sharpening on high-contrast aliasing edges and isolated specular fireflies. 1.0 = full protection, 0.0 = off.")
> = 1.0;

uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Sharpening Map\0Residual Magnitude\0";
    ui_label = "Debug View";
    NSHDR_TOOLTIP("Normal: Final sharpened output. Sharpening Map: Signed sharpening delta around mid-gray. Residual Magnitude: Raw neural edge response (tone-mapped to ~203 nits in HDR).")
> = 0;

// Internal color-space mode IDs
#define MODE_SDR   1
#define MODE_SCRGB 2
#define MODE_PQ    3

// Flat-patch early-out thresholds (eliminates OLED pure-black DC bias & skips flat sky/bars/UI)
static const float FLAT_SKIP_RANGE = 0.0005;
static const float FLAT_FADE_END   = 0.0025;

// BT.709 / BT.2020 luminance weights
static const float3 LUMA_709  = float3(0.2126, 0.7152, 0.0722);
static const float3 LUMA_2020 = float3(0.2627, 0.6780, 0.0593);

// ST.2084 PQ constants
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

// scRGB linear 1.0 = 80 nits; PQ linear 1.0 = 10000 nits
static const float SCRGB_TO_PQ_LINEAR = 80.0 / 10000.0;

// Comfortable ~203 nit reference white for HDR debug visualizations
static const float DEBUG_HDR_WHITE_NITS = 203.0;


// ============================================================================
// 3. GPU RESOURCES
// ============================================================================

texture TexLuma
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R16F;
};

sampler sTexLuma
{
    Texture = TexLuma;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

#if NSHDR_USE_PQ_LUT
// Static 4 KB 1024-entry ST.2084 PQ -> Linear [0, 1] LUT (exact at all 10-bit PQ codes).
// Built once on frame 0 via CS_InitPQLUT; fits entirely in L1 cache.
texture TexPQToLinearLUT
{
    Width = 1024;
    Height = 1;
    Format = R32F;
};

sampler sPQToLinearLUT
{
    Texture = TexPQToLinearLUT;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;
};

storage2D<float> StoragePQToLinearLUT
{
    Texture = TexPQToLinearLUT;
};
#endif


// ============================================================================
// 4. COLOR SCIENCE & ST.2084 PQ MATH
// ============================================================================

int ResolveColorSpaceMode()
{
#if NSHDR_PIPELINE_COMPILED
#if BUFFER_COLOR_SPACE == 3
    int mode = MODE_PQ;
#elif BUFFER_COLOR_SPACE == 2
    int mode = MODE_SCRGB;
#else
    int mode = MODE_SDR;
#endif
    if (ColorSpaceMode != 0)
        mode = ColorSpaceMode;
    return mode;
#else
    return MODE_SDR;
#endif
}

bool IsShaderActive()
{
    return (Intensity > 0.0001) || (ViewMode != 0);
}

#if NSHDR_PIPELINE_COMPILED
float PQToLinearScalar(float v)
{
    float vp = pow(saturate(v), 1.0 / PQ_M2);
    float num = max(vp - PQ_C1, 0.0);
    float den = PQ_C2 - PQ_C3 * vp;
    return pow(num / max(den, 1e-6), 1.0 / PQ_M1);
}

float LinearToPQScalar(float linearVal)
{
    float Lm1 = pow(saturate(linearVal), PQ_M1);
    float num = PQ_C1 + PQ_C2 * Lm1;
    float den = 1.0 + PQ_C3 * Lm1;
    return pow(num / max(den, 1e-6), PQ_M2);
}

float3 LinearToPQ3(float3 v)
{
    float3 Lm1 = pow(saturate(v), PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lm1;
    float3 den = 1.0 + PQ_C3 * Lm1;
    return pow(num / max(den, 1e-6), PQ_M2);
}

float PQToLinearFast(float v)
{
#if NSHDR_USE_PQ_LUT
    float u = saturate(v) * (1023.0 / 1024.0) + (0.5 / 1024.0);
    return tex2Dlod(sPQToLinearLUT, float4(u, 0.5, 0.0, 0.0)).r;
#else
    return PQToLinearScalar(v);
#endif
}

float3 PQToLinearFast3(float3 v)
{
#if NSHDR_USE_PQ_LUT
    return float3(
        PQToLinearFast(v.r),
        PQToLinearFast(v.g),
        PQToLinearFast(v.b)
    );
#else
    float3 vp = pow(saturate(v), 1.0 / PQ_M2);
    float3 num = max(vp - PQ_C1, 0.0);
    float3 den = PQ_C2 - PQ_C3 * vp;
    return pow(num / max(den, 1e-6), 1.0 / PQ_M1);
#endif
}

// Maps a [0, 1] sRGB-like debug scalar to comfortable ~203-nit HDR output in the active color space.
float3 EncodeDebugGray(float gray, int mode)
{
    float g = saturate(gray);
    float lin = (g <= 0.04045) ? (g / 12.92) : pow((g + 0.055) / 1.055, 2.4);
    float3 result = g.xxx;

    if (mode == MODE_PQ)
        result = LinearToPQScalar(lin * (DEBUG_HDR_WHITE_NITS / 10000.0)).xxx;
    else if (mode == MODE_SCRGB)
        result = (lin * (DEBUG_HDR_WHITE_NITS / 80.0)).xxx;

    return result;
}
#else
float3 EncodeDebugGray(float gray, int mode)
{
    return saturate(gray).xxx;
}
#endif


// ============================================================================
// 5. 12-CHANNEL NEURAL NETWORK (MODEL A / MODEL B)
// ============================================================================

min16float4 PReLU4(min16float4 v, min16float4 a)
{
    const min16float4 z = min16float4(0.0, 0.0, 0.0, 0.0);
    return (v >= z) ? v : (v * a);
}

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 1)
min16float RunModelA(
    min16float p0, min16float p1, min16float p2,
    min16float p3, min16float p4, min16float p5,
    min16float p6, min16float p7, min16float p8)
{
    // Layer 1: 3x3 Conv (1 -> 12 channels), pre-unrolled FP16 vector FMAs
    min16float4 L1_0 = min16float4(0.013958196, -0.034602143, 0.017349342, 0.051420901);
    min16float4 L1_1 = min16float4(-0.002181380, 0.005415063, -0.001871437, -0.040856525);
    min16float4 L1_2 = min16float4(0.021341776, 0.016098073, -0.027485607, -0.000488010);

    L1_0 += p0 * min16float4(0.157846898, -0.124734171, -0.127510592, -0.142722160);
    L1_1 += p0 * min16float4(-0.081266180, -0.035668157, -0.128144190, 0.029555680);
    L1_2 += p0 * min16float4(0.210917711, -0.152093813, -0.240400285, 0.259769797);
    L1_0 += p1 * min16float4(-0.018440859, -0.024522314, -0.128880441, -0.146906286);
    L1_1 += p1 * min16float4(0.013216234, 0.053734783, 0.141454548, 0.003527662);
    L1_2 += p1 * min16float4(-0.405102581, -0.348911375, -0.129119381, 0.220478624);
    L1_0 += p2 * min16float4(0.160163894, 0.025609946, -0.154936627, 0.000249316);
    L1_1 += p2 * min16float4(0.003316377, 0.011136279, 0.057656068, -0.143218592);
    L1_2 += p2 * min16float4(0.270902693, 0.031090865, -0.147552043, 0.070456423);
    L1_0 += p3 * min16float4(-0.024166472, -0.277416319, 0.124570601, -0.051763635);
    L1_1 += p3 * min16float4(-0.063075431, 0.229678214, 0.024368009, 0.056972858);
    L1_2 += p3 * min16float4(-0.169356808, 0.238923475, -0.007092264, 0.328647912);
    L1_0 += p4 * min16float4(-0.493815362, 0.346248239, 0.331841409, 0.500756621);
    L1_1 += p4 * min16float4(-0.000619092, -0.624130189, -0.436149389, 0.508056283);
    L1_2 += p4 * min16float4(0.077492014, 0.104612425, -0.268455684, -0.233587697);
    L1_0 += p5 * min16float4(-0.042992529, -0.168081835, -0.129785717, 0.158413351);
    L1_1 += p5 * min16float4(0.051667836, 0.455829173, 0.141271994, -0.077929094);
    L1_2 += p5 * min16float4(0.113687925, 0.209046662, -0.484994620, -0.358390838);
    L1_0 += p6 * min16float4(0.177939489, 0.052703582, -0.057644196, -0.147503659);
    L1_1 += p6 * min16float4(-0.029891005, -0.118537180, 0.029043604, 0.004845856);
    L1_2 += p6 * min16float4(0.132930592, -0.184339643, -0.218361959, 0.174553171);
    L1_0 += p7 * min16float4(-0.058614843, -0.305103749, 0.254780143, -0.034503106);
    L1_1 += p7 * min16float4(0.153854638, 0.107778028, 0.209778413, -0.009218552);
    L1_2 += p7 * min16float4(-0.360170394, -0.132584587, 0.632588685, -0.138400137);
    L1_0 += p8 * min16float4(0.190503091, 0.291831911, -0.109605379, -0.063311636);
    L1_1 += p8 * min16float4(0.040927604, -0.064029731, 0.092040010, -0.070735231);
    L1_2 += p8 * min16float4(0.178212926, 0.068329029, -0.240759194, -0.314655423);

    // PReLU 1
    L1_0 = PReLU4(L1_0, min16float4(-2.387171745, -0.272749335, -2.923240185, -0.953309059));
    L1_1 = PReLU4(L1_1, min16float4(-2.168637514, 0.855447471, -0.171175957, 0.460944146));
    L1_2 = PReLU4(L1_2, min16float4(0.312436283, -0.096803114, -0.173512161, -1.026450276));

    // Layer 2: 1x1 Dense (12 -> 12 channels), column-major vector FMAs (HFMA2-friendly)
    min16float4 L2_0 = min16float4(0.031355277, 0.220565408, -0.028643996, -0.000396760);
    min16float4 L2_1 = min16float4(0.053762030, 0.191098660, 0.028867338, -0.012472775);
    min16float4 L2_2 = min16float4(0.009898989, -0.032807596, 0.000105271, -0.024801077);

    L2_0 += L1_0.x * min16float4(-0.073890910, -0.354415715, 0.864738226, -0.159830540);
    L2_1 += L1_0.x * min16float4(0.471214026, -0.918480992, -0.821215689, 0.160448000);
    L2_2 += L1_0.x * min16float4(-1.269040704, -0.877907932, -1.432761431, -0.141609788);
    L2_0 += L1_0.y * min16float4(0.442915738, 0.255197883, -0.477833390, 0.038192000);
    L2_1 += L1_0.y * min16float4(0.549063981, -0.645617664, -0.385795057, 0.688369751);
    L2_2 += L1_0.y * min16float4(-0.059846044, -0.114583023, -0.421258569, -0.390305877);
    L2_0 += L1_0.z * min16float4(0.062885523, -0.292075455, -0.162099168, -0.158631608);
    L2_1 += L1_0.z * min16float4(-0.014970109, -0.418581784, 0.239962846, 0.691993833);
    L2_2 += L1_0.z * min16float4(0.084006079, 0.211630091, 0.412224859, 0.055484608);
    L2_0 += L1_0.w * min16float4(-0.237529442, -0.725630701, -0.246071026, 0.733967006);
    L2_1 += L1_0.w * min16float4(-0.376401037, 0.288688481, 0.137052476, -0.424544305);
    L2_2 += L1_0.w * min16float4(0.706222057, 0.710830331, 0.558480740, 0.364328563);
    L2_0 += L1_1.x * min16float4(-0.008712847, 0.136713073, -0.014323278, 0.160120249);
    L2_1 += L1_1.x * min16float4(0.332775176, 0.145794243, -0.084028967, 0.312038898);
    L2_2 += L1_1.x * min16float4(0.175326273, 0.375253499, -0.483496517, -0.040529206);
    L2_0 += L1_1.y * min16float4(2.043683290, 1.056981683, -1.849385142, -1.340586066);
    L2_1 += L1_1.y * min16float4(-0.024426594, -0.145882696, -1.247046232, 1.263586879);
    L2_2 += L1_1.y * min16float4(-0.270830810, 0.119935840, -1.311672688, -1.643486500);
    L2_0 += L1_1.z * min16float4(0.222858787, 0.002627724, -0.047311105, -0.508502781);
    L2_1 += L1_1.z * min16float4(0.425284505, -0.765440702, -0.231615975, 0.364498079);
    L2_2 += L1_1.z * min16float4(-0.473862469, -0.175328150, 0.200376466, -0.170545414);
    L2_0 += L1_1.w * min16float4(-0.048534606, -0.462668061, -0.018056788, 0.158020362);
    L2_1 += L1_1.w * min16float4(0.222185135, -0.024649546, 0.664888382, 0.522409737);
    L2_2 += L1_1.w * min16float4(0.326092124, 0.249779537, -0.228427365, -0.095221654);
    L2_0 += L1_2.x * min16float4(-1.483291268, -0.399437845, 1.759939194, 0.717141092);
    L2_1 += L1_2.x * min16float4(0.051209208, 0.305358291, 0.262363881, -0.222656667);
    L2_2 += L1_2.x * min16float4(-0.319517046, -0.355080426, -0.786878586, 0.965228558);
    L2_0 += L1_2.y * min16float4(-0.006965798, 0.087671049, 0.065428883, -0.080086760);
    L2_1 += L1_2.y * min16float4(0.561254084, -0.254941225, -2.439566135, -0.202635348);
    L2_2 += L1_2.y * min16float4(-0.295663029, -0.660095990, 0.552567244, -0.123915620);
    L2_0 += L1_2.z * min16float4(0.160122022, -0.143500865, -0.316779077, 0.126877829);
    L2_1 += L1_2.z * min16float4(-0.455035418, -0.346134901, -0.590004444, -0.225798979);
    L2_2 += L1_2.z * min16float4(-0.056098495, 0.286984503, -0.875478148, 0.122943603);
    L2_0 += L1_2.w * min16float4(0.114788808, -0.267752349, -0.108699806, -0.354385585);
    L2_1 += L1_2.w * min16float4(-0.320654094, -0.292018741, -0.150398821, -0.067541674);
    L2_2 += L1_2.w * min16float4(-0.444056034, 0.303410649, 0.512069345, 0.157081172);

    // PReLU 2
    L2_0 = PReLU4(L2_0, min16float4(-1.785561442, -0.498042375, -0.242182761, -1.958031893));
    L2_1 = PReLU4(L2_1, min16float4(-5.690943241, -0.729003668, -0.219683945, -3.025264025));
    L2_2 = PReLU4(L2_2, min16float4(-2.192382336, -1.276227236, -0.214304626, -2.779862165));

    // Layer 3: Output projection (12 -> 1), fused FP16 vector FMAs + single reduction
    min16float4 outAcc = L2_0 * min16float4(-1.723722577, -1.003309250, 1.881006837, 0.553851724);
    outAcc += L2_1 * min16float4(-0.404766232, 0.851177633, 0.785482407, -0.269340128);
    outAcc += L2_2 * min16float4(-0.382485956, 0.502349734, -0.833594918, 1.040693045);
    return (min16float)0.041042306 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
min16float RunModelB(
    min16float p0, min16float p1, min16float p2,
    min16float p3, min16float p4, min16float p5,
    min16float p6, min16float p7, min16float p8)
{
    // Layer 1: 3x3 Conv (1 -> 12 channels), pre-unrolled FP16 vector FMAs
    min16float4 L1_0 = min16float4(0.076806329, -0.002794162, 0.061713792, -0.002923975);
    min16float4 L1_1 = min16float4(0.026175125, -0.153714553, 0.074806444, -0.009952844);
    min16float4 L1_2 = min16float4(-0.002780868, -0.010205056, -0.021362821, 0.022398105);

    L1_0 += p0 * min16float4(-0.180299282, 0.014662747, 0.062037338, 0.162913963);
    L1_1 += p0 * min16float4(-0.096845955, -0.120704472, 0.018369380, 0.099958651);
    L1_2 += p0 * min16float4(-0.069051631, 0.084565893, 0.073071629, 0.168920413);
    L1_0 += p1 * min16float4(-0.040158097, -0.134107947, 0.069352902, 0.256246716);
    L1_1 += p1 * min16float4(-0.170313105, -0.429769307, -0.140258655, -0.072976060);
    L1_2 += p1 * min16float4(0.006422973, -0.032386098, 0.088514760, -0.294761181);
    L1_0 += p2 * min16float4(-0.186079368, 0.002835276, 0.152039975, 0.139015242);
    L1_1 += p2 * min16float4(-0.001664574, 0.045396138, 0.168260053, -0.068442158);
    L1_2 += p2 * min16float4(0.247336522, 0.189809218, -0.182572842, 0.225187257);
    L1_0 += p3 * min16float4(-0.070876308, 0.080869898, 0.216372654, -0.001347039);
    L1_1 += p3 * min16float4(0.298240304, -0.487868726, -0.011222760, 0.326351106);
    L1_2 += p3 * min16float4(-0.563833356, 0.083959579, -0.075837053, -0.235026225);
    L1_0 += p4 * min16float4(0.436923504, 0.340757638, -0.517588019, 0.083469912);
    L1_1 += p4 * min16float4(-0.197693869, 0.114102028, 0.453443885, -0.177208096);
    L1_2 += p4 * min16float4(0.035827920, -0.366167694, 0.453837216, 0.398048311);
    L1_0 += p5 * min16float4(-0.022785911, -0.090594806, 0.096829072, -0.151360869);
    L1_1 += p5 * min16float4(-0.159646124, 0.201200619, -0.038751587, 0.155050546);
    L1_2 += p5 * min16float4(0.400151104, -0.125081092, -0.035440426, -0.357304394);
    L1_0 += p6 * min16float4(0.123323008, 0.150079280, -0.071160279, -0.032094222);
    L1_1 += p6 * min16float4(-0.247537255, 0.395797133, -0.195361555, -0.067078769);
    L1_2 += p6 * min16float4(-0.094768256, 0.182795376, -0.047163446, 0.144929826);
    L1_0 += p7 * min16float4(0.147065058, 0.056526992, -0.022634532, -0.406054407);
    L1_1 += p7 * min16float4(0.090287514, -0.481248379, -0.089255974, -0.089840382);
    L1_2 += p7 * min16float4(-0.188313007, 0.200342402, -0.209137321, -0.289527595);
    L1_0 += p8 * min16float4(-0.157484218, -0.062786549, 0.087097272, -0.055152047);
    L1_1 += p8 * min16float4(-0.246287793, -0.003776026, -0.153421938, 0.065058291);
    L1_2 += p8 * min16float4(0.229671970, 0.031603150, 0.160108879, 0.226898223);

    // PReLU 1
    L1_0 = PReLU4(L1_0, min16float4(-1.442547083, -1.845595598, -1.306786060, -0.781065464));
    L1_1 = PReLU4(L1_1, min16float4(-0.109224759, -0.269420534, -0.060583767, 4.149562359));
    L1_2 = PReLU4(L1_2, min16float4(-0.792430758, -1.190022588, 0.142513722, 0.115136072));

    // Layer 2: 1x1 Dense (12 -> 12 channels), column-major vector FMAs (HFMA2-friendly)
    min16float4 L2_0 = min16float4(0.017644040, -0.120309018, -0.029212048, 0.068806812);
    min16float4 L2_1 = min16float4(0.053569581, -0.141184986, 0.036499061, -0.027951837);
    min16float4 L2_2 = min16float4(0.221204460, -0.030860605, 0.061900575, 0.100701667);

    L2_0 += L1_0.x * min16float4(-0.229082733, 0.480870575, 0.789031744, -0.600302875);
    L2_1 += L1_0.x * min16float4(-0.351149797, 0.535218537, 0.165266067, -0.848744035);
    L2_2 += L1_0.x * min16float4(-0.475019783, 0.582823992, 0.399218798, -0.224198133);
    L2_0 += L1_0.y * min16float4(-0.512871683, 0.391051710, -0.195746094, 0.046118792);
    L2_1 += L1_0.y * min16float4(0.594739795, 0.025757920, -0.195548356, -0.241835013);
    L2_2 += L1_0.y * min16float4(0.175065041, -0.069464915, 0.489937872, -0.468186438);
    L2_0 += L1_0.z * min16float4(0.736965120, -0.417410672, -0.737288594, 0.435882509);
    L2_1 += L1_0.z * min16float4(-0.068887308, -0.459538639, -0.235718101, -0.068041831);
    L2_2 += L1_0.z * min16float4(0.764335096, -0.495929092, -0.335512757, 0.389229238);
    L2_0 += L1_0.w * min16float4(-0.114540882, 0.369169980, -0.646142125, -0.587501466);
    L2_1 += L1_0.w * min16float4(0.360132426, 0.198079765, 0.526902139, -0.782595038);
    L2_2 += L1_0.w * min16float4(-0.890451252, 0.401011854, -0.769207895, 0.309193313);
    L2_0 += L1_1.x * min16float4(0.311843961, 0.009561459, 0.175059706, 0.547766745);
    L2_1 += L1_1.x * min16float4(-0.879847527, -0.176006883, -0.411046326, -0.665538311);
    L2_2 += L1_1.x * min16float4(0.180819586, 0.271378279, 0.378717065, -0.264145225);
    L2_0 += L1_1.y * min16float4(0.229555234, -0.459208190, 0.691861987, 0.373978168);
    L2_1 += L1_1.y * min16float4(0.880598426, -0.002988719, -0.018310603, -0.007513349);
    L2_2 += L1_1.y * min16float4(-0.296335191, 0.528470814, 0.111551061, 0.326991796);
    L2_0 += L1_1.z * min16float4(-0.347491890, 0.736723900, 0.824216008, -0.503097653);
    L2_1 += L1_1.z * min16float4(-1.479037523, 0.832145870, 0.489419967, -0.313240737);
    L2_2 += L1_1.z * min16float4(-0.511490107, 0.503377557, 0.790561855, -0.568252385);
    L2_0 += L1_1.w * min16float4(-0.145946071, 0.021142924, -0.017000724, -0.090789184);
    L2_1 += L1_1.w * min16float4(-0.044767305, 0.274749726, -0.265991926, -0.339458108);
    L2_2 += L1_1.w * min16float4(0.079025216, 0.125316218, -0.215333298, 0.097437136);
    L2_0 += L1_2.x * min16float4(-0.323113024, 0.269746661, -0.435575575, -0.345212072);
    L2_1 += L1_2.x * min16float4(0.112935498, -0.025276352, -0.114177100, 0.182142422);
    L2_2 += L1_2.x * min16float4(-0.549045265, 0.272755206, -0.553730965, 0.228109702);
    L2_0 += L1_2.y * min16float4(0.993536174, -0.397482157, -0.646290779, 0.522237957);
    L2_1 += L1_2.y * min16float4(-0.615179121, -0.236055046, -1.120269060, -0.052835885);
    L2_2 += L1_2.y * min16float4(0.121311098, -0.557049394, -0.940150917, 0.454378009);
    L2_0 += L1_2.z * min16float4(0.120587423, 0.399009675, 0.176628053, -0.069926046);
    L2_1 += L1_2.z * min16float4(-0.769895375, 0.435621321, -0.591441274, -0.442020327);
    L2_2 += L1_2.z * min16float4(-0.427249581, -0.097753078, 0.235617280, -0.301024586);
    L2_0 += L1_2.w * min16float4(0.773548186, 3.502569199, -3.576125145, -1.903903365);
    L2_1 += L1_2.w * min16float4(-0.700175762, -1.177589297, -2.321050406, -1.241039634);
    L2_2 += L1_2.w * min16float4(0.354185373, -2.256846189, 1.546710849, -3.406354189);

    // PReLU 2
    L2_0 = PReLU4(L2_0, min16float4(-1.701528668, -2.895941973, -0.477188736, -0.332290858));
    L2_1 = PReLU4(L2_1, min16float4(0.147276983, -0.674965382, 0.295771748, -0.229368791));
    L2_2 = PReLU4(L2_2, min16float4(-0.975452960, -3.820577860, -1.948044658, -1.310231686));

    // Layer 3: Output projection (12 -> 1), fused FP16 vector FMAs + single reduction
    min16float4 outAcc = L2_0 * min16float4(-0.875323415, 1.467422843, -1.980630398, -0.803634167);
    outAcc += L2_1 * min16float4(-1.479813099, -1.135522127, -0.968504190, 0.906366646);
    outAcc += L2_2 * min16float4(-1.197628736, 1.004730344, 1.426059246, -2.517558813);
    return (min16float)0.065220654 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

min16float RunNeuralNet(
    min16float p0, min16float p1, min16float p2,
    min16float p3, min16float p4, min16float p5,
    min16float p6, min16float p7, min16float p8)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#else
    min16float res = (min16float)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    else
        res = RunModelB(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    return res;
#endif
}


// ============================================================================
// 6. PASSES & TECHNIQUE
// ============================================================================

#if NSHDR_USE_PQ_LUT
groupshared bool gs_PQLUTSkip;

void CS_InitPQLUT(uint3 tid : SV_DispatchThreadID)
{
    // Single-lane skip/readiness check before unconditional barrier:
    // guarantees uniform sync convergence under FXC (no X3663), DXC, and Vulkan.
    if (tid.x == 0u)
    {
        bool skipInit = !IsShaderActive() || (ResolveColorSpaceMode() == MODE_SDR);
        if (!skipInit)
            skipInit = (tex2Dfetch(StoragePQToLinearLUT, int2(1023, 0)) > 0.5);
        gs_PQLUTSkip = skipInit;
    }
    barrier();

    if (gs_PQLUTSkip)
        return;

    [unroll]
    for (int k = 0; k < 8; ++k)
    {
        int idx = int(tid.x) * 8 + k;
        float pq = float(idx) / 1023.0;
        tex2Dstore(StoragePQToLinearLUT, int2(idx, 0), PQToLinearScalar(pq));
    }
}
#endif

void VS_GetLuma(
    in uint id : SV_VertexID,
    out float4 position : SV_Position,
    out float2 texcoord : TEXCOORD0,
    out nointerpolation int outMode : TEXCOORD1)
{
    outMode = ResolveColorSpaceMode();
    if (!IsShaderActive())
    {
        position = float4(0.0, 0.0, 0.0, 0.0);
        texcoord = float2(0.0, 0.0);
        return;
    }

    texcoord.x = (id == 2u) ? 2.0 : 0.0;
    texcoord.y = (id == 1u) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

void PS_GetLuma(
    float4 vpos : SV_Position,
    float2 uv : TEXCOORD0,
    nointerpolation int mode : TEXCOORD1,
    out float outLuma : SV_Target)
{
    float3 c = tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0, 0.0)).rgb;

#if NSHDR_PIPELINE_COMPILED
    [branch]
    if (mode == MODE_PQ)
    {
        // True Rec.2020 linear luminance re-encoded to perceptual ST.2084 PQ [0, 1].
        float3 cLin = PQToLinearFast3(c);
        float yLin = max(dot(cLin, LUMA_2020), 0.0);
        outLuma = LinearToPQScalar(yLin);
    }
    else if (mode == MODE_SCRGB)
    {
        // Signed BT.709 dot product gives exact luminance even for wide-gamut Rec.2020 scRGB colors.
        float yLin = max(dot(c, LUMA_709), 0.0) * SCRGB_TO_PQ_LINEAR;
        outLuma = LinearToPQScalar(yLin);
    }
    else
    {
        outLuma = dot(saturate(c), LUMA_709);
    }
#else
    outLuma = dot(saturate(c), LUMA_709);
#endif
}

void VS_Apply(
    in uint id : SV_VertexID,
    out float4 position : SV_Position,
    out float2 uv : TEXCOORD0,
    out float2 gatherUV : TEXCOORD1,
    out nointerpolation int outMode : TEXCOORD2,
    out nointerpolation float4 outSuppressParams : TEXCOORD3)
{
    outMode = ResolveColorSpaceMode();
    if (!IsShaderActive())
    {
        position = float4(0.0, 0.0, 0.0, 0.0);
        uv = float2(0.0, 0.0);
        gatherUV = float2(0.0, 0.0);
        outSuppressParams = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    uv.x = (id == 2u) ? 2.0 : 0.0;
    uv.y = (id == 1u) ? 2.0 : 0.0;
    // Half-texel shift aligns GatherRed(gatherUV) so .w=(x-1,y-1), .z=(x,y-1), .x=(x-1,y), .y=(x,y).
    gatherUV = uv - 0.5 * BUFFER_PIXEL_SIZE;

    // (aliasStart, aliasScale, fireflyStart, fireflyScale)
    // HDR PQ has ~0.7x the per-stop slope of sRGB, so PQ thresholds are scaled to match perceptual stops.
    if (outMode == MODE_SDR)
        outSuppressParams = float4(0.25, 4.0, 0.10, 6.0);
    else
        outSuppressParams = float4(0.18, 5.0, 0.075, 8.0);

    position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

void PS_Apply(
    float4 vpos : SV_Position,
    float2 uv : TEXCOORD0,
    float2 gatherUV : TEXCOORD1,
    nointerpolation int mode : TEXCOORD2,
    nointerpolation float4 suppressParams : TEXCOORD3,
    out float4 outColor : SV_Target)
{
    // 4 tex2DgatherR fetches load the entire 3x3 luma neighborhood (vs 13 scalar tex2D fetches).
    float4 g00 = tex2DgatherR(sTexLuma, gatherUV);
    float4 g10 = tex2DgatherR(sTexLuma, gatherUV, int2(1, 0));
    float4 g01 = tex2DgatherR(sTexLuma, gatherUV, int2(0, 1));
    float4 g11 = tex2DgatherR(sTexLuma, gatherUV, int2(1, 1));

    // Row -1: TL, T(N), TR | Row 0: L(W), C, R(E) | Row +1: BL, B(S), BR
    float p0 = g00.w; float p1 = g00.z; float p2 = g10.z;
    float p3 = g00.x; float p4 = g00.y; float p5 = g10.y;
    float p6 = g01.x; float p7 = g01.y; float p8 = g11.y;

    // Cross-neighborhood stats (N=p1, S=p7, W=p3, E=p5) reused for both range & artifact protection.
    float maxCross = max(max(p1, p7), max(p3, p5));
    float minCross = min(min(p1, p7), min(p3, p5));
    float maxDiag  = max(max(p0, p2), max(p6, p8));
    float minDiag  = min(min(p0, p2), min(p6, p8));

    float range3x3 = max(max(maxCross, maxDiag), p4) - min(min(minCross, minDiag), p4);

    // Pre-CNN artifact protection (aliasing high-contrast edge + isolated specular firefly suppression).
    float localContrast  = maxCross - minCross;
    float aliasingSignal = saturate((localContrast - suppressParams.x) * suppressParams.y);

    float brightSpike    = max(0.0, p4 - maxCross);
    float darkSpike      = max(0.0, minCross - p4);
    float spikeIntensity = max(brightSpike, darkSpike);
    float fireflySignal  = saturate((spikeIntensity - suppressParams.z) * suppressParams.w);

    float artifactSignal  = saturate(aliasingSignal + fireflySignal);
    float suppressionMask = 1.0 - artifactSignal * ArtifactProtection;

    // Flat-region fade (0 below FLAT_SKIP_RANGE -> 1 at FLAT_FADE_END) eliminates CNN DC bias on OLED black/flat skies.
    float flatMask = saturate((range3x3 - FLAT_SKIP_RANGE) * (1.0 / (FLAT_FADE_END - FLAT_SKIP_RANGE)));

    // Warp-coherent pre-CNN early-out: skips the entire 12-channel CNN and HDR re-encoding on flat/black/suppressed tiles.
    [branch]
    if ((ViewMode == 0) && ((range3x3 <= FLAT_SKIP_RANGE) || (suppressionMask <= 0.001)))
    {
        outColor = float4(tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0, 0.0)).rgb, 1.0);
        return;
    }

    // Run 12-channel CNN in native FP16 (min16float4)
    float rawResidual = (float)RunNeuralNet(
        (min16float)p0, (min16float)p1, (min16float)p2,
        (min16float)p3, (min16float)p4, (min16float)p5,
        (min16float)p6, (min16float)p7, (min16float)p8);

    float normalizedResidual = saturate(abs(rawResidual) * 2.63); // 2.63 ~ 1 / 0.38

    [branch]
    if (ViewMode == 2)
    {
        outColor = float4(EncodeDebugGray(normalizedResidual * flatMask, mode), 1.0);
        return;
    }

    float residual = clamp(rawResidual, -0.5, 0.5) * 0.22;

    if (abs(EdgeResponse) > 0.001)
    {
        float adaptationMask = (EdgeResponse < 0.0)
            ? lerp(1.0, 1.0 - normalizedResidual, -EdgeResponse)
            : lerp(1.0, 1.0 + normalizedResidual,  EdgeResponse);
        residual *= adaptationMask;
    }

    [branch]
    if (ViewMode == 1)
    {
        float debugRes = residual * flatMask * suppressionMask * Intensity;
        outColor = float4(EncodeDebugGray(0.5 + debugRes, mode), 1.0);
        return;
    }

    float delta = residual * flatMask * suppressionMask * Intensity;
    float3 origColor = tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0, 0.0)).rgb;

#if NSHDR_PIPELINE_COMPILED
    [branch]
    if (mode == MODE_PQ || mode == MODE_SCRGB)
    {
        if (abs(delta) < 1e-6)
        {
            outColor = float4(origColor, 1.0);
            return;
        }

        // Protect near-black (< ~0.01 nits) from positive multiplicative noise blow-up while keeping darkening free.
        float blackGuard = (delta > 0.0) ? saturate(p4 * 50.0) : 1.0;
        float targetPQ = saturate(p4 + delta * blackGuard);

        float yOld = PQToLinearFast(p4);
        float yNew = PQToLinearFast(targetPQ);
        float ratio = (yOld > 1e-7) ? clamp(yNew / yOld, 0.0, 4.0) : 1.0;

        if (mode == MODE_PQ)
        {
            float3 cLin = PQToLinearFast3(origColor);
            // Prevent channel clipping hue shifts near 10,000 nits.
            float maxCh = max(cLin.r, max(cLin.g, cLin.b));
            if ((ratio > 1.0) && (maxCh * ratio > 1.0))
                ratio = 1.0 / max(maxCh, 1e-6);

            outColor = float4(LinearToPQ3(cLin * ratio), 1.0);
        }
        else
        {
            // scRGB linear: direct luminance ratio scaling preserves signed wide-gamut negative channels intact.
            outColor = float4(origColor * ratio, 1.0);
        }
        return;
    }
#endif

    // SDR (sRGB): adding luma delta directly to RGB is algebraically identical to YCbCr Y-only sharpening
    // without YCbCr matrix round-trip overhead or rounding drift.
    outColor = float4(max(0.0, origColor + delta), 1.0);
}


technique Neural_Sharpen_HDR <
    ui_label = "Neural Sharpen HDR";
    ui_tooltip = "HDR-aware, GPU-optimized 12-channel Neural Sharpening (based on Barbatos Neural Sharpen 2.8, MIT). Supports HDR10 PQ, scRGB, and SDR.";
>
{
#if NSHDR_USE_PQ_LUT
    pass InitPQLUT
    {
        ComputeShader = CS_InitPQLUT<128, 1, 1>;
        DispatchSizeX = 1;
        DispatchSizeY = 1;
    }
#endif
    pass GetLuma
    {
        VertexShader = VS_GetLuma;
        PixelShader  = PS_GetLuma;
        RenderTarget = TexLuma;
    }
    pass Apply
    {
        VertexShader = VS_Apply;
        PixelShader  = PS_Apply;
    }
}

} // namespace NeuralSharpenHDR
