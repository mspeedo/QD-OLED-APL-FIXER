/*
    Neural Sharpen HDR
    Single-pass, HDR-aware, GPU-optimized standalone adaptation of Barbatos Neural Sharpening (CNN v2.8).

    Credits & License:
    - Original Neural Sharpening 12-channel CNN architecture and pretrained Model A / Model B weights:
      Barbatos (https://github.com/BarbatosAWLS/Reshade-Shaders), MIT License.
    - Single-pass HDR PQ/scRGB pipeline, 0-pow rational PQ approximations, OLED black-level protection,
      D4-symmetrized distilled compact neural profiles (Fast5 / Fast9), POINT/bilinear sampling optimization,
      and compile-time specialization: MIT License.

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

    Performance Profiles (Preprocessor NEURAL_SHARPEN_PROFILE):
      0 = Ultra Fast 5-Tap Bilinear (default): 5 TEX fetches covering full 3x3 via 4 bilinear quadrants,
          distilled 5->4->4->1 CNN (40 FMAs, 6.6x less ALU, 44% fewer TEX fetches).
      1 = Balanced Fast 9-Tap: 9 POINT TEX fetches, distilled 9->8->4->1 CNN (108 FMAs, 2.5x less ALU, R2=0.97).
      2 = Exact 12-Channel: 9 POINT TEX fetches, bit-exact Barbatos 1->12->12->1 CNN (264 FMAs).

    File layout:
      1. Compile-Time Preprocessor Switches
      2. UI Controls, Samplers & Constants
      3. Color Science & 0-pow Fast PQ Math
      4. Neural Network Profiles (Fast5 / Fast9 / Exact12)
      5. Single-Pass Shader & Technique
*/

#include "ReShade.fxh"


// ============================================================================
// 1. COMPILE-TIME PREPROCESSOR SWITCHES
// ============================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 0
#endif

// 0 = Ultra Fast 5-Tap Bilinear Neural (5 TEX, 40 FMAs - default, fastest)
// 1 = Balanced Fast 9-Tap Neural (9 POINT TEX, 108 FMAs, R2=0.97 vs Exact)
// 2 = Exact 12-Channel Barbatos Neural (9 POINT TEX, 264 FMAs)
#ifndef NEURAL_SHARPEN_PROFILE
    #define NEURAL_SHARPEN_PROFILE 0
#endif

// 1 = Compile-time Model A only (Balanced - default, smallest binary & lowest register pressure)
// 2 = Compile-time Model B only (Edge-Enhanced)
// 0 = Both models compiled with runtime UI switcher
#ifndef NEURAL_SHARPEN_MODEL
    #define NEURAL_SHARPEN_MODEL 1
#endif

// 0 = Strip debug views at compile time (default)
// 1 = Enable Sharpening Map and Residual Magnitude debug views in UI
#ifndef ENABLE_DEBUG_VIEWS
    #define ENABLE_DEBUG_VIEWS 0
#endif

// 0 = Compile ONLY detected BUFFER_COLOR_SPACE when known (1=SDR, 2=scRGB, 3=HDR10 PQ).
//     If BUFFER_COLOR_SPACE == 0 (Unknown), runtime Color Space dropdown is shown automatically.
// 1 = Always compile all color spaces and expose the runtime Color Space selector.
#ifndef ALLOW_MANUAL_COLOR_SPACE
    #define ALLOW_MANUAL_COLOR_SPACE 0
#endif

#if (BUFFER_COLOR_SPACE == 0) || (ALLOW_MANUAL_COLOR_SPACE != 0)
    #define NSHDR_DYNAMIC_CSPACE 1
#else
    #define NSHDR_DYNAMIC_CSPACE 0
#endif

// 0 = Full FP32 float4 inside CNN (default - faster on D3D11 SM5.0 / NVIDIA Ada)
// 1 = min16float4 / mediump vec4 inside CNN
#ifndef USE_FP16_CNN
    #define USE_FP16_CNN 0
#endif

#if USE_FP16_CNN
    #define NS_F1 min16float
    #define NS_F4 min16float4
#else
    #define NS_F1 float
    #define NS_F4 float4
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
// 2. UI CONTROLS, SAMPLERS & CONSTANTS
// ============================================================================

#if NSHDR_DYNAMIC_CSPACE
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
    NSHDR_TOOLTIP("Model A: Balanced detail sharpening. Model B: Edge-enhanced. Set NEURAL_SHARPEN_MODEL=1 or 2 for smallest binary.")
> = 0;
#endif

uniform float Intensity <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Intensity";
    NSHDR_TOOLTIP("Overall neural sharpening strength. At 0.0, the pass is vertex-culled for zero shader cost.")
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

#if ENABLE_DEBUG_VIEWS
uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Sharpening Map\0Residual Magnitude\0";
    ui_label = "Debug View";
    NSHDR_TOOLTIP("Normal: Final sharpened output. Sharpening Map: Signed delta around mid-gray. Residual Magnitude: Raw neural response (tone-mapped to ~203 nits in HDR).")
> = 0;
#endif

// Dedicated POINT sampler for exact 1-texel BackBuffer loads (avoids 2x2 bilinear L1 tag lookup).
sampler sBackBufferPoint
{
    Texture = ReShade::BackBufferTex;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

#define MODE_SDR   1
#define MODE_SCRGB 2
#define MODE_PQ    3

// Flat-patch fade thresholds (eliminates DC bias on OLED pure black and flat skies).
#if NEURAL_SHARPEN_PROFILE == 0
static const float FLAT_SKIP_RANGE = 0.00035;
static const float FLAT_INV_SPAN   = 1.0 / (0.00180 - 0.00035);
#else
static const float FLAT_SKIP_RANGE = 0.00050;
static const float FLAT_INV_SPAN   = 1.0 / (0.00250 - 0.00050);
#endif

static const float3 LUMA_709  = float3(0.2126, 0.7152, 0.0722);
static const float3 LUMA_2020 = float3(0.2627, 0.6780, 0.0593);
static const float SCRGB_TO_PQ_LINEAR = 80.0 / 10000.0;


// ============================================================================
// 3. COLOR SCIENCE & 0-POW FAST PQ MATH
// ============================================================================

#if NSHDR_DYNAMIC_CSPACE
int ResolveColorSpaceMode()
{
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
}
#endif

bool IsShaderActive()
{
#if ENABLE_DEBUG_VIEWS
    return (Intensity > 0.0001) || (ViewMode != 0);
#else
    return (Intensity > 0.0001);
#endif
}

#if NSHDR_DYNAMIC_CSPACE || (BUFFER_COLOR_SPACE == 2) || ENABLE_DEBUG_VIEWS
// 0-pow ST.2084 Linear [0, 1] -> PQ [0, 1] rational fit on u = L^0.25 (2 sqrts + 5 FMAs + 1 rcp).
float FastLinearToPQ(float linearVal)
{
    float u = sqrt(sqrt(saturate(linearVal)));
    float num = u * (-0.937040 + u * (60.908867 + u * 3.589653));
    float den = 1.0 + u * (20.243761 + u * 42.309906);
    return saturate(num / den);
}
#endif

#if NSHDR_DYNAMIC_CSPACE || (BUFFER_COLOR_SPACE == 2)
// 0-pow, 0-sqrt ST.2084 PQ [0, 1] -> Linear [0, 1] rational fit via u^4 (5 FMAs + 1 rcp + 2 muls).
float FastPQToLinear(float pq)
{
    float v = saturate(pq);
    float u = max(0.0, (v * (1.811813 + v * (8.965278 + v * 1.185166))) / (1.0 + v * (27.571310 - v * 16.625550)));
    float u2 = u * u;
    return u2 * u2;
}
#endif

#if ENABLE_DEBUG_VIEWS
float3 EncodeDebugGray(float gray, int mode)
{
    float g = saturate(gray);
    float lin = g * g;
    float3 result = g.xxx;

    if (mode == MODE_PQ)
        result = FastLinearToPQ(lin * (203.0 / 10000.0)).xxx;
    else if (mode == MODE_SCRGB)
        result = (lin * (203.0 / 80.0)).xxx;

    return result;
}
#endif

float ExtractLuma(float3 c, int mode)
{
#if !NSHDR_DYNAMIC_CSPACE
    #if BUFFER_COLOR_SPACE == 3
        return dot(saturate(c), LUMA_2020);
    #elif BUFFER_COLOR_SPACE == 2
        return FastLinearToPQ(max(dot(c, LUMA_709), 0.0) * SCRGB_TO_PQ_LINEAR);
    #else
        return dot(saturate(c), LUMA_709);
    #endif
#else
    float luma = dot(saturate(c), LUMA_709);
    if (mode == MODE_PQ)
        luma = dot(saturate(c), LUMA_2020);
    else if (mode == MODE_SCRGB)
        luma = FastLinearToPQ(max(dot(c, LUMA_709), 0.0) * SCRGB_TO_PQ_LINEAR);
    return luma;
#endif
}


// ============================================================================
// 4. NEURAL NETWORK PROFILES (FAST5 / FAST9 / EXACT12)
// ============================================================================

NS_F4 PReLU4(NS_F4 v, NS_F4 a)
{
    const NS_F4 z = NS_F4(0.0, 0.0, 0.0, 0.0);
    return (v >= z) ? v : (v * a);
}

#if NEURAL_SHARPEN_PROFILE == 0

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 1)
NS_F1 RunModelA_Fast5(NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3)
{
    // Layer 1: 5 -> 4 (5 float4 FMAs)
    NS_F4 L1 = NS_F4(0.044041615, -0.464537889, 0.254479736, -0.278363824);
    L1 += c * NS_F4(-0.724115193, 0.554412425, 0.006577393, 1.313981652);
    L1 += d0 * NS_F4(0.470940173, -1.002664804, -2.012996912, 1.113625169);
    L1 += d1 * NS_F4(-0.408214957, -0.990056753, -1.895673871, 0.969342709);
    L1 += d2 * NS_F4(-0.395975262, -0.965034127, -1.898413658, 1.011839151);
    L1 += d3 * NS_F4(0.428595275, -0.971805334, -2.014953375, 1.168593287);
    L1 = PReLU4(L1, NS_F4(0.231460378, -0.131050378, -0.300561428, 2.181897640));

    // Layer 2: 4 -> 4 (4 float4 FMAs)
    NS_F4 L2 = NS_F4(-0.098853096, 0.261688620, -0.391664416, 0.305713117);
    L2 += L1.x * NS_F4(0.620059669, -0.060512822, -0.479162425, 2.558356047);
    L2 += L1.y * NS_F4(-2.420872450, 0.023130698, -0.021687264, -0.769975364);
    L2 += L1.z * NS_F4(2.035257816, -0.618476212, 0.529927671, 0.751240253);
    L2 += L1.w * NS_F4(0.503898799, 0.197636768, -0.269512057, -0.195414096);
    L2 = PReLU4(L2, NS_F4(-0.374701828, -1.780340195, -0.111108810, -2.096511126));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(0.197443008, -0.337130219, 0.244023353, 0.102219202);
    return (NS_F1)-0.070484601 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
NS_F1 RunModelB_Fast5(NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3)
{
    // Layer 1: 5 -> 4 (5 float4 FMAs)
    NS_F4 L1 = NS_F4(0.262699813, 0.175043285, 0.004409165, -0.493532926);
    L1 += c * NS_F4(-1.006822586, 0.042879995, -0.003238662, 0.011929572);
    L1 += d0 * NS_F4(-0.978823006, 3.065347433, -0.606804430, 1.061139107);
    L1 += d1 * NS_F4(-0.591226280, 1.846954226, 4.010735035, -0.184501663);
    L1 += d2 * NS_F4(-0.661103189, 1.885742188, -2.807567120, 3.971651554);
    L1 += d3 * NS_F4(-0.923658133, 3.059687614, -0.590139270, 1.153838634);
    L1 = PReLU4(L1, NS_F4(0.143238261, 0.311831802, -1.398360968, 0.028653659));

    // Layer 2: 4 -> 4 (4 float4 FMAs)
    NS_F4 L2 = NS_F4(0.233327508, 0.351621002, -0.161384359, 0.018692229);
    L2 += L1.x * NS_F4(-0.926041603, -0.786513031, 0.045969095, -0.981017470);
    L2 += L1.y * NS_F4(-1.331356406, -1.728933334, 0.977817178, 0.859942198);
    L2 += L1.z * NS_F4(-1.405285001, 0.678171337, -3.152433634, -0.065504521);
    L2 += L1.w * NS_F4(1.738002658, -0.383949339, 0.960667074, -2.419680595);
    L2 = PReLU4(L2, NS_F4(-0.496564537, 0.031802882, 0.033218812, -0.767354369));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(-0.429979295, 0.441342771, 0.382009625, -0.299290389);
    return (NS_F1)0.058439214 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

NS_F1 RunNeuralFast5(NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA_Fast5(c, d0, d1, d2, d3);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB_Fast5(c, d0, d1, d2, d3);
#else
    NS_F1 res = (NS_F1)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA_Fast5(c, d0, d1, d2, d3);
    else
        res = RunModelB_Fast5(c, d0, d1, d2, d3);
    return res;
#endif
}

#elif NEURAL_SHARPEN_PROFILE == 1

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 1)
NS_F1 RunModelA_Fast9(
    NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3,
    NS_F1 d4, NS_F1 d5, NS_F1 d6, NS_F1 d7)
{
    // Layer 1: 9 -> 8 (18 float4 FMAs)
    NS_F4 L1_0 = NS_F4(-0.031998970, 0.031589180, -0.011161512, -0.093157247);
    NS_F4 L1_1 = NS_F4(0.082864963, -0.527208328, 0.031276781, 0.074128695);
    L1_0 += c * NS_F4(0.554125190, 0.455940366, -0.302275836, -0.228231400);
    L1_1 += c * NS_F4(-0.068743333, 0.900165081, 0.074868083, -0.833535612);
    L1_0 += d0 * NS_F4(0.200752094, 1.045886517, 0.338040113, -0.223041028);
    L1_1 += d0 * NS_F4(1.183322906, -0.258691579, -1.151645422, 0.647287846);
    L1_0 += d1 * NS_F4(-1.536124468, 0.091142580, -0.972179592, -0.823013783);
    L1_1 += d1 * NS_F4(-2.458814144, -0.069077872, 1.368291497, -0.705865324);
    L1_0 += d2 * NS_F4(0.180009082, 1.079992414, -0.721875131, 0.856232107);
    L1_1 += d2 * NS_F4(1.156480789, -0.145132154, -1.143059969, 0.665071130);
    L1_0 += d3 * NS_F4(-1.539264798, 0.052741200, 0.811920881, -1.010602832);
    L1_1 += d3 * NS_F4(-2.446686983, -0.204797328, 1.362010717, -0.998185039);
    L1_0 += d4 * NS_F4(-1.570204258, -0.048205778, -0.968622923, 0.255360544);
    L1_1 += d4 * NS_F4(-2.423376799, -0.274419278, 1.313420057, -1.262315273);
    L1_0 += d5 * NS_F4(0.137203023, 1.057691097, 1.330402970, 0.924584925);
    L1_1 += d5 * NS_F4(1.116716743, -0.446109384, -1.149516821, -0.004869843);
    L1_0 += d6 * NS_F4(-1.557010531, -0.085837789, 0.027410431, -0.193212390);
    L1_1 += d6 * NS_F4(-2.406188250, -0.301070154, 1.350923538, -1.403443575);
    L1_0 += d7 * NS_F4(0.106081411, 1.099354863, 0.400695503, 2.062373877);
    L1_1 += d7 * NS_F4(1.117820024, -0.323592067, -1.152697325, 0.155280858);
    L1_0 = PReLU4(L1_0, NS_F4(-0.101925738, -1.594390869, -0.618606448, -0.542654216));
    L1_1 = PReLU4(L1_1, NS_F4(-0.217854485, 0.135697380, -2.421566725, -0.583996952));

    // Layer 2: 8 -> 4 (8 float4 FMAs)
    NS_F4 L2 = NS_F4(-0.300556898, 0.461136729, -0.703779042, 0.221698195);
    L2 += L1_0.x * NS_F4(0.601879120, 0.807774723, 0.855787218, -0.250622272);
    L2 += L1_0.y * NS_F4(0.653280914, -0.575191021, 0.231344372, -0.195182651);
    L2 += L1_0.z * NS_F4(-0.316202760, -0.207855701, 0.268863440, 0.496142805);
    L2 += L1_0.w * NS_F4(-0.203538418, -0.287749171, 0.230785713, 0.379764140);
    L2 += L1_1.x * NS_F4(0.837859094, -0.143739209, -0.316334307, -2.585501432);
    L2 += L1_1.y * NS_F4(-0.694278657, -0.697551250, -0.390520781, -0.581950366);
    L2 += L1_1.z * NS_F4(-0.900669754, -0.049759720, 0.496678412, 0.132044002);
    L2 += L1_1.w * NS_F4(-0.409483910, -0.732794225, 0.539905965, 0.354147643);
    L2 = PReLU4(L2, NS_F4(-2.338482380, -0.040564131, -1.723849773, -0.228897572));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(0.107067667, 0.416742414, -0.149709746, -0.311124414);
    return (NS_F1)-0.041589886 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
NS_F1 RunModelB_Fast9(
    NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3,
    NS_F1 d4, NS_F1 d5, NS_F1 d6, NS_F1 d7)
{
    // Layer 1: 9 -> 8 (18 float4 FMAs)
    NS_F4 L1_0 = NS_F4(0.589212298, -0.006024884, -0.408899695, -0.019932732);
    NS_F4 L1_1 = NS_F4(-0.006232130, 0.041004732, 0.010023184, -0.051134340);
    L1_0 += c * NS_F4(-0.650064826, -0.008097425, 0.007536399, 0.071693853);
    L1_1 += c * NS_F4(0.002698497, 0.000686864, -0.028130081, 0.591559589);
    L1_0 += d0 * NS_F4(0.772505283, 1.174628377, 0.467936307, 2.178745031);
    L1_1 += d0 * NS_F4(1.127161145, -1.402289987, 2.066283226, -0.846293688);
    L1_0 += d1 * NS_F4(0.350459993, -0.473415226, 1.495615721, 0.053659558);
    L1_1 += d1 * NS_F4(2.120580673, 2.854192734, -1.937204361, 1.053054333);
    L1_0 += d2 * NS_F4(0.759836614, -0.555079877, 0.486662894, 2.174031973);
    L1_1 += d2 * NS_F4(0.809313357, -1.426777482, 2.073208809, -0.683421135);
    L1_0 += d3 * NS_F4(0.331205934, 1.690163374, 1.627286077, 0.075821288);
    L1_1 += d3 * NS_F4(-0.125565350, 2.853359461, -1.921966553, 1.025691628);
    L1_0 += d4 * NS_F4(0.353371739, -2.582031727, 1.707540274, 0.043890920);
    L1_1 += d4 * NS_F4(-0.030699311, 2.884803057, -1.965960264, 1.041918635);
    L1_0 += d5 * NS_F4(0.785464168, 1.196134686, 0.302724302, 2.143142462);
    L1_1 += d5 * NS_F4(-0.940354407, -1.417217851, 2.056678295, -0.760092854);
    L1_0 += d6 * NS_F4(0.267994076, -0.437827647, 1.828844905, 0.075968511);
    L1_1 += d6 * NS_F4(-2.297657251, 2.861240864, -1.949587584, 1.025820494);
    L1_0 += d7 * NS_F4(0.797478616, -0.615675867, 0.239265487, 2.148883104);
    L1_1 += d7 * NS_F4(-0.680544019, -1.413342237, 2.064162254, -0.684442818);
    L1_0 = PReLU4(L1_0, NS_F4(0.464792371, -0.990023613, 0.275134206, 0.725356936));
    L1_1 = PReLU4(L1_1, NS_F4(-0.984553337, -0.359404445, 0.170000821, -1.091895700));

    // Layer 2: 8 -> 4 (8 float4 FMAs)
    NS_F4 L2 = NS_F4(0.208050773, 0.020678975, -0.021543082, 0.383862555);
    L2 += L1_0.x * NS_F4(-0.496527582, -0.214418501, -0.172350034, -0.752626836);
    L2 += L1_0.y * NS_F4(0.321045756, 0.286656410, -0.690625608, -0.196537599);
    L2 += L1_0.z * NS_F4(-0.359548122, -1.210520506, -0.927967787, -0.207719088);
    L2 += L1_0.w * NS_F4(-0.830253601, 1.139095426, 0.792401016, 1.094697833);
    L2 += L1_1.x * NS_F4(0.315937132, 0.275918543, -0.691122353, -0.195112929);
    L2 += L1_1.y * NS_F4(-2.133630276, 1.153848886, 0.014148850, -0.168930069);
    L2 += L1_1.z * NS_F4(2.002983570, 1.312960863, 2.198209763, -0.667793453);
    L2 += L1_1.w * NS_F4(-0.380124539, -0.444612563, -0.005287662, 0.527688920);
    L2 = PReLU4(L2, NS_F4(-0.152752519, -1.030996323, -0.005886404, -4.763410091));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(0.661731601, -0.449330628, 0.371045768, -0.084763408);
    return (NS_F1)0.037533216 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

NS_F1 RunNeuralFast9(
    NS_F1 c, NS_F1 d0, NS_F1 d1, NS_F1 d2, NS_F1 d3,
    NS_F1 d4, NS_F1 d5, NS_F1 d6, NS_F1 d7)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA_Fast9(c, d0, d1, d2, d3, d4, d5, d6, d7);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB_Fast9(c, d0, d1, d2, d3, d4, d5, d6, d7);
#else
    NS_F1 res = (NS_F1)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA_Fast9(c, d0, d1, d2, d3, d4, d5, d6, d7);
    else
        res = RunModelB_Fast9(c, d0, d1, d2, d3, d4, d5, d6, d7);
    return res;
#endif
}

#else

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 1)
NS_F1 RunModelA_Exact12(
    NS_F1 p0, NS_F1 p1, NS_F1 p2,
    NS_F1 p3, NS_F1 p4, NS_F1 p5,
    NS_F1 p6, NS_F1 p7, NS_F1 p8)
{
    // Layer 1: 3x3 Conv (1 -> 12 channels)
    NS_F4 L1_0 = NS_F4(0.013958196, -0.034602143, 0.017349342, 0.051420901);
    NS_F4 L1_1 = NS_F4(-0.002181380, 0.005415063, -0.001871437, -0.040856525);
    NS_F4 L1_2 = NS_F4(0.021341776, 0.016098073, -0.027485607, -0.000488010);

    L1_0 += p0 * NS_F4(0.157846898, -0.124734171, -0.127510592, -0.142722160);
    L1_1 += p0 * NS_F4(-0.081266180, -0.035668157, -0.128144190, 0.029555680);
    L1_2 += p0 * NS_F4(0.210917711, -0.152093813, -0.240400285, 0.259769797);
    L1_0 += p1 * NS_F4(-0.018440859, -0.024522314, -0.128880441, -0.146906286);
    L1_1 += p1 * NS_F4(0.013216234, 0.053734783, 0.141454548, 0.003527662);
    L1_2 += p1 * NS_F4(-0.405102581, -0.348911375, -0.129119381, 0.220478624);
    L1_0 += p2 * NS_F4(0.160163894, 0.025609946, -0.154936627, 0.000249316);
    L1_1 += p2 * NS_F4(0.003316377, 0.011136279, 0.057656068, -0.143218592);
    L1_2 += p2 * NS_F4(0.270902693, 0.031090865, -0.147552043, 0.070456423);
    L1_0 += p3 * NS_F4(-0.024166472, -0.277416319, 0.124570601, -0.051763635);
    L1_1 += p3 * NS_F4(-0.063075431, 0.229678214, 0.024368009, 0.056972858);
    L1_2 += p3 * NS_F4(-0.169356808, 0.238923475, -0.007092264, 0.328647912);
    L1_0 += p4 * NS_F4(-0.493815362, 0.346248239, 0.331841409, 0.500756621);
    L1_1 += p4 * NS_F4(-0.000619092, -0.624130189, -0.436149389, 0.508056283);
    L1_2 += p4 * NS_F4(0.077492014, 0.104612425, -0.268455684, -0.233587697);
    L1_0 += p5 * NS_F4(-0.042992529, -0.168081835, -0.129785717, 0.158413351);
    L1_1 += p5 * NS_F4(0.051667836, 0.455829173, 0.141271994, -0.077929094);
    L1_2 += p5 * NS_F4(0.113687925, 0.209046662, -0.484994620, -0.358390838);
    L1_0 += p6 * NS_F4(0.177939489, 0.052703582, -0.057644196, -0.147503659);
    L1_1 += p6 * NS_F4(-0.029891005, -0.118537180, 0.029043604, 0.004845856);
    L1_2 += p6 * NS_F4(0.132930592, -0.184339643, -0.218361959, 0.174553171);
    L1_0 += p7 * NS_F4(-0.058614843, -0.305103749, 0.254780143, -0.034503106);
    L1_1 += p7 * NS_F4(0.153854638, 0.107778028, 0.209778413, -0.009218552);
    L1_2 += p7 * NS_F4(-0.360170394, -0.132584587, 0.632588685, -0.138400137);
    L1_0 += p8 * NS_F4(0.190503091, 0.291831911, -0.109605379, -0.063311636);
    L1_1 += p8 * NS_F4(0.040927604, -0.064029731, 0.092040010, -0.070735231);
    L1_2 += p8 * NS_F4(0.178212926, 0.068329029, -0.240759194, -0.314655423);

    L1_0 = PReLU4(L1_0, NS_F4(-2.387171745, -0.272749335, -2.923240185, -0.953309059));
    L1_1 = PReLU4(L1_1, NS_F4(-2.168637514, 0.855447471, -0.171175957, 0.460944146));
    L1_2 = PReLU4(L1_2, NS_F4(0.312436283, -0.096803114, -0.173512161, -1.026450276));

    // Layer 2: 1x1 Dense (12 -> 12 channels), column-major vector FMAs
    NS_F4 L2_0 = NS_F4(0.031355277, 0.220565408, -0.028643996, -0.000396760);
    NS_F4 L2_1 = NS_F4(0.053762030, 0.191098660, 0.028867338, -0.012472775);
    NS_F4 L2_2 = NS_F4(0.009898989, -0.032807596, 0.000105271, -0.024801077);

    L2_0 += L1_0.x * NS_F4(-0.073890910, -0.354415715, 0.864738226, -0.159830540);
    L2_1 += L1_0.x * NS_F4(0.471214026, -0.918480992, -0.821215689, 0.160448000);
    L2_2 += L1_0.x * NS_F4(-1.269040704, -0.877907932, -1.432761431, -0.141609788);
    L2_0 += L1_0.y * NS_F4(0.442915738, 0.255197883, -0.477833390, 0.038192000);
    L2_1 += L1_0.y * NS_F4(0.549063981, -0.645617664, -0.385795057, 0.688369751);
    L2_2 += L1_0.y * NS_F4(-0.059846044, -0.114583023, -0.421258569, -0.390305877);
    L2_0 += L1_0.z * NS_F4(0.062885523, -0.292075455, -0.162099168, -0.158631608);
    L2_1 += L1_0.z * NS_F4(-0.014970109, -0.418581784, 0.239962846, 0.691993833);
    L2_2 += L1_0.z * NS_F4(0.084006079, 0.211630091, 0.412224859, 0.055484608);
    L2_0 += L1_0.w * NS_F4(-0.237529442, -0.725630701, -0.246071026, 0.733967006);
    L2_1 += L1_0.w * NS_F4(-0.376401037, 0.288688481, 0.137052476, -0.424544305);
    L2_2 += L1_0.w * NS_F4(0.706222057, 0.710830331, 0.558480740, 0.364328563);
    L2_0 += L1_1.x * NS_F4(-0.008712847, 0.136713073, -0.014323278, 0.160120249);
    L2_1 += L1_1.x * NS_F4(0.332775176, 0.145794243, -0.084028967, 0.312038898);
    L2_2 += L1_1.x * NS_F4(0.175326273, 0.375253499, -0.483496517, -0.040529206);
    L2_0 += L1_1.y * NS_F4(2.043683290, 1.056981683, -1.849385142, -1.340586066);
    L2_1 += L1_1.y * NS_F4(-0.024426594, -0.145882696, -1.247046232, 1.263586879);
    L2_2 += L1_1.y * NS_F4(-0.270830810, 0.119935840, -1.311672688, -1.643486500);
    L2_0 += L1_1.z * NS_F4(0.222858787, 0.002627724, -0.047311105, -0.508502781);
    L2_1 += L1_1.z * NS_F4(0.425284505, -0.765440702, -0.231615975, 0.364498079);
    L2_2 += L1_1.z * NS_F4(-0.473862469, -0.175328150, 0.200376466, -0.170545414);
    L2_0 += L1_1.w * NS_F4(-0.048534606, -0.462668061, -0.018056788, 0.158020362);
    L2_1 += L1_1.w * NS_F4(0.222185135, -0.024649546, 0.664888382, 0.522409737);
    L2_2 += L1_1.w * NS_F4(0.326092124, 0.249779537, -0.228427365, -0.095221654);
    L2_0 += L1_2.x * NS_F4(-1.483291268, -0.399437845, 1.759939194, 0.717141092);
    L2_1 += L1_2.x * NS_F4(0.051209208, 0.305358291, 0.262363881, -0.222656667);
    L2_2 += L1_2.x * NS_F4(-0.319517046, -0.355080426, -0.786878586, 0.965228558);
    L2_0 += L1_2.y * NS_F4(-0.006965798, 0.087671049, 0.065428883, -0.080086760);
    L2_1 += L1_2.y * NS_F4(0.561254084, -0.254941225, -2.439566135, -0.202635348);
    L2_2 += L1_2.y * NS_F4(-0.295663029, -0.660095990, 0.552567244, -0.123915620);
    L2_0 += L1_2.z * NS_F4(0.160122022, -0.143500865, -0.316779077, 0.126877829);
    L2_1 += L1_2.z * NS_F4(-0.455035418, -0.346134901, -0.590004444, -0.225798979);
    L2_2 += L1_2.z * NS_F4(-0.056098495, 0.286984503, -0.875478148, 0.122943603);
    L2_0 += L1_2.w * NS_F4(0.114788808, -0.267752349, -0.108699806, -0.354385585);
    L2_1 += L1_2.w * NS_F4(-0.320654094, -0.292018741, -0.150398821, -0.067541674);
    L2_2 += L1_2.w * NS_F4(-0.444056034, 0.303410649, 0.512069345, 0.157081172);

    L2_0 = PReLU4(L2_0, NS_F4(-1.785561442, -0.498042375, -0.242182761, -1.958031893));
    L2_1 = PReLU4(L2_1, NS_F4(-5.690943241, -0.729003668, -0.219683945, -3.025264025));
    L2_2 = PReLU4(L2_2, NS_F4(-2.192382336, -1.276227236, -0.214304626, -2.779862165));

    // Layer 3: Output projection (12 -> 1)
    NS_F4 outAcc = L2_0 * NS_F4(-1.723722577, -1.003309250, 1.881006837, 0.553851724);
    outAcc += L2_1 * NS_F4(-0.404766232, 0.851177633, 0.785482407, -0.269340128);
    outAcc += L2_2 * NS_F4(-0.382485956, 0.502349734, -0.833594918, 1.040693045);
    return (NS_F1)0.041042306 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
NS_F1 RunModelB_Exact12(
    NS_F1 p0, NS_F1 p1, NS_F1 p2,
    NS_F1 p3, NS_F1 p4, NS_F1 p5,
    NS_F1 p6, NS_F1 p7, NS_F1 p8)
{
    // Layer 1: 3x3 Conv (1 -> 12 channels)
    NS_F4 L1_0 = NS_F4(0.076806329, -0.002794162, 0.061713792, -0.002923975);
    NS_F4 L1_1 = NS_F4(0.026175125, -0.153714553, 0.074806444, -0.009952844);
    NS_F4 L1_2 = NS_F4(-0.002780868, -0.010205056, -0.021362821, 0.022398105);

    L1_0 += p0 * NS_F4(-0.180299282, 0.014662747, 0.062037338, 0.162913963);
    L1_1 += p0 * NS_F4(-0.096845955, -0.120704472, 0.018369380, 0.099958651);
    L1_2 += p0 * NS_F4(-0.069051631, 0.084565893, 0.073071629, 0.168920413);
    L1_0 += p1 * NS_F4(-0.040158097, -0.134107947, 0.069352902, 0.256246716);
    L1_1 += p1 * NS_F4(-0.170313105, -0.429769307, -0.140258655, -0.072976060);
    L1_2 += p1 * NS_F4(0.006422973, -0.032386098, 0.088514760, -0.294761181);
    L1_0 += p2 * NS_F4(-0.186079368, 0.002835276, 0.152039975, 0.139015242);
    L1_1 += p2 * NS_F4(-0.001664574, 0.045396138, 0.168260053, -0.068442158);
    L1_2 += p2 * NS_F4(0.247336522, 0.189809218, -0.182572842, 0.225187257);
    L1_0 += p3 * NS_F4(-0.070876308, 0.080869898, 0.216372654, -0.001347039);
    L1_1 += p3 * NS_F4(0.298240304, -0.487868726, -0.011222760, 0.326351106);
    L1_2 += p3 * NS_F4(-0.563833356, 0.083959579, -0.075837053, -0.235026225);
    L1_0 += p4 * NS_F4(0.436923504, 0.340757638, -0.517588019, 0.083469912);
    L1_1 += p4 * NS_F4(-0.197693869, 0.114102028, 0.453443885, -0.177208096);
    L1_2 += p4 * NS_F4(0.035827920, -0.366167694, 0.453837216, 0.398048311);
    L1_0 += p5 * NS_F4(-0.022785911, -0.090594806, 0.096829072, -0.151360869);
    L1_1 += p5 * NS_F4(-0.159646124, 0.201200619, -0.038751587, 0.155050546);
    L1_2 += p5 * NS_F4(0.400151104, -0.125081092, -0.035440426, -0.357304394);
    L1_0 += p6 * NS_F4(0.123323008, 0.150079280, -0.071160279, -0.032094222);
    L1_1 += p6 * NS_F4(-0.247537255, 0.395797133, -0.195361555, -0.067078769);
    L1_2 += p6 * NS_F4(-0.094768256, 0.182795376, -0.047163446, 0.144929826);
    L1_0 += p7 * NS_F4(0.147065058, 0.056526992, -0.022634532, -0.406054407);
    L1_1 += p7 * NS_F4(0.090287514, -0.481248379, -0.089255974, -0.089840382);
    L1_2 += p7 * NS_F4(-0.188313007, 0.200342402, -0.209137321, -0.289527595);
    L1_0 += p8 * NS_F4(-0.157484218, -0.062786549, 0.087097272, -0.055152047);
    L1_1 += p8 * NS_F4(-0.246287793, -0.003776026, -0.153421938, 0.065058291);
    L1_2 += p8 * NS_F4(0.229671970, 0.031603150, 0.160108879, 0.226898223);

    L1_0 = PReLU4(L1_0, NS_F4(-1.442547083, -1.845595598, -1.306786060, -0.781065464));
    L1_1 = PReLU4(L1_1, NS_F4(-0.109224759, -0.269420534, -0.060583767, 4.149562359));
    L1_2 = PReLU4(L1_2, NS_F4(-0.792430758, -1.190022588, 0.142513722, 0.115136072));

    // Layer 2: 1x1 Dense (12 -> 12 channels), column-major vector FMAs
    NS_F4 L2_0 = NS_F4(0.017644040, -0.120309018, -0.029212048, 0.068806812);
    NS_F4 L2_1 = NS_F4(0.053569581, -0.141184986, 0.036499061, -0.027951837);
    NS_F4 L2_2 = NS_F4(0.221204460, -0.030860605, 0.061900575, 0.100701667);

    L2_0 += L1_0.x * NS_F4(-0.229082733, 0.480870575, 0.789031744, -0.600302875);
    L2_1 += L1_0.x * NS_F4(-0.351149797, 0.535218537, 0.165266067, -0.848744035);
    L2_2 += L1_0.x * NS_F4(-0.475019783, 0.582823992, 0.399218798, -0.224198133);
    L2_0 += L1_0.y * NS_F4(-0.512871683, 0.391051710, -0.195746094, 0.046118792);
    L2_1 += L1_0.y * NS_F4(0.594739795, 0.025757920, -0.195548356, -0.241835013);
    L2_2 += L1_0.y * NS_F4(0.175065041, -0.069464915, 0.489937872, -0.468186438);
    L2_0 += L1_0.z * NS_F4(0.736965120, -0.417410672, -0.737288594, 0.435882509);
    L2_1 += L1_0.z * NS_F4(-0.068887308, -0.459538639, -0.235718101, -0.068041831);
    L2_2 += L1_0.z * NS_F4(0.764335096, -0.495929092, -0.335512757, 0.389229238);
    L2_0 += L1_0.w * NS_F4(-0.114540882, 0.369169980, -0.646142125, -0.587501466);
    L2_1 += L1_0.w * NS_F4(0.360132426, 0.198079765, 0.526902139, -0.782595038);
    L2_2 += L1_0.w * NS_F4(-0.890451252, 0.401011854, -0.769207895, 0.309193313);
    L2_0 += L1_1.x * NS_F4(0.311843961, 0.009561459, 0.175059706, 0.547766745);
    L2_1 += L1_1.x * NS_F4(-0.879847527, -0.176006883, -0.411046326, -0.665538311);
    L2_2 += L1_1.x * NS_F4(0.180819586, 0.271378279, 0.378717065, -0.264145225);
    L2_0 += L1_1.y * NS_F4(0.229555234, -0.459208190, 0.691861987, 0.373978168);
    L2_1 += L1_1.y * NS_F4(0.880598426, -0.002988719, -0.018310603, -0.007513349);
    L2_2 += L1_1.y * NS_F4(-0.296335191, 0.528470814, 0.111551061, 0.326991796);
    L2_0 += L1_1.z * NS_F4(-0.347491890, 0.736723900, 0.824216008, -0.503097653);
    L2_1 += L1_1.z * NS_F4(-1.479037523, 0.832145870, 0.489419967, -0.313240737);
    L2_2 += L1_1.z * NS_F4(-0.511490107, 0.503377557, 0.790561855, -0.568252385);
    L2_0 += L1_1.w * NS_F4(-0.145946071, 0.021142924, -0.017000724, -0.090789184);
    L2_1 += L1_1.w * NS_F4(-0.044767305, 0.274749726, -0.265991926, -0.339458108);
    L2_2 += L1_1.w * NS_F4(0.079025216, 0.125316218, -0.215333298, 0.097437136);
    L2_0 += L1_2.x * NS_F4(-0.323113024, 0.269746661, -0.435575575, -0.345212072);
    L2_1 += L1_2.x * NS_F4(0.112935498, -0.025276352, -0.114177100, 0.182142422);
    L2_2 += L1_2.x * NS_F4(-0.549045265, 0.272755206, -0.553730965, 0.228109702);
    L2_0 += L1_2.y * NS_F4(0.993536174, -0.397482157, -0.646290779, 0.522237957);
    L2_1 += L1_2.y * NS_F4(-0.615179121, -0.236055046, -1.120269060, -0.052835885);
    L2_2 += L1_2.y * NS_F4(0.121311098, -0.557049394, -0.940150917, 0.454378009);
    L2_0 += L1_2.z * NS_F4(0.120587423, 0.399009675, 0.176628053, -0.069926046);
    L2_1 += L1_2.z * NS_F4(-0.769895375, 0.435621321, -0.591441274, -0.442020327);
    L2_2 += L1_2.z * NS_F4(-0.427249581, -0.097753078, 0.235617280, -0.301024586);
    L2_0 += L1_2.w * NS_F4(0.773548186, 3.502569199, -3.576125145, -1.903903365);
    L2_1 += L1_2.w * NS_F4(-0.700175762, -1.177589297, -2.321050406, -1.241039634);
    L2_2 += L1_2.w * NS_F4(0.354185373, -2.256846189, 1.546710849, -3.406354189);

    L2_0 = PReLU4(L2_0, NS_F4(-1.701528668, -2.895941973, -0.477188736, -0.332290858));
    L2_1 = PReLU4(L2_1, NS_F4(0.147276983, -0.674965382, 0.295771748, -0.229368791));
    L2_2 = PReLU4(L2_2, NS_F4(-0.975452960, -3.820577860, -1.948044658, -1.310231686));

    // Layer 3: Output projection (12 -> 1)
    NS_F4 outAcc = L2_0 * NS_F4(-0.875323415, 1.467422843, -1.980630398, -0.803634167);
    outAcc += L2_1 * NS_F4(-1.479813099, -1.135522127, -0.968504190, 0.906366646);
    outAcc += L2_2 * NS_F4(-1.197628736, 1.004730344, 1.426059246, -2.517558813);
    return (NS_F1)0.065220654 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

NS_F1 RunNeuralExact12(
    NS_F1 p0, NS_F1 p1, NS_F1 p2,
    NS_F1 p3, NS_F1 p4, NS_F1 p5,
    NS_F1 p6, NS_F1 p7, NS_F1 p8)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA_Exact12(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB_Exact12(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#else
    NS_F1 res = (NS_F1)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA_Exact12(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    else
        res = RunModelB_Exact12(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    return res;
#endif
}

#endif


// ============================================================================
// 5. SINGLE-PASS SHADER & TECHNIQUE
// ============================================================================

void VS_Apply(
    in uint id : SV_VertexID,
    out float4 position : SV_Position,
    out float2 uv : TEXCOORD0
#if NSHDR_DYNAMIC_CSPACE
    , out nointerpolation int outMode : TEXCOORD1
    , out nointerpolation float4 outSuppressParams : TEXCOORD2
#endif
    )
{
#if NSHDR_DYNAMIC_CSPACE
    outMode = ResolveColorSpaceMode();
#endif
    if (!IsShaderActive())
    {
        position = float4(0.0, 0.0, 0.0, 0.0);
        uv = float2(0.0, 0.0);
#if NSHDR_DYNAMIC_CSPACE
        outSuppressParams = float4(0.0, 0.0, 0.0, 0.0);
#endif
        return;
    }

    uv.x = (id == 2u) ? 2.0 : 0.0;
    uv.y = (id == 1u) ? 2.0 : 0.0;
#if NSHDR_DYNAMIC_CSPACE
    if (outMode == MODE_SDR)
        outSuppressParams = float4(0.25, 4.0, 0.10, 6.0);
    else
        outSuppressParams = float4(0.18, 5.0, 0.075, 8.0);
#endif
    position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

void PS_Apply(
    float4 vpos : SV_Position,
    float2 uv : TEXCOORD0
#if NSHDR_DYNAMIC_CSPACE
    , nointerpolation int mode : TEXCOORD1
    , nointerpolation float4 suppressParams : TEXCOORD2
#endif
    , out float4 outColor : SV_Target)
{
#if !NSHDR_DYNAMIC_CSPACE
    #if BUFFER_COLOR_SPACE == 3
        const int mode = MODE_PQ;
        const float4 suppressParams = float4(0.18, 5.0, 0.075, 8.0);
    #elif BUFFER_COLOR_SPACE == 2
        const int mode = MODE_SCRGB;
        const float4 suppressParams = float4(0.18, 5.0, 0.075, 8.0);
    #else
        const int mode = MODE_SDR;
        const float4 suppressParams = float4(0.25, 4.0, 0.10, 6.0);
    #endif
#endif

    const float px = BUFFER_RCP_WIDTH;
    const float py = BUFFER_RCP_HEIGHT;

    float3 origColor = tex2Dlod(sBackBufferPoint, float4(uv, 0.0, 0.0)).rgb;
    float p4 = ExtractLuma(origColor, mode);

#if NEURAL_SHARPEN_PROFILE == 0
    // Ultra Fast 5-Tap Bilinear: 4 half-pixel LINEAR taps cover all 9 pixels of the 3x3 box!
    const float hx = 0.5 * px;
    const float hy = 0.5 * py;

    float q0 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(uv + float2(-hx, -hy), 0.0, 0.0)).rgb, mode);
    float q1 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(uv + float2( hx, -hy), 0.0, 0.0)).rgb, mode);
    float q2 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(uv + float2(-hx,  hy), 0.0, 0.0)).rgb, mode);
    float q3 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(uv + float2( hx,  hy), 0.0, 0.0)).rgb, mode);

    float maxQ = max(max(q0, q1), max(q2, q3));
    float minQ = min(min(q0, q1), min(q2, q3));

    float rangeQ   = max(maxQ, p4) - min(minQ, p4);
    float flatMask = saturate((rangeQ - FLAT_SKIP_RANGE) * FLAT_INV_SPAN);

    float localContrast  = (maxQ - minQ) * 1.5;
    float aliasingSignal = saturate((localContrast - suppressParams.x) * suppressParams.y);

    float spikeIntensity = max(max(0.0, p4 - maxQ), max(0.0, minQ - p4)) * 1.3333333;
    float fireflySignal  = saturate((spikeIntensity - suppressParams.z) * suppressParams.w);

    float suppressionMask = 1.0 - saturate(aliasingSignal + fireflySignal) * ArtifactProtection;

    float rawResidual = (float)RunNeuralFast5(
        (NS_F1)p4,
        (NS_F1)(q0 - p4), (NS_F1)(q1 - p4), (NS_F1)(q2 - p4), (NS_F1)(q3 - p4));
#else
    // 9-Tap POINT sampling (1-texel loads via sBackBufferPoint avoid 2x2 bilinear L1 tag lookups).
    float p0 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2(-px, -py), 0.0, 0.0)).rgb, mode);
    float p1 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2(0.0, -py), 0.0, 0.0)).rgb, mode);
    float p2 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2( px, -py), 0.0, 0.0)).rgb, mode);
    float p3 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2(-px, 0.0), 0.0, 0.0)).rgb, mode);
    float p5 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2( px, 0.0), 0.0, 0.0)).rgb, mode);
    float p6 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2(-px,  py), 0.0, 0.0)).rgb, mode);
    float p7 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2(0.0,  py), 0.0, 0.0)).rgb, mode);
    float p8 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv + float2( px,  py), 0.0, 0.0)).rgb, mode);

    float maxCross = max(max(p1, p7), max(p3, p5));
    float minCross = min(min(p1, p7), min(p3, p5));
    float maxDiag  = max(max(p0, p2), max(p6, p8));
    float minDiag  = min(min(p0, p2), min(p6, p8));

    float range3x3 = max(max(maxCross, maxDiag), p4) - min(min(minCross, minDiag), p4);
    float flatMask = saturate((range3x3 - FLAT_SKIP_RANGE) * FLAT_INV_SPAN);

    float localContrast  = maxCross - minCross;
    float aliasingSignal = saturate((localContrast - suppressParams.x) * suppressParams.y);

    float spikeIntensity = max(max(0.0, p4 - maxCross), max(0.0, minCross - p4));
    float fireflySignal  = saturate((spikeIntensity - suppressParams.z) * suppressParams.w);

    float suppressionMask = 1.0 - saturate(aliasingSignal + fireflySignal) * ArtifactProtection;

    #if NEURAL_SHARPEN_PROFILE == 1
        float rawResidual = (float)RunNeuralFast9(
            (NS_F1)p4,
            (NS_F1)(p0 - p4), (NS_F1)(p1 - p4), (NS_F1)(p2 - p4), (NS_F1)(p3 - p4),
            (NS_F1)(p5 - p4), (NS_F1)(p6 - p4), (NS_F1)(p7 - p4), (NS_F1)(p8 - p4));
    #else
        float rawResidual = (float)RunNeuralExact12(
            (NS_F1)p0, (NS_F1)p1, (NS_F1)p2,
            (NS_F1)p3, (NS_F1)p4, (NS_F1)p5,
            (NS_F1)p6, (NS_F1)p7, (NS_F1)p8);
    #endif
#endif

    float normalizedResidual = saturate(abs(rawResidual) * 2.63); // 2.63 ~ 1 / 0.38

#if ENABLE_DEBUG_VIEWS
    [branch]
    if (ViewMode == 2)
    {
        outColor = float4(EncodeDebugGray(normalizedResidual * flatMask, mode), 1.0);
        return;
    }
#endif

    float residual = clamp(rawResidual, -0.5, 0.5) * 0.22;
    residual *= (1.0 + EdgeResponse * normalizedResidual);

    float delta = residual * flatMask * suppressionMask * Intensity;

#if ENABLE_DEBUG_VIEWS
    [branch]
    if (ViewMode == 1)
    {
        outColor = float4(EncodeDebugGray(0.5 + delta, mode), 1.0);
        return;
    }
#endif

#if !NSHDR_DYNAMIC_CSPACE
    #if BUFFER_COLOR_SPACE == 3
        float3 blackGuard = (delta > 0.0) ? saturate(origColor * 32.0) : float3(1.0, 1.0, 1.0);
        outColor = float4(saturate(origColor + delta * blackGuard), 1.0);
    #elif BUFFER_COLOR_SPACE == 2
        float blackGuard = (delta > 0.0) ? saturate(p4 * 50.0) : 1.0;
        float yOld = FastPQToLinear(p4);
        float yNew = FastPQToLinear(saturate(p4 + delta * blackGuard));
        float ratio = (yOld > 1e-7) ? clamp(yNew / yOld, 0.0, 4.0) : 1.0;
        outColor = float4(origColor * ratio, 1.0);
    #else
        outColor = float4(max(0.0, origColor + delta), 1.0);
    #endif
#else
    if (mode == MODE_PQ)
    {
        float3 blackGuard = (delta > 0.0) ? saturate(origColor * 32.0) : float3(1.0, 1.0, 1.0);
        outColor = float4(saturate(origColor + delta * blackGuard), 1.0);
    }
    else if (mode == MODE_SCRGB)
    {
        float blackGuard = (delta > 0.0) ? saturate(p4 * 50.0) : 1.0;
        float yOld = FastPQToLinear(p4);
        float yNew = FastPQToLinear(saturate(p4 + delta * blackGuard));
        float ratio = (yOld > 1e-7) ? clamp(yNew / yOld, 0.0, 4.0) : 1.0;
        outColor = float4(origColor * ratio, 1.0);
    }
    else
    {
        outColor = float4(max(0.0, origColor + delta), 1.0);
    }
#endif
}


technique Neural_Sharpen_HDR <
    ui_label = "Neural Sharpen HDR";
    ui_tooltip = "Single-pass, HDR-aware, GPU-optimized Neural Sharpening (based on Barbatos Neural Sharpen 2.8, MIT). Set NEURAL_SHARPEN_PROFILE=0 (Ultra Fast 5-Tap), 1 (Fast 9-Tap), or 2 (Exact 12-Ch).";
>
{
    pass Apply
    {
        VertexShader = VS_Apply;
        PixelShader  = PS_Apply;
    }
}

} // namespace NeuralSharpenHDR
