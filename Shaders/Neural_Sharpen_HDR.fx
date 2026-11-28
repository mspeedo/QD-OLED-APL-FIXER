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
      1 = Balanced Fast 9-Tap (default): 9 POINT TEX fetches, D4-invariant 9->4->4->1 CNN (56 FMAs, 4.7x less ALU, R2=0.974).
      0 = Ultra Fast 5-Tap Bilinear: 5 TEX fetches covering full 3x3 via 4 bilinear quadrants,
          D4-invariant 6->4->4->1 CNN (44 FMAs, 6.0x less ALU, 44% fewer TEX fetches, R2=0.92).
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

// 1 = Balanced Fast 9-Tap D4-Symmetric Neural (9 POINT TEX, 56 FMAs, R2=0.974 vs Exact - default)
// 0 = Ultra Fast 5-Tap Bilinear D4-Symmetric Neural (5 TEX, 44 FMAs, R2=0.92 - fastest)
// 2 = Exact 12-Channel Barbatos Neural (9 POINT TEX, 264 FMAs)
#ifndef NEURAL_SHARPEN_PROFILE
    #define NEURAL_SHARPEN_PROFILE 1
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

// 1 = Enable FrameGen-safe wall-clock camera-pan motion adaptation (~0.001-0.003 ms, D3D11+/Vulkan/GL4.3+)
// 0 = Strip motion compute pass and textures completely at compile time
#ifndef ENABLE_MOTION_ADAPTATION
    #define ENABLE_MOTION_ADAPTATION 1
#endif

#if ENABLE_MOTION_ADAPTATION && defined(__RENDERER__) && (__RENDERER__ >= 0x0B000)
    #define NSHDR_MOTION_COMPILED 1
#else
    #define NSHDR_MOTION_COMPILED 0
#endif

// 0 = Strip diagnostic motion OSD completely at compile time (default)
// 1 = Enable compact top-right OSD showing live total intensity, motion boost, and live pan meter
#ifndef ENABLE_MOTION_OSD
    #define ENABLE_MOTION_OSD 0
#endif

#if ENABLE_MOTION_OSD && NSHDR_MOTION_COMPILED
    #define NSHDR_OSD_COMPILED 1
#else
    #define NSHDR_OSD_COMPILED 0
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
    NSHDR_TOOLTIP("Base neural sharpening strength when stationary. Set to 0.0 with Motion Boost > 0 for pan-only sharpening (vertex-culled when still).")
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

#if NSHDR_MOTION_COMPILED
uniform float MotionBoost <
    ui_category = "Motion Adaptive";
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Motion Boost";
    NSHDR_TOOLTIP("Extra sharpening intensity added during camera pans to counteract TAA / FrameGen motion blur. 0.0 = disabled.")
> = 0.50;

uniform float MotionSensitivity <
    ui_category = "Motion Adaptive";
    ui_type = "drag";
    ui_min = 0.25;
    ui_max = 3.0;
    ui_step = 0.05;
    ui_label = "Motion Sensitivity";
    NSHDR_TOOLTIP("How easily camera pans reach full Motion Boost. Higher = triggers on slower pans; lower = requires faster pans.")
> = 1.0;

uniform float MotionDecayMs <
    ui_category = "Motion Adaptive";
    ui_type = "drag";
    ui_min = 40.0;
    ui_max = 400.0;
    ui_step = 10.0;
    ui_label = "Motion Decay (ms)";
    NSHDR_TOOLTIP("Wall-clock decay time after stopping a camera pan. Higher values fade back more gradually without sharpness pop.")
> = 140.0;

#if NSHDR_OSD_COMPILED
uniform bool ShowMotionOSD <
    ui_category = "Motion Adaptive";
    ui_label = "Show Motion OSD";
    NSHDR_TOOLTIP("Displays a compact top-right overlay with live total intensity, added motion boost, and a real-time pan meter.")
> = true;
#endif

// Wall-clock milliseconds (FrameGen-safe; does not rely on frametime which freezes on generated frames).
uniform uint MotionClockMs < source = "timer"; >;
#endif

#if ENABLE_DEBUG_VIEWS
uniform int ViewMode <
    ui_type = "combo";
    ui_items = "Normal\0Sharpening Map\0Residual Magnitude\0";
    ui_label = "Debug View";
    NSHDR_TOOLTIP("Normal: Final sharpened output. Sharpening Map: Signed delta around mid-gray. Residual Magnitude: Raw neural response (tone-mapped to ~203 nits in HDR).")
> = 0;
#endif

#if ENABLE_UI_TOOLTIPS
uniform int _PreprocessorHelp <
    ui_category = "Preprocessor Definitions Help";
    ui_category_closed = true;
    ui_type = "radio";
    ui_label = " ";
    ui_text =
        "NEURAL_SHARPEN_PROFILE (default: 1)\n"
        "  1 = Balanced Fast 9-Tap (9 POINT TEX, 56 FMAs, R2=0.974 vs Exact)\n"
        "  0 = Ultra Fast 5-Tap Bilinear (5 TEX, 44 FMAs, R2=0.92, fastest)\n"
        "  2 = Exact 12-Channel Barbatos CNN (9 POINT TEX, 264 FMAs)\n"
        "\n"
        "NEURAL_SHARPEN_MODEL (default: 1)\n"
        "  1 = Model A only (Balanced, smallest shader)\n"
        "  2 = Model B only (Edge-Enhanced, smallest shader)\n"
        "  0 = Both models compiled with runtime UI switcher\n"
        "\n"
        "ENABLE_MOTION_ADAPTATION (default: 1)\n"
        "  1 = FrameGen-safe wall-clock camera-pan boost (~0.005 ms)\n"
        "  0 = Strip motion compute pass and textures completely\n"
        "\n"
        "ENABLE_MOTION_OSD (default: 0)\n"
        "  0 = Stripped at compile time (zero overhead)\n"
        "  1 = Compact top-right diagnostic OSD showing live motion boost\n"
        "\n"
        "ENABLE_DEBUG_VIEWS (default: 0)\n"
        "  0 = Stripped at compile time (fastest)\n"
        "  1 = Enables Sharpening Map & Residual Magnitude debug views\n"
        "\n"
        "ALLOW_MANUAL_COLOR_SPACE (default: 0)\n"
        "  0 = Compile-time specialized to detected BUFFER_COLOR_SPACE\n"
        "      (runtime Color Space selector still appears if Unknown)\n"
        "  1 = Always show manual SDR / scRGB / HDR10 PQ selector\n"
        "\n"
        "USE_FP16_CNN (default: 0)\n"
        "  0 = FP32 float4 CNN (recommended / faster on D3D11 SM5.0)\n"
        "  1 = FP16 min16float4 CNN\n"
        "\n"
        "ENABLE_UI_TOOLTIPS (default: 1)\n"
        "  1 = Show tooltips and this preprocessor help section\n"
        "  0 = Strip all UI tooltips and help text";
>;
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

#if NSHDR_MOTION_COMPILED
// 16x16 R32F previous-frame coarse luma snapshot (1 KB VRAM; single-channel R32F supports D3D11 typed UAV load).
texture TexMotionPrev
{
    Width = 16;
    Height = 16;
    Format = R32F;
};
storage2D<float> StorageMotionPrev
{
    Texture = TexMotionPrev;
};

// 2x1 R32I wall-clock state: [0,0]=initialized flag, [1,0]=last wall-clock ms.
texture TexMotionClock
{
    Width = 2;
    Height = 1;
    Format = R32I;
};
storage2D<int> StorageMotionClock
{
    Texture = TexMotionClock;
};

// 2x1 R32F motion state: [0,0]=smoothed motion [0,1] (read once per frame in VS_Apply), [1,0]=prev mean luma.
texture TexMotionState
{
    Width = 2;
    Height = 1;
    Format = R32F;
};
sampler sMotionState
{
    Texture = TexMotionState;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<float> StorageMotionState
{
    Texture = TexMotionState;
};
#endif

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

#if NSHDR_DYNAMIC_CSPACE || (BUFFER_COLOR_SPACE == 2) || ENABLE_DEBUG_VIEWS || NSHDR_OSD_COMPILED
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
NS_F1 RunModelA_Fast5(NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4, NS_F1 f5)
{
    // Layer 1: 6 D4-invariant quadrant features -> 4 (6 float4 FMAs, 1 float4 register)
    NS_F4 L1 = NS_F4(-0.730821788, -0.012679012, -0.011884286, -0.203455493);
    L1 += f0 * NS_F4(0.784533858, 0.461801618, 0.369383633, 0.038089700);
    L1 += f1 * NS_F4(-2.707480907, 4.825026512, -2.861585617, 0.167170525);
    L1 += f2 * NS_F4(-2.689104557, 1.570460081, -1.354166269, 1.572292686);
    L1 += f3 * NS_F4(-0.333779693, 0.844432175, -0.652857542, -0.179609820);
    L1 += f4 * NS_F4(3.167257786, 2.580765486, -2.669166803, -1.620874643);
    L1 += f5 * NS_F4(1.200607181, -0.656353354, 1.005273700, 2.967521906);
    L1 = PReLU4(L1, NS_F4(-0.659885526, -0.405795425, -5.364359856, 0.331523985));

    // Layer 2: 4 -> 4 (4 float4 FMAs, 1 float4 register)
    NS_F4 L2 = NS_F4(0.297295213, -0.505572319, 0.686525464, -0.234156102);
    L2 += L1.x * NS_F4(-0.783984542, -0.385230303, -0.732657433, 1.050563812);
    L2 += L1.y * NS_F4(-1.248377442, 1.311829448, -0.769987106, 0.640519083);
    L2 += L1.z * NS_F4(0.544887245, -0.021909865, 0.129431665, -0.265296131);
    L2 += L1.w * NS_F4(0.601746738, 0.395208925, -0.665749490, 0.236189306);
    L2 = PReLU4(L2, NS_F4(0.332469761, -0.260700166, -0.552000105, -0.615324855));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(0.219960108, 0.227117255, 0.241145313, -0.125508279);
    return (NS_F1)-0.100949124 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
NS_F1 RunModelB_Fast5(NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4, NS_F1 f5)
{
    // Layer 1: 6 D4-invariant quadrant features -> 4 (6 float4 FMAs, 1 float4 register)
    NS_F4 L1 = NS_F4(-0.498439014, -0.086211741, 0.080668695, 0.034546122);
    L1 += f0 * NS_F4(0.045874864, 0.791166902, -0.053344809, 0.023593465);
    L1 += f1 * NS_F4(1.555841923, 2.135283947, 4.186703205, 3.080390453);
    L1 += f2 * NS_F4(2.931004524, 0.199650809, -0.889571428, 1.885230184);
    L1 += f3 * NS_F4(1.550874114, 0.225837886, 1.671626925, -0.297964156);
    L1 += f4 * NS_F4(0.384031236, -5.218594074, 0.298344046, -1.024906278);
    L1 += f5 * NS_F4(1.380360603, -2.309120178, 0.051716380, 0.095078066);
    L1 = PReLU4(L1, NS_F4(-0.382867306, -1.903236389, 2.302712679, -1.544907331));

    // Layer 2: 4 -> 4 (4 float4 FMAs, 1 float4 register)
    NS_F4 L2 = NS_F4(-0.221982688, -0.322668999, 0.394862086, -0.031507511);
    L2 += L1.x * NS_F4(0.222056463, 0.615740299, -1.905480981, -1.438645840);
    L2 += L1.y * NS_F4(0.295358181, 0.176379740, -0.322972000, 0.105399743);
    L2 += L1.z * NS_F4(1.366834044, 2.617906332, 0.841546118, -0.457906306);
    L2 += L1.w * NS_F4(0.893518746, -0.315902114, 2.189673662, 1.249550939);
    L2 = PReLU4(L2, NS_F4(2.560443401, 0.114069745, -0.921200156, -0.033280935));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(-0.199998975, 0.505118191, -0.229397729, -0.235644013);
    return (NS_F1)0.030873140 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

NS_F1 RunNeuralFast5(NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4, NS_F1 f5)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA_Fast5(f0, f1, f2, f3, f4, f5);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB_Fast5(f0, f1, f2, f3, f4, f5);
#else
    NS_F1 res = (NS_F1)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA_Fast5(f0, f1, f2, f3, f4, f5);
    else
        res = RunModelB_Fast5(f0, f1, f2, f3, f4, f5);
    return res;
#endif
}

#elif NEURAL_SHARPEN_PROFILE == 1

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 1)
NS_F1 RunModelA_Fast9(
    NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4,
    NS_F1 f5, NS_F1 f6, NS_F1 f7, NS_F1 f8)
{
    // Layer 1: 9 D4-invariant features -> 4 (9 float4 FMAs, 1 float4 register)
    NS_F4 L1 = NS_F4(-0.021602197, -0.037732009, -0.031781912, 0.003739639);
    L1 += f0 * NS_F4(0.490581721, 0.470121324, 0.032057382, -0.304590225);
    L1 += f1 * NS_F4(0.515552700, -3.694405079, 4.940162659, -5.152735710);
    L1 += f2 * NS_F4(2.292732239, -0.257025272, -3.482381105, 1.976642013);
    L1 += f3 * NS_F4(0.605435491, 0.066763438, 0.347984612, -0.520679593);
    L1 += f4 * NS_F4(0.388647079, -0.061712220, 0.022914547, 0.111411117);
    L1 += f5 * NS_F4(0.395523220, 0.052985273, 0.120051131, -0.413724184);
    L1 += f6 * NS_F4(0.229209855, -0.176119968, 0.017499449, 0.384687394);
    L1 += f7 * NS_F4(-0.246686488, -0.209233299, -0.273828566, 0.398813099);
    L1 += f8 * NS_F4(0.037717190, 0.020286521, -0.018663839, 0.283268660);
    L1 = PReLU4(L1, NS_F4(-0.576455534, -0.116835482, -0.668418586, -0.363026619));

    // Layer 2: 4 -> 4 (4 float4 FMAs, 1 float4 register)
    NS_F4 L2 = NS_F4(0.528029621, 0.066236846, -0.068031587, 0.424274743);
    L2 += L1.x * NS_F4(-1.067868233, -0.235955998, 0.821404099, -0.244782045);
    L2 += L1.y * NS_F4(0.310634166, -0.758111000, -0.275063127, -0.956197739);
    L2 += L1.z * NS_F4(-0.256804377, -1.819971204, 2.701256275, 0.277401924);
    L2 += L1.w * NS_F4(-0.018259861, 2.818995237, -1.965307236, 0.453616083);
    L2 = PReLU4(L2, NS_F4(0.020508764, -0.887271762, -0.660406947, -0.984633923));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(0.488787711, -0.190372065, 0.301346272, -0.362045377);
    return (NS_F1)-0.091703318 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

#if (NEURAL_SHARPEN_MODEL == 0) || (NEURAL_SHARPEN_MODEL == 2)
NS_F1 RunModelB_Fast9(
    NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4,
    NS_F1 f5, NS_F1 f6, NS_F1 f7, NS_F1 f8)
{
    // Layer 1: 9 D4-invariant features -> 4 (9 float4 FMAs, 1 float4 register)
    NS_F4 L1 = NS_F4(-0.043293942, 0.400390416, -0.116353638, 0.020093702);
    L1 += f0 * NS_F4(0.550856352, 0.070300192, -0.156522706, -0.013008763);
    L1 += f1 * NS_F4(4.360252380, -1.881465673, -0.017534792, 9.729352951);
    L1 += f2 * NS_F4(-0.539573669, -2.096323729, -0.235640705, -4.499947071);
    L1 += f3 * NS_F4(-0.642835140, -0.897718787, -2.686426878, 0.083114818);
    L1 += f4 * NS_F4(1.110328317, 0.050820790, 1.611313581, -0.282710671);
    L1 += f5 * NS_F4(-0.244483963, -0.488649815, -1.500734448, 0.159409881);
    L1 += f6 * NS_F4(-0.240425289, -0.234503448, 1.156003237, -0.126181141);
    L1 += f7 * NS_F4(0.637926877, 0.660809159, 2.168094397, 0.115314506);
    L1 += f8 * NS_F4(-0.004000456, 0.249163449, 1.148556948, -0.164119169);
    L1 = PReLU4(L1, NS_F4(-1.050242424, -0.756138861, -0.938530564, -0.242162332));

    // Layer 2: 4 -> 4 (4 float4 FMAs, 1 float4 register)
    NS_F4 L2 = NS_F4(0.501681566, -0.149832264, -0.566783309, 0.689273655);
    L2 += L1.x * NS_F4(0.105667859, -0.559347391, 0.398490667, 0.281221420);
    L2 += L1.y * NS_F4(-1.372257352, 0.243086413, 0.897467256, -1.479187012);
    L2 += L1.z * NS_F4(0.442376077, 1.213447571, -0.289834768, -0.457043797);
    L2 += L1.w * NS_F4(-1.711187601, -4.710763454, -1.940149069, -1.514082551);
    L2 = PReLU4(L2, NS_F4(0.438383222, 0.103595130, -0.154791519, 0.058535647));

    // Layer 3: 4 -> 1 (1 float4 mul + sum)
    NS_F4 outAcc = L2 * NS_F4(-0.804578483, 0.577563882, -0.618271053, 0.833986282);
    return (NS_F1)0.006116488 + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);
}
#endif

NS_F1 RunNeuralFast9(
    NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4,
    NS_F1 f5, NS_F1 f6, NS_F1 f7, NS_F1 f8)
{
#if NEURAL_SHARPEN_MODEL == 1
    return RunModelA_Fast9(f0, f1, f2, f3, f4, f5, f6, f7, f8);
#elif NEURAL_SHARPEN_MODEL == 2
    return RunModelB_Fast9(f0, f1, f2, f3, f4, f5, f6, f7, f8);
#else
    NS_F1 res = (NS_F1)0.0;
    [branch]
    if (ModelSelector == 0)
        res = RunModelA_Fast9(f0, f1, f2, f3, f4, f5, f6, f7, f8);
    else
        res = RunModelB_Fast9(f0, f1, f2, f3, f4, f5, f6, f7, f8);
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
// 5. MOTION ADAPTATION PASS, APPLY SHADER & TECHNIQUE
// ============================================================================

#if NSHDR_MOTION_COMPILED
groupshared int   gs_MotionAction; // 0 = skip/hold, 2 = init snapshot, 3 = update motion
groupshared int   gs_MotionMode;
groupshared float gs_MotionDtMs;
groupshared float gs_MotionPrevMean;
groupshared float gs_MotionCurrMean;
groupshared float gs_MotionCurr[256];
groupshared float gs_MotionDiff[256];
groupshared float gs_MotionMove[256];
groupshared float gs_MotionColSum[16];
groupshared float gs_MotionColDiff[16];
groupshared float gs_MotionColMove[16];

// Single 1x1x1 threadgroup (256 threads) measuring camera-pan motion across a 16x16 inner-80% grid.
// Uses ReShade wall-clock timer (source="timer") gated at >=12 ms (~60-80 Hz) so FrameGen generated
// frames (dtMs == 0) never flicker or false-drop motion, and holds boosted intensity across them.
// All barriers sit unconditionally at top level (zero returns before any barrier -> 100% FXC X3663-safe).
void CS_UpdateMotion(uint3 gtid : SV_GroupThreadID)
{
    uint lane = gtid.y * 16u + gtid.x;

    if (lane == 0u)
    {
        int action = 0;
        float dtMs = 16.666667;
        float prevMean = 0.0;

#if NSHDR_DYNAMIC_CSPACE
        gs_MotionMode = ResolveColorSpaceMode();
#elif BUFFER_COLOR_SPACE == 3
        gs_MotionMode = MODE_PQ;
#elif BUFFER_COLOR_SPACE == 2
        gs_MotionMode = MODE_SCRGB;
#else
        gs_MotionMode = MODE_SDR;
#endif

        if (MotionBoost <= 0.0001)
        {
            if (tex2Dfetch(StorageMotionClock, int2(0, 0)) != 0)
            {
                tex2Dstore(StorageMotionClock, int2(0, 0), 0);
                tex2Dstore(StorageMotionState, int2(0, 0), 0.0);
            }
        }
        else
        {
            int initialized = tex2Dfetch(StorageMotionClock, int2(0, 0));
            uint nowMs = MotionClockMs;
            if (initialized == 0)
            {
                tex2Dstore(StorageMotionClock, int2(0, 0), 1);
                tex2Dstore(StorageMotionClock, int2(1, 0), int(nowMs));
                tex2Dstore(StorageMotionState, int2(0, 0), 0.0);
                action = 2;
            }
            else
            {
                uint lastMs = uint(tex2Dfetch(StorageMotionClock, int2(1, 0)));
                uint deltaMs = nowMs - lastMs;
                // Skip generated frames (deltaMs == 0) and sub-12 ms intervals; hold smoothed motion.
                if (deltaMs >= 12u)
                {
                    tex2Dstore(StorageMotionClock, int2(1, 0), int(nowMs));
                    dtMs = float(min(deltaMs, 250u));
                    prevMean = tex2Dfetch(StorageMotionState, int2(1, 0));
                    action = 3;
                }
            }
        }

        gs_MotionAction = action;
        gs_MotionDtMs = dtMs;
        gs_MotionPrevMean = prevMean;
    }
    barrier();

    // Stage 1: Sample 16x16 grid across inner 80% of screen (avoids HUD edges) and swap with previous snapshot.
    float curr = 0.0;
    float prev = 0.0;
    if (gs_MotionAction != 0)
    {
        float2 gridUV = float2(0.10, 0.10) + (float2(gtid.xy) + 0.5) * (0.80 / 16.0);
        float3 sampleRGB = tex2Dlod(ReShade::BackBuffer, float4(gridUV, 0.0, 0.0)).rgb;
        curr = ExtractLuma(sampleRGB, gs_MotionMode);
        prev = tex2Dfetch(StorageMotionPrev, int2(gtid.xy));
        tex2Dstore(StorageMotionPrev, int2(gtid.xy), curr);
    }
    gs_MotionCurr[lane] = curr;
    barrier();

    // Stage 2: Compute current frame mean luma (16 lanes sum 16 entries each, then lane 0 sums 16).
    if (lane < 16u)
    {
        float s = 0.0;
        if (gs_MotionAction != 0)
        {
            uint baseIdx = lane * 16u;
            [unroll]
            for (uint k = 0u; k < 16u; ++k)
                s += gs_MotionCurr[baseIdx + k];
        }
        gs_MotionColSum[lane] = s;
    }
    barrier();

    if (lane == 0u)
    {
        float meanCurr = 0.0;
        if (gs_MotionAction != 0)
        {
            float total = 0.0;
            [unroll]
            for (uint k = 0u; k < 16u; ++k)
                total += gs_MotionColSum[k];
            meanCurr = total * (1.0 / 256.0);
            if (gs_MotionAction == 2)
                tex2Dstore(StorageMotionState, int2(1, 0), meanCurr);
        }
        gs_MotionCurrMean = meanCurr;
    }
    barrier();

    // Stage 3: min(rawDiff, zeroMeanDiff) normalized to 60 Hz wall-clock rate:
    // - zmDiff == 0 rejects global flashes / exposure shifts.
    // - rawDiff == 0 on static background rejects localized bright NPCs / spells shifting the frame mean.
    float diffVal = 0.0;
    float moveVal = 0.0;
    if (gs_MotionAction == 3)
    {
        float rateScale = 16.666667 / clamp(gs_MotionDtMs, 12.0, 100.0);
        float rawDiff = abs(curr - prev);
        float zmDiff  = abs((curr - gs_MotionCurrMean) - (prev - gs_MotionPrevMean));
        float d = min(rawDiff, zmDiff) * rateScale;
        diffVal = min(d, 0.25);
        moveVal = (d > 0.008) ? 1.0 : 0.0;
    }
    gs_MotionDiff[lane] = diffVal;
    gs_MotionMove[lane] = moveVal;
    barrier();

    // Stage 4: Reduce diff magnitude & screen coverage fraction.
    if (lane < 16u)
    {
        float sd = 0.0;
        float sm = 0.0;
        if (gs_MotionAction == 3)
        {
            uint baseIdx = lane * 16u;
            [unroll]
            for (uint k = 0u; k < 16u; ++k)
            {
                sd += gs_MotionDiff[baseIdx + k];
                sm += gs_MotionMove[baseIdx + k];
            }
        }
        gs_MotionColDiff[lane] = sd;
        gs_MotionColMove[lane] = sm;
    }
    barrier();

    // Stage 5: Coverage-gated camera-pan score + asymmetric wall-clock smoothing (fast attack, smooth decay).
    if ((lane == 0u) && (gs_MotionAction == 3))
    {
        float sumDiff = 0.0;
        float sumMove = 0.0;
        [unroll]
        for (uint k = 0u; k < 16u; ++k)
        {
            sumDiff += gs_MotionColDiff[k];
            sumMove += gs_MotionColMove[k];
        }

        float meanDiff = sumDiff * (1.0 / 256.0);
        float coverage = sumMove * (1.0 / 256.0);

        // Require broad screen shift (>25% moving -> full at 60%) to ignore local NPCs, grass, and HUD.
        float coverageGate = saturate((coverage - 0.25) * (1.0 / 0.35));
        float magSignal    = saturate((meanDiff * MotionSensitivity - 0.006) * (1.0 / 0.030));
        float panTarget    = (gs_MotionDtMs > 200.0) ? 0.0 : (coverageGate * magSignal);

        float prevSmooth = tex2Dfetch(StorageMotionState, int2(0, 0));
        float tauMs      = (panTarget > prevSmooth) ? 30.0 : max(MotionDecayMs, 20.0);
        float alpha      = 1.0 - exp(-gs_MotionDtMs / tauMs);
        float newSmooth  = lerp(prevSmooth, panTarget, alpha);

        tex2Dstore(StorageMotionState, int2(0, 0), newSmooth);
        tex2Dstore(StorageMotionState, int2(1, 0), gs_MotionCurrMean);
    }
}
#endif

void VS_Apply(
    in uint id : SV_VertexID,
    out float4 position : SV_Position,
    out float2 uv : TEXCOORD0
#if NSHDR_MOTION_COMPILED
    , out nointerpolation float outEffectiveIntensity : TEXCOORD1
#endif
#if NSHDR_DYNAMIC_CSPACE
    , out nointerpolation int outMode : TEXCOORD2
    , out nointerpolation float4 outSuppressParams : TEXCOORD3
#endif
    )
{
#if NSHDR_DYNAMIC_CSPACE
    outMode = ResolveColorSpaceMode();
#endif

    float effectiveIntensity = Intensity;
#if NSHDR_MOTION_COMPILED
    if (MotionBoost > 0.0001)
    {
        float motionFactor = saturate(tex2Dlod(sMotionState, float4(0.25, 0.5, 0.0, 0.0)).r);
        effectiveIntensity += MotionBoost * motionFactor;
    }
    outEffectiveIntensity = effectiveIntensity;
#endif

#if ENABLE_DEBUG_VIEWS
    bool active = (effectiveIntensity > 0.0001) || (ViewMode != 0);
#else
    bool active = (effectiveIntensity > 0.0001);
#endif
    if (!active)
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
#if NSHDR_MOTION_COMPILED
    , nointerpolation float effectiveIntensity : TEXCOORD1
#endif
#if NSHDR_DYNAMIC_CSPACE
    , nointerpolation int mode : TEXCOORD2
    , nointerpolation float4 suppressParams : TEXCOORD3
#endif
    , out float4 outColor : SV_Target)
{
#if !NSHDR_MOTION_COMPILED
    const float effectiveIntensity = Intensity;
#endif
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
    // Ultra Fast 5-Tap Bilinear: 4 half-pixel LINEAR taps cover all 9 pixels of the 3x3 box (4 scalar UV adds).
    const float hx = 0.5 * px;
    const float hy = 0.5 * py;
    float xL = uv.x - hx;
    float xR = uv.x + hx;
    float yT = uv.y - hy;
    float yB = uv.y + hy;

    float q0 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(xL, yT, 0.0, 0.0)).rgb, mode);
    float q1 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(xR, yT, 0.0, 0.0)).rgb, mode);
    float q2 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(xL, yB, 0.0, 0.0)).rgb, mode);
    float q3 = ExtractLuma(tex2Dlod(ReShade::BackBuffer, float4(xR, yB, 0.0, 0.0)).rgb, mode);

    float maxQ = max(max(q0, q1), max(q2, q3));
    float minQ = min(min(q0, q1), min(q2, q3));

    float rangeQ   = max(maxQ, p4) - min(minQ, p4);
    float flatMask = saturate((rangeQ - FLAT_SKIP_RANGE) * FLAT_INV_SPAN);

    float localContrast  = (maxQ - minQ) * 1.5;
    float aliasingSignal = saturate((localContrast - suppressParams.x) * suppressParams.y);

    float spikeIntensity = max(max(0.0, p4 - maxQ), max(0.0, minQ - p4)) * 1.3333333;
    float fireflySignal  = saturate((spikeIntensity - suppressParams.z) * suppressParams.w);

    float suppressionMask = 1.0 - saturate(aliasingSignal + fireflySignal) * ArtifactProtection;

    // Exact D4-invariant 5-tap quadrant features reusing maxQ/minQ (44 FMAs, R2=0.92).
    float avgQDiff = 0.25 * ((q0 + q3) + (q1 + q2)) - p4;
    float anisoQ   = 0.5  * abs((q0 + q3) - (q1 + q2));
    float gradQ    = 0.5  * abs((q0 + q1) - (q2 + q3)) + 0.5 * abs((q0 + q2) - (q1 + q3));

    float rawResidual = (float)RunNeuralFast5(
        (NS_F1)p4, (NS_F1)avgQDiff,
        (NS_F1)(maxQ - p4), (NS_F1)(minQ - p4),
        (NS_F1)anisoQ, (NS_F1)gradQ);
#else
    // 9-Tap POINT sampling with pre-reduced coordinates (4 scalar adds instead of 16).
    float xL = uv.x - px;
    float xR = uv.x + px;
    float yT = uv.y - py;
    float yB = uv.y + py;

    float p0 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xL,   yT,   0.0, 0.0)).rgb, mode);
    float p1 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv.x, yT,   0.0, 0.0)).rgb, mode);
    float p2 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xR,   yT,   0.0, 0.0)).rgb, mode);
    float p3 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xL,   uv.y, 0.0, 0.0)).rgb, mode);
    float p5 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xR,   uv.y, 0.0, 0.0)).rgb, mode);
    float p6 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xL,   yB,   0.0, 0.0)).rgb, mode);
    float p7 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(uv.x, yB,   0.0, 0.0)).rgb, mode);
    float p8 = ExtractLuma(tex2Dlod(sBackBufferPoint, float4(xR,   yB,   0.0, 0.0)).rgb, mode);

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
        // Exact D4-invariant 9-tap features reusing maxCross/minCross/maxDiag/minDiag (56 FMAs vs 108, R2=0.974).
        float avgCrossDiff = 0.25 * ((p1 + p7) + (p3 + p5)) - p4;
        float avgDiagDiff  = 0.25 * ((p0 + p8) + (p2 + p6)) - p4;
        float anisoHV      = 0.5  * abs((p1 + p7) - (p3 + p5));
        float anisoDiag    = 0.5  * abs((p0 + p8) - (p2 + p6));

        float rawResidual = (float)RunNeuralFast9(
            (NS_F1)p4,
            (NS_F1)avgCrossDiff,   (NS_F1)avgDiagDiff,
            (NS_F1)(maxCross - p4), (NS_F1)(minCross - p4),
            (NS_F1)(maxDiag  - p4), (NS_F1)(minDiag  - p4),
            (NS_F1)anisoHV,        (NS_F1)anisoDiag);
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

    float delta = residual * flatMask * suppressionMask * effectiveIntensity;

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


#if NSHDR_OSD_COMPILED
// Compact 3x5 bitmap font for diagnostic OSD ("INT X.XX +X.XX").
uint GetOSDCharBits(uint glyphId)
{
    static const uint GLYPHS[15] = {
        31599u, // 0: '0'
        11415u, // 1: '1'
        29671u, // 2: '2'
        29647u, // 3: '3'
        23497u, // 4: '4'
        31183u, // 5: '5'
        31215u, // 6: '6'
        29266u, // 7: '7'
        31727u, // 8: '8'
        31695u, // 9: '9'
        2u,     // 10: '.'
        1488u,  // 11: '+'
        29847u, // 12: 'I'
        24557u, // 13: 'N'
        29842u  // 14: 'T'
    };
    return (glyphId < 15u) ? GLYPHS[glyphId] : 0u;
}

float3 EncodeOSDColor(float3 srgbCol, int mode)
{
    float3 lin = srgbCol * srgbCol; // fast ~gamma 2.0 to linear (0 pow())
    float3 result = srgbCol;
    if (mode == MODE_PQ)
    {
        float3 n = lin * (180.0 / 10000.0);
        result = float3(FastLinearToPQ(n.r), FastLinearToPQ(n.g), FastLinearToPQ(n.b));
    }
    else if (mode == MODE_SCRGB)
    {
        result = lin * (180.0 / 80.0);
    }
    return result;
}

void VS_MotionOSD(
    in uint id : SV_VertexID,
    out float4 position : SV_Position,
    out float2 localPx : TEXCOORD0,
    out nointerpolation float3 outMotionData : TEXCOORD1, // (motionFactor, addedBoost, effectiveIntensity)
    out nointerpolation int outMode : TEXCOORD2)
{
#if NSHDR_DYNAMIC_CSPACE
    outMode = ResolveColorSpaceMode();
#elif BUFFER_COLOR_SPACE == 3
    outMode = MODE_PQ;
#elif BUFFER_COLOR_SPACE == 2
    outMode = MODE_SCRGB;
#else
    outMode = MODE_SDR;
#endif

    if (!ShowMotionOSD)
    {
        position = float4(0.0, 0.0, 0.0, 0.0);
        localPx = float2(0.0, 0.0);
        outMotionData = float3(0.0, 0.0, 0.0);
        return;
    }

    float motionFactor = (MotionBoost > 0.0001)
        ? saturate(tex2Dlod(sMotionState, float4(0.25, 0.5, 0.0, 0.0)).r)
        : 0.0;
    float addedBoost = MotionBoost * motionFactor;
    outMotionData = float3(motionFactor, addedBoost, Intensity + addedBoost);

    // Tight 188x46 px quad anchored 16 px from top-right corner (avoids EOTF_Boost_X36 top-left OSD).
    const float boxW = 188.0;
    const float boxH = 46.0;
    const float u0 = (BUFFER_WIDTH - 16.0 - boxW) * BUFFER_RCP_WIDTH;
    const float u1 = (BUFFER_WIDTH - 16.0) * BUFFER_RCP_WIDTH;
    const float v0 = 16.0 * BUFFER_RCP_HEIGHT;
    const float v1 = (16.0 + boxH) * BUFFER_RCP_HEIGHT;

    float cx = (id == 1u || id == 3u || id == 4u) ? 1.0 : 0.0;
    float cy = (id == 2u || id == 4u || id == 5u) ? 1.0 : 0.0;

    float2 uv = float2(lerp(u0, u1, cx), lerp(v0, v1, cy));
    localPx = float2(cx * boxW, cy * boxH);
    position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

void PS_MotionOSD(
    float4 vpos : SV_Position,
    float2 localPx : TEXCOORD0,
    nointerpolation float3 motionData : TEXCOORD1,
    nointerpolation int mode : TEXCOORD2,
    out float4 outColor : SV_Target)
{
    uint px = uint(max(localPx.x, 0.0));
    uint py = uint(max(localPx.y, 0.0));

    float motionFactor = motionData.x;
    uint boostCenti = uint(round(clamp(motionData.y, 0.0, 9.99) * 100.0));
    uint totalCenti = uint(round(clamp(motionData.z, 0.0, 9.99) * 100.0));

    // Dark translucent backdrop with subtle 1px border
    float3 srgbCol = float3(0.06, 0.07, 0.09);
    float alpha = 0.80;
    if (px == 0u || px == 187u || py == 0u || py == 45u)
    {
        srgbCol = float3(0.32, 0.38, 0.46);
        alpha = 0.90;
    }

    // Row 1 (y = 8..22, 3x scale = 9x15 px glyphs, 12 px pitch): "INT X.XX +X.XX" (14 cells)
    if (px >= 10u && px < 178u && py >= 8u && py < 23u)
    {
        uint relX = px - 10u;
        uint relY = py - 8u;
        uint cell = relX / 12u;
        uint subX = relX - cell * 12u;
        if (subX < 9u)
        {
            uint col = subX / 3u;
            uint row = relY / 3u;
            uint glyphId = 99u; // space
            bool isBoostText = (cell >= 9u);

            if (cell == 0u) glyphId = 12u; // I
            else if (cell == 1u) glyphId = 13u; // N
            else if (cell == 2u) glyphId = 14u; // T
            else if (cell == 4u) glyphId = totalCenti / 100u;
            else if (cell == 5u) glyphId = 10u; // .
            else if (cell == 6u) glyphId = (totalCenti / 10u) % 10u;
            else if (cell == 7u) glyphId = totalCenti % 10u;
            else if (cell == 9u) glyphId = 11u; // +
            else if (cell == 10u) glyphId = boostCenti / 100u;
            else if (cell == 11u) glyphId = 10u; // .
            else if (cell == 12u) glyphId = (boostCenti / 10u) % 10u;
            else if (cell == 13u) glyphId = boostCenti % 10u;

            uint bits = GetOSDCharBits(glyphId);
            uint bitIdx = 14u - (row * 3u + col);
            if (((bits >> bitIdx) & 1u) != 0u)
            {
                srgbCol = isBoostText
                    ? lerp(float3(0.65, 0.75, 0.85), float3(0.25, 0.95, 0.65), motionFactor)
                    : float3(0.92, 0.95, 0.98);
                alpha = 1.0;
            }
        }
    }

    // Row 2 (y = 28..37, x = 10..177): Live motion meter bar (0% -> 100% of MotionBoost)
    if (px >= 10u && px < 178u && py >= 28u && py < 38u)
    {
        if (px == 10u || px == 177u || py == 28u || py == 37u)
        {
            srgbCol = float3(0.35, 0.42, 0.50);
            alpha = 0.95;
        }
        else
        {
            float barPos = float(px - 11u) / 165.0;
            if (barPos <= motionFactor)
            {
                srgbCol = lerp(float3(0.18, 0.72, 0.95), float3(0.25, 0.98, 0.58), barPos);
                alpha = 0.96;
            }
            else
            {
                srgbCol = float3(0.13, 0.15, 0.19);
                alpha = 0.85;
            }
        }
    }

    outColor = float4(EncodeOSDColor(srgbCol, mode), alpha);
}
#endif


technique Neural_Sharpen_HDR <
    ui_label = "Neural Sharpen HDR";
    ui_tooltip =
        "HDR-aware, GPU-optimized Neural Sharpening with FrameGen-safe camera-pan motion adaptation (based on Barbatos Neural Sharpen 2.8, MIT).\n"
        "Supports HDR10 PQ, scRGB Linear, and SDR.\n\n"
        "Preprocessor definitions:\n"
        "  NEURAL_SHARPEN_PROFILE: 1=Fast 9-Tap (default), 0=Ultra Fast 5-Tap, 2=Exact 12-Ch\n"
        "  NEURAL_SHARPEN_MODEL: 1=Model A (default), 2=Model B, 0=Both (UI selector)\n"
        "  ENABLE_MOTION_ADAPTATION: 1=On (default, ~0.005 ms), 0=Off\n"
        "  ENABLE_MOTION_OSD: 0=Off (default), 1=Live motion boost OSD (top-right)\n"
        "  ENABLE_DEBUG_VIEWS: 0=Off (default), 1=On\n"
        "  ALLOW_MANUAL_COLOR_SPACE: 0=Auto compile-time (default), 1=Manual selector\n"
        "  USE_FP16_CNN: 0=FP32 (default, faster on D3D11), 1=FP16\n"
        "  ENABLE_UI_TOOLTIPS: 1=On (default), 0=Off";
>
{
#if NSHDR_MOTION_COMPILED
    pass UpdateMotion
    {
        ComputeShader = CS_UpdateMotion<16, 16, 1>;
        DispatchSizeX = 1;
        DispatchSizeY = 1;
    }
#endif
    pass Apply
    {
        VertexShader = VS_Apply;
        PixelShader  = PS_Apply;
    }
#if NSHDR_OSD_COMPILED
    pass MotionOSD
    {
        VertexShader = VS_MotionOSD;
        PixelShader  = PS_MotionOSD;
        VertexCount  = 6;
        BlendEnable  = true;
        SrcBlend     = SRCALPHA;
        DestBlend    = INVSRCALPHA;
    }
#endif
}

} // namespace NeuralSharpenHDR
