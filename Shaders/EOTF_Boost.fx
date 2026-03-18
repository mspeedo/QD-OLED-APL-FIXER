/*
    EOTF Boost v8.0 - 1D APL-Only Lookup
    =================================

    Purpose
    -------
    This shader boosts HDR luminance to compensate for OLED / display ABL behavior
    using a simplified measured 1D lookup table and a hybrid pixel participation model.

    This version applies compensation as a multiplicative gain in absolute nits:

        scene_gain = measured_compensation(APL) shaped by LUT weight and strength
        pixel_gain = scene_gain ^ participation

    where:
        - APL is the scene average picture level metric (0..1, shown as 0..100%)
        - compensation > 1 means the display measured darker than the requested target

    This version collapses the original measured 2D APL x nits LUT into a single
    representative compensation value per APL row (anchored near 109 nits), because
    the per-row variation across target nits was small.

    The lookup table is NOT used as a direct inverse solve.
    Instead, it is used as a SHAPE / WEIGHT map that drives a capped boost model.

*/

#include "ReShade.fxh"

// --- COMPILE-TIME DEBUG FEATURE SWITCHES ---
// Set to 1 to compile the graph feature in, or 0 to strip it out completely.
#ifndef ENABLE_APL_GRAPH
    #define ENABLE_APL_GRAPH 0
#endif

// --- UI SETTINGS ---

uniform int APLInputMode <
    ui_type = "combo";
    ui_items = "scRGB Normalized\0PQ Decoded Normalized\0";
    ui_label = "APL Input Mode";
    ui_tooltip = "Selects how the shader interprets scene luminance for the APL metric. scRGB uses BT.709 luma scaled by Reference White. PQ uses ST.2084-decoded BT.2020 luma scaled by Reference White.";
> = 1;

uniform float APLReferenceWhiteNits <
    ui_type = "slider";
    ui_min = 10.0; ui_max = 1000.0;
    ui_label = "APL Reference White (nits)";
    ui_tooltip = "Reference white used only for the APL metric normalization. It does not directly clamp output nits or change the graph axes.";
> = 250.0;

uniform int APLGridSize <
    ui_type = "slider";
    ui_min = 4; ui_max = 32;
    ui_label = "APL Grid Size";
    ui_tooltip = "APL sample grid resolution. Total samples = Grid Size x Grid Size. Higher values are more stable but cost more.";
> = 32;

uniform float APLTrigger <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.95;
    ui_label = "APL Trigger";
    ui_tooltip = "Fade-in threshold for the boost based on the smoothed APL metric. Below this level the effect is reduced or disabled. 10% APL on the graph is exactly the threshold when this is set to 0.10.";
> = 0.12;

uniform float MaxAPLBoostStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "Max APL Boost Strength";
    ui_tooltip = "Scales the measured APL compensation in log-gain space before per-pixel participation is applied. 1.0 means full measured compensation at maximum LUT weight. Values below 1.0 under-compensate. Values above 1.0 intentionally over-compensate.";
> = 0.5;

uniform float BoostTGamma <
    ui_type = "slider";
    ui_min = 0.25; ui_max = 2.50;
    ui_label = "Boost LUT Gamma";
    ui_tooltip = "Reshapes the LUT-derived boost weight before Max APL Boost Strength is applied. Values below 1.0 increase lower-weight regions more. Values above 1.0 reduce them more.";
> = 1.0;

uniform float BoostRollOff <
    ui_type = "slider";
    ui_min = 1000.0; ui_max = 1500.0;
    ui_label = "Boost roll off end";
    ui_tooltip = "Desired output anchor of the PQ rational highlight shoulder in nits. The shader dynamically places the knee from the current smoothed APL so the boosted curve lands on this endpoint more consistently across APL levels.";
> = 1000.0;

uniform float BoostRollOffLogFactor <
    ui_type = "slider";
    ui_min = 0; ui_max = 1.0;
    ui_label = "Boost roll off softness";
    ui_tooltip = "Controls the PQ rational shoulder softness after boost. Lower values place the knee later and preserve highlight separation more strongly. Higher values start the shoulder earlier and make the compression more gradual.";
> = 1.0;

static const float PixelParticipationStartNits = 1.0;

static const float PixelParticipationFullNits = 40.0;

static const float PixelParticipationGamma = 1.0;

uniform float PixelParticipationFloor <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_label = "Shadow Protection Floor";
    ui_tooltip = "Minimum share of the APL-derived scene compensation applied to every pixel before the luminance-weighted participation ramp adds the remainder. Higher values track the measured ABL behavior more faithfully. Lower values behave more like a perceptual shadow-protection model.";
> = 1.0;

uniform float TransitionSpeed <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "APL Smoothing Time (s)";
    ui_tooltip = "Temporal smoothing time constant for the live APL metric in seconds. 0 = disabled. FPS-independent. This affects live boosting and OSD values, but the graph uses its own Graph APL % slider.";
> = 0.25;

uniform float SaturationComp <
    ui_type = "slider";
    ui_min = 0.5; ui_max = 1.5;
    ui_label = "Saturation Compensation";
    ui_tooltip = "Scales chroma reinjection after luma boosting. 1.0 = neutral. Values above 1.0 restore or exaggerate saturation in boosted regions.";
> = 1.0;

uniform float SIGNAL_REFERENCE_NITS <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 200.0;
    ui_label = "scRGB Signal Reference (nits)";
    ui_tooltip = "Reference nits for scRGB signal conversion. Standard scRGB uses 80 nits per 1.0 signal. Used only when APL Input Mode = scRGB Normalized.";
> = 80.0;

uniform bool ShowOSD <
    ui_label = "Show APL / Metric Stats";
    ui_tooltip = "Displays the current smoothed APL percentage and the maximum sampled raw luma value from the APL analysis pass.";
> = false;

uniform float OSDBrightness <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_label = "OSD Brightness";
    ui_tooltip = "Controls OSD and graph overlay brightness.";
> = 0.5;

uniform float FrameTime < source = "frametime"; >;

#if ENABLE_APL_GRAPH
uniform bool ShowAPLGraph <
    ui_label = "Show APL EOTF Debug Graph";
    ui_tooltip = "Shows the analysis graph. Blue dashed = reference, Light blue = real 2D measured LUT output, Green = shader remapped target using the live 1D nits-domain multiplicative model with the PQ rational shoulder, Gray = projected measured output after that compensation using the real 2D table.";
> = false;

uniform float GraphAPLIndex <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 100.0;
    ui_label = "Graph APL (%)";
    ui_tooltip = "Continuous APL value used only by the graph overlay. The graph interpolates between measured LUT rows like the live shader instead of snapping to discrete APL rows.";
> = 50.0;

uniform float GraphAxisMaxNits <
    ui_type = "slider";
    ui_min = 1000.0; ui_max = 10000.0;
    ui_label = "Graph Axis Max (nits)";
    ui_tooltip = "Maximum nits shown on both graph axes. Raising it lets you inspect curve behavior beyond 1000-nit input without changing the live shader.";
> = 1000.0;

uniform float GraphOpacity <
    ui_type = "slider";
    ui_min = 0.05; ui_max = 1.0;
    ui_label = "Graph Opacity";
    ui_tooltip = "Opacity of the graph overlay background and curves.";
> = 0.5;

uniform bool GraphUsePQSpace <
    ui_label = "Graph PQ-Encoded Axes";
    ui_tooltip = "Renders the graph in PQ-encoded space instead of linear nits. Axis labels remain in nits.";
> = true;
#endif

// --- TEXTURES ---

texture TexAPL
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerAPL
{
    Texture = TexAPL;
};

texture TexAPLInstant
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerAPLInstant
{
    Texture = TexAPLInstant;
};

texture TexAPLPrev
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerAPLPrev
{
    Texture = TexAPLPrev;
};

#if ENABLE_APL_GRAPH
texture TexGraphParams
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerGraphParams
{
    Texture = TexGraphParams;
};

texture TexBoosted
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler SamplerBoosted
{
    Texture = TexBoosted;
};
#endif


// --- FUNCTIONS ---

float GetLuma709(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float GetLuma2020(float3 color)
{
    return dot(color, float3(0.2627, 0.6780, 0.0593));
}

float3 PQToLinearBT2100(float3 v)
{
    // ST.2084 / PQ EOTF
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;

    float3 vp = pow(saturate(v), 1.0 / m2);
    float3 num = max(vp - c1, 0.0);
    float3 den = c2 - c3 * vp;
    return pow(num / max(den, 1e-6), 1.0 / m1); // 0..1 relative to 10000 nits
}

float LinearToPQBT2100(float linearValue)
{
    // ST.2084 / PQ OETF, input is 0..1 relative to 10000 nits
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;

    float L = saturate(linearValue);
    float Lm1 = pow(L, m1);
    float num = c1 + c2 * Lm1;
    float den = 1.0 + c3 * Lm1;
    return pow(num / max(den, 1e-6), m2);
}

float NitsToPQ(float nits)
{
    return LinearToPQBT2100(saturate(nits / 10000.0));
}

float GetSignalLuma(float3 color)
{
    return (APLInputMode == 1) ? GetLuma2020(color) : GetLuma709(color);
}

float GetSceneNitsFromColor(float3 color)
{
    if (APLInputMode == 1)
    {
        float3 linearPQ = PQToLinearBT2100(color);
        return GetLuma2020(linearPQ) * 10000.0;
    }

    return GetLuma709(max(color, 0.0.xxx)) * SIGNAL_REFERENCE_NITS;
}

float GetAPLMetricSample(float3 color)
{
    return saturate(GetSceneNitsFromColor(color) / max(APLReferenceWhiteNits, 1.0));
}

float GetDigit(int digit, float2 uv)
{
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 0.0;

    int patterns[10] = { 31599, 9362, 29671, 29391, 23497, 31183, 31215, 29257, 31727, 31695 };
    int num = patterns[clamp(digit, 0, 9)];
    int x = int(uv.x * 3.0);
    int y = int((1.0 - uv.y) * 5.0);

    return (num >> (x + y * 3)) & 1;
}

float GetPercent(float2 uv)
{
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 0.0;

    bool slash = abs(uv.x - (1.0 - uv.y)) < 0.15;
    bool circles = (distance(uv, float2(0.3, 0.25)) < 0.2) || (distance(uv, float2(0.7, 0.75)) < 0.2);

    return (slash || circles) ? 1.0 : 0.0;
}

float GetDot(float2 uv)
{
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 0.0;

    return (uv.x > 0.35 && uv.x < 0.65 && uv.y > 0.00 && uv.y < 0.20) ? 1.0 : 0.0;
}

float Remap01(float x, float a, float b)
{
    return saturate((x - a) / max(b - a, 1e-6));
}

float SmootherStep01(float x)
{
    x = saturate(x);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

float SegmentLerp(float x, float x0, float y0, float x1, float y1)
{
    return lerp(y0, y1, Remap01(x, x0, x1));
}

static const int APL_COUNT = 10;
static const int NIT_COUNT = 24;

static const float APL_POINTS[APL_COUNT] =
{
    3.000000, 5.000000, 7.000000, 10.000000, 14.000000, 18.000000, 22.000000, 25.000000, 35.000000, 50.000000
};

static const float NIT_POINTS[NIT_COUNT] =
{
    10.302358, 13.132379, 16.611621, 20.870708, 26.064850, 32.378420, 40.030390, 49.280814, 60.438551, 73.870492, 90.012580, 109.383004, 132.598006, 160.390856, 193.634650, 233.369755, 280.836899, 337.517130, 405.180165, 485.942985, 582.341000, 697.414628, 834.814884, 998.932391
};

// Original 2D table collapsed to one representative compensation value per APL row.
// These anchors are taken near 109 nits, which tracks the row average very closely
// while preserving the stronger APL dependence that matters most.
static const float COMP_APL_1D[APL_COUNT] =
{
    1.000000, // APL 3
    1.352556, // APL 5
    1.673813, // APL 7
    2.151376, // APL 10
    2.455437, // APL 14
    2.636610, // APL 18
    2.760512, // APL 22
    2.839076, // APL 25
    3.064304, // APL 35
    3.345667  // APL 50
};

static const float COMP_MIN = 1.0;
static const float COMP_MAX = 3.345667;

int FindAPLIndex(float aplPct)
{
    int idx = 0;

    [loop]
    for (int i = 0; i < APL_COUNT - 1; ++i)
    {
        if (aplPct >= APL_POINTS[i + 1])
            idx = i + 1;
    }

    return clamp(idx, 0, APL_COUNT - 2);
}

float LookupMeasuredComp1D(float aplPct)
{
    float clampedAPL = clamp(aplPct, APL_POINTS[0], APL_POINTS[APL_COUNT - 1]);
    int a0 = FindAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        APL_POINTS[a0], COMP_APL_1D[a0],
        APL_POINTS[a1], COMP_APL_1D[a1]
    );
}

// LUT shapes the scene-compensation weight only. Final response is a nits-domain gain.
float MeasuredCompToBoostT(float comp)
{
    return saturate((comp - COMP_MIN) / max(COMP_MAX - COMP_MIN, 1e-6));
}

float ShapeBoostT(float t)
{
    float safeT = saturate(t);

    if (abs(BoostTGamma - 1.0) <= 1e-6)
        return safeT;

    return pow(safeT, BoostTGamma);
}

float LookupMeasuredBoostT1D(float aplPct)
{
    return ShapeBoostT(MeasuredCompToBoostT(LookupMeasuredComp1D(aplPct)));
}

float ComputeAPLBoostFader(float currentAPL)
{
    return smoothstep(APLTrigger, min(APLTrigger + 0.05, 1.0), currentAPL);
}

float ComputeTemporalBlendFactor(float smoothingSeconds)
{
    if (smoothingSeconds <= 1e-6)
        return 1.0;

    float dtSeconds = max(FrameTime, 0.0) * 0.001;
    return saturate(1.0 - exp(-dtSeconds / max(smoothingSeconds, 1e-6)));
}

float ComputePixelParticipationWeight(float inputNits)
{
    float safeInputNits = max(inputNits, 1e-4);
    float startNits = max(PixelParticipationStartNits, 1e-4);
    float fullNits = max(PixelParticipationFullNits, startNits + 1e-4);

    float logStart = log2(startNits);
    float logFull = log2(fullNits);
    float t = (log2(safeInputNits) - logStart) / max(logFull - logStart, 1e-6);
    t = SmootherStep01(t);

    return pow(saturate(t), PixelParticipationGamma);
}

float ComputePixelParticipation(float inputNits)
{
    float floorShare = saturate(PixelParticipationFloor);

    if (floorShare >= 0.9999)
        return 1.0;

    float w_pix = ComputePixelParticipationWeight(inputNits);
    return lerp(floorShare, 1.0, w_pix);
}

float ComputePixelGainFromSceneLogGain(float sceneLogGain, float inputNits)
{
    return exp2(sceneLogGain * ComputePixelParticipation(inputNits));
}

static const float RATIONAL_SHOULDER_MIN_SPAN_FACTOR = 0.08;

float ComputeRationalShoulderStartPQ(float anchorBoostedPQ, float anchorOutPQ)
{
    float compressionPQ = max(anchorBoostedPQ - anchorOutPQ, 0.0);
    float spanFactor = lerp(RATIONAL_SHOULDER_MIN_SPAN_FACTOR, 1.0, saturate(BoostRollOffLogFactor));
    float startPQ = anchorOutPQ - compressionPQ * spanFactor;

    return min(max(startPQ, 0.0), max(anchorOutPQ - 1e-6, 0.0));
}

float EvaluateRationalShoulderPQ(float boostedPQ, float rollOffStartPQ, float anchorBoostedPQ, float anchorOutPQ)
{
    if (rollOffStartPQ <= 0.0 || boostedPQ <= rollOffStartPQ)
        return boostedPQ;

    float inSpan = max(anchorBoostedPQ - rollOffStartPQ, 1e-6);
    float outSpan = max(anchorOutPQ - rollOffStartPQ, 1e-6);

    if (anchorBoostedPQ <= rollOffStartPQ + 1e-6 || anchorOutPQ <= rollOffStartPQ + 1e-6)
        return min(boostedPQ, 1.0);

    // Anchored Michaelis-Menten style shoulder in PQ space.
    // u = 1 lands exactly on the chosen output anchor and the slope at the knee stays at 1.
    float q = saturate(outSpan / inSpan);
    float a = max(1.0 / max(q, 1e-6) - 1.0, 0.0);
    float u = max((boostedPQ - rollOffStartPQ) / inSpan, 0.0);
    float v = u / (1.0 + a * u);

    return min(rollOffStartPQ + inSpan * v, 1.0);
}

float EvaluateRationalShoulderNits(float boostedNits, float rollOffStartNits, float anchorBoostedNits, float anchorOutNits)
{
    float safeBoostedNits = max(boostedNits, 0.0);
    float safeRollOffStartNits = max(rollOffStartNits, 0.0);
    float safeAnchorBoostedNits = max(anchorBoostedNits, 0.0);
    float safeAnchorOutNits = max(anchorOutNits, 0.0);

    if (safeRollOffStartNits <= 0.0 || safeBoostedNits <= safeRollOffStartNits)
        return safeBoostedNits;

    float boostedPQ = NitsToPQ(safeBoostedNits);
    float rollOffStartPQ = NitsToPQ(safeRollOffStartNits);
    float anchorBoostedPQ = NitsToPQ(safeAnchorBoostedNits);
    float anchorOutPQ = NitsToPQ(safeAnchorOutNits);
    float rolledPQ = EvaluateRationalShoulderPQ(boostedPQ, rollOffStartPQ, anchorBoostedPQ, anchorOutPQ);

    return max(PQToLinearBT2100(rolledPQ.xxx).x * 10000.0, 0.0);
}

float ApplyRationalShoulderToNits(float boostedNits, float originalNits, float rollOffStartNits, float anchorBoostedNits, float anchorOutNits)
{
    float rolledNits = EvaluateRationalShoulderNits(boostedNits, rollOffStartNits, anchorBoostedNits, anchorOutNits);
    return max(rolledNits, originalNits);
}

float ComputeSceneGainNoRolloff(float currentAPL, float pixelBoostT)
{
    float aplPct = saturate(currentAPL) * 100.0;
    float measuredComp = max(LookupMeasuredComp1D(aplPct), 1.0);
    float fader = ComputeAPLBoostFader(currentAPL);

    // Build scene compensation in log-gain space so:
    //   - measuredComp = 1.0  -> no change
    //   - strength = 1.0      -> full measured compensation at maximum LUT weight
    //   - strength < 1.0      -> partial compensation
    //   - strength > 1.0      -> intentional over-compensation
    float gainExponent = max(MaxAPLBoostStrength * pixelBoostT * fader, 0.0);
    return exp2(log2(measuredComp) * gainExponent);
}

float ComputePixelGainNoRolloff(float currentAPL, float inputNits, float pixelBoostT)
{
    float sceneGain = max(ComputeSceneGainNoRolloff(currentAPL, pixelBoostT), 1.0);
    float participation = ComputePixelParticipation(inputNits);

    // Hybrid participation keeps some global compensation on all pixels, while
    // brighter pixels smoothly receive the remaining share.
    return exp2(log2(sceneGain) * participation);
}

float SignalLumaToNits(float signalLuma)
{
    if (APLInputMode == 1)
        return max(PQToLinearBT2100(signalLuma.xxx).x * 10000.0, 0.0);

    return max(signalLuma, 0.0) * SIGNAL_REFERENCE_NITS;
}

float NitsToSignalLuma(float nits)
{
    if (APLInputMode == 1)
        return NitsToPQ(max(nits, 0.0));

    return max(nits, 0.0) / SIGNAL_REFERENCE_NITS;
}

float ComputeBoostedTargetNitsFromBoostTNoRolloff(float currentAPL, float inputNits, float pixelBoostT)
{
    float safeInputNits = max(inputNits, 0.0);
    float pixelGain = ComputePixelGainNoRolloff(currentAPL, safeInputNits, pixelBoostT);

    return safeInputNits * pixelGain;
}

float ComputeBoostedTargetNitsNoRolloff(float currentAPL, float inputNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float pixelBoostT = LookupMeasuredBoostT1D(currentAPL * 100.0);
    return ComputeBoostedTargetNitsFromBoostTNoRolloff(currentAPL, safeInputNits, pixelBoostT);
}

float ComputeRollOffAnchorBoostedNits(float currentAPL)
{
    float rollOffEndNits = max(BoostRollOff, 0.0);

    if (rollOffEndNits <= 0.0)
        return 0.0;

    return ComputeBoostedTargetNitsNoRolloff(currentAPL, max(rollOffEndNits, 1e-4));
}

float SolveDynamicRollOffStartNits(float currentAPL)
{
    float rollOffEndNits = max(BoostRollOff, 0.0);

    if (rollOffEndNits <= 0.0)
        return 0.0;

    float referenceInputNits = max(rollOffEndNits, 1e-4);
    float referenceBoostedNits = ComputeRollOffAnchorBoostedNits(currentAPL);

    if (referenceBoostedNits <= referenceInputNits + 1e-4)
        return referenceBoostedNits;

    float anchorOutPQ = NitsToPQ(referenceInputNits);
    float anchorBoostedPQ = NitsToPQ(referenceBoostedNits);
    float rollOffStartPQ = ComputeRationalShoulderStartPQ(anchorBoostedPQ, anchorOutPQ);

    return max(PQToLinearBT2100(rollOffStartPQ.xxx).x * 10000.0, 0.0);
}

float ApplyBoostWithRationalRolloff(float signalLuma, float currentAPL, float pixelBoostT, float rollOffStartNits, float anchorBoostedNits)
{
    float originalNits = SignalLumaToNits(signalLuma);
    float fullyBoostedNits = ComputeBoostedTargetNitsFromBoostTNoRolloff(currentAPL, originalNits, pixelBoostT);
    float rolledNits = ApplyRationalShoulderToNits(fullyBoostedNits, originalNits, rollOffStartNits, anchorBoostedNits, BoostRollOff);

    return NitsToSignalLuma(rolledNits);
}

float ApplyBoostWithRationalRolloffFromSceneLogGain(float signalLuma, float sceneLogGain, float rollOffStartNits, float anchorBoostedNits)
{
    float originalNits = SignalLumaToNits(signalLuma);
    float safeOriginalNits = max(originalNits, 0.0);
    float fullyBoostedNits = safeOriginalNits * ComputePixelGainFromSceneLogGain(sceneLogGain, safeOriginalNits);
    float rolledNits = ApplyRationalShoulderToNits(fullyBoostedNits, safeOriginalNits, rollOffStartNits, anchorBoostedNits, BoostRollOff);

    return NitsToSignalLuma(rolledNits);
}

float ComputeBoostedTargetNitsFromBoostT(float currentAPL, float inputNits, float pixelBoostT, float rollOffStartNits, float anchorBoostedNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float signalLuma = NitsToSignalLuma(safeInputNits);

    return SignalLumaToNits(ApplyBoostWithRationalRolloff(signalLuma, currentAPL, pixelBoostT, rollOffStartNits, anchorBoostedNits));
}




#if ENABLE_APL_GRAPH
// Restored graph-only 2D measurement table from the original shader.
// Live boost logic stays on the simplified 1D LUT path.
static const float GRAPH_COMP_TABLE_2D[APL_COUNT * NIT_COUNT] =
{
    // APL 3
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,

    // APL 5
    1.275485, 1.279098, 1.298112, 1.311262, 1.320233, 1.306626, 1.336365, 1.320144,
    1.337008, 1.337614, 1.354279, 1.352556, 1.347158, 1.353409, 1.354995, 1.357722,
    1.360410, 1.379355, 1.375188, 1.377428, 1.380128, 1.378296, 1.394903, 1.403779,

    // APL 7
    1.636865, 1.586585, 1.600228, 1.632120, 1.646603, 1.656140, 1.653732, 1.648574,
    1.659723, 1.638967, 1.676518, 1.673813, 1.667162, 1.672429, 1.678772, 1.682167,
    1.684833, 1.688457, 1.682479, 1.705485, 1.708849, 1.722217, 1.732755, 1.743012,

    // APL 10
    2.222370, 2.139135, 2.131274, 2.103239, 2.119450, 2.139539, 2.144950, 2.119860,
    2.154030, 2.144674, 2.181553, 2.151376, 2.139477, 2.150906, 2.153600, 2.140693,
    2.162291, 2.190399, 2.181303, 2.170379, 2.194950, 2.195467, 2.208124, 2.226317,

    // APL 14
    2.574646, 2.466301, 2.477981, 2.440763, 2.439207, 2.453153, 2.457554, 2.437744,
    2.459988, 2.467005, 2.474901, 2.455437, 2.446400, 2.453859, 2.454635, 2.467066,
    2.467625, 2.465686, 2.466095, 2.466487, 2.476404, 2.484286, 2.496364, 2.518080,

    // APL 18
    2.748700, 2.653554, 2.657380, 2.630880, 2.595614, 2.617234, 2.624612, 2.604644,
    2.621808, 2.624117, 2.640117, 2.636610, 2.604680, 2.617061, 2.615594, 2.612188,
    2.617207, 2.621120, 2.624654, 2.626987, 2.640365, 2.633628, 2.665580, 2.675877,

    // APL 22
    2.916959, 2.818433, 2.786699, 2.777913, 2.748502, 2.748234, 2.764812, 2.754077,
    2.767587, 2.764558, 2.776648, 2.760512, 2.734844, 2.742903, 2.740379, 2.745571,
    2.753666, 2.757212, 2.750390, 2.758649, 2.770630, 2.772743, 2.790399, 2.814688,

    // APL 25
    3.005716, 2.911831, 2.892784, 2.875547, 2.824210, 2.833016, 2.856895, 2.834469,
    2.853785, 2.843423, 2.864118, 2.839076, 2.816207, 2.827746, 2.825887, 2.840115,
    2.842497, 2.844660, 2.835615, 2.839335, 2.859106, 2.857170, 2.876447, 2.900269,

    // APL 35
    3.227778, 3.139621, 3.086106, 3.101464, 3.053503, 3.050585, 3.063989, 3.034956,
    3.074032, 3.056235, 3.082187, 3.064304, 3.033025, 3.038294, 3.037854, 3.051513,
    3.057893, 3.059339, 3.042766, 3.063924, 3.067941, 3.076905, 3.098045, 3.124520,

    // APL 50
    3.605147, 3.469041, 3.422317, 3.396852, 3.379085, 3.350383, 3.331431, 3.333736,
    3.385554, 3.346382, 3.380603, 3.345667, 3.325029, 3.308191, 3.317324, 3.329362,
    3.338411, 3.331323, 3.321922, 3.330550, 3.342307, 3.347892, 3.376754, 3.407354
};

float GraphTableComp2D(int aplIdx, int nitIdx)
{
    return GRAPH_COMP_TABLE_2D[aplIdx * NIT_COUNT + nitIdx];
}

float GetGraphMeasuredMaxInputNits()
{
    return NIT_POINTS[NIT_COUNT - 1];
}

int FindNitIndex(float inputNits)
{
    int idx = 0;

    [loop]
    for (int i = 0; i < NIT_COUNT - 1; ++i)
    {
        if (inputNits >= NIT_POINTS[i + 1])
            idx = i + 1;
    }

    return clamp(idx, 0, NIT_COUNT - 2);
}

float LookupGraphCompForAPLRow2D(int aplIdx, float inputNits)
{
    float clampedNits = clamp(inputNits, NIT_POINTS[0], NIT_POINTS[NIT_COUNT - 1]);
    int n0 = FindNitIndex(clampedNits);
    int n1 = min(n0 + 1, NIT_COUNT - 1);

    return SegmentLerp(
        clampedNits,
        NIT_POINTS[n0], GraphTableComp2D(aplIdx, n0),
        NIT_POINTS[n1], GraphTableComp2D(aplIdx, n1)
    );
}

float LookupMeasuredComp2DGraph(float aplPct, float inputNits)
{
    float clampedAPL = clamp(aplPct, APL_POINTS[0], APL_POINTS[APL_COUNT - 1]);
    int a0 = FindAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        APL_POINTS[a0], LookupGraphCompForAPLRow2D(a0, inputNits),
        APL_POINTS[a1], LookupGraphCompForAPLRow2D(a1, inputNits)
    );
}

float SampleRealMeasuredOutputNitsForAPL(float aplPct, float targetNits)
{
    float comp = max(LookupMeasuredComp2DGraph(aplPct, targetNits), 1e-6);
    return targetNits / comp;
}

float SampleApproxMeasuredOutputNitsForAPL(float aplPct, float targetNits)
{
    float comp = max(LookupMeasuredComp1D(aplPct), 1e-6);
    return targetNits / comp;
}

float ComputeGraphBoostedTargetNitsWithStart(float aplPct, float inputNits, float rollOffStartNits, float anchorBoostedNits)
{
    float currentAPL = saturate(aplPct / 100.0);
    float safeInputNits = max(inputNits, 0.0);
    float pixelBoostT = LookupMeasuredBoostT1D(aplPct);

    return ComputeBoostedTargetNitsFromBoostT(currentAPL, safeInputNits, pixelBoostT, rollOffStartNits, anchorBoostedNits);
}

float GetAPLMaxMeasuredNits(float aplPct)
{
    float maxMeasured = 0.0;

    [unroll]
    for (int i = 0; i < NIT_COUNT; ++i)
    {
        float targetNits = NIT_POINTS[i];
        maxMeasured = max(maxMeasured, SampleRealMeasuredOutputNitsForAPL(aplPct, targetNits));
    }

    return maxMeasured;
}

float SampleCorrectedOutputNitsForAPL(float aplPct, float boostedTargetNits, float maxMeasuredNits)
{
    float comp = max(LookupMeasuredComp2DGraph(aplPct, boostedTargetNits), 1e-6);
    return min(boostedTargetNits / comp, maxMeasuredNits);
}

float GraphAxisCoordinateWithPQMax(float nits, float axisMaxNits, float axisMaxPQ)
{
    float safeNits = max(nits, 0.0);
    float safeAxisMaxNits = max(axisMaxNits, 1.0);

    if (GraphUsePQSpace)
        return NitsToPQ(safeNits) / max(axisMaxPQ, 1e-6);

    return safeNits / safeAxisMaxNits;
}

float GraphAxisCoordinate(float nits, float axisMaxNits)
{
    float safeAxisMaxNits = max(axisMaxNits, 1.0);
    float axisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(safeAxisMaxNits), 1e-6) : 0.0;
    return GraphAxisCoordinateWithPQMax(nits, safeAxisMaxNits, axisMaxPQ);
}

float GraphTickValueFromFractionWithPQMax(float frac, float axisMaxNits, float axisMaxPQ)
{
    float safeAxisMaxNits = max(axisMaxNits, 1e-6);
    float safeFrac = saturate(frac);

    if (!GraphUsePQSpace)
        return safeAxisMaxNits * safeFrac;

    float tickPQ = axisMaxPQ * safeFrac;
    return max(PQToLinearBT2100(tickPQ.xxx).x * 10000.0, 0.0);
}

float GraphTickValueFromFraction(float frac, float axisMaxNits)
{
    float safeAxisMaxNits = max(axisMaxNits, 1e-6);
    float axisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(safeAxisMaxNits), 1e-6) : 0.0;
    return GraphTickValueFromFractionWithPQMax(frac, safeAxisMaxNits, axisMaxPQ);
}

float GraphSampleNitsFromFraction(float frac, float axisMaxNits, float axisMaxPQ)
{
    return GraphTickValueFromFractionWithPQMax(frac, axisMaxNits, axisMaxPQ);
}

float2 ToGraphPointWithPQMax(float2 graphPos, float2 graphSize, float axisMaxNits, float axisMaxPQ, float xNits, float yNits)
{
    float nx = saturate(GraphAxisCoordinateWithPQMax(xNits, axisMaxNits, axisMaxPQ));
    float ny = GraphAxisCoordinateWithPQMax(yNits, axisMaxNits, axisMaxPQ);
    return graphPos + float2(nx * graphSize.x, (1.0 - ny) * graphSize.y);
}

float2 ToGraphPoint(float2 graphPos, float2 graphSize, float axisMaxNits, float xNits, float yNits)
{
    float safeAxisMaxNits = max(axisMaxNits, 1.0);
    float axisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(safeAxisMaxNits), 1e-6) : 0.0;
    return ToGraphPointWithPQMax(graphPos, graphSize, safeAxisMaxNits, axisMaxPQ, xNits, yNits);
}

float DistanceToSegment(float2 p, float2 a, float2 b, out float h)
{
    float2 pa = p - a;
    float2 ba = b - a;
    float denom = max(dot(ba, ba), 1e-6);
    h = saturate(dot(pa, ba) / denom);
    return length(pa - ba * h);
}

float DrawGraphLine(float2 p, float2 a, float2 b, float thickness)
{
    float pad = thickness * 2.2;
    float2 bbMin = min(a, b) - pad.xx;
    float2 bbMax = max(a, b) + pad.xx;

    if (p.x < bbMin.x || p.x > bbMax.x || p.y < bbMin.y || p.y > bbMax.y)
        return 0.0;

    float h;
    float d = DistanceToSegment(p, a, b, h);
    return 1.0 - smoothstep(thickness, thickness * 1.8, d);
}

float DrawGraphDashedLine(float2 p, float2 a, float2 b, float thickness, float dashCount)
{
    float pad = thickness * 2.2;
    float2 bbMin = min(a, b) - pad.xx;
    float2 bbMax = max(a, b) + pad.xx;

    if (p.x < bbMin.x || p.x > bbMax.x || p.y < bbMin.y || p.y > bbMax.y)
        return 0.0;

    float h;
    float d = DistanceToSegment(p, a, b, h);
    float lineMask = 1.0 - smoothstep(thickness, thickness * 1.8, d);
    float dashMask = step(frac(h * dashCount), 0.55);
    return lineMask * dashMask;
}

float DrawGraphRect(float2 p, float2 minP, float2 maxP, float thickness)
{
    float inside = step(minP.x, p.x) * step(minP.y, p.y) * step(p.x, maxP.x) * step(p.y, maxP.y);
    float left   = 1.0 - smoothstep(thickness, thickness * 1.8, abs(p.x - minP.x));
    float right  = 1.0 - smoothstep(thickness, thickness * 1.8, abs(p.x - maxP.x));
    float top    = 1.0 - smoothstep(thickness, thickness * 1.8, abs(p.y - minP.y));
    float bottom = 1.0 - smoothstep(thickness, thickness * 1.8, abs(p.y - maxP.y));
    return inside * saturate(left + right + top + bottom);
}

int CountDigitsInt(int value)
{
    int v = max(value, 0);

    if (v >= 10000) return 5;
    if (v >= 1000) return 4;
    if (v >= 100) return 3;
    if (v >= 10) return 2;
    return 1;
}

float DrawDigitAt(float2 texcoord, float2 topRight, float scale, float aspect, int digit)
{
    float2 uv = texcoord;
    uv.x *= aspect;

    float2 anchor = topRight;
    anchor.x *= aspect;

    uv -= anchor;
    uv.x = -uv.x;

    return GetDigit(digit, uv / scale);
}

float DrawNumberRightAligned(float2 texcoord, float2 topRight, float scale, float aspect, int value)
{
    int v = max(value, 0);
    int digits = CountDigitsInt(v);
    float stepX = (scale * 0.82) / max(aspect, 1e-6);
    float mask = 0.0;

    mask += DrawDigitAt(texcoord, topRight, scale, aspect, v % 10);

    if (digits >= 2)
        mask += DrawDigitAt(texcoord, topRight - float2(stepX, 0.0), scale, aspect, (v / 10) % 10);

    if (digits >= 3)
        mask += DrawDigitAt(texcoord, topRight - float2(stepX * 2.0, 0.0), scale, aspect, (v / 100) % 10);

    if (digits >= 4)
        mask += DrawDigitAt(texcoord, topRight - float2(stepX * 3.0, 0.0), scale, aspect, (v / 1000) % 10);

    if (digits >= 5)
        mask += DrawDigitAt(texcoord, topRight - float2(stepX * 4.0, 0.0), scale, aspect, (v / 10000) % 10);

    return saturate(mask);
}

float DrawNumberCentered(float2 texcoord, float2 centerTop, float scale, float aspect, int value)
{
    int digits = CountDigitsInt(value);
    float stepX = (scale * 0.82) / max(aspect, 1e-6);
    float totalWidth = stepX * float(max(digits - 1, 0));
    float2 topRight = centerTop + float2(totalWidth * 0.5, 0.0);
    return DrawNumberRightAligned(texcoord, topRight, scale, aspect, value);
}

float3 DrawAPLGraphOverlay(float2 texcoord, float3 sceneColor)
{
    const int GRAPH_CURVE_SAMPLES = 64;
    float aspect = ReShade::ScreenSize.x / ReShade::ScreenSize.y;
    float2 p = texcoord;
    p.x *= aspect;

    // Bottom-left quarter layout with room for axis labels
    float2 graphPos = float2(0.055 * aspect, 0.48);
    float2 graphSize = float2(0.43 * aspect, 0.44);
    float2 graphMin = graphPos;
    float2 graphMax = graphPos + graphSize;

    float thickness = 0.00105;
    float curveThickness = thickness * 0.95;
    float refThickness = thickness * 0.90;
    float gridThickness = 0.00050;
    float tickThickness = 0.00075;
    float tickLen = graphSize.y * 0.018;
    float labelScale = 0.014;
    float digitStepScaled = labelScale * 0.82;
    float margin = thickness * 5.0;

    float graphAPLPercent = clamp(GraphAPLIndex, 0.0, 100.0);
    float graphCurrentAPL = saturate(graphAPLPercent / 100.0);
    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1000.0, 10000.0);
    float4 graphParams = tex2Dlod(SamplerGraphParams, float4(0.5, 0.5, 0.0, 0.0));
    float graphRollOffStartNits = graphParams.r;
    float graphMaxMeasuredNits = graphParams.g;
    float graphAxisMaxPQ = GraphUsePQSpace ? max(graphParams.b, 1e-6) : 0.0;

    float graphXMin = graphMin.x - margin;
    float graphXMax = graphMax.x + margin;

    float xLabelMinX = graphMin.x - labelScale * 2.0;
    float xLabelMaxX = graphMax.x + labelScale * 2.0;
    float xLabelMinY = graphMax.y - margin;
    float xLabelMaxY = graphMax.y + 0.010 + labelScale * 1.25;

    int maxAxisLabelDigits = CountDigitsInt((int)round(graphAxisMaxNits));
    float yLabelAnchorX = graphPos.x - graphSize.x * 0.035;
    float yLabelMinX = yLabelAnchorX - digitStepScaled * (float(maxAxisLabelDigits) + 0.25) - labelScale * 0.35;
    float yLabelMaxX = graphPos.x + margin;
    float yLabelMinY = graphMin.y - labelScale * 0.6;
    float yLabelMaxY = graphMax.y + labelScale * 0.6;

    if (p.x < min(yLabelMinX, xLabelMinX) || p.x > max(graphXMax, xLabelMaxX))
        return sceneColor;

    float maxRemappedNits = graphAxisMaxNits;
    float graphAnchorBoostedNits = graphParams.a;

    bool inGraphX = (p.x >= graphXMin && p.x <= graphXMax);

    if (inGraphX)
    {
        maxRemappedNits = ComputeGraphBoostedTargetNitsWithStart(graphAPLPercent, graphAxisMaxNits, graphRollOffStartNits, graphAnchorBoostedNits);
        float overflowCoord = max(GraphAxisCoordinateWithPQMax(maxRemappedNits, graphAxisMaxNits, graphAxisMaxPQ) - 1.0, 0.0);
        float overflowHeight = graphSize.y * overflowCoord;
    }

    bool inGraphCore = inGraphX && (p.y >= graphMin.y - margin) && (p.y <= graphMax.y + margin);
    bool inXLabelRegion = (p.x >= xLabelMinX) && (p.x <= xLabelMaxX) && (p.y >= xLabelMinY) && (p.y <= xLabelMaxY);
    bool inYLabelRegion = (p.x >= yLabelMinX) && (p.x <= yLabelMaxX) && (p.y >= yLabelMinY) && (p.y <= yLabelMaxY);

    if (!inGraphCore && !inXLabelRegion && !inYLabelRegion)
        return sceneColor;

    float inside = step(graphMin.x, p.x) * step(graphMin.y, p.y) * step(p.x, graphMax.x) * step(p.y, graphMax.y);

    float frameMask = 0.0;
    float gridMask = 0.0;
    float tickMask = 0.0;
    float labelMask = 0.0;
    float refMask = 0.0;
    float measuredMask = 0.0;
    float remappedMask = 0.0;
    float correctedMask = 0.0;

    if (inGraphCore)
    {
        frameMask = DrawGraphRect(p, graphMin, graphMax, thickness);

        [unroll]
        for (int i = 1; i < 10; i++)
        {
            float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 10.0, graphAxisMaxNits, graphAxisMaxPQ);
            float2 v0 = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, 0.0);
            float2 v1 = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, graphAxisMaxNits);
            float2 h0 = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0, tickValue);
            float2 h1 = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, graphAxisMaxNits, tickValue);

            gridMask += DrawGraphLine(p, v0, v1, gridThickness) * 0.32;
            gridMask += DrawGraphLine(p, h0, h1, gridThickness) * 0.32;
        }

        [unroll]
        for (int i = 0; i <= 5; i++)
        {
            float tickFrac = float(i) / 5.0;
            float tickValue = GraphTickValueFromFractionWithPQMax(tickFrac, graphAxisMaxNits, graphAxisMaxPQ);
            int tickLabel = (int)round(tickValue);

            float2 xTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, 0.0);
            float2 yTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0, tickValue);

            tickMask += DrawGraphLine(p, xTick + float2(0.0, -tickLen), xTick, tickThickness);
            tickMask += DrawGraphLine(p, yTick, yTick + float2(tickLen, 0.0), tickThickness);
        }

        refMask = DrawGraphDashedLine(
            p,
            ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0, 0.0),
            ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, graphAxisMaxNits, graphAxisMaxNits),
            refThickness,
            22.0
        );
    }

    if (inXLabelRegion || inYLabelRegion)
    {
        [unroll]
        for (int i = 0; i <= 5; i++)
        {
            float tickFrac = float(i) / 5.0;
            float tickValue = GraphTickValueFromFractionWithPQMax(tickFrac, graphAxisMaxNits, graphAxisMaxPQ);
            int tickLabel = (int)round(tickValue);

            float2 xTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, 0.0);
            float2 yTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0, tickValue);

            if (inXLabelRegion)
            {
                float2 xLabelCenter = float2(xTick.x / max(aspect, 1e-6), graphMax.y + 0.010);
                labelMask += DrawNumberCentered(texcoord, xLabelCenter, labelScale, aspect, tickLabel);
            }

            if (inYLabelRegion)
            {
                float2 yLabelTopRight = float2(yLabelAnchorX / max(aspect, 1e-6), yTick.y - labelScale * 0.52);
                labelMask += DrawNumberRightAligned(texcoord, yLabelTopRight, labelScale, aspect, tickLabel);
            }
        }
    }

    if (inGraphCore)
    {
        [loop]
        for (int s = 0; s < GRAPH_CURVE_SAMPLES - 1; ++s)
        {
            float t0 = float(s) / float(GRAPH_CURVE_SAMPLES - 1);
            float t1 = float(s + 1) / float(GRAPH_CURVE_SAMPLES - 1);

            float x0 = GraphSampleNitsFromFraction(t0, graphAxisMaxNits, graphAxisMaxPQ);
            float x1 = GraphSampleNitsFromFraction(t1, graphAxisMaxNits, graphAxisMaxPQ);

            float y0Remapped = ComputeGraphBoostedTargetNitsWithStart(graphAPLPercent, x0, graphRollOffStartNits, graphAnchorBoostedNits);
            float y1Remapped = ComputeGraphBoostedTargetNitsWithStart(graphAPLPercent, x1, graphRollOffStartNits, graphAnchorBoostedNits);

            float2 aRemapped = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0Remapped);
            float2 bRemapped = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1Remapped);
            remappedMask = max(remappedMask, DrawGraphLine(p, aRemapped, bRemapped, curveThickness * 0.95));

            if (inGraphCore)
            {
                float y0Corrected = SampleCorrectedOutputNitsForAPL(graphAPLPercent, y0Remapped, graphMaxMeasuredNits);
                float y1Corrected = SampleCorrectedOutputNitsForAPL(graphAPLPercent, y1Remapped, graphMaxMeasuredNits);

                float measuredMaxInputNits = GetGraphMeasuredMaxInputNits();

                if (x0 < measuredMaxInputNits)
                {
                    float mx0 = x0;
                    float mx1 = min(x1, measuredMaxInputNits);
                    float my0 = SampleRealMeasuredOutputNitsForAPL(graphAPLPercent, mx0);
                    float my1 = SampleRealMeasuredOutputNitsForAPL(graphAPLPercent, mx1);

                    float2 aMeasured = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, mx0, my0);
                    float2 bMeasured = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, mx1, my1);
                    measuredMask = max(measuredMask, DrawGraphLine(p, aMeasured, bMeasured, curveThickness));
                }

                float2 aCorrected = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0Corrected);
                float2 bCorrected = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1Corrected);

                correctedMask = max(correctedMask, DrawGraphLine(p, aCorrected, bCorrected, curveThickness));
            }
        }
    }

    float bgMask = inside * 0.12;
    float3 graphColor = sceneColor;
    graphColor = lerp(graphColor, float3(0.0, 0.0, 0.0), bgMask * saturate(GraphOpacity * 0.95));
    graphColor = lerp(graphColor, float3(0.58, 0.58, 0.58) * OSDBrightness * 1.25, saturate(gridMask) * GraphOpacity);
    graphColor = lerp(graphColor, float3(0.90, 0.90, 0.90) * OSDBrightness * 1.45, saturate(tickMask) * GraphOpacity);
    graphColor = lerp(graphColor, float3(1.0, 1.0, 1.0) * OSDBrightness * 1.8, saturate(frameMask) * GraphOpacity);
    graphColor = lerp(graphColor, float3(0.85, 0.85, 0.85) * OSDBrightness * 1.7, saturate(labelMask) * saturate(GraphOpacity + 0.05));
    graphColor = lerp(graphColor, float3(0.40, 0.65, 1.00) * OSDBrightness * 2.0, saturate(refMask) * GraphOpacity);
    graphColor = lerp(graphColor, float3(0.62, 0.82, 1.00) * OSDBrightness * 1.95, saturate(measuredMask) * saturate(GraphOpacity * 0.95));
    graphColor = lerp(graphColor, float3(0.30, 0.88, 0.42) * OSDBrightness * 1.55, saturate(remappedMask) * saturate(GraphOpacity + 0.06));
    graphColor = lerp(graphColor, float3(0.62, 0.62, 0.62) * OSDBrightness * 1.65, saturate(correctedMask) * saturate(GraphOpacity + 0.20));

    return saturate(graphColor);
}

#endif

// --- SHADERS ---

// PASS 1: Calculate raw APL / Metric + Max Sampled Raw Luma
float4 PS_CalcAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    const int MAX_GRID_SIZE = 32;
    int steps = clamp(APLGridSize, 4, MAX_GRID_SIZE);
    float invSteps = 1.0 / float(steps);

    float totalMetric = 0.0;
    float totalSamples = 0.0;
    float maxSampledRawLuma = 0.0;

    [loop]
    for (int x = 0; x < MAX_GRID_SIZE; ++x)
    {
        if (x >= steps)
            break;

        [loop]
        for (int y = 0; y < MAX_GRID_SIZE; ++y)
        {
            if (y >= steps)
                break;

            float2 sampleUV = (float2(float(x), float(y)) + 0.5) * invSteps;
            float3 color = tex2Dlod(ReShade::BackBuffer, float4(sampleUV, 0.0, 0.0)).rgb;

            maxSampledRawLuma = max(maxSampledRawLuma, GetLuma709(max(color, 0.0.xxx)));
            totalMetric += GetAPLMetricSample(color);
            totalSamples += 1.0;
        }
    }

    float apl = totalMetric / max(totalSamples, 1.0);

    // r = raw current-frame APL metric, g = max sampled raw luma, a = valid
    return float4(apl, maxSampledRawLuma, 0.0, 1.0);
}

float4 PS_CopyAPLState(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    return tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
}

float4 PS_SmoothAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float4 currentData = tex2Dlod(SamplerAPLInstant, float4(0.5, 0.5, 0.0, 0.0));
    float4 prevData = tex2Dlod(SamplerAPLPrev, float4(0.5, 0.5, 0.0, 0.0));

    float currentAPL = saturate(currentData.r);
    float prevAPLRaw = prevData.r;
    float prevSmoothedAPL = saturate(prevAPLRaw);

    float alpha = ComputeTemporalBlendFactor(TransitionSpeed);
    float hasPrev = (prevData.a > 0.5 && prevAPLRaw >= 0.0 && prevAPLRaw <= 1.0) ? 1.0 : 0.0;

    float smoothedAPL = lerp(currentAPL, lerp(prevSmoothedAPL, currentAPL, alpha), hasPrev);
    float dynamicRollOffStartNits = SolveDynamicRollOffStartNits(smoothedAPL);
    float rollOffAnchorBoostedNits = ComputeRollOffAnchorBoostedNits(smoothedAPL);

    // r = smoothed APL metric, g = current max sampled raw luma, b = dynamic roll off start from smoothed APL, a = boosted anchor nits used by the PQ rational shoulder
    return float4(smoothedAPL, currentData.g, dynamicRollOffStartNits, rollOffAnchorBoostedNits);
}

float3 DrawStatsOverlay(float2 texcoord, float3 sceneColor, float currentAPL, float maxSampledLuma, float fader)
{
    float2 posStart = float2(0.84, 0.05);
    float scale = 0.04;
    float lineSpacing = scale * 1.35;
    float charSpacing = scale * 0.5;
    float aspect = ReShade::ScreenSize.x / ReShade::ScreenSize.y;

    float textMask = 0.0;

    // Line 1: APL percentage
    int aplVal = clamp(int(currentAPL * 100.0), 0, 99);
    int a0 = aplVal / 10;
    int a1 = aplVal % 10;

    float2 uvDigits1 = texcoord - posStart;
    uvDigits1.x *= aspect;

    float2 uvA0 = uvDigits1;
    float2 uvA1 = uvDigits1 - float2(charSpacing * aspect, 0.0);
    float2 uvA2 = uvDigits1 - float2(charSpacing * 2.0 * aspect, 0.0);

    uvA0.x = -uvA0.x;
    uvA1.x = -uvA1.x;

    textMask += GetDigit(a0, uvA0 / scale);
    textMask += GetDigit(a1, uvA1 / scale);
    textMask += GetPercent(uvA2 / scale);

    // Line 2: Max sampled raw luma as XX.XX
    float maxDisplay = clamp(maxSampledLuma, 0.0, 99.99);
    int maxWhole = int(floor(maxDisplay));
    int maxFrac = int(floor(frac(maxDisplay) * 100.0 + 0.5));

    if (maxFrac >= 100)
    {
        maxFrac -= 100;
        maxWhole = min(maxWhole + 1, 99);
    }

    int m0 = (maxWhole / 10) % 10;
    int m1 = maxWhole % 10;
    int m2 = (maxFrac / 10) % 10;
    int m3 = maxFrac % 10;

    float2 uvDigits2 = texcoord - (posStart + float2(0.0, lineSpacing));
    uvDigits2.x *= aspect;

    float2 uvM0 = uvDigits2;
    float2 uvM1 = uvDigits2 - float2(charSpacing * aspect, 0.0);
    float2 uvM2 = uvDigits2 - float2(charSpacing * 2.0 * aspect, 0.0);
    float2 uvM3 = uvDigits2 - float2(charSpacing * 3.0 * aspect, 0.0);
    float2 uvM4 = uvDigits2 - float2(charSpacing * 4.0 * aspect, 0.0);

    uvM0.x = -uvM0.x;
    uvM1.x = -uvM1.x;
    uvM2.x = -uvM2.x;
    uvM3.x = -uvM3.x;
    uvM4.x = -uvM4.x;

    textMask += GetDigit(m0, uvM0 / scale);
    textMask += GetDigit(m1, uvM1 / scale);
    textMask += GetDot(uvM2 / scale);
    textMask += GetDigit(m2, uvM3 / scale);
    textMask += GetDigit(m3, uvM4 / scale);

    float3 baseColor = lerp(float3(1, 1, 1), float3(0, 1, 0), fader);
    float3 textColor = baseColor * OSDBrightness;

    return lerp(sceneColor, textColor, saturate(textMask));
}

// PASS 2: Main Rendering (1D APL-only measured scene gain + hybrid luminance participation)
float4 PS_MainPass(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float pixelLuma = GetSignalLuma(color);
    float safePixelLuma = (APLInputMode == 1) ? pixelLuma : max(pixelLuma, 0.0);

    float4 aplData = tex2D(SamplerAPL, float2(0.5, 0.5));
    float currentAPL = aplData.r;
    float dynamicRollOffStartNits = aplData.b;

    float aplPct = saturate(currentAPL) * 100.0;
    float measuredComp = max(LookupMeasuredComp1D(aplPct), 1.0);
    float pixelBoostT = ShapeBoostT(MeasuredCompToBoostT(measuredComp));
    float fader = ComputeAPLBoostFader(currentAPL);
    float sceneGainExponent = max(MaxAPLBoostStrength * pixelBoostT * fader, 0.0);
    float sceneLogGain = log2(measuredComp) * sceneGainExponent;

    if (sceneGainExponent <= 0.0 && abs(SaturationComp - 1.0) <= 1e-6 && (ShowOSD == false))
        return float4(color, 1.0);

    float boostedLuma = ApplyBoostWithRationalRolloffFromSceneLogGain(safePixelLuma, sceneLogGain, dynamicRollOffStartNits, aplData.a);

    float chromaScale = lerp(1.0, SaturationComp, fader);
    float3 chroma = (color - safePixelLuma.xxx) * chromaScale;
    float3 finalColorLumaChroma = boostedLuma.xxx + chroma;
    float scale = (safePixelLuma > 1e-4) ? (boostedLuma / safePixelLuma) : 1.0;
    float3 finalColorScaled = color * scale;
    float darkBlend = ((APLInputMode == 0) && (pixelLuma <= 0.0)) ? 1.0 : smoothstep(0.003, 0.015, max(safePixelLuma, 0.0));
    float3 finalColor = lerp(finalColorScaled, finalColorLumaChroma, darkBlend);

    if (ShowOSD)
        finalColor = DrawStatsOverlay(texcoord, finalColor, currentAPL, aplData.g, fader);

    return float4(finalColor, 1.0);
}

#if ENABLE_APL_GRAPH
float4 PS_CalcGraphParams(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float graphAPLPercent = clamp(GraphAPLIndex, 0.0, 100.0);
    float graphCurrentAPL = saturate(graphAPLPercent / 100.0);
    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1000.0, 10000.0);
    float graphAxisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(graphAxisMaxNits), 1e-6) : 0.0;
    float graphRollOffStartNits = SolveDynamicRollOffStartNits(graphCurrentAPL);
    float maxMeasuredNits = GetAPLMaxMeasuredNits(graphAPLPercent);
    float graphAnchorBoostedNits = ComputeRollOffAnchorBoostedNits(graphCurrentAPL);

    return float4(graphRollOffStartNits, maxMeasuredNits, graphAxisMaxPQ, graphAnchorBoostedNits);
}

float4 PS_DebugOverlay(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 finalColor = tex2D(SamplerBoosted, texcoord).rgb;

    if (ShowAPLGraph)
    {
        finalColor = DrawAPLGraphOverlay(texcoord, finalColor);
    }

    return float4(finalColor, 1.0);
}
#endif

technique EOTF_Boost_1D_APL_LUT 
{
    pass APL_Calculation
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcAPL;
        RenderTarget = TexAPLInstant;
    }

    pass APL_CopyState
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CopyAPLState;
        RenderTarget = TexAPLPrev;
    }

    pass APL_Smoothing
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_SmoothAPL;
        RenderTarget = TexAPL;
    }

    pass Main_Boost
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_MainPass;
#if ENABLE_APL_GRAPH
        RenderTarget = TexBoosted;
#endif
    }

#if ENABLE_APL_GRAPH
    pass Graph_Params
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphParams;
        RenderTarget = TexGraphParams;
    }

    pass Debug_Overlay
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_DebugOverlay;
    }
#endif
}
