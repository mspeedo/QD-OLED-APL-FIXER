/*
    EOTF Boost v8.16.4 - 1D APL Lookup
    Calibrated for monitor MSI MPG 341CQR QD-OLED X36
    ================================================================

    Purpose
    -------
    This shader boosts HDR luminance to compensate for OLED / display ABL behavior
    using a simplified measured 1D APL lookup table.

    This version applies compensation as a scene-uniform multiplicative gain in
    absolute nits (the pixel-participation / shadow-protection-floor model was
    removed in v8.11; every pixel receives the full scene gain):

        scene_gain = measured_compensation(APL) shaped by LUT weight and strength
        pixel_gain = scene_gain

    where:
        - APL is the scene average picture level metric (0..1, shown as 0..100%)
        - compensation > 1 means the display measured darker than the requested target

    This version collapses the original measured 2D APL x nits LUT into a single
    representative compensation value per APL row (anchored near 109 nits), because
    the per-row variation across target nits was small.

    The live boost strength comes from Global APL Boost Strength. Optional per-APL
    controls can be compiled in with PER_APL_BOOST_STRENGTH_ENABLE.

    The lookup table is NOT used as a direct inverse solve.
    Instead, it is used as a shape / weight map that drives a capped boost model.

    Optional final HDR calibration LUT layer
    ----------------------------------------
    If enabled, PQHDRLUT.cube is applied as a separate final display-calibration
    layer after the EOTF boost calculation, or directly to the source signal when
    EOTF Boost is disabled. The original boost logic remains intact.

    Optional APL-driven LUT compensation mode
    -----------------------------------------
    This experimental mode is compile-time stripped by default. Set
    PQHDRLUT_ENABLE_APL_DRIVEN to 1 to compile its UI, resources, functions, and
    three additional post-boost APL passes.

    Disabled: original fixed PQHDRLUT path.

    Enabled with EOTF Boost + PQHDRLUT: after boost parameters are computed, a cheap
    APL_DECODE_SIZE^2 sampled-grid pass applies the same boost math to those sampled
    pixels and measures the real post-boost / pre-LUT APL. That value drives the
    luma-preserving APL-driven LUT lookup only; the boost model does not consume it.

    Enabled with PQHDRLUT only: there is no post-boost intermediate, so the mode
    falls back to the smoothed raw source APL.

    Closed-loop display-side APL
    ----------------------------
    The compensation lookup is indexed by the display-side (post-boost) APL. The
    decode pass evaluates the exact boost pipeline (BT.2390 rolloff, never-darken
    clamp, color-preserving limiting with each sample's own max channel) for every
    grid sample at three candidate gains; the reductions average them, yielding the
    scene's aggregate post-boost APL response A(g) at four gain nodes (g = 0 is the
    raw APL). A(g) is smooth and near-log-linear, so the 1x1 solve runs a damped
    fixed-point iteration on the log-interpolated response — matching an exact
    per-sample solve to ~0.03% while leaving only a handful of scalar ALU ops in
    the single-threaded pass.

*/

#include "ReShade.fxh"

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 0
#endif

// Optional final HDR calibration 3D LUT.
// Set PQHDRLUT_ENABLE to 0 to strip PQHDRLUT.cube texture loading, UI controls,
// and sampling code from the compiled shader.
#ifndef PQHDRLUT_ENABLE
    #define PQHDRLUT_ENABLE 0
#endif

// Experimental APL-driven LUT compensation is available only when PQHDRLUT is
// compiled in. This keeps its preprocessor option hidden when PQHDRLUT_ENABLE = 0.
// Set to 1 to compile its UI control, resources, functions, and three extra passes.
#if PQHDRLUT_ENABLE
    #ifndef PQHDRLUT_ENABLE_APL_DRIVEN
        #define PQHDRLUT_ENABLE_APL_DRIVEN 0
    #endif
#endif

// Optional per-APL boost-strength controls are stripped by default, including the
// runtime checkbox, ten sliders, lookup function, and associated branches. The
// shader then uses Global APL Boost Strength directly. Set to 1 to compile them in.
#ifndef PER_APL_BOOST_STRENGTH_ENABLE
    #define PER_APL_BOOST_STRENGTH_ENABLE 0
#endif


// --- COMPILE-TIME DEBUG FEATURE SWITCHES ---
// Set to 1 to compile the graph feature in, or 0 to strip it out completely.
// Variant: built-in window projection graph + BT.2390-style reference rolloff overlay.
#ifndef ENABLE_APL_GRAPH
    #define ENABLE_APL_GRAPH 0
#endif

#ifndef ENABLE_UI_TOOLTIPS
    #define ENABLE_UI_TOOLTIPS 0
#endif

// APL sample budget: the aspect-matched decode grid uses at most APL_DECODE_SIZE^2
// samples (see APL_GRID_W/H below). Must be a power of two between 8 and 64.
// 64 (up to 4096 samples) is the default for stable APL estimation; override at
// compile time with: #define APL_DECODE_SIZE 32
#ifndef APL_DECODE_SIZE
    #define APL_DECODE_SIZE 64
#endif

#if (APL_DECODE_SIZE < 8) || (APL_DECODE_SIZE > 64) || (APL_DECODE_SIZE & (APL_DECODE_SIZE - 1))
    #error "APL_DECODE_SIZE must be a power of two between 8 and 64"
#endif

// Aspect-matched sample grid: APL_DECODE_SIZE^2 is the sample BUDGET; the actual
// grid is APL_GRID_W x APL_GRID_H with W/H chosen per aspect bucket so sample
// spacing is near-isotropic in screen pixels (a square 64x64 grid on a 21:9 screen
// samples ~2.4x denser vertically than horizontally). Row count H ~= budget_side /
// sqrt(aspect), evaluated with integer math per bucket; W fills the budget, so
// W * H <= APL_DECODE_SIZE^2 always (never more samples than the square grid).
// BUFFER_WIDTH/HEIGHT are preprocessor values, so this re-derives automatically
// whenever ReShade recompiles on a resolution change.
#if (BUFFER_WIDTH * 10 >= BUFFER_HEIGHT * 31)         // ~32:9 super-ultrawide and wider
    #define APL_GRID_H ((APL_DECODE_SIZE * 34 + 32) / 64)
#elif (BUFFER_WIDTH * 10 >= BUFFER_HEIGHT * 21)       // ~21:9 ultrawide
    #define APL_GRID_H ((APL_DECODE_SIZE * 41 + 32) / 64)
#elif (BUFFER_WIDTH * 10 >= BUFFER_HEIGHT * 15)       // 16:9 / 16:10
    #define APL_GRID_H ((APL_DECODE_SIZE * 48 + 32) / 64)
#elif (BUFFER_WIDTH >= BUFFER_HEIGHT)                 // 3:2 / 4:3 / 5:4 / square
    #define APL_GRID_H ((APL_DECODE_SIZE * 56 + 32) / 64)
#else                                                 // portrait orientations
    #define APL_GRID_H ((APL_DECODE_SIZE * 85 + 32) / 64)
#endif
#define APL_GRID_W ((APL_DECODE_SIZE * APL_DECODE_SIZE) / APL_GRID_H)


#if ENABLE_UI_TOOLTIPS
    #define UI_TOOLTIP(text) ui_tooltip = text;
#else
    #define UI_TOOLTIP(text)
#endif


#if PQHDRLUT_ENABLE
// Override at compile time if the .cube file is not 65^3.
#ifndef PQHDRLUT_SIZE
    #define PQHDRLUT_SIZE 65
#endif
#endif

// --- UI SETTINGS ---

uniform bool EnableEOTFBoost <
    ui_label = "Enable EOTF Boost";
    UI_TOOLTIP("Enables the APL-based EOTF boost/ABL compensation layer. Turn this off to bypass boost. Live APL sampling is skipped unless APL-driven LUT compensation is enabled.")
> = true;

uniform int APLInputMode <
    ui_type = "combo";
    ui_items = "scRGB Normalized\0PQ Decoded Normalized\0";
    ui_label = "APL Input Mode";
    UI_TOOLTIP("Selects how the shader interprets scene luminance for the APL metric. scRGB uses BT.709 luma scaled by Reference White. PQ uses ST.2084-decoded BT.2020 luma scaled by Reference White.")
> = 1;

uniform float APLReferenceWhiteNits <
    ui_type = "slider";
    ui_min = 10.0; ui_max = 1500.0; ui_step = 1.0;
    ui_label = "APL Reference White (nits)";
    UI_TOOLTIP("Reference white used only for the APL metric normalization. It does not directly clamp output nits or change the graph axes.")
> = 1350.0;

uniform float APLTrigger <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.95; ui_step = 0.01;
    ui_label = "APL Trigger";
    UI_TOOLTIP("Boost fade-in start threshold based on the smoothed APL metric. Below this level the effect is disabled. With APL Trigger Fade Width = 0, this remains a hard on/off threshold. 10% APL on the graph is exactly the threshold when this is set to 0.10.")
> = 0.00;

uniform float APLTriggerFadeWidth <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.50; ui_step = 0.01;
    ui_label = "APL Trigger Fade Width";
    UI_TOOLTIP("Width of the APL Trigger fade-in range. 0 = original hard trigger. Example: Trigger 0.10 and Fade Width 0.05 means boost fades from 0 at 10% APL to full strength at 15% APL.")
> = 0.00;

uniform float CompensationFreezeAPLPercent <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 50.0; ui_step = 0.1;
    ui_label = "Compensation Freeze APL %";
    UI_TOOLTIP("Freezes the measured compensation lookup above the selected APL percentage. Example: 10.0 means APL values above 10% keep using the 10% compensation row. 0 = disabled.")
> = 0.0;

uniform float MaxAPLBoostStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "Global APL Boost Strength";
    UI_TOOLTIP("Scales the measured APL compensation in log-gain space before it is applied. 1.0 means full measured compensation at maximum LUT weight. Values below 1.0 under-compensate. Values above 1.0 intentionally over-compensate.")
> = 0.5;


#if PER_APL_BOOST_STRENGTH_ENABLE
uniform bool EnablePerAPLBoostStrength <
    ui_label = "Enable Per-APL Boost Strength";
    UI_TOOLTIP("Enables the advanced per-APL boost-strength controls below. When disabled, the shader uses only Global APL Boost Strength.")
> = true;

uniform float APLBoostStrength03 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 3% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 3% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.4;

uniform float APLBoostStrength05 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 5% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 5% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.8;

uniform float APLBoostStrength07 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 7% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 7% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.9;

uniform float APLBoostStrength10 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 10% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 10% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.8;

uniform float APLBoostStrength14 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 14% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 14% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.65;

uniform float APLBoostStrength18 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 18% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 18% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.54;

uniform float APLBoostStrength22 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 22% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 22% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.5;

uniform float APLBoostStrength25 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 25% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 25% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.5;

uniform float APLBoostStrength35 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 35% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 35% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.5;

uniform float APLBoostStrength50 <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Advanced Per-APL Boost Strength";
    ui_category_closed = true;
    ui_label = "APL 50% Boost Strength";
    UI_TOOLTIP("Per-APL boost strength override for the measured 50% APL point. Used only when Enable Per-APL Boost Strength is enabled.")
> = 0.5;
#endif


uniform float BoostRollOff <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1500.0; ui_step = 1.0;
    ui_label = "Boost Roll-Off Target (nits)";
    UI_TOOLTIP("Desired output anchor of the PQ highlight rolloff in nits. The shader dynamically places the knee from the current smoothed APL so the boosted curve lands on this endpoint more consistently across APL levels.")
> = 1350.0;

uniform float BoostRollOffShape <
    ui_type = "slider";
    ui_min = 0.25; ui_max = 4.0; ui_step = 0.01;
    ui_label = "Boost Roll-Off Shape";
    UI_TOOLTIP("Adjusts the live roll off character by moving the roll off start together with the shoulder curvature so the transition stays smooth and monotonic. 1.0 = standard BT.2390. Values below 1.0 start later and hold highlights higher longer. Values above 1.0 start earlier and compress highlights harder.")
> = 1.25;


uniform float TransitionSpeed <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "APL Smoothing Time (s)";
    UI_TOOLTIP("Temporal smoothing time constant for the live APL-related metrics in seconds. 0 = disabled. FPS-independent. This affects live boosting and OSD values, but the graph uses its own Graph APL % slider.")
> = 0.25;

uniform bool EnableColorPreservingBoostMode <
    ui_label = "Preserve Color by Reducing Boost";
    UI_TOOLTIP("When enabled, saturated colors keep their RGB ratio by reducing only the added boost before channels would exceed the Boost Roll-Off Target. Uses a fixed soft knee of 0.85. Original behavior is unchanged when disabled.")
> = false;

static const float COLOR_PRESERVING_BOOST_KNEE = 0.85;

uniform float SIGNAL_REFERENCE_NITS <
    ui_type = "slider";
    ui_min = 1.0; ui_max = 200.0; ui_step = 1.0;
    ui_label = "scRGB Signal Reference (nits)";
    UI_TOOLTIP("Reference nits for scRGB signal conversion. Standard scRGB uses 80 nits per 1.0 signal. Used only when APL Input Mode = scRGB Normalized.")
> = 80.0;


#if PQHDRLUT_ENABLE
uniform bool EnablePQHDRLUT <
    ui_label = "Enable PQ HDR LUT";
    UI_TOOLTIP("Applies PQHDRLUT.cube as a final display-calibration layer after the EOTF boost. Default is on.")
> = true;

#if PQHDRLUT_ENABLE_APL_DRIVEN
uniform bool EnableAPLDrivenLUTCompensationMode <
    ui_category = "Optional HDR LUT Calibration";
    ui_category_closed = true;
    ui_label = "APL-driven LUT Compensation Mode";
    UI_TOOLTIP("Experimental. Disabled = original fixed LUT path. Enabled = boost behavior unchanged; sampled real post-boost/pre-LUT APL is used only for luma-preserving APL-driven LUT compensation.")
> = true;
#endif

uniform int PQHDRLUTInputColorSpace <
    ui_category = "Optional HDR LUT Calibration";
    ui_category_closed = true;
    ui_type = "combo";
    ui_items = "Auto\0HDR10 PQ / Rec.2020\0scRGB / linear Rec.709\0";
    ui_label = "LUT Input Color Space";
    UI_TOOLTIP("Auto follows ReShade BUFFER_COLOR_SPACE. For HDR10 swapchains use HDR10 PQ / Rec.2020; for Windows scRGB HDR paths use scRGB / linear Rec.709.")
> = 0;
#endif

uniform bool ShowOSD <
    ui_label = "Show APL Stats";
    UI_TOOLTIP("Displays raw input APL (green), boost-model APL or sampled post-boost/pre-LUT APL in APL-driven LUT mode (yellow), and sampled max nits for the same source (cyan).")
> = false;

uniform float OSDBrightness <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_label = "OSD Brightness";
    UI_TOOLTIP("Controls OSD and graph overlay brightness.")
> = 0.5;

uniform float FrameTime < source = "frametime"; >;

#if ENABLE_APL_GRAPH
uniform bool ShowAPLGraph <
    ui_label = "Show APL EOTF Debug Graph";
    UI_TOOLTIP("Shows the analysis graph. Standard mode: Blue dashed = identity reference, optional Magenta dashed = BT.2390-style reference tone map using the projected measured peak for the selected raw APL input, Light blue = real 2D measured LUT output for that raw input APL, Green = shader remapped target after the closed-loop APL solve, Gray = projected measured output at the solved display-side operating point. Window projection mode: Blue dashed = identity reference, optional Magenta dashed = BT.2390-style reference tone map using the selected window peak, Light blue = measured window EOTF for the raw input, Gray = projected window output after the closed-loop APL solve.")
> = true;

uniform bool GraphShowBT2390Reference <
    ui_label = "Graph Show BT.2390 Reference";
    UI_TOOLTIP("Shows or hides the optional BT.2390-style Hermite rolloff reference overlay. It uses the measured peak for the selected APL or selected window size.")
> = false;

uniform bool GraphUseFullFieldWindowProjection <
    ui_label = "Graph Use Window Projection";
    UI_TOOLTIP("Switches the debug graph to the built-in window PQ measurement projection overlay. In this mode, Graph APL (%) is ignored. Use the window selector below to choose between the built-in 100%, 50%, 25%, 15%, and 10% window measurements. Blue dashed = identity reference, optional Magenta dashed = BT.2390-style reference tone map using the selected window peak, Light blue = measured window EOTF only, Gray = projected window output only.")
> = true;

uniform int GraphProjectionWindowSize <
    ui_type = "combo";
    ui_items = "100% Window\0 50% Window\0 25% Window\0 15% Window\0 10% Window\0";
    ui_label = "Graph Projection Window Size";
    UI_TOOLTIP("Selects which built-in measured window set is used by the full-field projection graph mode.")
> = 0;

uniform float GraphAPLIndex <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 50.0; ui_step = 0.01;
    ui_label = "Graph APL (%)";
    UI_TOOLTIP("Continuous raw / pre-boost input APL value used by the standard APL-slice graph mode. Light blue = measured curve for that raw input APL. Green = shader remapped target projected from that raw input through the closed-loop APL solve. Gray = projected measured output at the solved display-side operating point. Ignored when Graph Use Window Projection is enabled.")
> = 50.0;

uniform float GraphInputSignalLimitNits <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 4000.0; ui_step = 1.0;
    ui_label = "Graph Input Signal Limit (nits)";
    UI_TOOLTIP("Graph-only input signal cap used for curve preview. 0 = disabled. When set above 0, the graph behaves as if input signal values above this nit level were clipped to the specified value. This affects only graph curves and references, not the live shader or OSD.")
> = 0.0;

uniform float GraphAxisMaxNits <
    ui_type = "slider";
    ui_min = 100.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label = "Graph Axis Max (nits)";
    UI_TOOLTIP("Maximum nits shown on both graph axes. Raising it lets you inspect curve behavior beyond 1000-nit input without changing the live shader.")
> = 1350.0;

uniform float GraphOpacity <
    ui_type = "slider";
    ui_min = 0.05; ui_max = 1.0; ui_step = 0.01;
    ui_label = "Graph Opacity";
    UI_TOOLTIP("Opacity of the graph overlay background and curves.")
> = 0.5;

uniform bool GraphUsePQSpace <
    ui_label = "Graph PQ-Encoded Axes";
    UI_TOOLTIP("Renders the graph in PQ-encoded space instead of linear nits. Axis labels remain in nits.")
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

// Scene-uniform boost/rolloff parameters precomputed after APL smoothing.
// RGBA layout:
//   .r = scene gain exp2(sceneLogGain), uniform for all pixels
//   .g = BT.2390 PQ range; <= 0 means rolloff inactive
//   .b = BT.2390 shaped knee start in normalized PQ range
//   .a = BT.2390 compression span in normalized PQ range
texture TexBoostParams
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerBoostParams
{
    Texture = TexBoostParams;
};

#if PQHDRLUT_ENABLE
texture3D PQHDRLUTTexture <
    source = "PQHDRLUT.cube";
>
{
    Width = PQHDRLUT_SIZE;
    Height = PQHDRLUT_SIZE;
    Depth = PQHDRLUT_SIZE;
    Format = RGBA32F;
};

sampler3D PQHDRLUTSampler
{
    Texture = PQHDRLUTTexture;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
    MagFilter = LINEAR;
    MinFilter = LINEAR;
    MipFilter = POINT;
};

#endif

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

// Parallel APL decode target.
// Each of the APL_GRID_W x APL_GRID_H texels is written by PS_DecodeAPL,
// which runs on that many GPU threads simultaneously (grid aspect matches the
// screen, so sample spacing is near-isotropic in pixels). In PQ mode each
// thread decodes the sampled RGB triplet once and derives both luma and optional
// max-channel nits from that same decoded value, instead of serialising all
// decodes inside a single 1x1 pixel shader loop.
// RGBA32F layout: .r = raw pre-boost APL metric, .gba = post-boost metric of
// this sample at the three candidate gains of the response-curve solve
// (gMax/3, 2*gMax/3, gMax). Per-sample luma nits for the OSD max readout are
// stored separately in TexAPLDecodedNits.
texture TexAPLDecoded
{
    Width  = APL_GRID_W;
    Height = APL_GRID_H;
    Format = RGBA32F;
};
sampler SamplerAPLDecoded
{
    Texture   = TexAPLDecoded;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Companion target of the decode MRT pass: per-sample luma nits, only consumed by
// the OSD max-nits readout via the column reduction.
texture TexAPLDecodedNits
{
    Width  = APL_GRID_W;
    Height = APL_GRID_H;
    Format = R32F;
};
sampler SamplerAPLDecodedNits
{
    Texture   = TexAPLDecodedNits;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Stage-1 reduction target: one texel per column of TexAPLDecoded.
// PS_ReduceAPLColumns runs on APL_GRID_W parallel threads, each summing one
// column (APL_GRID_H fetches). PS_CalcAPL then only sums these APL_GRID_W
// column results, replacing the previous single-thread whole-grid serial
// fetch loop (a latency-bound pattern) with two short, parallel stages.
// RGBA32F: column sums of (raw metric, candidate metric g1, g2, g3) — must be
// full-width or the upper candidate channels are silently dropped on write.
texture TexAPLReduced
{
    Width  = APL_GRID_W;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerAPLReduced
{
    Texture   = TexAPLReduced;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Column max of the per-sample luma nits (OSD readout only).
texture TexAPLReducedNits
{
    Width  = APL_GRID_W;
    Height = 1;
    Format = R32F;
};
sampler SamplerAPLReducedNits
{
    Texture   = TexAPLReducedNits;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Aggregate post-boost APL response of the scene at the three candidate gains.
// .rgb = A(gMax/3), A(2*gMax/3), A(gMax); .a = valid. A(0) is the raw APL itself.
texture TexAPLResponse
{
    Width  = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerAPLResponse
{
    Texture = TexAPLResponse;
};


#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
// Optional real post-boost / pre-LUT APL path for APL-driven LUT compensation.
// This does not render a full extra boosted image. It re-runs the exact boost math only
// on the APL sampling grid, then averages those boosted samples in a 1x1 pass.
texture TexPostBoostAPLDecoded
{
    Width  = APL_GRID_W;
    Height = APL_GRID_H;
    Format = RG32F;
};
sampler SamplerPostBoostAPLDecoded
{
    Texture   = TexPostBoostAPLDecoded;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Stage-1 reduction target for the post-boost APL path (same scheme as TexAPLReduced).
texture TexPostBoostAPLReduced
{
    Width  = APL_GRID_W;
    Height = 1;
    Format = RG32F;
};
sampler SamplerPostBoostAPLReduced
{
    Texture   = TexPostBoostAPLReduced;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

texture TexPostBoostAPL
{
    Width = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerPostBoostAPL
{
    Texture = TexPostBoostAPL;
};
#endif

#if ENABLE_APL_GRAPH
// Curve-precompute constants — must match DrawAPLGraphOverlay.
// Defined here (not as a const int inside the function) so the texture Width attribute
// can reference it at compile time and both the precompute pass and the draw pass agree.
#define GRAPH_CURVE_SAMPLES 64

// Row indices inside TexGraphCurves (height = 4).
// Each texel stores float4(ax, ay, bx, by) in p-space screen coords
// (texcoord with p.x *= aspect).  The precompute pass converts from nits
// to screen space so the per-pixel draw loop needs zero NitsToPQ / pow calls.
// Texels with x < 0 are sentinels: the segment should be skipped.
#define GCURVE_REMAPPED  0   // green re-mapped curve (APL mode only)
#define GCURVE_CORRECTED 1   // gray projected-output / corrected curve
#define GCURVE_MEASURED  2   // light-blue measured raw curve
#define GCURVE_BT2390REF 3   // magenta BT.2390 reference (optional)

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

// Precomputed per-segment screen-space endpoints for all four curve rows.
// Width = GRAPH_CURVE_SAMPLES (one texel per segment), Height = 4 (one row per curve).
texture TexGraphCurves
{
    Width  = GRAPH_CURVE_SAMPLES;
    Height = 4;
    Format = RGBA32F;
};
sampler SamplerGraphCurves
{
    Texture   = TexGraphCurves;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};

// Precomputed grid/tick/ref line endpoints.  All positions are purely uniform-derived —
// computing them per-pixel with NitsToPQ (2 pow calls each) wastes ~200 pow calls per
// inGraphCore pixel.  Layout (one float4(ax,ay,bx,by) per texel in p-space):
//   0–8:   grid vertical lines   (i = 1..9)
//   9–17:  grid horizontal lines (i = 1..9)
//   18–23: x-tick marks          (i = 0..5)
//   24–29: y-tick marks          (i = 0..5)
//   30:    identity reference dashed line
//   31:    (padding / sentinel)
#define GRAPH_LINE_COUNT 32
texture TexGraphLines
{
    Width  = GRAPH_LINE_COUNT;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerGraphLines
{
    Texture   = TexGraphLines;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
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

// float3 overload — encodes all three channels in one pair of vector pow calls instead of
// three pairs of scalar calls.  Used by ApplyBoostPreserveColorFromPrecomputedParams to re-encode
// the boosted PQ output without serialising the per-channel work.
float3 LinearToPQBT2100(float3 v)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;

    float3 Lm1 = pow(saturate(v), m1);
    float3 num = c1 + c2 * Lm1;
    float3 den = 1.0 + c3 * Lm1;
    return pow(num / max(den, 1e-6), m2);
}

// Scalar version of PQ EOTF — avoids float3 construction overhead in scalar-only contexts.
float PQToLinearScalar(float v)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;

    float vp = pow(saturate(v), 1.0 / m2);
    float num = max(vp - c1, 0.0);
    float den = c2 - c3 * vp;
    return pow(num / max(den, 1e-6), 1.0 / m1);
}

float NitsToPQ(float nits)
{
    return LinearToPQBT2100(saturate(nits / 10000.0));
}

bool NeedsAPLProcessing()
{
#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
    return EnableEOTFBoost || (EnablePQHDRLUT && EnableAPLDrivenLUTCompensationMode);
#else
    return EnableEOTFBoost;
#endif
}

#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
bool NeedsRealPostBoostAPLProcessing()
{
    // Only needed when boost is active and the APL-driven LUT path is active.
    // LUT-only mode has no boosted intermediate; its effective APL is the raw source APL.
    // The boost model itself no longer consumes this measurement: the closed-loop APL
    // is solved in-frame from the decoded grid's aggregate response curve.
    return EnableEOTFBoost && EnablePQHDRLUT && EnableAPLDrivenLUTCompensationMode;
}
#endif

#if PQHDRLUT_ENABLE
// --- Optional final PQ HDR calibration LUT layer ---


float3 PQHDRLUT_PQ_To_Nits(float3 pq)
{
    return PQToLinearBT2100(max(pq, 0.0)) * 10000.0;
}

float3 PQHDRLUT_Nits_To_PQ(float3 nits)
{
    return LinearToPQBT2100(max(nits, 0.0) / 10000.0);
}

// Linear-light Rec.709/scRGB to CIE XYZ, D65.
float3 PQHDRLUT_Rec709_To_XYZ(float3 rgb)
{
    return float3(
        dot(rgb, float3(0.4123907993, 0.3575843394, 0.1804807884)),
        dot(rgb, float3(0.2126390059, 0.7151686788, 0.0721923154)),
        dot(rgb, float3(0.0193308187, 0.1191947798, 0.9505321522))
    );
}

// CIE XYZ, D65 to linear-light Rec.709/scRGB.
float3 PQHDRLUT_XYZ_To_Rec709(float3 xyz)
{
    return float3(
        dot(xyz, float3( 3.2409699419, -1.5373831776, -0.4986107603)),
        dot(xyz, float3(-0.9692436363,  1.8759675015,  0.0415550574)),
        dot(xyz, float3( 0.0556300797, -0.2039769589,  1.0569715142))
    );
}

// Linear-light Rec.2020 to CIE XYZ, D65.
float3 PQHDRLUT_Rec2020_To_XYZ(float3 rgb)
{
    return float3(
        dot(rgb, float3(0.6369580483, 0.1446169036, 0.1688809752)),
        dot(rgb, float3(0.2627002120, 0.6779980715, 0.0593017165)),
        dot(rgb, float3(0.0000000000, 0.0280726930, 1.0609850577))
    );
}

// CIE XYZ, D65 to linear-light Rec.2020.
float3 PQHDRLUT_XYZ_To_Rec2020(float3 xyz)
{
    return float3(
        dot(xyz, float3( 1.7166511880, -0.3556707838, -0.2533662814)),
        dot(xyz, float3(-0.6666843518,  1.6164812366,  0.0157685458)),
        dot(xyz, float3( 0.0176398574, -0.0427706133,  0.9421031212))
    );
}

int PQHDRLUT_Resolve_Input_Mode()
{
    if (PQHDRLUTInputColorSpace == 1)
        return 1;

    if (PQHDRLUTInputColorSpace == 2)
        return 2;

#if BUFFER_COLOR_SPACE == 2
    // scRGB
    return 2;
#elif BUFFER_COLOR_SPACE == 3
    // HDR10 ST.2084 / Rec.2020
    return 1;
#else
    // HDR-only fallback when the runtime reports an unknown buffer color space.
    return 1;
#endif
}

float3 PQHDRLUT_Buffer_To_PQ2020(float3 color, int mode)
{
    if (mode == 2)
    {
        // scRGB is linear Rec.709 with 1.0 representing 80 nits.
        // Keep signed components through the matrix conversion because scRGB can
        // represent wide-gamut colours with negative Rec.709 components.
        float3 rec709_nits = color * 80.0;
        float3 xyz_nits = PQHDRLUT_Rec709_To_XYZ(rec709_nits);
        float3 rec2020_nits = PQHDRLUT_XYZ_To_Rec2020(xyz_nits);
        return PQHDRLUT_Nits_To_PQ(max(rec2020_nits, 0.0));
    }

    return max(color, 0.0);
}

float3 PQHDRLUT_PQ2020_To_Buffer(float3 pq, int mode)
{
    if (mode == 2)
    {
        float3 rec2020_nits = PQHDRLUT_PQ_To_Nits(max(pq, 0.0));
        float3 xyz_nits = PQHDRLUT_Rec2020_To_XYZ(rec2020_nits);
        float3 rec709_nits = PQHDRLUT_XYZ_To_Rec709(xyz_nits);

        // Do not clamp: negative scRGB components can legitimately represent
        // colours outside the Rec.709 triangle.
        return rec709_nits / 80.0;
    }

    return pq;
}

float3 PQHDRLUT_Sample(float3 pq)
{
    float3 coordinate = saturate(pq);

    // Half-texel corrected coordinate for the 3D .cube texture.
    coordinate =
        (coordinate * (PQHDRLUT_SIZE - 1.0) + 0.5) /
        PQHDRLUT_SIZE;

    return tex3D(PQHDRLUTSampler, coordinate).rgb;
}


float3 PQHDRLUT_Apply(float3 color)
{
    int inputMode = PQHDRLUT_Resolve_Input_Mode();
    float3 sourcePQ = PQHDRLUT_Buffer_To_PQ2020(color, inputMode);
    float3 calibratedPQ = PQHDRLUT_Sample(sourcePQ);
    return PQHDRLUT_PQ2020_To_Buffer(calibratedPQ, inputMode);
}
#endif

// NitsToPQ(0.0) = LinearToPQBT2100(0.0) = c1^m2 — pure compile-time constant.
// Replaces two NitsToPQ(0.0) calls per ComputeBT2390ReferenceOutputNits invocation.
static const float PQ_BLACK = 7.309559025783966e-07;

// BT.2390 highlight rolloff in PQ space.
// This follows the Report ITU-R BT.2390 EETF construction when shapeControl = 1.0.
// For other values we keep the same normalized source/target endpoints, then move the
// knee and rebuild the shoulder with a monotonic power form that preserves a slope of 1
// where the rolloff begins and a slope of 0 at the peak. This avoids the S-shaped bend
// that appears when only the Hermite parameterization is warped.
float ComputeBT2390ShapedKneeStart(float maxLum, float shapeControl)
{
    float standardKneeStart = saturate(1.5 * maxLum - 0.5);

    // Standard BT.2390 fast path: avoid log2() and extra shaping math when the control
    // is effectively at its neutral value.
    if (abs(shapeControl - 1.0) <= 1e-4)
        return standardKneeStart;

    float safeShapeControl = max(shapeControl, 1e-4);
    float shapeBias = log2(safeShapeControl);

    if (shapeBias > 0.0)
    {
        float hardT = saturate(shapeBias * 0.5);
        float aggressiveKneeStart = standardKneeStart * 0.15;
        return saturate(lerp(standardKneeStart, aggressiveKneeStart, hardT));
    }

    if (shapeBias < 0.0)
    {
        float softT = saturate(-shapeBias * 0.5);
        float softerKneeStart = standardKneeStart + (maxLum - standardKneeStart) * 0.85;
        return min(lerp(standardKneeStart, softerKneeStart, softT), maxLum - 1e-6);
    }

    return standardKneeStart;
}

#if ENABLE_APL_GRAPH
float ApplyBT2390EETFToPQWithShape(float inputPQ, float sourcePeakNits, float targetPeakNits, float shapeControl)
{
    float safeSourcePeakNits = max(sourcePeakNits, 1e-4);
    float safeTargetPeakNits = max(targetPeakNits, 0.0);

    if (safeTargetPeakNits <= 0.0)
        return PQ_BLACK;

    if (safeTargetPeakNits >= safeSourcePeakNits - 1e-4)
        return saturate(inputPQ);

    float sourceBlackPQ = PQ_BLACK;
    float sourceWhitePQ = max(NitsToPQ(safeSourcePeakNits), sourceBlackPQ + 1e-6);
    float targetWhitePQ = min(NitsToPQ(safeTargetPeakNits), sourceWhitePQ - 1e-6);

    float pqRange = max(sourceWhitePQ - sourceBlackPQ, 1e-6);
    float e1 = saturate((saturate(inputPQ) - sourceBlackPQ) / pqRange);
    float maxLum = saturate((targetWhitePQ - sourceBlackPQ) / pqRange);

    if (maxLum >= 1.0 - 1e-6)
        return saturate(inputPQ);

    float kneeStart = ComputeBT2390ShapedKneeStart(maxLum, shapeControl);
    float e2 = e1;

    if (e1 >= kneeStart)
    {
        float shoulderSpan = max(1.0 - kneeStart, 1e-6);
        float compressionSpan = max(maxLum - kneeStart, 1e-6);
        float u = saturate((e1 - kneeStart) / shoulderSpan);
        float shoulderPower = max(shoulderSpan / compressionSpan, 1.0);

        e2 = kneeStart + compressionSpan * (1.0 - pow(1.0 - u, shoulderPower));
    }

    // In this shader the source and target black levels are both PQ black, so the BT.2390
    // black-lift tail stage is mathematically a no-op and can be skipped.
    return saturate(e2 * pqRange + sourceBlackPQ);
}

float ApplyBT2390EETFToNitsWithShape(float inputNits, float sourcePeakNits, float targetPeakNits, float shapeExponent)
{
    float safeInputNits = max(inputNits, 0.0);
    float outputPQ = ApplyBT2390EETFToPQWithShape(NitsToPQ(safeInputNits), sourcePeakNits, targetPeakNits, shapeExponent);
    return max(PQToLinearScalar(outputPQ) * 10000.0, 0.0);
}

float ApplyBT2390EETFToNits(float inputNits, float sourcePeakNits, float targetPeakNits)
{
    return ApplyBT2390EETFToNitsWithShape(inputNits, sourcePeakNits, targetPeakNits, 1.0);
}
#endif

float GetSceneNitsFromColor(float3 color)
{
    if (APLInputMode == 1)
    {
        float3 linearPQ = PQToLinearBT2100(color);
        return GetLuma2020(linearPQ) * 10000.0;
    }

    // scRGB may use negative Rec.709 components to represent wide-gamut colours.
    // Preserve those signed components in the luma dot product, then clamp only
    // the resulting physical luminance to zero.
    return max(GetLuma709(color) * SIGNAL_REFERENCE_NITS, 0.0);
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

    float2 d0 = uv - float2(0.3, 0.25);
    float2 d1 = uv - float2(0.7, 0.75);
    bool slash = abs(uv.x - (1.0 - uv.y)) < 0.15;
    bool circles = (dot(d0, d0) < 0.04) || (dot(d1, d1) < 0.04);

    return (slash || circles) ? 1.0 : 0.0;
}

float GetDot(float2 uv)
{
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 0.0;

    float y = 1.0 - uv.y;
    return (uv.x > 0.35 && uv.x < 0.65 && y > 0.00 && y < 0.20) ? 1.0 : 0.0;
}

float Remap01(float x, float a, float b)
{
    return saturate((x - a) / max(b - a, 1e-6));
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
    3.575635, 5.171928, 7.225205, 10.050671, 13.609937, 18.423039, 24.669117, 32.378420, 42.624646, 55.159547, 71.694443, 92.698470, 118.169439, 151.523348, 191.827692, 244.458256, 307.922168, 390.672851, 494.833309, 620.319592, 783.927695, 981.175502, 1238.660348, 1350.000000
};

// Original 2D table collapsed to one representative compensation value per APL row.
// These anchors are taken near 100 nits, which tracks the row average very closely
// while preserving the stronger APL dependence that matters most.
static const float COMP_APL_1D[APL_COUNT] =
{
    1.000000, // APL 3
    1.438764, // APL 5
    1.864640, // APL 7
    2.594132, // APL 10
    2.833818, // APL 14
    3.005215, // APL 18
    3.164954, // APL 22
    3.255290, // APL 25
    3.513427, // APL 35
    3.805490  // APL 50
};

static const float COMP_MIN = 1.0;
static const float COMP_MAX = 3.805490;

int FindAPLIndex(float aplPct)
{
    // Branchless: all APL_COUNT-1 comparisons are independent and emit in parallel.
    // [loop]+branch forces a serial dependency chain; [unroll]+step() removes it.
    int idx = 0;
    [unroll]
    for (int i = 0; i < APL_COUNT - 1; ++i)
        idx += int(step(APL_POINTS[i + 1], aplPct));
    return min(idx, APL_COUNT - 2);
}

float ApplyCompensationFreezeAPL(float aplPct)
{
    float freezeAPL = CompensationFreezeAPLPercent;
    if (freezeAPL > 0.0)
        aplPct = min(aplPct, freezeAPL);

    return aplPct;
}

float LookupMeasuredComp1D(float aplPct)
{
    float lookupAPL = ApplyCompensationFreezeAPL(aplPct);
    float clampedAPL = clamp(lookupAPL, APL_POINTS[0], APL_POINTS[APL_COUNT - 1]);
    int a0 = FindAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        APL_POINTS[a0], COMP_APL_1D[a0],
        APL_POINTS[a1], COMP_APL_1D[a1]
    );
}

#if PER_APL_BOOST_STRENGTH_ENABLE
float GetPerAPLBoostStrengthAtIndex(int idx)
{
    if (idx == 0) return APLBoostStrength03;
    if (idx == 1) return APLBoostStrength05;
    if (idx == 2) return APLBoostStrength07;
    if (idx == 3) return APLBoostStrength10;
    if (idx == 4) return APLBoostStrength14;
    if (idx == 5) return APLBoostStrength18;
    if (idx == 6) return APLBoostStrength22;
    if (idx == 7) return APLBoostStrength25;
    if (idx == 8) return APLBoostStrength35;
    return APLBoostStrength50;
}

float LookupPerAPLBoostStrength(float aplPct)
{
    if (!EnablePerAPLBoostStrength)
        return MaxAPLBoostStrength;

    float clampedAPL = clamp(aplPct, APL_POINTS[0], APL_POINTS[APL_COUNT - 1]);
    int a0 = FindAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        APL_POINTS[a0], GetPerAPLBoostStrengthAtIndex(a0),
        APL_POINTS[a1], GetPerAPLBoostStrengthAtIndex(a1)
    );
}
#endif

// LUT shapes the scene-compensation weight only. Final response is a nits-domain gain.
float ComputeAPLBoostFader(float currentAPL)
{
    float triggerStart = saturate(APLTrigger);
    float fadeWidth = max(APLTriggerFadeWidth, 0.0);

    if (fadeWidth <= 1e-6)
        return step(triggerStart, currentAPL);

    float triggerEnd = min(triggerStart + fadeWidth, 1.0);
    return Remap01(currentAPL, triggerStart, max(triggerEnd, triggerStart + 1e-6));
}

float ComputeTemporalBlendFactor(float smoothingSeconds)
{
    if (smoothingSeconds <= 1e-6)
        return 1.0;

    float dtSeconds = max(FrameTime, 0.0) * 0.001;
    return saturate(1.0 - exp(-dtSeconds / max(smoothingSeconds, 1e-6)));
}


float ComputeSceneBoostStrength(float currentAPL)
{
    float fader = ComputeAPLBoostFader(currentAPL);
#if PER_APL_BOOST_STRENGTH_ENABLE
    float aplPct = saturate(currentAPL) * 100.0;
    float boostStrength = LookupPerAPLBoostStrength(aplPct);
#else
    float boostStrength = MaxAPLBoostStrength;
#endif
    return max(boostStrength * fader, 0.0);
}


float ComputeSceneLogGainFromMeasuredComp(float measuredComp, float currentAPL)
{
    float safeMeasuredComp = max(measuredComp, 1.0);
    return log2(safeMeasuredComp) * ComputeSceneBoostStrength(currentAPL);
}

float ComputeSceneLogGainFromAPL(float currentAPL)
{
    float aplPct = saturate(currentAPL) * 100.0;
    float measuredComp = max(LookupMeasuredComp1D(aplPct), 1.0);
    return ComputeSceneLogGainFromMeasuredComp(measuredComp, currentAPL);
}

// Conservative upper bound of the achievable scene log-gain under the current
// settings. The compensation table is monotonic, so the maximum measured
// compensation is the last APL anchor after applying the optional freeze clamp.
// Strength is bounded independently; the trigger fader is <= 1. This keeps the
// response-curve candidate gains wide enough to cover the live gain without
// over-spacing them unnecessarily when Compensation Freeze APL % is active.
float ComputeCandidateMaxLogGain()
{
    float maxComp = max(LookupMeasuredComp1D(APL_POINTS[APL_COUNT - 1]), 1.0);
    float maxStrength = MaxAPLBoostStrength;

#if PER_APL_BOOST_STRENGTH_ENABLE
    if (EnablePerAPLBoostStrength)
    {
        maxStrength = 0.0;
        [unroll]
        for (int j = 0; j < APL_COUNT; ++j)
            maxStrength = max(maxStrength, GetPerAPLBoostStrengthAtIndex(j));
    }
#endif

    return max(log2(maxComp) * max(maxStrength, 0.0), 0.0);
}

#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
float3 PQHDRLUT_Apply_APLDriven(float3 color, float currentAPL)
{
    int inputMode = PQHDRLUT_Resolve_Input_Mode();
    float3 sourcePQ = PQHDRLUT_Buffer_To_PQ2020(color, inputMode);
    float3 sourceNits = PQHDRLUT_PQ_To_Nits(sourcePQ);
    float sourceLumaNits = max(GetLuma2020(sourceNits), 0.0);

    // Always compute the original fixed-LUT result first. In APL-driven mode this
    // result supplies the final luminance, so disabling this mode is still exactly
    // the original path and enabling it changes only the colour-direction behaviour.
    float3 fixedCalibratedPQ = PQHDRLUT_Sample(sourcePQ);
    float3 fixedCalibratedNits = PQHDRLUT_PQ_To_Nits(fixedCalibratedPQ);
    float fixedCalibratedLumaNits = max(GetLuma2020(fixedCalibratedNits), 0.0);

    // At black / near-black, fall back to the fixed LUT. Chroma-only extraction is
    // unstable when luma is tiny and would magnify harmless LUT black-offset noise.
    if (sourceLumaNits <= 1e-4 || fixedCalibratedLumaNits <= 1e-4)
        return PQHDRLUT_PQ2020_To_Buffer(fixedCalibratedPQ, inputMode);

    float aplPct = saturate(currentAPL) * 100.0;
    float measuredComp = max(LookupMeasuredComp1D(aplPct), 1.0);

    if (measuredComp <= 1.0001)
        return PQHDRLUT_PQ2020_To_Buffer(fixedCalibratedPQ, inputMode);

    // measuredComp > 1 means the monitor is physically darker than requested.
    // effectiveScale estimates the real post-ABL brightness relative to the signal.
    // Example: measuredComp = 2 => a nominal 1000-nit pixel behaves like ~500 nits.
    float effectiveScale = 1.0 / max(measuredComp, 1.0);

    // Sample the LUT at the estimated real post-ABL brightness, using the same
    // RGB/chromaticity direction. This lower-brightness LUT slice is used only
    // to obtain the corrected colour direction: hue, saturation and channel balance.
    float3 effectiveNits = sourceNits * effectiveScale;
    float3 effectivePQ = PQHDRLUT_Nits_To_PQ(effectiveNits);
    float3 aplColourPQ = PQHDRLUT_Sample(effectivePQ);
    float3 aplColourNits = PQHDRLUT_PQ_To_Nits(aplColourPQ);
    float aplColourLumaNits = max(GetLuma2020(aplColourNits), 0.0);

    if (aplColourLumaNits <= 1e-4)
        return PQHDRLUT_PQ2020_To_Buffer(fixedCalibratedPQ, inputMode);

    // Luma-preserving reconstruction:
    //   direction = lower-brightness APL-driven LUT colour
    //   luminance = original fixed-LUT luminance
    // This prevents the lower slice from adding a second luminance correction while
    // still allowing its hue/saturation/channel-balance behaviour to be used.
    float3 lumaPreservedNits = max(aplColourNits * (fixedCalibratedLumaNits / aplColourLumaNits), 0.0);

    return PQHDRLUT_PQ2020_To_Buffer(PQHDRLUT_Nits_To_PQ(lumaPreservedNits), inputMode);
}
#endif

// SolveClosedLoopDisplayAPLFromRaw is defined further below, after the precomputed
// boost/rolloff parameter functions it now uses for its forward model.

#if ENABLE_APL_GRAPH
float ComputeSceneGainNoRolloff(float currentAPL)
{
    float aplPct = saturate(currentAPL) * 100.0;
    float measuredComp = max(LookupMeasuredComp1D(aplPct), 1.0);
    float sceneLogGain = ComputeSceneLogGainFromMeasuredComp(measuredComp, currentAPL);
    return exp2(sceneLogGain);
}

float SignalLumaToNits(float signalLuma)
{
    if (APLInputMode == 1)
        return max(PQToLinearScalar(signalLuma) * 10000.0, 0.0);

    return max(signalLuma, 0.0) * SIGNAL_REFERENCE_NITS;
}

float NitsToSignalLuma(float nits)
{
    if (APLInputMode == 1)
        return NitsToPQ(max(nits, 0.0));

    return max(nits, 0.0) / SIGNAL_REFERENCE_NITS;
}

float ComputeBoostedTargetNitsFromBoostTNoRolloff(float currentAPL, float inputNits)
{
    // Gain is scene-uniform (the participation model was removed in v8.11).
    return max(inputNits, 0.0) * max(ComputeSceneGainNoRolloff(currentAPL), 1.0);
}

#endif

float ComputeRollOffAnchorBoostedNitsFromSceneLogGain(float sceneLogGain)
{
    float rollOffEndNits = max(BoostRollOff, 0.0);

    if (rollOffEndNits <= 0.0)
        return 0.0;

    float referenceInputNits = max(rollOffEndNits, 1e-4);
    return referenceInputNits * exp2(sceneLogGain);
}

#if ENABLE_APL_GRAPH
float ComputeRollOffAnchorBoostedNits(float currentAPL)
{
    float sceneLogGain = ComputeSceneLogGainFromAPL(currentAPL);
    return ComputeRollOffAnchorBoostedNitsFromSceneLogGain(sceneLogGain);
}
#endif

// Precompute all scene-uniform BT.2390 rolloff setup in a 1x1 pass.
// The fullscreen pass still computes the pixel-dependent NitsToPQ/PQToNits work,
// but no longer recomputes source/target white, knee placement, or anchor per pixel.
float4 ComputeBoostRolloffParamsFromSceneLogGain(float sceneLogGain)
{
    float sceneGain = exp2(sceneLogGain);
    float rollOffEndNits = max(BoostRollOff, 0.0);
    float anchorBoostedNits = ComputeRollOffAnchorBoostedNitsFromSceneLogGain(sceneLogGain);

    if (rollOffEndNits <= 0.0)
        return float4(sceneGain, 0.0, 1.0, 0.0);

    float sourcePeakNits = max(anchorBoostedNits, rollOffEndNits + 1e-4);
    float sourceWhitePQ = max(NitsToPQ(sourcePeakNits), PQ_BLACK + 1e-6);
    float targetWhitePQ = min(NitsToPQ(rollOffEndNits), sourceWhitePQ - 1e-6);

    float pqRange = max(sourceWhitePQ - PQ_BLACK, 1e-6);
    float maxLum = saturate((targetWhitePQ - PQ_BLACK) / pqRange);

    if (maxLum >= 1.0 - 1e-6)
        return float4(sceneGain, 0.0, 1.0, 0.0);

    float kneeStart = ComputeBT2390ShapedKneeStart(maxLum, BoostRollOffShape);
    float compressionSpan = max(maxLum - kneeStart, 1e-6);

    return float4(sceneGain, pqRange, kneeStart, compressionSpan);
}

float ApplyBT2390EETFToPQWithPrecomputedParams(float inputPQ, float4 boostParams)
{
    float pqRange = boostParams.g;

    if (pqRange <= 0.0)
        return saturate(inputPQ);

    float kneeStart = saturate(boostParams.b);
    float compressionSpan = max(boostParams.a, 1e-6);

    float e1 = saturate((saturate(inputPQ) - PQ_BLACK) / pqRange);
    float e2 = e1;

    if (e1 >= kneeStart)
    {
        float shoulderSpan = max(1.0 - kneeStart, 1e-6);
        float u = saturate((e1 - kneeStart) / shoulderSpan);
        float shoulderPower = max(shoulderSpan / compressionSpan, 1.0);

        e2 = kneeStart + compressionSpan * (1.0 - pow(1.0 - u, shoulderPower));
    }

    return saturate(e2 * pqRange + PQ_BLACK);
}

float ApplyBT2390EETFToNitsWithPrecomputedParams(float inputNits, float4 boostParams)
{
    float safeInputNits = max(inputNits, 0.0);

    if (boostParams.g <= 0.0)
        return safeInputNits;

    float outputPQ = ApplyBT2390EETFToPQWithPrecomputedParams(NitsToPQ(safeInputNits), boostParams);
    return max(PQToLinearScalar(outputPQ) * 10000.0, 0.0);
}

#if ENABLE_APL_GRAPH
float ApplyBoostWithBT2390Rolloff(float signalLuma, float currentAPL, float anchorBoostedNits)
{
    float originalNits = SignalLumaToNits(signalLuma);
    float fullyBoostedNits = ComputeBoostedTargetNitsFromBoostTNoRolloff(currentAPL, originalNits);
    float rollOffEndNits = max(BoostRollOff, 0.0);

    if (rollOffEndNits <= 0.0)
        return NitsToSignalLuma(fullyBoostedNits);

    float sourcePeakNits = max(anchorBoostedNits, rollOffEndNits + 1e-4);
    float rolledNits = max(ApplyBT2390EETFToNitsWithShape(fullyBoostedNits, sourcePeakNits, rollOffEndNits, BoostRollOffShape), originalNits);

    return NitsToSignalLuma(rolledNits);
}

float ApplyBoostWithSelectedRolloff(float signalLuma, float currentAPL, float anchorBoostedNits)
{
    return ApplyBoostWithBT2390Rolloff(signalLuma, currentAPL, anchorBoostedNits);
}

#endif

float ComputeBoostedLumaNitsFromPrecomputedParams(float inputLumaNits, float4 boostParams)
{
    // Gain is scene-uniform: boostParams.r holds the precomputed exp2(sceneLogGain).
    float safeInputLumaNits = max(inputLumaNits, 0.0);
    float fullyBoostedNits = safeInputLumaNits * max(boostParams.r, 0.0);

    if (boostParams.g <= 0.0)
        return fullyBoostedNits;

    return max(ApplyBT2390EETFToNitsWithPrecomputedParams(fullyBoostedNits, boostParams), safeInputLumaNits);
}

// Mean-pixel closed-loop projection solve. The forward model runs the mean scene
// nits through the exact per-pixel luma pipeline (BT.2390 rolloff, never-darken
// clamp; the boost gain itself is scene-uniform).
//
// Used by graph projections (hypothetical APL slices have no live pixel data)
// and as a first-frame fallback. The live boost path uses
// SolveClosedLoopDisplayAPLFromResponse, which solves on the scene's aggregate
// post-boost response sampled exactly per grid sample by the decode pass.
float SolveClosedLoopDisplayAPLFromRaw(float rawAPL)
{
    float safeRawAPL = saturate(rawAPL);

    if (safeRawAPL <= 1e-6)
        return 0.0;

    float referenceWhite = max(APLReferenceWhiteNits, 1.0);
    float meanInputNits = safeRawAPL * referenceWhite;
    float displayAPL = safeRawAPL;

    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        float sceneLogGain = ComputeSceneLogGainFromAPL(displayAPL);
        float4 rolloffParams = ComputeBoostRolloffParamsFromSceneLogGain(sceneLogGain);
        float boostedMeanNits = ComputeBoostedLumaNitsFromPrecomputedParams(meanInputNits, rolloffParams);
        float estimatedDisplayAPL = saturate(boostedMeanNits / referenceWhite);

        // Mild damping keeps the closed-loop estimate stable with very short smoothing times.
        displayAPL = lerp(displayAPL, estimatedDisplayAPL, 0.85);
    }

    return displayAPL;
}

#if ENABLE_APL_GRAPH
float ComputeGraphClosedLoopAPLFromRawPercent(float rawAPLPercent)
{
    float rawAPL = saturate(rawAPLPercent * 0.01);

    // Graph projections are hypothetical APL slices / window patterns, not the live
    // scene, so the mean-pixel projection model applies here.
    return SolveClosedLoopDisplayAPLFromRaw(rawAPL);
}
#endif

float Max3(float3 v)
{
    return max(max(v.r, v.g), v.b);
}

float ComputeMaxHuePreservingScale(float3 rgb, float channelLimit)
{
    float maxChannel = Max3(rgb);

    if (maxChannel <= 1e-6)
        return 1.0;

    return max(channelLimit / maxChannel, 0.0);
}

float SoftLimitBoostScale(float desiredScale, float maxScale, float kneeFraction)
{
    // This limiter operates only on added boost. It never returns below 1.0,
    // so source pixels are not darkened when they already have no channel headroom.
    float safeDesired = max(desiredScale, 1.0);
    float safeMax = max(maxScale, 1.0);

    if (safeDesired <= 1.0 || safeMax <= 1.0 + 1e-6)
        return 1.0;

    float kneeStart = lerp(1.0, safeMax, saturate(kneeFraction));

    if (safeDesired <= kneeStart)
        return safeDesired;

    float span = max(safeMax - kneeStart, 1e-6);
    float x = (safeDesired - kneeStart) / span;

    return min(kneeStart + span * (x / (1.0 + x)), safeMax);
}

float3 ApplyBoostPreserveColorFromPrecomputedParams(float3 color, float4 boostParams)
{
    if (APLInputMode == 1)
    {
        float3 linearColorNits = PQToLinearBT2100(saturate(color)) * 10000.0;
        float originalLumaNits = max(GetLuma2020(linearColorNits), 0.0);

        if (originalLumaNits <= 1e-6)
            return color;

        float boostedLumaNits = ComputeBoostedLumaNitsFromPrecomputedParams(originalLumaNits, boostParams);
        float colorScale = boostedLumaNits / originalLumaNits;

        if (EnableColorPreservingBoostMode)
        {
            // BoostRollOff = 0 means roll-off disabled: use the full PQ range as the
            // channel limit instead of clamping to 1 nit, which would kill the boost.
            float outputChannelLimitNits = (BoostRollOff > 0.0) ? clamp(BoostRollOff, 1.0, 10000.0) : 10000.0;
            float maxHuePreservingScale = ComputeMaxHuePreservingScale(linearColorNits, outputChannelLimitNits);
            colorScale = SoftLimitBoostScale(colorScale, maxHuePreservingScale, COLOR_PRESERVING_BOOST_KNEE);
        }

        float3 boostedColorNits = linearColorNits * colorScale;

        return LinearToPQBT2100(saturate(max(boostedColorNits, 0.0) / 10000.0));
    }

    float3 linearColor = color;
    // Preserve signed scRGB components for the Rec.709 luma calculation. Negative
    // components legitimately encode colours outside Rec.709; clamp only final luma.
    float originalLumaNits = max(GetLuma709(linearColor) * SIGNAL_REFERENCE_NITS, 0.0);

    if (originalLumaNits <= 1e-6)
        return color;

    float boostedLumaNits = ComputeBoostedLumaNitsFromPrecomputedParams(originalLumaNits, boostParams);
    float colorScale = boostedLumaNits / originalLumaNits;

    if (EnableColorPreservingBoostMode)
    {
        // BoostRollOff = 0 means roll-off disabled: use the full PQ range as the
        // channel limit instead of clamping to 1 nit, which would kill the boost.
        float outputChannelLimit = ((BoostRollOff > 0.0) ? clamp(BoostRollOff, 1.0, 10000.0) : 10000.0) / max(SIGNAL_REFERENCE_NITS, 1.0);
        float maxHuePreservingScale = ComputeMaxHuePreservingScale(max(linearColor, 0.0.xxx), outputChannelLimit);
        colorScale = SoftLimitBoostScale(colorScale, maxHuePreservingScale, COLOR_PRESERVING_BOOST_KNEE);
    }

    return linearColor * colorScale;
}

// Exact per-sample post-boost APL metric for one candidate gain. Mirrors the luma
// path of ApplyBoostPreserveColorFromPrecomputedParams, including the
// color-preserving limiter evaluated with this sample's own max channel.
float ComputePostBoostMetricForSample(float lumaNits, float maxChannelNits, float4 boostParams)
{
    if (lumaNits <= 1e-6)
        return 0.0;

    float boostedLumaNits = ComputeBoostedLumaNitsFromPrecomputedParams(lumaNits, boostParams);

    if (EnableColorPreservingBoostMode && maxChannelNits > 1e-6)
    {
        // BoostRollOff = 0 means roll-off disabled: full-range channel limit.
        float outputChannelLimitNits = (BoostRollOff > 0.0) ? clamp(BoostRollOff, 1.0, 10000.0) : 10000.0;
        float maxHuePreservingScale = max(outputChannelLimitNits / maxChannelNits, 0.0);
        float colorScale = SoftLimitBoostScale(boostedLumaNits / lumaNits, maxHuePreservingScale, COLOR_PRESERVING_BOOST_KNEE);
        boostedLumaNits = lumaNits * colorScale;
    }

    return saturate(boostedLumaNits / max(APLReferenceWhiteNits, 1.0));
}

// Response-curve closed-loop display-side APL solve.
//
// The solve only needs the scene's aggregate post-boost response A(g), not the
// per-sample distribution: the decode pass evaluates the exact boost per sample at
// three candidate gains (in parallel, with per-sample max-channel limiting), the
// reductions average them, and this 1x1 function runs the damped fixed point on
// the log-interpolated four-node curve (A(0) = raw APL). A(g) is near-log-linear,
// so this matches an exact per-sample solve to ~0.03% (validated), while the
// single-threaded work here is one fetch and a handful of ALU ops.
float SolveClosedLoopDisplayAPLFromResponse(float rawAPL)
{
    float safeRawAPL = saturate(rawAPL);

    if (safeRawAPL <= 1e-6)
        return 0.0;

    float gMax = ComputeCandidateMaxLogGain();

    // All strengths zero: gain is identically 1.0, display APL equals raw APL.
    if (gMax <= 1e-6)
        return safeRawAPL;

    float4 responseData = tex2Dlod(SamplerAPLResponse, float4(0.5, 0.5, 0.0, 0.0));

    // Response not populated yet (e.g. first frame after a reload): fall back to
    // the mean-pixel projection model rather than reporting zero.
    if (responseData.a <= 0.0)
        return SolveClosedLoopDisplayAPLFromRaw(safeRawAPL);

    // Log-domain response nodes at g = 0, gMax/3, 2*gMax/3, gMax.
    float4 logA = log2(max(float4(safeRawAPL, responseData.rgb), 1e-6));

    float displayAPL = safeRawAPL;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float g = clamp(ComputeSceneLogGainFromAPL(displayAPL), 0.0, gMax);
        float t = (g / gMax) * 3.0; // node-space position in [0, 3]

        float logEst;
        if (t <= 1.0)
            logEst = lerp(logA.x, logA.y, t);
        else if (t <= 2.0)
            logEst = lerp(logA.y, logA.z, t - 1.0);
        else
            logEst = lerp(logA.z, logA.w, t - 2.0);

        float estimatedDisplayAPL = saturate(exp2(logEst));

        // Mild damping keeps the closed-loop estimate stable with very short smoothing times.
        displayAPL = lerp(displayAPL, estimatedDisplayAPL, 0.85);
    }

    return displayAPL;
}
#if ENABLE_APL_GRAPH
float ComputeBoostedTargetNitsFromBoostT(float currentAPL, float inputNits, float anchorBoostedNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float signalLuma = NitsToSignalLuma(safeInputNits);

    return SignalLumaToNits(ApplyBoostWithSelectedRolloff(signalLuma, currentAPL, anchorBoostedNits));
}


// Restored graph-only 2D measurement table from the original shader.
// Live boost logic stays on the simplified 1D LUT path.
static const float GRAPH_COMP_TABLE_2D[APL_COUNT * NIT_COUNT] =
{
    // APL 3
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,

    // APL 5.
    1.426002, 1.418319, 1.418362, 1.408464, 1.446689, 1.439896, 1.430032, 1.417518,
    1.426299, 1.433991, 1.438851, 1.438764, 1.444610, 1.457619, 1.461260, 1.451388,
    1.439402, 1.439348, 1.449156, 1.454828, 1.475658, 1.465801, 1.461834, 1.465964,

    // APL 7.
    1.901008, 1.891043, 1.866986, 1.852415, 1.892284, 1.878469, 1.864525, 1.844185,
    1.871305, 1.866372, 1.897690, 1.864640, 1.867602, 1.873403, 1.906579, 1.916905,
    1.903288, 1.889973, 1.892973, 1.909843, 1.900878, 1.893409, 1.890150, 1.903270,

    // APL 10.
    2.696380, 2.656127, 2.571548, 2.531986, 2.572484, 2.587114, 2.581274, 2.553721,
    2.578727, 2.589062, 2.576027, 2.594132, 2.565634, 2.567442, 2.589947, 2.602294,
    2.608675, 2.599693, 2.594153, 2.625117, 2.597062, 2.594686, 2.627796, 2.652832,

    // APL 14.
    3.014935, 2.957599, 2.863537, 2.788778, 2.823049, 2.829380, 2.853258, 2.837809,
    2.827572, 2.835808, 2.841031, 2.833818, 2.847280, 2.824416, 2.807268, 2.818432,
    2.831186, 2.839942, 2.825752, 2.826951, 2.820289, 2.822078, 2.817427, 2.881472,

    // APL 18.
    3.261780, 3.193873, 3.084611, 2.979779, 3.016971, 3.011393, 3.028951, 3.025158,
    3.010275, 3.016307, 3.016287, 3.005215, 3.010005, 3.005951, 2.977315, 2.976826,
    2.990090, 3.010028, 2.998662, 2.990941, 2.983052, 2.984053, 2.983156, 3.050799,

    // APL 22.
    3.447681, 3.384312, 3.275227, 3.149785, 3.193180, 3.166548, 3.178603, 3.166691,
    3.149251, 3.159883, 3.163959, 3.164954, 3.146424, 3.145451, 3.112629, 3.112500,
    3.125895, 3.142294, 3.139984, 3.133374, 3.112062, 3.117090, 3.123962, 3.183895,

    // APL 25.
    3.581481, 3.477656, 3.383956, 3.274945, 3.311163, 3.272880, 3.270689, 3.271470,
    3.238493, 3.255894, 3.252702, 3.255290, 3.245266, 3.237040, 3.207700, 3.208200,
    3.209342, 3.225015, 3.219945, 3.227690, 3.204559, 3.204962, 3.212599, 3.278812,

    // APL 35.
    3.930382, 3.875004, 3.743497, 3.582293, 3.612637, 3.575576, 3.551885, 3.544735,
    3.527227, 3.523647, 3.522499, 3.513427, 3.497814, 3.499382, 3.462916, 3.449010,
    3.444877, 3.457831, 3.466321, 3.467338, 3.438985, 3.438391, 3.457077, 3.529337,

    // APL 50.
    4.419177, 4.341897, 4.162438, 3.959669, 3.972455, 3.940871, 3.907002, 3.879731,
    3.879384, 3.844997, 3.833234, 3.805490, 3.812932, 3.811731, 3.775736, 3.739971,
    3.720363, 3.743520, 3.761325, 3.752846, 3.717274, 3.713838, 3.738359, 3.835342,
};


static const int FULLFIELD_100_COUNT = 33;

static const float FULLFIELD_100_INPUT_NITS[FULLFIELD_100_COUNT] =
{
    0.000000, 0.014036, 0.059603, 0.157591, 0.322069, 0.595511, 0.992458, 1.592344,
    2.444660, 3.575635, 5.171928, 7.225205, 10.050671, 13.609937, 18.423039, 24.669117,
    32.378420, 42.624646, 55.159547, 71.694443, 92.698470, 118.169439, 151.523348, 191.827692,
    244.458256, 307.922168, 390.672851, 494.833309, 620.319592, 783.927695, 981.175502, 1238.660348,
    1350.000000
};

static const float FULLFIELD_100_MEASURED_NITS[FULLFIELD_100_COUNT] =
{
    0.000000, 0.014648, 0.088214, 0.193289, 0.318997, 0.661295, 0.989292, 1.447618,
    2.236296, 3.376542, 4.920436, 6.930280, 9.705878, 13.055362, 17.754820, 23.893304,
    31.387134, 40.600344, 45.011469, 47.620761, 50.368808, 52.891721, 56.604813, 68.190830,
    82.588125, 98.494626, 118.746210, 143.304828, 172.126608, 209.385482, 251.637523, 297.368081,
    309.321642
};


static const int FULLFIELD_50_COUNT = 33;

static const float FULLFIELD_50_MEASURED_NITS[FULLFIELD_50_COUNT] =
{
    0.000000, 0.015862, 0.091231, 0.191387, 0.312262, 0.671993, 1.005095, 1.466901,
    2.286369, 3.449248, 5.001989, 7.038000, 9.801644, 13.156916, 17.924338, 24.163637,
    31.789419, 41.683506, 53.627316, 69.185620, 84.080554, 92.334532, 96.520524, 101.008793,
    104.432241, 114.183568, 137.227399, 166.211743, 199.071008, 240.682466, 287.937407, 345.551770,
    363.939963
};

static const int FULLFIELD_25_COUNT = 33;

static const float FULLFIELD_25_MEASURED_NITS[FULLFIELD_25_COUNT] =
{
    0.000000, 0.017065, 0.092684, 0.188695, 0.316532, 0.675762, 1.014760, 1.479035,
    2.288388, 3.459781, 4.990658, 7.024088, 9.770902, 13.179701, 17.913331, 24.091540,
    31.696396, 41.565991, 53.429244, 68.944165, 88.666517, 112.012512, 143.437406, 170.146931,
    184.837337, 192.847953, 201.161006, 207.413333, 229.887560, 275.787536, 329.782519, 395.968809,
    415.892980
};

static const int FULLFIELD_15_COUNT = 33;

static const float FULLFIELD_15_MEASURED_NITS[FULLFIELD_15_COUNT] =
{
    0.000000, 0.016253, 0.094834, 0.187706, 0.317148, 0.676951, 1.015021, 1.479980,
    2.287189, 3.448004, 4.994607, 7.041970, 9.784785, 13.150748, 17.905736, 24.049399,
    31.633274, 41.473215, 53.281468, 69.049461, 88.784032, 112.346140, 143.356248, 182.754138,
    234.099656, 278.571999, 300.521740, 315.031157, 328.402913, 344.764374, 366.215873, 440.577317,
    462.133337
};

static const int FULLFIELD_10_COUNT = 33;

static const float FULLFIELD_10_MEASURED_NITS[FULLFIELD_10_COUNT] =
{
    0.000000, 0.015782, 0.096480, 0.189964, 0.315951, 0.677001, 1.013099, 1.478316,
    2.295147, 3.465727, 5.010710, 7.056289, 9.789478, 13.159489, 17.861715, 24.058224,
    31.705232, 41.587231, 53.534239, 69.184716, 88.953048, 112.100670, 143.263472, 182.691020,
    234.475162, 294.818124, 372.854232, 426.292189, 459.043502, 483.394124, 503.668346, 517.617448,
    513.378666
};


static const int GRAPH_WINDOW_MODE_100 = 0;
static const int GRAPH_WINDOW_MODE_50  = 1;
static const int GRAPH_WINDOW_MODE_25  = 2;
static const int GRAPH_WINDOW_MODE_15  = 3;
static const int GRAPH_WINDOW_MODE_10  = 4;

int FindFullFieldWindowInputIndex(float inputNits)
{
    [loop]
    for (int i = 0; i < FULLFIELD_100_COUNT - 1; ++i)
    {
        if (inputNits < FULLFIELD_100_INPUT_NITS[i + 1])
            return i;
    }

    return FULLFIELD_100_COUNT - 2;
}

int GetFullFieldWindowCountByMode(int mode)
{
    switch (mode)
    {
        case GRAPH_WINDOW_MODE_10: return FULLFIELD_10_COUNT;
        case GRAPH_WINDOW_MODE_15: return FULLFIELD_15_COUNT;
        case GRAPH_WINDOW_MODE_25: return FULLFIELD_25_COUNT;
        case GRAPH_WINDOW_MODE_50: return FULLFIELD_50_COUNT;
        default:                   return FULLFIELD_100_COUNT;
    }
}

float GetFullFieldWindowScaleByMode(int mode)
{
    switch (mode)
    {
        case GRAPH_WINDOW_MODE_10: return 0.10;
        case GRAPH_WINDOW_MODE_15: return 0.15;
        case GRAPH_WINDOW_MODE_25: return 0.25;
        case GRAPH_WINDOW_MODE_50: return 0.50;
        default:                   return 1.00;
    }
}

float GetFullFieldMeasuredNitsByModeAndIndex(int mode, int idx)
{
    switch (mode)
    {
        case GRAPH_WINDOW_MODE_10: return FULLFIELD_10_MEASURED_NITS[idx];
        case GRAPH_WINDOW_MODE_15: return FULLFIELD_15_MEASURED_NITS[idx];
        case GRAPH_WINDOW_MODE_25: return FULLFIELD_25_MEASURED_NITS[idx];
        case GRAPH_WINDOW_MODE_50: return FULLFIELD_50_MEASURED_NITS[idx];
        default:                   return FULLFIELD_100_MEASURED_NITS[idx];
    }
}

float SampleMeasuredOutputNitsFullFieldByMode(int mode, float targetNits)
{
    float clampedNits = clamp(targetNits, FULLFIELD_100_INPUT_NITS[0], FULLFIELD_100_INPUT_NITS[FULLFIELD_100_COUNT - 1]);
    int i0 = FindFullFieldWindowInputIndex(clampedNits);
    int i1 = min(i0 + 1, GetFullFieldWindowCountByMode(mode) - 1);

    return SegmentLerp(
        clampedNits,
        FULLFIELD_100_INPUT_NITS[i0], GetFullFieldMeasuredNitsByModeAndIndex(mode, i0),
        FULLFIELD_100_INPUT_NITS[i1], GetFullFieldMeasuredNitsByModeAndIndex(mode, i1)
    );
}

float GetFullFieldMeasuredMaxInputNitsByMode(int mode)
{
    return FULLFIELD_100_INPUT_NITS[GetFullFieldWindowCountByMode(mode) - 1];
}

float GetFullFieldMeasuredMaxOutputNitsByMode(int mode)
{
    int count = GetFullFieldWindowCountByMode(mode);
    float maxMeasuredNits = 0.0;

    [loop]
    for (int i = 0; i < count; ++i)
        maxMeasuredNits = max(maxMeasuredNits, GetFullFieldMeasuredNitsByModeAndIndex(mode, i));

    return maxMeasuredNits;
}

float ComputeFullFieldRemappedTargetNitsByMode(int mode, float inputNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float currentAPL = SolveClosedLoopDisplayAPLFromRaw(
        saturate(safeInputNits * GetFullFieldWindowScaleByMode(mode) / max(APLReferenceWhiteNits, 1e-4))
    );
    float anchorBoostedNits = ComputeRollOffAnchorBoostedNits(currentAPL);

    return ComputeBoostedTargetNitsFromBoostT(currentAPL, safeInputNits, anchorBoostedNits);
}

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
    [loop]
    for (int i = 0; i < NIT_COUNT - 1; ++i)
    {
        if (inputNits < NIT_POINTS[i + 1])
            return i;
    }

    return NIT_COUNT - 2;
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

float ComputeGraphBoostedTargetNits(float aplPct, float inputNits, float anchorBoostedNits)
{
    float currentAPL = saturate(aplPct / 100.0);
    float safeInputNits = max(inputNits, 0.0);

    return ComputeBoostedTargetNitsFromBoostT(currentAPL, safeInputNits, anchorBoostedNits);
}

float GetAPLMaxMeasuredNits(float aplPct)
{
    float maxMeasured = 0.0;

    [loop]
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


#if ENABLE_APL_GRAPH
float ApplyGraphInputSignalLimitNits(float inputNits)
{
    float safeInputNits = max(inputNits, 0.0);

    if (GraphInputSignalLimitNits > 0.0)
        return min(safeInputNits, GraphInputSignalLimitNits);

    return safeInputNits;
}

float ComputeGraphCurveRemappedTargetNits(bool useFullFieldWindowProjection, int fullFieldWindowMode, float graphClosedLoopAPLPercent, float graphAnchorBoostedNits, float inputNits)
{
    float limitedInputNits = ApplyGraphInputSignalLimitNits(inputNits);

    if (useFullFieldWindowProjection)
        return ComputeFullFieldRemappedTargetNitsByMode(fullFieldWindowMode, limitedInputNits);

    return ComputeGraphBoostedTargetNits(graphClosedLoopAPLPercent, limitedInputNits, graphAnchorBoostedNits);
}

float ComputeGraphCurveCorrectedOutputNits(bool useFullFieldWindowProjection, int fullFieldWindowMode, float graphClosedLoopAPLPercent, float graphMaxMeasuredNits, float remappedTargetNits)
{
    if (useFullFieldWindowProjection)
        return SampleMeasuredOutputNitsFullFieldByMode(fullFieldWindowMode, remappedTargetNits);

    return SampleCorrectedOutputNitsForAPL(graphClosedLoopAPLPercent, remappedTargetNits, graphMaxMeasuredNits);
}

float ComputeGraphCurveMeasuredRawOutputNits(bool useFullFieldWindowProjection, int fullFieldWindowMode, float graphRawAPLPercent, float inputNits)
{
    float limitedInputNits = ApplyGraphInputSignalLimitNits(inputNits);

    if (useFullFieldWindowProjection)
        return SampleMeasuredOutputNitsFullFieldByMode(fullFieldWindowMode, limitedInputNits);

    return SampleRealMeasuredOutputNitsForAPL(graphRawAPLPercent, limitedInputNits);
}
#endif

float ComputeBT2390ReferenceOutputNits(float inputNits, float sourcePeakNits, float targetPeakNits)
{
    return ApplyBT2390EETFToNits(inputNits, sourcePeakNits, targetPeakNits);
}

float GraphAxisCoordinateWithPQMax(float nits, float axisMaxNits, float axisMaxPQ)
{
    float safeNits = max(nits, 0.0);
    float safeAxisMaxNits = max(axisMaxNits, 1.0);

    if (GraphUsePQSpace)
        return NitsToPQ(safeNits) / max(axisMaxPQ, 1e-6);

    return safeNits / safeAxisMaxNits;
}

float GraphTickValueFromFractionWithPQMax(float frac, float axisMaxNits, float axisMaxPQ)
{
    float safeAxisMaxNits = max(axisMaxNits, 1e-6);
    float safeFrac = saturate(frac);

    if (!GraphUsePQSpace)
        return safeAxisMaxNits * safeFrac;

    float tickPQ = axisMaxPQ * safeFrac;
    return max(PQToLinearScalar(tickPQ) * 10000.0, 0.0);
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
    float labelScale = 0.014;
    float digitStepScaled = labelScale * 0.82;
    float margin = thickness * 5.0;

    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);
    bool useFullFieldWindowProjection = GraphUseFullFieldWindowProjection;
    int fullFieldWindowMode = GraphProjectionWindowSize;
    float4 graphParams = tex2Dlod(SamplerGraphParams, float4(0.5, 0.5, 0.0, 0.0));
    float graphMaxMeasuredNits = useFullFieldWindowProjection
        ? GetFullFieldMeasuredMaxOutputNitsByMode(fullFieldWindowMode)
        : graphParams.g;
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

    bool inGraphX = (p.x >= graphXMin && p.x <= graphXMax);

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
    float idealPQRefMask = 0.0;
    float measuredMask = 0.0;
    float remappedMask = 0.0;
    float correctedMask = 0.0;

    if (inGraphCore)
    {
        frameMask = DrawGraphRect(p, graphMin, graphMax, thickness);

        // Grid lines: indices 0–8 (vertical) and 9–17 (horizontal).
        // All endpoints precomputed in PS_CalcGraphLines — zero NitsToPQ/pow here.
        [loop]
        for (int i = 0; i < 9; i++)
        {
            float uV = (float(i)     + 0.5) / float(GRAPH_LINE_COUNT);
            float uH = (float(i + 9) + 0.5) / float(GRAPH_LINE_COUNT);
            float4 segV = tex2Dlod(SamplerGraphLines, float4(uV, 0.5, 0.0, 0.0));
            float4 segH = tex2Dlod(SamplerGraphLines, float4(uH, 0.5, 0.0, 0.0));
            gridMask += DrawGraphLine(p, segV.xy, segV.zw, gridThickness) * 0.32;
            gridMask += DrawGraphLine(p, segH.xy, segH.zw, gridThickness) * 0.32;
        }

        // Tick marks: x-ticks at indices 18–23, y-ticks at 24–29.
        [loop]
        for (int i = 0; i < 6; i++)
        {
            float uX = (float(i + 18) + 0.5) / float(GRAPH_LINE_COUNT);
            float uY = (float(i + 24) + 0.5) / float(GRAPH_LINE_COUNT);
            float4 segX = tex2Dlod(SamplerGraphLines, float4(uX, 0.5, 0.0, 0.0));
            float4 segY = tex2Dlod(SamplerGraphLines, float4(uY, 0.5, 0.0, 0.0));
            tickMask += DrawGraphLine(p, segX.xy, segX.zw, tickThickness);
            tickMask += DrawGraphLine(p, segY.xy, segY.zw, tickThickness);
        }

        // Identity reference dashed line: index 30.
        {
            float uRef = (30.0 + 0.5) / float(GRAPH_LINE_COUNT);
            float4 segRef = tex2Dlod(SamplerGraphLines, float4(uRef, 0.5, 0.0, 0.0));
            refMask = DrawGraphDashedLine(p, segRef.xy, segRef.zw, refThickness, 22.0);
        }

    }

    if (inXLabelRegion || inYLabelRegion)
    {
        [loop]
        for (int i = 0; i < 6; i++)
        {
            // Fetch x-tick (idx 18+i) and y-tick (idx 24+i) endpoints from precomputed texture.
            // xTick = segX.zw (the 'b' endpoint = the tick base on the axis).
            // yTick = segY.xy (the 'a' endpoint = the tick base on the axis).
            float uX = (float(i + 18) + 0.5) / float(GRAPH_LINE_COUNT);
            float uY = (float(i + 24) + 0.5) / float(GRAPH_LINE_COUNT);
            float4 segX = tex2Dlod(SamplerGraphLines, float4(uX, 0.5, 0.0, 0.0));
            float4 segY = tex2Dlod(SamplerGraphLines, float4(uY, 0.5, 0.0, 0.0));
            float2 xTick = segX.zw; // 'b' endpoint is the base of the x-tick (on the axis line)
            float2 yTick = segY.xy; // 'a' endpoint is the base of the y-tick (on the axis line)

            // tickValue in nits is needed only for the integer label.
            // GraphTickValueFromFractionWithPQMax is cheap in linear-space mode; in PQ mode
            // it calls PQToLinearScalar but this block is gated by inXLabelRegion/inYLabelRegion
            // which is a small strip — the pow cost here is acceptable and unavoidable.
            float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 5.0, graphAxisMaxNits, graphAxisMaxPQ);
            int tickLabel = (int)round(tickValue);

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
        // -------------------------------------------------------------------
        // Curve drawing — all segment endpoints precomputed in PS_CalcGraphCurves.
        // Each texel = float4(ax, ay, bx, by) in p-space (texcoord with x*=aspect).
        // Sentinels (x < 0) mark segments to skip (invalid range or disabled curve).
        // Per-pixel cost: GRAPH_CURVE_SAMPLES tex fetches + DrawGraphLine bbox tests.
        // All expensive LUT math + NitsToPQ/pow moved to the 64x4 precompute pass.
        // -------------------------------------------------------------------
        [loop]
        for (int s = 0; s < GRAPH_CURVE_SAMPLES - 1; ++s)
        {
            float u = (float(s) + 0.5) / float(GRAPH_CURVE_SAMPLES);

            // Remapped (green) curve — APL mode only; precompute stores sentinel in window mode
            if (!useFullFieldWindowProjection)
            {
                float4 seg = tex2Dlod(SamplerGraphCurves, float4(u, 0.125, 0.0, 0.0));
                if (seg.x >= 0.0)
                    remappedMask = max(remappedMask, DrawGraphLine(p, seg.xy, seg.zw, curveThickness * 0.95));
            }

            // Corrected / gray projected-output curve
            {
                float4 seg = tex2Dlod(SamplerGraphCurves, float4(u, 0.375, 0.0, 0.0));
                if (seg.x >= 0.0)
                    correctedMask = max(correctedMask, DrawGraphLine(p, seg.xy, seg.zw, curveThickness));
            }

            // Measured raw / light-blue curve
            {
                float4 seg = tex2Dlod(SamplerGraphCurves, float4(u, 0.625, 0.0, 0.0));
                if (seg.x >= 0.0)
                    measuredMask = max(measuredMask, DrawGraphLine(p, seg.xy, seg.zw, curveThickness));
            }
        }

        // BT.2390 reference (magenta dashed) — row 3; precompute stores sentinels when disabled
        if (GraphShowBT2390Reference && max(graphMaxMeasuredNits, 0.0) > 0.0)
        {
            [loop]
            for (int s = 0; s < GRAPH_CURVE_SAMPLES - 1; ++s)
            {
                float u = (float(s) + 0.5) / float(GRAPH_CURVE_SAMPLES);
                float4 seg = tex2Dlod(SamplerGraphCurves, float4(u, 0.875, 0.0, 0.0));
                if (seg.x >= 0.0)
                    idealPQRefMask = max(idealPQRefMask, DrawGraphDashedLine(p, seg.xy, seg.zw, refThickness * 0.95, 18.0));
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
    float measuredMaskSat = saturate(measuredMask);
    float correctedMaskSat = saturate(correctedMask);

    float3 measuredCurveColor = float3(0.62, 0.82, 1.00);
    float3 correctedCurveColor = float3(0.62, 0.62, 0.62);

    graphColor = lerp(graphColor, float3(0.40, 0.65, 1.00) * OSDBrightness * 2.0, saturate(refMask) * GraphOpacity);
    graphColor = lerp(graphColor, float3(1.00, 0.35, 0.92) * OSDBrightness * 2.0, saturate(idealPQRefMask) * saturate(GraphOpacity + 0.02));
    graphColor = lerp(graphColor, measuredCurveColor * OSDBrightness * 1.95, measuredMaskSat * saturate(GraphOpacity * 0.95));
    graphColor = lerp(graphColor, float3(0.30, 0.88, 0.42) * OSDBrightness * 1.55, saturate(remappedMask) * saturate(GraphOpacity + 0.06));
    graphColor = lerp(graphColor, correctedCurveColor * OSDBrightness * 1.85, correctedMaskSat * saturate(GraphOpacity + 0.20));

    return saturate(graphColor);
}

#endif

// --- SHADERS ---

// PASS 0: Parallel APL decode — runs on APL_GRID_W x APL_GRID_H threads.
// Each thread samples the backbuffer at its own UV and performs the full PQ decode
// (or scRGB luma) exactly once, writing the result to TexAPLDecoded.
// This spreads the grid's transcendental calls across that many parallel GPU
// threads instead of serialising them all inside a single 1x1 pixel shader loop.
// PS_CalcAPL then only needs to read and sum precomputed scalars — zero transcendentals.
void PS_DecodeAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord,
                  out float4 outMetrics : SV_Target0, out float4 outNits : SV_Target1)
{
    outMetrics = 0.0.xxxx;
    outNits    = 0.0.xxxx;

    // Runtime pass skipping is not available here, so make the pass a cheap no-op
    // unless EOTF boost or APL-driven LUT mode needs live APL data.
    if (!NeedsAPLProcessing())
        return;

    float3 color = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0.0, 0.0)).rgb;

    float sceneNits   = 0.0;
    float maxChanNits = 0.0;

    // Decode the APL sample only once. In PQ mode the previous path called the
    // ST.2084 EOTF separately for luma and max-channel nits; both values now come
    // from the same decoded RGB triplet. Max-channel nits are only needed when the
    // color-preserving limiter participates in the candidate response solve.
    if (APLInputMode == 1)
    {
        float3 linearNits = PQToLinearBT2100(color) * 10000.0;
        sceneNits = GetLuma2020(linearNits);

        if (EnableEOTFBoost && EnableColorPreservingBoostMode)
            maxChanNits = max(max(linearNits.r, linearNits.g), linearNits.b);
    }
    else
    {
        // Signed scRGB components must participate in the Rec.709 luma dot product;
        // clamp only the resulting luminance. The channel-headroom limiter still
        // uses positive channel magnitudes because negative channels do not clip.
        sceneNits = max(GetLuma709(color) * SIGNAL_REFERENCE_NITS, 0.0);

        if (EnableEOTFBoost && EnableColorPreservingBoostMode)
        {
            float3 positiveColor = max(color, 0.0.xxx);
            maxChanNits = max(max(positiveColor.r, positiveColor.g), positiveColor.b) * SIGNAL_REFERENCE_NITS;
        }
    }

    float rawMetric = saturate(sceneNits / max(APLReferenceWhiteNits, 1.0));

    // Response-curve candidates: this sample's exact post-boost metric at three
    // candidate gains covering the achievable range. Rolloff params are a function
    // of gain only; recomputing them per thread is trivially parallel work.
    float3 candidateMetrics = rawMetric.xxx;

    if (EnableEOTFBoost)
    {
        float gMax = ComputeCandidateMaxLogGain();

        if (gMax > 1e-6)
        {
            [unroll]
            for (int k = 1; k <= 3; ++k)
            {
                float4 candidateParams = ComputeBoostRolloffParamsFromSceneLogGain(gMax * (float(k) / 3.0));
                candidateMetrics[k - 1] = ComputePostBoostMetricForSample(sceneNits, maxChanNits, candidateParams);
            }
        }
    }

    // RT0: .r = raw pre-boost metric, .gba = post-boost metric at the candidate gains
    outMetrics = float4(rawMetric, candidateMetrics);
    // RT1: .r = luma nits (OSD max readout)
    outNits = float4(sceneNits, 0.0, 0.0, 1.0);
}


// PASS 1a: Column reduction — runs on APL_GRID_W parallel threads.
// Each thread sums one column of TexAPLDecoded (APL_GRID_H fetches) and tracks
// the column max nits. Every texel is read exactly once at full fp32 precision, so
// the result is the exact sum/max — no sample is skipped or approximated.
void PS_ReduceAPLColumns(float4 vpos : SV_Position, float2 texcoord : TexCoord,
                         out float4 outSums : SV_Target0, out float4 outMax : SV_Target1)
{
    outSums = 0.0.xxxx;
    outMax  = 0.0.xxxx;

    if (!NeedsAPLProcessing())
        return;

    static const float invW = 1.0 / float(APL_GRID_W);
    static const float invH = 1.0 / float(APL_GRID_H);

    // SV_Position is pixel-center based (x + 0.5), so this hits column texel centers exactly.
    float u = vpos.x * invW;

    float4 columnSums   = 0.0.xxxx;
    float columnMaxNits = 0.0;

    [loop]
    for (int y = 0; y < APL_GRID_H; ++y)
    {
        float v = (float(y) + 0.5) * invH;
        columnSums    += tex2Dlod(SamplerAPLDecoded, float4(u, v, 0.0, 0.0));
        columnMaxNits  = max(columnMaxNits, tex2Dlod(SamplerAPLDecodedNits, float4(u, v, 0.0, 0.0)).r);
    }

    // RT0: column sums of (raw metric, candidate metrics); RT1: column max nits
    outSums = columnSums;
    outMax  = float4(columnMaxNits, 0.0, 0.0, 1.0);
}

// PASS 1b: Final accumulation — runs on a single 1x1 pixel.
// Only APL_GRID_W column results remain to sum, instead of the previous
// whole-grid serial fetches in one thread (a latency-bound pattern).
// The total is mathematically identical; only fp summation order changes.

void PS_CalcAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord,
                out float4 outInstant : SV_Target0, out float4 outResponse : SV_Target1)
{
    outInstant  = 0.0.xxxx;
    outResponse = 0.0.xxxx;

    // No APL accumulation unless EOTF boost or APL-driven LUT mode needs it.
    if (!NeedsAPLProcessing())
        return;

    float4 totals             = 0.0.xxxx;
    float maxSampledSceneNits = 0.0;
    static const float invW            = 1.0 / float(APL_GRID_W);
    static const float invTotalSamples = 1.0 / float(APL_GRID_W * APL_GRID_H);

    [loop]
    for (int x = 0; x < APL_GRID_W; ++x)
    {
        float u = (float(x) + 0.5) * invW;
        totals              += tex2Dlod(SamplerAPLReduced, float4(u, 0.5, 0.0, 0.0));
        maxSampledSceneNits  = max(maxSampledSceneNits, tex2Dlod(SamplerAPLReducedNits, float4(u, 0.5, 0.0, 0.0)).r);
    }

    float4 averages = totals * invTotalSamples; // .r = raw APL, .gba = A(g1), A(g2), A(g3)

    // RT0 (TexAPLInstant, layout unchanged):
    // r = raw current-frame APL metric, g = max sampled decoded scene nits, b = unused, a = valid
    outInstant = float4(averages.r, maxSampledSceneNits, 0.0, 1.0);
    // RT1 (TexAPLResponse): aggregate post-boost response at the candidate gains
    outResponse = float4(averages.gba, 1.0);
}

#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
// Optional real post-boost / pre-LUT APL decode. Evaluated after PS_SmoothAPL
// and PS_CalcBoostParams, so it can reuse the already-computed boost parameters.
// This pass does not feed back into boost setup; the closed-loop APL is solved
// in-frame from the sampled response curve. It only measures the sampled post-boost APL used
// by the final APL-driven LUT path.
float4 PS_DecodePostBoostAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!NeedsRealPostBoostAPLProcessing())
        return float4(0.0, 0.0, 0.0, 0.0);

    float3 color = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0.0, 0.0)).rgb;
    float4 boostParams = tex2Dlod(SamplerBoostParams, float4(0.5, 0.5, 0.0, 0.0));

    float3 boostedColor = ApplyBoostPreserveColorFromPrecomputedParams(color, boostParams);
    float sceneNits = GetSceneNitsFromColor(boostedColor);
    float metric = saturate(sceneNits / max(APLReferenceWhiteNits, 1.0));

    // .r = real post-boost/pre-LUT normalised APL metric
    // .g = real post-boost/pre-LUT sampled nits, for OSD max-nits when this path is active
    return float4(metric, sceneNits, 0.0, 1.0);
}

// Post-boost column reduction — same two-stage scheme as PS_ReduceAPLColumns.
float4 PS_ReducePostBoostAPLColumns(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!NeedsRealPostBoostAPLProcessing())
        return float4(0.0, 0.0, 0.0, 0.0);

    static const float invW = 1.0 / float(APL_GRID_W);
    static const float invH = 1.0 / float(APL_GRID_H);

    float u = vpos.x * invW;

    float columnMetricSum = 0.0;
    float columnMaxNits   = 0.0;

    [loop]
    for (int y = 0; y < APL_GRID_H; ++y)
    {
        float2 data = tex2Dlod(SamplerPostBoostAPLDecoded, float4(u, (float(y) + 0.5) * invH, 0.0, 0.0)).rg;
        columnMetricSum += data.r;
        columnMaxNits    = max(columnMaxNits, data.g);
    }

    return float4(columnMetricSum, columnMaxNits, 0.0, 1.0);
}

float4 PS_CalcPostBoostAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!NeedsRealPostBoostAPLProcessing())
        return float4(0.0, 0.0, 0.0, 0.0);

    float totalMetric = 0.0;
    float maxSampledSceneNits = 0.0;
    static const float invW            = 1.0 / float(APL_GRID_W);
    static const float invTotalSamples = 1.0 / float(APL_GRID_W * APL_GRID_H);

    [loop]
    for (int x = 0; x < APL_GRID_W; ++x)
    {
        float2 data = tex2Dlod(SamplerPostBoostAPLReduced, float4((float(x) + 0.5) * invW, 0.5, 0.0, 0.0)).rg;

        totalMetric += data.r;
        maxSampledSceneNits = max(maxSampledSceneNits, data.g);
    }

    float apl = totalMetric * invTotalSamples;

    // r = real raw post-boost/pre-LUT APL, g = post-boost max sampled nits, a = valid
    return float4(apl, maxSampledSceneNits, 0.0, 1.0);
}
#endif

float4 PS_CopyAPLState(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!NeedsAPLProcessing())
        return float4(0.0, 0.0, 0.0, 0.0);

    return tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
}


float4 PS_SmoothAPL(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!NeedsAPLProcessing())
        return float4(0.0, 0.0, 0.0, 0.0);

    float4 currentData = tex2Dlod(SamplerAPLInstant, float4(0.5, 0.5, 0.0, 0.0));
    float4 prevData = tex2Dlod(SamplerAPLPrev, float4(0.5, 0.5, 0.0, 0.0));

    float rawAPL = saturate(currentData.r);
    float prevSmoothedAPL = saturate(prevData.r);

    float alpha = ComputeTemporalBlendFactor(TransitionSpeed);

    // .b > 0 marks valid history. A dedicated flag (instead of testing r/g/a)
    // prevents a long true-black scene from being misread as "no history", which
    // would snap the smoothing on fade-in.
    float hasPrev = (prevData.b > 0.0) ? 1.0 : 0.0;

    // Boost path: closed-loop display-side APL solved in-frame on the scene's
    // aggregate post-boost response curve. Rolloff, the never-darken clamp and
    // color-preserving limiting were evaluated exactly per sample by the decode
    // pass, so every parameter shapes the estimate directly.
    // When EOTF Boost is disabled but APL-driven LUT mode is enabled, keep a
    // smoothed raw APL only so the LUT-only path still has an APL value.
    float currentAPLForModel = EnableEOTFBoost ? SolveClosedLoopDisplayAPLFromResponse(rawAPL) : rawAPL;
    float smoothedAPL = lerp(currentAPLForModel, lerp(prevSmoothedAPL, currentAPLForModel, alpha), hasPrev);

    // Precompute scene-uniform sceneLogGain here (1x1 pass) so PS_MainPass reads it from the
    // texture instead of recomputing the LUT lookup + log2 chain for every pixel.
    float sceneLogGain = EnableEOTFBoost ? ComputeSceneLogGainFromAPL(smoothedAPL) : 0.0;

    // r = smoothed APL used by the boost model and LUT-only fallback
    //     EOTF Boost ON: closed-loop display-side APL from the response-curve solve
    //     EOTF Boost OFF: smoothed raw APL for APL-driven LUT-only mode
    // g = OSD output-APL fallback; overwritten by sampled post-boost APL in Main_Boost
    //     when APL-driven LUT mode is active and the post-boost pass is valid
    // b = history-valid flag
    // a = precomputed scene log-gain (uniform across all pixels)
    return float4(smoothedAPL, smoothedAPL, 1.0, sceneLogGain);
}

float4 PS_CalcBoostParams(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!EnableEOTFBoost)
        return float4(1.0, 0.0, 1.0, 0.0);

    float4 aplData = tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
    return ComputeBoostRolloffParamsFromSceneLogGain(aplData.a);
}


float DrawOSDDigitAt(float2 texcoord, float2 topRight, float scale, float aspect, int digit)
{
    float2 uv = texcoord;
    uv.x *= aspect;

    float2 anchor = topRight;
    anchor.x *= aspect;

    uv -= anchor;
    uv.x = -uv.x;

    return GetDigit(digit, uv / scale);
}

float DrawOSDPercentAt(float2 texcoord, float2 topRight, float scale, float aspect)
{
    float2 uv = texcoord;
    uv.x *= aspect;

    float2 anchor = topRight;
    anchor.x -= scale / max(aspect, 1e-6);
    anchor.x *= aspect;

    uv -= anchor;

    return GetPercent(uv / scale);
}

float DrawOSDDotAt(float2 texcoord, float2 topRight, float scale, float aspect)
{
    float2 uv = texcoord;
    uv.x *= aspect;

    float2 anchor = topRight;
    anchor.x *= aspect;

    uv -= anchor;
    uv.x = -uv.x;

    return GetDot(uv / scale);
}

float DrawOSDAPLPercent2(float2 texcoord, float2 topRight, float scale, float stepX, float aspect, float currentAPL)
{
    int aplPctX100 = clamp(int(floor(saturate(currentAPL) * 10000.0 + 0.5)), 0, 10000);
    int integerPart = aplPctX100 / 100;
    int fractionalPart = aplPctX100 % 100;

    float mask = 0.0;

    // Fixed 2-decimal percent layout, right-aligned to the hundredths digit:
    // [hundreds][tens][ones].[tenths][hundredths]
    mask += DrawOSDDigitAt(texcoord, topRight, scale, aspect, fractionalPart % 10);
    mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX, 0.0), scale, aspect, (fractionalPart / 10) % 10);
    mask += DrawOSDDotAt(texcoord, topRight - float2(stepX * 2.0, 0.0), scale, aspect);
    mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 3.0, 0.0), scale, aspect, integerPart % 10);

    if (integerPart >= 10)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 4.0, 0.0), scale, aspect, (integerPart / 10) % 10);

    if (integerPart >= 100)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 5.0, 0.0), scale, aspect, (integerPart / 100) % 10);

    return saturate(mask);
}

float DrawOSDRow5(float2 texcoord, float2 topRight, float scale, float stepX, float aspect, int value)
{
    uint v = (uint)clamp(value, 0, 99999);
    float mask = 0.0;

    mask += DrawOSDDigitAt(texcoord, topRight, scale, aspect, (int)(v % 10));

    if (v >= 10)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX, 0.0), scale, aspect, (int)((v / 10) % 10));

    if (v >= 100)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 2.0, 0.0), scale, aspect, (int)((v / 100) % 10));

    if (v >= 1000)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 3.0, 0.0), scale, aspect, (int)((v / 1000) % 10));

    if (v >= 10000)
        mask += DrawOSDDigitAt(texcoord, topRight - float2(stepX * 4.0, 0.0), scale, aspect, (int)((v / 10000) % 10));

    return saturate(mask);
}


#define OSD_CHAR_SPACE -1
#define OSD_CHAR_A 0
#define OSD_CHAR_I 8
#define OSD_CHAR_L 11
#define OSD_CHAR_M 12
#define OSD_CHAR_N 13
#define OSD_CHAR_O 14
#define OSD_CHAR_P 15
#define OSD_CHAR_R 17
#define OSD_CHAR_S 18
#define OSD_CHAR_T 19
#define OSD_CHAR_U 20
#define OSD_CHAR_W 22
#define OSD_CHAR_X 23

float GetOSDLetter(int letter, float2 uv)
{
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 0.0;

    int pattern = 0;

    // 3x5 uppercase glyphs, encoded with the same y-layout as GetDigit.
    if (letter == OSD_CHAR_A) pattern = 11245;
    else if (letter == OSD_CHAR_I) pattern = 29847;
    else if (letter == OSD_CHAR_L) pattern = 4687;
    else if (letter == OSD_CHAR_M) pattern = 24557;
    else if (letter == OSD_CHAR_N) pattern = 24573;
    else if (letter == OSD_CHAR_O) pattern = 31599;
    else if (letter == OSD_CHAR_P) pattern = 15049;
    else if (letter == OSD_CHAR_R) pattern = 15085;
    else if (letter == OSD_CHAR_S) pattern = 29671;
    else if (letter == OSD_CHAR_T) pattern = 29842;
    else if (letter == OSD_CHAR_U) pattern = 23407;
    else if (letter == OSD_CHAR_W) pattern = 23421;
    else if (letter == OSD_CHAR_X) pattern = 23213;

    int x = int(uv.x * 3.0);
    int y = int((1.0 - uv.y) * 5.0);

    return (pattern >> (x + y * 3)) & 1;
}

float DrawOSDLetterAt(float2 texcoord, float2 topLeft, float scale, float aspect, int letter)
{
    float2 uv = texcoord;
    uv.x *= aspect;

    float2 anchor = topLeft;
    anchor.x *= aspect;

    uv -= anchor;

    return GetOSDLetter(letter, uv / scale);
}

float DrawOSDLabel8(float2 texcoord, float2 topLeft, float scale, float stepX, float aspect, int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7)
{
    float mask = 0.0;
    if (c0 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft, scale, aspect, c0);
    if (c1 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX, 0.0), scale, aspect, c1);
    if (c2 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 2.0, 0.0), scale, aspect, c2);
    if (c3 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 3.0, 0.0), scale, aspect, c3);
    if (c4 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 4.0, 0.0), scale, aspect, c4);
    if (c5 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 5.0, 0.0), scale, aspect, c5);
    if (c6 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 6.0, 0.0), scale, aspect, c6);
    if (c7 >= 0) mask += DrawOSDLetterAt(texcoord, topLeft + float2(stepX * 7.0, 0.0), scale, aspect, c7);
    return saturate(mask);
}


float3 DrawStatsOverlay(float2 texcoord, float3 sceneColor, float rawInputAPL, float outputAPL, float maxSampledNits)
{
    float aspect = ReShade::ScreenSize.x / ReShade::ScreenSize.y;
    float invAspect = 1.0 / max(aspect, 1e-6);

    // Smaller compact OSD than previous versions.
    // Row labels:
    //   RAW APL  = current raw input scene APL
    //   OUT APL  = smoothed output/display-side APL used by the boost logic
    //   MAX NITS = current-frame max sampled decoded scene luminance in nits
    float scale = 0.022;
    float glyphWidth = scale * invAspect;
    float stepX = glyphWidth * 1.06;
    float lineSpacing = scale * 1.18;
    float percentGap = glyphWidth * 0.20;

    float labelScale = scale * 0.92;
    float labelGlyphWidth = labelScale * invAspect;
    float labelStepX = labelGlyphWidth * 1.18;
    float labelTotalWidth = labelStepX * 7.0 + labelGlyphWidth;
    float labelGap = glyphWidth * 0.82;

    // Compact three-row numeric OSD on the right.
    float rightMargin = 0.016;
    float2 inputPercentRight = float2(1.0 - rightMargin, 0.034);
    float2 inputRowRight = inputPercentRight - float2(glyphWidth + percentGap, 0.0);
    float2 outputPercentRight = inputPercentRight + float2(0.0, lineSpacing);
    float2 outputRowRight = inputRowRight + float2(0.0, lineSpacing);
    float2 nitsRowRight = inputRowRight + float2(0.0, lineSpacing * 2.0);

    float valueLeft = inputRowRight.x - stepX * 5.0 - glyphWidth;
    float2 inputLabelTopLeft = float2(valueLeft - labelGap - labelTotalWidth, inputRowRight.y);
    float2 outputLabelTopLeft = inputLabelTopLeft + float2(0.0, lineSpacing);
    float2 nitsLabelTopLeft = inputLabelTopLeft + float2(0.0, lineSpacing * 2.0);

    float left = inputLabelTopLeft.x;
    float right = inputPercentRight.x;
    float top = inputRowRight.y;
    float bottom = nitsRowRight.y + scale;

    float padX = glyphWidth * 0.48;
    float padY = scale * 0.20;

    if (texcoord.x < left - padX || texcoord.x > right + padX || texcoord.y < top - padY || texcoord.y > bottom + padY)
        return sceneColor;

    int nitDisplay = clamp(int(floor(max(maxSampledNits, 0.0) + 0.5)), 0, 99999);

    float bgMask = (texcoord.x >= left - padX && texcoord.x <= right + padX && texcoord.y >= top - padY && texcoord.y <= bottom + padY) ? 1.0 : 0.0;

    float rawLabelMask = DrawOSDLabel8(texcoord, inputLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_R, OSD_CHAR_A, OSD_CHAR_W, OSD_CHAR_SPACE, OSD_CHAR_A, OSD_CHAR_P, OSD_CHAR_L, OSD_CHAR_SPACE);
    float outputLabelMask = DrawOSDLabel8(texcoord, outputLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_O, OSD_CHAR_U, OSD_CHAR_T, OSD_CHAR_SPACE, OSD_CHAR_A, OSD_CHAR_P, OSD_CHAR_L, OSD_CHAR_SPACE);
    float nitsLabelMask = DrawOSDLabel8(texcoord, nitsLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_M, OSD_CHAR_A, OSD_CHAR_X, OSD_CHAR_SPACE, OSD_CHAR_N, OSD_CHAR_I, OSD_CHAR_T, OSD_CHAR_S);

    float inputMask = 0.0;
    inputMask += DrawOSDAPLPercent2(texcoord, inputRowRight, scale, stepX, aspect, rawInputAPL);
    inputMask += DrawOSDPercentAt(texcoord, inputPercentRight, scale, aspect);
    inputMask = saturate(inputMask);

    float outputMask = 0.0;
    outputMask += DrawOSDAPLPercent2(texcoord, outputRowRight, scale, stepX, aspect, outputAPL);
    outputMask += DrawOSDPercentAt(texcoord, outputPercentRight, scale, aspect);
    outputMask = saturate(outputMask);

    float nitsMask = DrawOSDRow5(texcoord, nitsRowRight, scale, stepX, aspect, nitDisplay);

    float bgAlpha = 0.18 * OSDBrightness * bgMask;
    float3 shadedScene = sceneColor * (1.0 - bgAlpha);

    float3 inputColor = float3(0.30, 1.00, 0.30) * OSDBrightness;
    float3 outputColor = float3(1.00, 0.90, 0.18) * OSDBrightness;
    float3 nitsColor = float3(0.60, 0.85, 1.00) * OSDBrightness;

    float3 result = shadedScene;
    result = lerp(result, inputColor, saturate(rawLabelMask + inputMask));
    result = lerp(result, outputColor, saturate(outputLabelMask + outputMask));
    result = lerp(result, nitsColor, saturate(nitsLabelMask + nitsMask));

    return result;
}

// PASS 2b: Main Rendering (1D APL-only measured scene gain, scene-uniform)

float4 PS_MainPass(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    bool boostEnabled = EnableEOTFBoost;

#if PQHDRLUT_ENABLE
    bool lutEnabled = EnablePQHDRLUT;
#endif

    // EOTF Boost OFF = clean passthrough or LUT-only path.
    // APL texture reads happen here only when APL-driven LUT mode is enabled.
    if (!boostEnabled)
    {
#if PQHDRLUT_ENABLE
        if (lutEnabled)
        {
#if PQHDRLUT_ENABLE_APL_DRIVEN
            if (EnableAPLDrivenLUTCompensationMode)
            {
                float4 aplData = tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
                color = PQHDRLUT_Apply_APLDriven(color, aplData.r);
            }
            else
#endif
            {
                color = PQHDRLUT_Apply(color);
            }
        }
#endif
#if ENABLE_APL_GRAPH
        if (ShowAPLGraph)
            color = DrawAPLGraphOverlay(texcoord, color);
#endif
        return float4(color, 1.0);
    }

    float4 aplData = tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
    float outputAPLForOSD = aplData.g;

#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
    bool useRealPostBoostAPLForLUT = lutEnabled && EnableAPLDrivenLUTCompensationMode;
    float4 postBoostAPLData = float4(0.0, 0.0, 0.0, 0.0);
    if (useRealPostBoostAPLForLUT)
    {
        postBoostAPLData = tex2Dlod(SamplerPostBoostAPL, float4(0.5, 0.5, 0.0, 0.0));
        if (postBoostAPLData.a > 0.0)
            outputAPLForOSD = postBoostAPLData.r;
    }
#endif

    // sceneLogGain is scene-uniform (depends only on APL + uniforms).
    // It is precomputed once in PS_SmoothAPL and stored in aplData.a,
    // eliminating the LUT lookup + log2 + conditional pow chain per pixel.
    float sceneLogGain = aplData.a;

#if PQHDRLUT_ENABLE
    if ((sceneLogGain <= 0.0) && (lutEnabled == false) && (ShowOSD == false))
    {
#if ENABLE_APL_GRAPH
        if (ShowAPLGraph)
            color = DrawAPLGraphOverlay(texcoord, color);
#endif
        return float4(color, 1.0);
    }
#else
    if ((sceneLogGain <= 0.0) && (ShowOSD == false))
    {
#if ENABLE_APL_GRAPH
        if (ShowAPLGraph)
            color = DrawAPLGraphOverlay(texcoord, color);
#endif
        return float4(color, 1.0);
    }
#endif

    float3 finalColor = color;

    float4 boostParams = tex2Dlod(SamplerBoostParams, float4(0.5, 0.5, 0.0, 0.0));
    finalColor = ApplyBoostPreserveColorFromPrecomputedParams(color, boostParams);

#if PQHDRLUT_ENABLE
    // Optional calibration layer is applied after the boost layer when boost is enabled,
    // or directly to the original HDR signal when boost is disabled. APL-driven LUT
    // mode changes only the LUT-compensation lookup; boost math above is unchanged.
    if (lutEnabled)
    {
#if PQHDRLUT_ENABLE_APL_DRIVEN
        if (EnableAPLDrivenLUTCompensationMode)
        {
            float aplForLUT = aplData.r;
            if (postBoostAPLData.a > 0.0)
                aplForLUT = postBoostAPLData.r;

            finalColor = PQHDRLUT_Apply_APLDriven(finalColor, aplForLUT);
        }
        else
#endif
        {
            finalColor = PQHDRLUT_Apply(finalColor);
        }
    }
#endif

    if (ShowOSD)
    {
        float4 instantData = tex2Dlod(SamplerAPLInstant, float4(0.5, 0.5, 0.0, 0.0));
        float rawInputAPL = saturate(instantData.r);
        float currentMaxSampledNits = max(instantData.g, 0.0);
#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
        if (postBoostAPLData.a > 0.0)
            currentMaxSampledNits = max(postBoostAPLData.g, 0.0);
#endif
        finalColor = DrawStatsOverlay(texcoord, finalColor, rawInputAPL, outputAPLForOSD, currentMaxSampledNits);
    }

#if ENABLE_APL_GRAPH
    // Graph overlay is drawn directly in the main pass (uniform branch — free when hidden).
    // This removes the full-resolution TexBoosted intermediate and the extra fullscreen
    // Debug_Overlay pass that previously cost ~120 MB/frame of bandwidth even with the
    // graph toggled off. Drawn after the OSD to preserve the original overlay order.
    if (ShowAPLGraph)
        finalColor = DrawAPLGraphOverlay(texcoord, finalColor);
#endif

    return float4(finalColor, 1.0);
}

#if ENABLE_APL_GRAPH
float4 PS_CalcGraphParams(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);
    float graphAxisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(graphAxisMaxNits), 1e-6) : 0.0;

    if (GraphUseFullFieldWindowProjection)
    {
        return float4(0.0, 0.0, graphAxisMaxPQ, 0.0);
    }

    float graphRawAPLPercent = clamp(GraphAPLIndex, 0.0, 100.0);
    float graphClosedLoopAPL = ComputeGraphClosedLoopAPLFromRawPercent(graphRawAPLPercent);
    float graphClosedLoopAPLPercent = graphClosedLoopAPL * 100.0;
    float maxMeasuredNits = GetAPLMaxMeasuredNits(graphClosedLoopAPLPercent);
    float graphAnchorBoostedNits = ComputeRollOffAnchorBoostedNits(graphClosedLoopAPL);

    // r = solved closed-loop APL %, g = max measured nits, b = axis max PQ, a = rolloff anchor.
    // The old r value was graphRollOffStartNits but PS_CalcGraphCurves never used it.
    return float4(graphClosedLoopAPLPercent, maxMeasuredNits, graphAxisMaxPQ, graphAnchorBoostedNits);
}

// GRAPH PASS 1b: Precompute grid/tick/ref line screen-space endpoints (32 pixels — free).
// Eliminates ~200 NitsToPQ/pow calls per inGraphCore pixel in the fullscreen draw pass.
float4 PS_CalcGraphLines(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    int idx = int(vpos.x);

    float aspect       = ReShade::ScreenSize.x / ReShade::ScreenSize.y;
    float2 graphPos    = float2(0.055 * aspect, 0.48);
    float2 graphSize   = float2(0.43  * aspect, 0.44);
    float  tickLen     = graphSize.y * 0.018;

    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);
    float4 graphParams     = tex2Dlod(SamplerGraphParams, float4(0.5, 0.5, 0.0, 0.0));
    float  graphAxisMaxPQ  = GraphUsePQSpace ? max(graphParams.b, 1e-6) : 0.0;

    float2 a = 0.0, b = 0.0;

    if (idx < 9) // grid vertical lines i=1..9
    {
        int i = idx + 1;
        float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 10.0, graphAxisMaxNits, graphAxisMaxPQ);
        a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, 0.0);
        b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, graphAxisMaxNits);
    }
    else if (idx < 18) // grid horizontal lines i=1..9
    {
        int i = idx - 9 + 1;
        float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 10.0, graphAxisMaxNits, graphAxisMaxPQ);
        a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0,           tickValue);
        b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, graphAxisMaxNits, tickValue);
    }
    else if (idx < 24) // x-tick marks i=0..5
    {
        int i = idx - 18;
        float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 5.0, graphAxisMaxNits, graphAxisMaxPQ);
        float2 xTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, tickValue, 0.0);
        a = xTick + float2(0.0, -tickLen);
        b = xTick;
    }
    else if (idx < 30) // y-tick marks i=0..5
    {
        int i = idx - 24;
        float tickValue = GraphTickValueFromFractionWithPQMax(float(i) / 5.0, graphAxisMaxNits, graphAxisMaxPQ);
        float2 yTick = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0, tickValue);
        a = yTick;
        b = yTick + float2(tickLen, 0.0);
    }
    else if (idx == 30) // identity reference dashed line
    {
        a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, 0.0,             0.0);
        b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, graphAxisMaxNits, graphAxisMaxNits);
    }
    else // idx == 31 — padding sentinel
    {
        return float4(-1.0, -1.0, -1.0, -1.0);
    }

    return float4(a, b);
}

// GRAPH PASS 2: Precompute all curve segment endpoints (64 x 4 = 256 pixels — free).
//
// Each texel (s, row) stores float4(ax, ay, bx, by) in p-space screen coords
// (texcoord with p.x *= aspect) for curve segment s, row = GCURVE_* index.
// Sentinel float4(-1,-1,-1,-1) marks segments to skip in the draw pass.
//
// This removes all expensive LUT math + NitsToPQ/pow calls from the fullscreen
// graph draw in PS_MainPass.  The per-pixel draw loop only does tex fetches + DrawGraphLine.
float4 PS_CalcGraphCurves(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    static const float4 SENTINEL = float4(-1.0, -1.0, -1.0, -1.0);

    int s   = int(vpos.x);   // 0 .. GRAPH_CURVE_SAMPLES-1
    int row = int(vpos.y);   // 0 .. 3

    // Only segments 0..62 are valid; column 63 is a pad, never fetched by the draw pass.
    if (s >= GRAPH_CURVE_SAMPLES - 1)
        return SENTINEL;

    // --- Shared graph layout (must exactly match DrawAPLGraphOverlay) ---
    float aspect       = ReShade::ScreenSize.x / ReShade::ScreenSize.y;
    float2 graphPos    = float2(0.055 * aspect, 0.48);
    float2 graphSize   = float2(0.43  * aspect, 0.44);
    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);

    float4 graphParams               = tex2Dlod(SamplerGraphParams, float4(0.5, 0.5, 0.0, 0.0));
    float  graphAxisMaxPQ            = GraphUsePQSpace ? max(graphParams.b, 1e-6) : 0.0;
    float  graphRawAPLPercent        = clamp(GraphAPLIndex, 0.0, 100.0);
    float  graphClosedLoopAPLPercent = graphParams.r;
    float  graphAnchorBoostedNits    = graphParams.a;

    bool useFF = GraphUseFullFieldWindowProjection;
    int fullFieldWindowMode = GraphProjectionWindowSize;

    // --- Sample the two nits x-values for this segment ---
    float t0 = float(s)     / float(GRAPH_CURVE_SAMPLES - 1);
    float t1 = float(s + 1) / float(GRAPH_CURVE_SAMPLES - 1);
    float x0 = GraphSampleNitsFromFraction(t0, graphAxisMaxNits, graphAxisMaxPQ);
    float x1 = GraphSampleNitsFromFraction(t1, graphAxisMaxNits, graphAxisMaxPQ);

    float graphMaxMeasuredNits = useFF
        ? GetFullFieldMeasuredMaxOutputNitsByMode(fullFieldWindowMode)
        : graphParams.g;

    float4 result = SENTINEL;

    if (row == GCURVE_REMAPPED)
    {
        // Green re-mapped curve: standard APL mode only, using the closed-loop display-side APL solved from the selected raw input APL.
        if (!useFF)
        {
            float y0 = ComputeGraphCurveRemappedTargetNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphAnchorBoostedNits, x0);
            float y1 = ComputeGraphCurveRemappedTargetNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphAnchorBoostedNits, x1);
            float2 a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0);
            float2 b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1);
            result = float4(a, b);
        }
    }
    else if (row == GCURVE_CORRECTED)
    {
        // Gray projected / corrected output curve (both modes).
        float y0r = ComputeGraphCurveRemappedTargetNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphAnchorBoostedNits, x0);
        float y1r = ComputeGraphCurveRemappedTargetNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphAnchorBoostedNits, x1);
        float y0  = ComputeGraphCurveCorrectedOutputNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphMaxMeasuredNits, y0r);
        float y1  = ComputeGraphCurveCorrectedOutputNits(useFF, fullFieldWindowMode, graphClosedLoopAPLPercent, graphMaxMeasuredNits, y1r);
        float2 a  = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0);
        float2 b  = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1);
        result = float4(a, b);
    }
    else if (row == GCURVE_MEASURED)
    {
        // Light-blue measured raw curve at the selected raw input APL / window set (clamped to measuredMaxInputNits).
        float measuredMaxInputNits = useFF
            ? GetFullFieldMeasuredMaxInputNitsByMode(fullFieldWindowMode)
            : GetGraphMeasuredMaxInputNits();

        if (x0 < measuredMaxInputNits)
        {
            float mx1 = min(x1, measuredMaxInputNits);
            float y0   = ComputeGraphCurveMeasuredRawOutputNits(useFF, fullFieldWindowMode, graphRawAPLPercent, x0);
            float y1   = ComputeGraphCurveMeasuredRawOutputNits(useFF, fullFieldWindowMode, graphRawAPLPercent, mx1);
            float2 a   = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0,  y0);
            float2 b   = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, mx1, y1);
            result = float4(a, b);
        }
    }
    else // row == GCURVE_BT2390REF (row 3)
    {
        // Magenta dashed BT.2390 reference curve (optional).
        // Source peak intentionally follows the graph axis max, so the reference
        // maps the currently visible input range down to the projected measured
        // peak (per user preference; the curve therefore reshapes with axis zoom).
        float idealReferencePeakNits = max(graphMaxMeasuredNits, 0.0);
        if (GraphShowBT2390Reference && idealReferencePeakNits > 0.0)
        {
            float lx0 = ApplyGraphInputSignalLimitNits(x0);
            float lx1 = ApplyGraphInputSignalLimitNits(x1);
            float y0  = ComputeBT2390ReferenceOutputNits(lx0, graphAxisMaxNits, idealReferencePeakNits);
            float y1  = ComputeBT2390ReferenceOutputNits(lx1, graphAxisMaxNits, idealReferencePeakNits);
            float2 a  = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0);
            float2 b  = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1);
            result = float4(a, b);
        }
    }
    return result;
}

#endif

technique EOTF_Boost_1D_APL_LUT 
{
    // Parallel decode: APL_GRID_W x APL_GRID_H threads each decode one APL sample.
    pass APL_Decode
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_DecodeAPL;
        RenderTarget0 = TexAPLDecoded;
        RenderTarget1 = TexAPLDecodedNits;
    }

    // Stage-1 reduction: APL_GRID_W threads each sum one decoded column.
    pass APL_Reduce
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_ReduceAPLColumns;
        RenderTarget0 = TexAPLReduced;
        RenderTarget1 = TexAPLReducedNits;
    }

    pass APL_Calculation
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcAPL;
        RenderTarget0 = TexAPLInstant;
        RenderTarget1 = TexAPLResponse;
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

    pass Boost_Params
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcBoostParams;
        RenderTarget = TexBoostParams;
    }

#if PQHDRLUT_ENABLE && PQHDRLUT_ENABLE_APL_DRIVEN
    pass PostBoost_APL_Decode
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_DecodePostBoostAPL;
        RenderTarget = TexPostBoostAPLDecoded;
    }

    pass PostBoost_APL_Reduce
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_ReducePostBoostAPLColumns;
        RenderTarget = TexPostBoostAPLReduced;
    }

    pass PostBoost_APL_Calculation
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcPostBoostAPL;
        RenderTarget = TexPostBoostAPL;
    }
#endif

#if ENABLE_APL_GRAPH
    // Graph precompute passes read only uniforms and TexGraphParams — no dependency on
    // the boosted image — so they run before Main_Boost, which then draws the overlay
    // itself. This eliminates the full-resolution TexBoosted intermediate and the extra
    // fullscreen Debug_Overlay pass entirely.
    pass Graph_Params
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphParams;
        RenderTarget = TexGraphParams;
    }

    pass Graph_Lines
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphLines;
        RenderTarget = TexGraphLines;
    }

    pass Graph_Curves
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphCurves;
        RenderTarget = TexGraphCurves;
    }
#endif

    pass Main_Boost
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_MainPass;
    }
}
