/*
    EOTF Boost v8.18.7.9 - Dense Independent 1D APL Table
    Calibrated for monitor MSI MPG 341CQR QD-OLED X36
    ================================================================

    Purpose
    -------
    This shader boosts HDR luminance to compensate for OLED / display ABL behavior
    using a simplified measured 1D APL lookup table.

    This version applies compensation as a scene-uniform multiplicative gain in
    absolute nits (the pixel-participation / shadow-protection-floor model was
    removed in v8.11; every pixel receives the full scene gain):

        scene_gain = measured_compensation(APL) ^ effective_strength(APL)
        pixel_gain = scene_gain

    where:
        - APL is the scene average picture level metric (0..1, shown as 0..100%)
        - compensation > 1 means the display measured darker than the requested target

    The live compensation uses an independent replaceable 1D APL table measured
    at one configurable target brightness (normally 100 nits). The dense table is
    intentionally decoupled from the legacy 2D debug-graph APL rows, so the Excel-
    generated COMP_APL_COUNT / COMP_APL_POINTS / COMP_APL_1D block can be pasted
    directly without changing graph resources or optional per-APL strength controls.

    The live boost strength comes from Global APL Boost Strength. Optional per-APL
    controls can be compiled in with PER_APL_BOOST_STRENGTH_ENABLE.

    The lookup table is NOT used as a direct per-pixel inverse solve. It supplies
    the scene-level log-gain target, while strength, closed-loop APL, highlight
    roll-off and optional channel-headroom limiting shape the final output.

    Optional final HDR calibration LUT layer
    ----------------------------------------
    If enabled, PQHDRLUT.cube is applied as a separate final display-calibration
    layer after the EOTF boost calculation, or directly to the source signal when
    EOTF Boost is disabled. The original boost logic remains intact.

    Closed-loop display-side APL
    ----------------------------
    This revision replaces the sparse live APL grid with a rolling full-frame compute
    histogram. Every source pixel is decoded once with a 128-segment piecewise-linear
    PQ EOTF approximation (no per-pixel pow calls), accumulated inside a 32x32 tile,
    then accumulated into persistent row histograms and reduced globally each frame.
    Normal mode uses 64 quasi-log luminance bins;
    color-preserving mode uses 64 luminance x 4 max/luma-headroom bins.

    One interleaved set of tile rows is refreshed per frame by default, while a final
    256-thread compute pass evaluates one occupied histogram bin per thread
    at the same three candidate gains and performs a shared-memory reduction. This
    produces the same four aggregate response nodes consumed by the v8.18.2 bracketed
    ten-step closed-loop solver, while eliminating sparse-grid spatial aliasing and
    the former long single-pixel histogram loop. The OSD maximum is tracked exactly
    for the rolling reconstruction in 0.25-nit units rather than inferred from the
    upper edge of the highest occupied histogram bin. Shared solved-APL smoothing,
    boost, fixed PQHDRLUT timing, graph logic and the response-node clamp fix remain unchanged.

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

// --- FULL-FRAME HISTOGRAM PATH ---
// One 16x16 compute group covers a 32x32 source tile; every thread processes a
// fixed 2x2 quad. Normal mode uses 64 quasi-log luma bins. Color-preserving mode
// uses 64 luma x 4 max/luma-ratio bins so the headroom limiter remains represented.
#define APL_FFH_TILE_SIZE             32
#define APL_FFH_THREADS_X             16
#define APL_FFH_THREADS_Y             16
#define APL_FFH_PIXELS_X               2
#define APL_FFH_PIXELS_Y               2
#define APL_FFH_LUMA_BINS              64
#define APL_FFH_COLOR_LUMA_BINS        64
#define APL_FFH_COLOR_RATIO_BINS        4
#define APL_FFH_MAX_BINS              256
#define APL_FFH_BLOCK_W                16
#define APL_FFH_BLOCK_H                16
#define APL_FFH_AUX_X                  APL_FFH_BLOCK_W
#define APL_FFH_REDUCE_THREADS         64

// Packed tile-local histogram: upper bits hold count, lower 20 bits hold the
// accumulated 10-bit within-bin luminance positions. A 32x32 tile has at most
// 1024 pixels, so 1024 * 1023 remains below 2^20 and cannot carry into count.
#define APL_FFH_PACK_COUNT_SHIFT       20
#define APL_FFH_PACK_OFFSET_MASK       ((1 << APL_FFH_PACK_COUNT_SHIFT) - 1)
#define APL_FFH_PACK_OFFSET_SCALE      1023
#define APL_FFH_PACK_ONE_COUNT         (1 << APL_FFH_PACK_COUNT_SHIFT)

#define APL_FFH_TILE_COUNT_X ((BUFFER_WIDTH  + APL_FFH_TILE_SIZE - 1) / APL_FFH_TILE_SIZE)
#define APL_FFH_TILE_COUNT_Y ((BUFFER_HEIGHT + APL_FFH_TILE_SIZE - 1) / APL_FFH_TILE_SIZE)

// One compact 16x16 histogram block per 32-pixel-high screen row. Tile groups
// accumulate directly into their row block, replacing separate per-tile
// histograms with only APL_FFH_TILE_COUNT_Y persistent row histograms.
// The count texture has one auxiliary column at x = APL_FFH_AUX_X. Its first
// texel in each row block stores that screen row's exact maximum luminance in Q4
// units. Offset and color-headroom textures keep the original 16-column width.
#define APL_FFH_ROW_HIST_TEX_W APL_FFH_BLOCK_W
#define APL_FFH_ROW_COUNT_TEX_W (APL_FFH_BLOCK_W + 1)
#define APL_FFH_ROW_TEX_H (APL_FFH_TILE_COUNT_Y * APL_FFH_BLOCK_H)

// Rolling full-frame update mode.
// 0 = disabled: refresh the complete frame every rendered frame.
// N > 0 = split the 32-pixel-high tile rows into N interleaved phases and
// refresh one phase per rendered frame. The global histogram and solved APL are
// still rebuilt every frame from the latest persistent histogram of every row.
// Default 6 updates roughly one sixth of the screen rows each frame, so every
// source pixel is refreshed once per six rendered frames.
#ifndef APL_FULLFRAME_ROLLING_PHASES
    #define APL_FULLFRAME_ROLLING_PHASES 6
#endif

#if APL_FULLFRAME_ROLLING_PHASES < 0
    #error "APL_FULLFRAME_ROLLING_PHASES must be >= 0"
#endif

#if APL_FULLFRAME_ROLLING_PHASES > APL_FFH_TILE_COUNT_Y
    #error "APL_FULLFRAME_ROLLING_PHASES cannot exceed the number of full-frame tile rows"
#endif

#if APL_FULLFRAME_ROLLING_PHASES > 0
    #define APL_FFH_ROLLING_ROW_SLOTS ((APL_FFH_TILE_COUNT_Y + APL_FULLFRAME_ROLLING_PHASES - 1) / APL_FULLFRAME_ROLLING_PHASES)
    // Row metadata packs a 12-bit runtime histogram signature above a 20-bit
    // pixel count. The count validates that a persistent row was fully rebuilt.
    #define APL_FFH_ROW_META_COUNT_BITS 20
    #define APL_FFH_ROW_META_COUNT_MASK ((1 << APL_FFH_ROW_META_COUNT_BITS) - 1)
    #if (BUFFER_WIDTH * APL_FFH_TILE_SIZE) > APL_FFH_ROW_META_COUNT_MASK
        #error "Rolling full-frame row pixel count exceeds the 20-bit metadata field"
    #endif
#else
    #define APL_FFH_ROLLING_ROW_SLOTS APL_FFH_TILE_COUNT_Y
#endif

#if (APL_FFH_THREADS_X * APL_FFH_PIXELS_X != APL_FFH_TILE_SIZE) || (APL_FFH_THREADS_Y * APL_FFH_PIXELS_Y != APL_FFH_TILE_SIZE)
    #error "Full-frame APL histogram tile/thread dimensions are inconsistent"
#endif
#if (APL_FFH_BLOCK_W * APL_FFH_BLOCK_H != APL_FFH_MAX_BINS)
    #error "Full-frame APL histogram block dimensions do not match maximum bin count"
#endif
#if (APL_FFH_COLOR_LUMA_BINS * APL_FFH_COLOR_RATIO_BINS != APL_FFH_MAX_BINS)
    #error "Color-preserving full-frame APL histogram dimensions are inconsistent"
#endif


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
    UI_TOOLTIP("Enables the APL-based EOTF boost/ABL compensation layer. Turn this off to bypass boost and skip live APL processing.")
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
> = 1300.0;

uniform float CompensationFreezeAPLPercent <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 95.0; ui_step = 0.1;
    ui_label = "Compensation Freeze APL %";
    UI_TOOLTIP("Freezes the measured compensation lookup above the selected solved display-side APL percentage. Default 50.0 means higher operating points keep using the 50% compensation value. Set 95.0 to use the full measured table; 0 = disabled.")
> = 50.0;

uniform float MaxAPLBoostStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "Global APL Boost Strength";
    UI_TOOLTIP("Scales measured APL compensation in log-gain space. 1.0 applies the full measured compensation. Values below 1.0 under-compensate; values above 1.0 intentionally over-compensate.")
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
> = 1300.0;

uniform float BoostRollOffShape <
    ui_type = "slider";
    ui_min = 0.25; ui_max = 4.0; ui_step = 0.01;
    ui_label = "Boost Roll-Off Shape";
    UI_TOOLTIP("Adjusts the live roll-off character by moving the knee together with the shoulder curvature so the transition remains smooth and monotonic. 1.0 = neutral BT.2390-style shoulder. Values below 1.0 start later and retain highlights longer. Values above 1.0 start earlier and compress highlights more strongly.")
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
    UI_TOOLTIP("Displays raw input APL (green), solved display-side APL used by the boost model (yellow), and the exact maximum input-scene luminance present in the rolling full-frame reconstruction (cyan, 0.25-nit analysis precision). Stats are available while EOTF Boost is active.")
> = false;

uniform float OSDBrightness <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_label = "OSD Brightness";
    UI_TOOLTIP("Controls OSD and graph overlay brightness.")
> = 0.5;

uniform float FrameTime < source = "frametime"; >;
uniform int FrameCount < source = "framecount"; >;

#if ENABLE_APL_GRAPH
uniform bool ShowAPLGraph <
    ui_label = "Show APL EOTF Debug Graph";
    UI_TOOLTIP("Shows the analysis graph. Standard mode: Blue dashed = identity reference, optional Magenta dashed = BT.2390-style reference tone map using the projected measured peak for the selected raw APL input, Light blue = real 2D measured LUT output for that raw input APL, Green = shader remapped target after the closed-loop APL solve, Gray = projected measured output at the solved display-side operating point. Window projection mode uses the same four-node response solve, APL lookup, rolloff, never-darken clamp and optional color-preserving limiter as the live shader. Light blue = measured raw window EOTF, Green = live-equivalent remapped signal, Gray = measured window output predicted from that signal.")
> = true;

uniform bool GraphShowBT2390Reference <
    ui_label = "Graph Show BT.2390-Style Reference";
    UI_TOOLTIP("Shows or hides the optional BT.2390-style roll-off reference overlay. It uses the projected measured peak for the selected APL or window size.")
> = false;

uniform bool GraphUseFullFieldWindowProjection <
    ui_label = "Graph Use Window Projection";
    UI_TOOLTIP("Switches the debug graph to the built-in window PQ measurement projection overlay. Graph APL (%) is ignored. The selected window is modeled as bright window pixels plus a black background, then processed through the same steady-state four-node closed-loop response solver and boost/rolloff math as the live shader. Blue dashed = identity, optional Magenta dashed = BT.2390 reference, Light blue = measured raw window EOTF, Green = remapped signal sent to the display, Gray = predicted measured window output.")
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
    UI_TOOLTIP("Continuous raw / pre-boost input APL value used by the standard fixed-APL slice graph. The live-equivalent closed-loop operating point is solved from a uniform representative scene at this APL and then held fixed across the plotted EOTF curve. Light blue = measured curve for the selected raw APL, Green = remapped signal, Gray = predicted measured output. Ignored when Graph Use Window Projection is enabled.")
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
> = 1300.0;

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

// Uniform candidate boost/roll-off parameters used by the histogram response solve.
// Width 3 stores the exact parameter sets for gMax/3, 2*gMax/3 and gMax.
// A tiny three-pixel setup pass computes these once per frame; every APL sample
// then reuses them instead of repeating uniform PQ/log/exp setup for every occupied bin.
texture TexCandidateBoostParams
{
    Width = 3;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerCandidateBoostParams
{
    Texture = TexCandidateBoostParams;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
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
storage2D<float4> StorageAPLInstant
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


// Tiny runtime-generated PQ decode LUT. A 129-pixel setup pass evaluates exact
// ST.2084 once per node; hardware linear filtering supplies the 128-segment
// approximation to the full-frame compute pass. This avoids dynamically indexing
// a 129-element const array, which D3D11 FXC tried to force-unroll (X3511).
texture TexAPLFastPQNits
{
    Width = 129;
    Height = 1;
    Format = R32F;
};
sampler SamplerAPLFastPQNits
{
    Texture = TexAPLFastPQNits;
    AddressU = CLAMP;
    AddressV = CLAMP;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;
};

// Compact row-aggregated and global histogram resources. The hot tile pass uses
// one packed group-shared count+luminance atomic per pixel in normal mode, merges
// equal bins inside each thread's 2x2 quad, then performs only one storage atomic
// per occupied tile/bin into the corresponding 32-pixel-high screen-row histogram.
// The count textures reserve one auxiliary column for the exact rolling maximum.
texture TexAPLRowHistCount
{
    Width = APL_FFH_ROW_COUNT_TEX_W;
    Height = APL_FFH_ROW_TEX_H;
    Format = R32I;
};
sampler2D<int> SamplerAPLRowHistCount
{
    Texture = TexAPLRowHistCount;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<int> StorageAPLRowHistCount
{
    Texture = TexAPLRowHistCount;
};

texture TexAPLRowHistOffset10
{
    Width = APL_FFH_ROW_HIST_TEX_W;
    Height = APL_FFH_ROW_TEX_H;
    Format = R32I;
};
sampler2D<int> SamplerAPLRowHistOffset10
{
    Texture = TexAPLRowHistOffset10;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<int> StorageAPLRowHistOffset10
{
    Texture = TexAPLRowHistOffset10;
};

// Color-preserving mode only: two-nit-unit max-channel sums. A single 32-pixel
// row contains at most BUFFER_WIDTH*32 pixels, so this stays safely in signed R32I.
texture TexAPLRowHistMaxNits
{
    Width = APL_FFH_ROW_HIST_TEX_W;
    Height = APL_FFH_ROW_TEX_H;
    Format = R32I;
};
sampler2D<int> SamplerAPLRowHistMaxNits
{
    Texture = TexAPLRowHistMaxNits;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<int> StorageAPLRowHistMaxNits
{
    Texture = TexAPLRowHistMaxNits;
};

#if APL_FULLFRAME_ROLLING_PHASES > 0
// Persistent rolling-row validity metadata. High bits store the current histogram
// interpretation signature; low 20 bits store the number of pixels accumulated
// into that row. A row participates in the global reduction only when both match.
texture TexAPLRowMeta
{
    Width = 1;
    Height = APL_FFH_TILE_COUNT_Y;
    Format = R32I;
};
sampler2D<int> SamplerAPLRowMeta
{
    Texture = TexAPLRowMeta;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<int> StorageAPLRowMeta
{
    Texture = TexAPLRowMeta;
};
#endif

texture TexAPLGlobalHistCount
{
    Width = APL_FFH_BLOCK_W + 1;
    Height = APL_FFH_BLOCK_H;
    Format = R32I;
};
sampler2D<int> SamplerAPLGlobalHistCount
{
    Texture = TexAPLGlobalHistCount;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<int> StorageAPLGlobalHistCount
{
    Texture = TexAPLGlobalHistCount;
};

// Full-frame offset/max sums can exceed signed 32-bit, so the short row reduction
// converts them to float only after all storage atomics are complete.
texture TexAPLGlobalHistOffset10
{
    Width = APL_FFH_BLOCK_W;
    Height = APL_FFH_BLOCK_H;
    Format = R32F;
};
sampler2D<float> SamplerAPLGlobalHistOffset10
{
    Texture = TexAPLGlobalHistOffset10;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<float> StorageAPLGlobalHistOffset10
{
    Texture = TexAPLGlobalHistOffset10;
};

texture TexAPLGlobalHistMaxNits
{
    Width = APL_FFH_BLOCK_W;
    Height = APL_FFH_BLOCK_H;
    Format = R32F;
};
sampler2D<float> SamplerAPLGlobalHistMaxNits
{
    Texture = TexAPLGlobalHistMaxNits;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
};
storage2D<float> StorageAPLGlobalHistMaxNits
{
    Texture = TexAPLGlobalHistMaxNits;
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
storage2D<float4> StorageAPLResponse
{
    Texture = TexAPLResponse;
};


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
#define GCURVE_REMAPPED  0   // green live-equivalent remapped signal curve
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

// Live-equivalent steady-state projection for each graph x-axis sample.
// Standard APL mode reuses one fixed bracketed operating point; window mode builds
// and solves its four-node response per x sample. Both apply the same gain lookup,
// roll-off, never-darken clamp and optional color limiter as the live shader.
// RGBA layout:
//   .r = remapped signal nits sent to the display
//   .g = predicted measured output from that remapped signal
//   .b = solved display-side APL
//   .a = valid
texture TexGraphProjection
{
    Width  = GRAPH_CURVE_SAMPLES;
    Height = 1;
    Format = RGBA32F;
};
sampler SamplerGraphProjection
{
    Texture   = TexGraphProjection;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
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
    return EnableEOTFBoost;
}

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

// BT.2390-style highlight roll-off in PQ space. The neutral setting uses the
// standard normalized knee placement, then constructs a monotonic power shoulder
// with slope 1 at the knee and slope 0 at the peak. Shape control moves the knee and
// shoulder together while preserving those endpoint constraints.
float ComputeBT2390ShapedKneeStart(float maxLum, float shapeControl)
{
    float standardKneeStart = saturate(1.5 * maxLum - 0.5);

    // Neutral-knee fast path: avoid log2() and extra shaping math when the control
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

static const int GRAPH_APL_COUNT = 10;
static const int NIT_COUNT = 24;

// Legacy APL rows retained only for the optional 2D debug graph and advanced
// per-APL strength controls. They are independent of the live dense 1D table.
static const float GRAPH_APL_POINTS[GRAPH_APL_COUNT] =
{
    3.000000, 5.000000, 7.000000, 10.000000, 14.000000, 18.000000, 22.000000, 25.000000, 35.000000, 50.000000
};

static const float NIT_POINTS[NIT_COUNT] =
{
    3.575635, 5.171928, 7.225205, 10.050671, 13.609937, 18.423039, 24.669117, 32.378420, 42.624646, 55.159547, 71.694443, 92.698470, 118.169439, 151.523348, 191.827692, 244.458256, 307.922168, 390.672851, 494.833309, 620.319592, 783.927695, 981.175502, 1238.660348, 1350.000000
};

// ==========================================================================
// BEGIN DENSE 1D APL TABLE
// Replace the three declarations below with the block generated in Excel at
// Shader Table!A74. Piecewise-linear interpolation is retained by the shader.
// ==========================================================================
// ReShade FX/HLSL has no portable static-array length query. Keep this explicit
// compile-time count synchronized with both arrays and the generated Excel block.
static const int COMP_APL_COUNT = 20;

static const float COMP_APL_POINTS[COMP_APL_COUNT] =
{
    2.000000,
    3.000000,
    4.000000,
    5.000000,
    6.000000,
    7.000000,
    8.000000,
    9.000000,
    10.000000,
    11.000000,
    15.000000,
    20.000000,
    25.000000,
    30.000000,
    35.000000,
    40.000000,
    50.000000,
    60.000000,
    70.000000,
    95.000000
};

static const float COMP_APL_1D[COMP_APL_COUNT] =
{
    1.000000, // APL 2.%
    1.000000, // APL 3.%
    1.222800, // APL 4.%
    1.429226, // APL 5.%
    1.654645, // APL 6.%
    1.896584, // APL 7.%
    2.111334, // APL 8.%
    2.346582, // APL 9.%
    2.598814, // APL 10.%
    2.710108, // APL 11.%
    2.911658, // APL 15.%
    3.127065, // APL 20.%
    3.286341, // APL 25.%
    3.429106, // APL 30.%
    3.558611, // APL 35.%
    3.665642, // APL 40.%
    3.855918, // APL 50.%
    4.029733, // APL 60.%
    4.184151, // APL 70.%
    4.530331  // APL 95.%
};
// ==========================================================================
// END DENSE 1D APL TABLE
// ==========================================================================

int FindCompAPLIndex(float aplPct)
{
    // Branchless: all COMP_APL_COUNT-1 comparisons are independent.
    int idx = 0;
    [unroll]
    for (int i = 0; i < COMP_APL_COUNT - 1; ++i)
        idx += int(step(COMP_APL_POINTS[i + 1], aplPct));
    return min(idx, COMP_APL_COUNT - 2);
}

int FindGraphAPLIndex(float aplPct)
{
    int idx = 0;
    [unroll]
    for (int i = 0; i < GRAPH_APL_COUNT - 1; ++i)
        idx += int(step(GRAPH_APL_POINTS[i + 1], aplPct));
    return min(idx, GRAPH_APL_COUNT - 2);
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
    float clampedAPL = clamp(lookupAPL, COMP_APL_POINTS[0], COMP_APL_POINTS[COMP_APL_COUNT - 1]);
    int a0 = FindCompAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, COMP_APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        COMP_APL_POINTS[a0], COMP_APL_1D[a0],
        COMP_APL_POINTS[a1], COMP_APL_1D[a1]
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

    float clampedAPL = clamp(aplPct, GRAPH_APL_POINTS[0], GRAPH_APL_POINTS[GRAPH_APL_COUNT - 1]);
    int a0 = FindGraphAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, GRAPH_APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        GRAPH_APL_POINTS[a0], GetPerAPLBoostStrengthAtIndex(a0),
        GRAPH_APL_POINTS[a1], GetPerAPLBoostStrengthAtIndex(a1)
    );
}
#endif

float ComputeTemporalBlendFactor(float smoothingSeconds)
{
    if (smoothingSeconds <= 1e-6)
        return 1.0;

    float dtSeconds = max(FrameTime, 0.0) * 0.001;
    return saturate(1.0 - exp(-dtSeconds / max(smoothingSeconds, 1e-6)));
}


float ComputeSceneBoostStrength(float currentAPL)
{
#if PER_APL_BOOST_STRENGTH_ENABLE
    float aplPct = saturate(currentAPL) * 100.0;
    float boostStrength = LookupPerAPLBoostStrength(aplPct);
#else
    float boostStrength = MaxAPLBoostStrength;
#endif
    return max(boostStrength, 0.0);
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
// Strength is bounded independently. This keeps the response-curve candidate
// gains wide enough to cover the live gain without over-spacing them unnecessarily
// when Compensation Freeze APL % is active.
float ComputeCandidateMaxLogGain()
{
    float maxComp = max(LookupMeasuredComp1D(COMP_APL_POINTS[COMP_APL_COUNT - 1]), 1.0);
    float maxStrength = MaxAPLBoostStrength;

#if PER_APL_BOOST_STRENGTH_ENABLE
    if (EnablePerAPLBoostStrength)
    {
        maxStrength = 0.0;
        [unroll]
        for (int j = 0; j < GRAPH_APL_COUNT; ++j)
            maxStrength = max(maxStrength, GetPerAPLBoostStrengthAtIndex(j));
    }
#endif

    return max(log2(maxComp) * max(maxStrength, 0.0), 0.0);
}


float ComputeRollOffAnchorBoostedNitsFromSceneLogGain(float sceneLogGain)
{
    float rollOffEndNits = max(BoostRollOff, 0.0);

    if (rollOffEndNits <= 0.0)
        return 0.0;

    float referenceInputNits = max(rollOffEndNits, 1e-4);
    return referenceInputNits * exp2(sceneLogGain);
}

// Build scene-uniform BT.2390-style roll-off parameters for one log gain.
// The final live parameters are computed in a 1x1 pass, while the three response-
// curve candidates are computed once in a three-pixel setup pass.
float4 ComputeBoostRolloffParamsFromSceneLogGain(float sceneLogGain)
{
    if (sceneLogGain <= 1e-6)
        return float4(1.0, 0.0, 1.0, 0.0);

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

float4 LoadCandidateBoostParams(int candidateIndex)
{
    int idx = clamp(candidateIndex, 1, 3);
    float u = (float(idx) - 0.5) / 3.0;
    return tex2Dlod(SamplerCandidateBoostParams, float4(u, 0.5, 0.0, 0.0));
}

// Three-pixel uniform setup pass: x = 0,1,2 correspond to gMax/3, 2*gMax/3,
// and gMax. This replaces identical transcendental setup in every APL sample.
float4 PS_CalcCandidateBoostParams(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!EnableEOTFBoost)
        return float4(1.0, 0.0, 1.0, 0.0);

    int candidateIndex = min(int(vpos.x) + 1, 3);
    float gMax = ComputeCandidateMaxLogGain();
    return ComputeBoostRolloffParamsFromSceneLogGain(gMax * (float(candidateIndex) / 3.0));
}

float ApplyBT2390EETFToNitsWithPrecomputedParams(float inputNits, float4 boostParams)
{
    float safeInputNits = max(inputNits, 0.0);
    float pqRange = boostParams.g;

    if (pqRange <= 0.0)
        return safeInputNits;

    float inputPQ = NitsToPQ(safeInputNits);
    float kneeStart = saturate(boostParams.b);
    float e1 = saturate((inputPQ - PQ_BLACK) / pqRange);

    // Below the knee the roll-off is exactly identity. Return the original nits
    // directly and avoid an unnecessary inverse-PQ pow round trip.
    if (e1 < kneeStart)
        return safeInputNits;

    float shoulderSpan = max(1.0 - kneeStart, 1e-6);
    float compressionSpan = max(boostParams.a, 1e-6);
    float u = saturate((e1 - kneeStart) / shoulderSpan);
    float shoulderPower = max(shoulderSpan / compressionSpan, 1.0);
    float e2 = kneeStart + compressionSpan * (1.0 - pow(1.0 - u, shoulderPower));
    float outputPQ = saturate(e2 * pqRange + PQ_BLACK);

    return max(PQToLinearScalar(outputPQ) * 10000.0, 0.0);
}

float ComputeBoostedLumaNitsFromPrecomputedParams(float inputLumaNits, float4 boostParams)
{
    // Gain is scene-uniform: boostParams.r holds the precomputed exp2(sceneLogGain).
    float safeInputLumaNits = max(inputLumaNits, 0.0);
    float fullyBoostedNits = safeInputLumaNits * max(boostParams.r, 0.0);

    if (boostParams.g <= 0.0)
        return fullyBoostedNits;

    return max(ApplyBT2390EETFToNitsWithPrecomputedParams(fullyBoostedNits, boostParams), safeInputLumaNits);
}

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
    if (boostParams.r <= 1.000001 && boostParams.g <= 0.0)
        return color;

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

// Exact per-sample post-boost luminance for one gain candidate. Mirrors the luma
// path of ApplyBoostPreserveColorFromPrecomputedParams, including BT.2390 rolloff,
// the never-darken clamp and the optional max-channel color-preserving limiter.
// Keeping this as the shared scalar path lets the live response decode and the
// graph simulators use exactly the same scalar luma math.
float ComputePostBoostNitsForSample(float lumaNits, float maxChannelNits, float4 boostParams)
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

    return boostedLumaNits;
}

// Exact per-sample post-boost APL metric for one candidate gain.
float ComputePostBoostMetricForSample(float lumaNits, float maxChannelNits, float4 boostParams)
{
    return saturate(
        ComputePostBoostNitsForSample(lumaNits, maxChannelNits, boostParams) /
        max(APLReferenceWhiteNits, 1.0)
    );
}

// Evaluate aggregate post-boost APL response A(g) from precomputed log-domain
// nodes at g = 0, gMax/3, 2*gMax/3 and gMax.
float EvaluateDisplayAPLResponseFromLogNodes(float4 logA, float logGain, float gainToNodeScale)
{
    float t = clamp(logGain * gainToNodeScale, 0.0, 3.0);
    float logEst;

    if (t <= 1.0)
        logEst = lerp(logA.x, logA.y, t);
    else if (t <= 2.0)
        logEst = lerp(logA.y, logA.z, t - 1.0);
    else
        logEst = lerp(logA.z, logA.w, t - 2.0);

    return saturate(exp2(logEst));
}

// Shared bracketed closed-loop display-side APL solve. The live shader and both
// graph simulators call this exact function. It solves
//
//     g = ComputeSceneLogGainFromAPL(A(g))
//
// directly in log-gain space. ComputeCandidateMaxLogGain guarantees that the
// requested gain lies within [0, gMax], so the endpoint signs form a valid bracket.
float SolveClosedLoopDisplayAPLFromResponseNodes(float rawAPL, float3 responseNodes, float gMax)
{
    float safeRawAPL = saturate(rawAPL);

    if (safeRawAPL <= 1e-6)
        return 0.0;

    if (gMax <= 1e-6)
        return safeRawAPL;

    // Log nodes and reciprocal gain scaling are invariant across all bisection
    // iterations; compute them once instead of repeating four log2 operations.
    float4 logA = log2(max(float4(safeRawAPL, responseNodes), 1e-6));
    float gainToNodeScale = 3.0 / gMax;

    float lo = 0.0;
    float hi = gMax;

    float aplLo = safeRawAPL;           // A(0)
    float fLo = clamp(ComputeSceneLogGainFromAPL(aplLo), 0.0, gMax);

    // Prefer the stable zero-gain solution when the configured strength requests
    // no boost at the unboosted operating point.
    if (fLo <= 1e-6)
        return aplLo;

    float aplHi = saturate(responseNodes.z); // A(gMax)
    float fHi = clamp(ComputeSceneLogGainFromAPL(aplHi), 0.0, gMax) - gMax;

    if (fHi >= -1e-6)
        return aplHi;

    // Ten bisection steps leave less than 1/1024 of the original log-gain interval.
    // Keep this as a runtime loop to avoid expanding graph-enabled compile time.
    [loop]
    for (int i = 0; i < 10; ++i)
    {
        float mid = 0.5 * (lo + hi);
        float aplMid = EvaluateDisplayAPLResponseFromLogNodes(logA, mid, gainToNodeScale);
        float fMid = clamp(ComputeSceneLogGainFromAPL(aplMid), 0.0, gMax) - mid;

        if (fMid > 0.0)
            lo = mid;
        else
            hi = mid;
    }

    float solvedGain = 0.5 * (lo + hi);
    return EvaluateDisplayAPLResponseFromLogNodes(logA, solvedGain, gainToNodeScale);
}

// Bracketed fallback for a uniform representative scene. This is used only if the
// aggregate response texture is not yet valid and by the standard fixed-APL graph.
// It uses the same precomputed candidate parameters and the same solver as live data.
float SolveUniformSceneDisplayAPL(float rawAPL)
{
    float safeRawAPL = saturate(rawAPL);

    if (safeRawAPL <= 1e-6 || !EnableEOTFBoost)
        return safeRawAPL;

    float gMax = ComputeCandidateMaxLogGain();

    if (gMax <= 1e-6)
        return safeRawAPL;

    float representativeNits = safeRawAPL * max(APLReferenceWhiteNits, 1.0);
    float3 responseNodes;

    [unroll]
    for (int k = 1; k <= 3; ++k)
    {
        responseNodes[k - 1] = ComputePostBoostMetricForSample(
            representativeNits,
            representativeNits,
            LoadCandidateBoostParams(k)
        );
    }

    return SolveClosedLoopDisplayAPLFromResponseNodes(safeRawAPL, responseNodes, gMax);
}

// Live response-curve solve. The decode pass evaluates the exact boost per sample
// at three candidate gains, reductions average those samples, and this 1x1 path
// solves on the resulting aggregate response curve.
float SolveClosedLoopDisplayAPLFromResponse(float rawAPL)
{
    float safeRawAPL = saturate(rawAPL);

    if (safeRawAPL <= 1e-6)
        return 0.0;

    float gMax = ComputeCandidateMaxLogGain();

    if (gMax <= 1e-6)
        return safeRawAPL;

    float4 responseData = tex2Dlod(SamplerAPLResponse, float4(0.5, 0.5, 0.0, 0.0));

    if (responseData.a <= 0.0)
        return SolveUniformSceneDisplayAPL(safeRawAPL);

    return SolveClosedLoopDisplayAPLFromResponseNodes(safeRawAPL, responseData.rgb, gMax);
}
#if ENABLE_APL_GRAPH
// Graph-only 2D measurement table used to project physical display output.
// Live boost logic intentionally remains on the simplified 1D compensation path.
static const float GRAPH_COMP_TABLE_2D[GRAPH_APL_COUNT * NIT_COUNT] =
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

// Live-equivalent steady-state projection of a grayscale test window on black.
// The window and background are treated as two sample populations, so raw APL and
// all three response nodes follow the same per-pixel saturation and averaging as
// the live histogram path. The shared four-node solver then produces
// the same display-side APL that the live shader would use for this static pattern.
// Returns: .r remapped signal nits, .g predicted measured output nits,
//          .b solved display-side APL, .a valid.
float4 ComputeFullFieldLiveProjectionByMode(int mode, float inputNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float windowFraction = saturate(GetFullFieldWindowScaleByMode(mode));
    float referenceWhite = max(APLReferenceWhiteNits, 1.0);

    // Exact aggregate raw metric for a uniform grayscale window on black:
    // each bright pixel is normalized and saturated first, then averaged with
    // the zero-valued background pixels, matching the live decode/reduction path.
    float rawWindowMetric = saturate(safeInputNits / referenceWhite);
    float rawAPL = saturate(windowFraction * rawWindowMetric);
    float displayAPL = rawAPL;
    float remappedSignalNits = safeInputNits;

    if (EnableEOTFBoost && rawAPL > 1e-6)
    {
        float gMax = ComputeCandidateMaxLogGain();

        if (gMax > 1e-6)
        {
            float3 responseNodes = rawAPL.xxx;

            [unroll]
            for (int k = 1; k <= 3; ++k)
            {
                float4 candidateParams = LoadCandidateBoostParams(k);
                float brightPixelMetric = ComputePostBoostMetricForSample(
                    safeInputNits,
                    safeInputNits, // grayscale max channel equals luma
                    candidateParams
                );

                // Black-background pixels contribute zero at every gain node.
                responseNodes[k - 1] = saturate(windowFraction * brightPixelMetric);
            }

            displayAPL = SolveClosedLoopDisplayAPLFromResponseNodes(rawAPL, responseNodes, gMax);
        }

        float sceneLogGain = ComputeSceneLogGainFromAPL(displayAPL);
        float4 liveBoostParams = ComputeBoostRolloffParamsFromSceneLogGain(sceneLogGain);
        remappedSignalNits = ComputePostBoostNitsForSample(
            safeInputNits,
            safeInputNits, // exact grayscale max-channel limiter input
            liveBoostParams
        );
    }

    float predictedMeasuredOutputNits = SampleMeasuredOutputNitsFullFieldByMode(mode, remappedSignalNits);
    return float4(remappedSignalNits, predictedMeasuredOutputNits, displayAPL, 1.0);
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
    float clampedAPL = clamp(aplPct, GRAPH_APL_POINTS[0], GRAPH_APL_POINTS[GRAPH_APL_COUNT - 1]);
    int a0 = FindGraphAPLIndex(clampedAPL);
    int a1 = min(a0 + 1, GRAPH_APL_COUNT - 1);

    return SegmentLerp(
        clampedAPL,
        GRAPH_APL_POINTS[a0], LookupGraphCompForAPLRow2D(a0, inputNits),
        GRAPH_APL_POINTS[a1], LookupGraphCompForAPLRow2D(a1, inputNits)
    );
}

float SampleRealMeasuredOutputNitsForAPL(float aplPct, float targetNits)
{
    float comp = max(LookupMeasuredComp2DGraph(aplPct, targetNits), 1e-6);
    return targetNits / comp;
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
        // All curves share the same uniformly spaced x endpoints in graph space.
        // A pixel can only be touched by its local segment or nearby neighbors.
        // Inspect five segments for line-thickness margin instead of scanning all 63.
        float curveX = saturate((p.x - graphMin.x) / max(graphSize.x, 1e-6));
        int centerSegment = clamp(
            int(floor(curveX * float(GRAPH_CURVE_SAMPLES - 1))),
            0,
            GRAPH_CURVE_SAMPLES - 2
        );

        [unroll]
        for (int neighbor = 0; neighbor < 5; ++neighbor)
        {
            int s = clamp(centerSegment + neighbor - 2, 0, GRAPH_CURVE_SAMPLES - 2);
            float u = (float(s) + 0.5) / float(GRAPH_CURVE_SAMPLES);

            float4 remappedSeg = tex2Dlod(SamplerGraphCurves, float4(u, 0.125, 0.0, 0.0));
            if (remappedSeg.x >= 0.0)
                remappedMask = max(remappedMask, DrawGraphLine(p, remappedSeg.xy, remappedSeg.zw, curveThickness * 0.95));

            float4 correctedSeg = tex2Dlod(SamplerGraphCurves, float4(u, 0.375, 0.0, 0.0));
            if (correctedSeg.x >= 0.0)
                correctedMask = max(correctedMask, DrawGraphLine(p, correctedSeg.xy, correctedSeg.zw, curveThickness));

            float4 measuredSeg = tex2Dlod(SamplerGraphCurves, float4(u, 0.625, 0.0, 0.0));
            if (measuredSeg.x >= 0.0)
                measuredMask = max(measuredMask, DrawGraphLine(p, measuredSeg.xy, measuredSeg.zw, curveThickness));

            if (GraphShowBT2390Reference && graphMaxMeasuredNits > 0.0)
            {
                float4 referenceSeg = tex2Dlod(SamplerGraphCurves, float4(u, 0.875, 0.0, 0.0));
                if (referenceSeg.x >= 0.0)
                    idealPQRefMask = max(
                        idealPQRefMask,
                        DrawGraphDashedLine(p, referenceSeg.xy, referenceSeg.zw, refThickness * 0.95, 18.0)
                    );
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


// --- FULL-FRAME HISTOGRAM HELPERS ---

// Runtime-generated 128-segment piecewise-linear ST.2084 EOTF approximation.
float PS_BuildAPLFastPQLUT(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    int node = min(int(vpos.x), 128);
    float pqCode = float(node) * (1.0 / 128.0);
    return PQToLinearScalar(pqCode) * 10000.0;
}

float APL_FastPQCodeToNits(float pqCode)
{
    // Node i is centered at (i + 0.5) / 129. Hardware bilinear filtering
    // interpolates between the two adjacent exact-PQ nodes.
    float u = (saturate(pqCode) * 128.0 + 0.5) * (1.0 / 129.0);
    return tex2Dlod(SamplerAPLFastPQNits, float4(u, 0.5, 0.0, 0.0)).r;
}

float3 APL_FastPQToNits(float3 pq)
{
    return float3(
        APL_FastPQCodeToNits(pq.r),
        APL_FastPQCodeToNits(pq.g),
        APL_FastPQCodeToNits(pq.b));
}

int APL_QuantizeNitsQ4(float nits)
{
    return clamp((int)floor(max(nits, 0.0) * 4.0 + 0.5), 0, 40000);
}

int APL_QuantizeMaxNits2(float nits)
{
    // Two-nit units keep per-row integer sums safely below signed R32 limits even
    // at very wide resolutions; the resulting mean max-channel error is negligible.
    return clamp((int)floor(max(nits, 0.0) * 0.5 + 0.5), 0, 5000);
}

int APL_ActiveHistogramBins()
{
    return (EnableEOTFBoost && EnableColorPreservingBoostMode) ? APL_FFH_MAX_BINS : APL_FFH_LUMA_BINS;
}

int APL_HistogramLumaBin64(int lumaQ4)
{
    if (lumaQ4 <= 0) return 0;
    int exponent = firstbithigh(lumaQ4);
    int subBin = (exponent >= 2) ? ((lumaQ4 >> (exponent - 2)) & 3) : 0;
    return min(exponent * 4 + subBin, APL_FFH_LUMA_BINS - 1);
}

int APL_HistogramColorRatioBin(int lumaQ4, int maxQ4)
{
    int safeLumaQ4 = max(lumaQ4, 1);
    int ratioBin = 0;
    ratioBin += (maxQ4 >= safeLumaQ4 * 2) ? 1 : 0;
    ratioBin += (maxQ4 >= safeLumaQ4 * 4) ? 1 : 0;
    ratioBin += (maxQ4 >= safeLumaQ4 * 8) ? 1 : 0;
    return min(ratioBin, APL_FFH_COLOR_RATIO_BINS - 1);
}

void APL_HistogramLumaBinBoundsQ4(int lumaBin, out int lowerQ4, out int upperQ4)
{
    lowerQ4 = 0;
    upperQ4 = 0;

    int exponent = lumaBin >> 2;
    int subBin = lumaBin & 3;

    if (exponent < 2)
    {
        lowerQ4 = (exponent == 0) ? 0 : (1 << exponent);
        upperQ4 = (1 << (exponent + 1)) - 1;
    }
    else
    {
        int stepQ4 = 1 << (exponent - 2);
        lowerQ4 = (1 << exponent) + subBin * stepQ4;
        upperQ4 = lowerQ4 + stepQ4 - 1;
    }

    upperQ4 = min(upperQ4, 40000);
    lowerQ4 = min(lowerQ4, upperQ4);
}

int APL_EncodeWithinBinOffset10(int lumaQ4, int lumaBin)
{
    int lowerQ4 = 0;
    int upperQ4 = 0;
    APL_HistogramLumaBinBoundsQ4(lumaBin, lowerQ4, upperQ4);
    int rangeQ4 = upperQ4 - lowerQ4;

    if (rangeQ4 <= 0)
        return 0;

    int numerator = (clamp(lumaQ4, lowerQ4, upperQ4) - lowerQ4) * APL_FFH_PACK_OFFSET_SCALE;
    return clamp((numerator + rangeQ4 / 2) / rangeQ4, 0, APL_FFH_PACK_OFFSET_SCALE);
}

float APL_ReconstructMeanLumaQ4(int binIndex, bool colorHistogram, float pixelCount, float offsetSum10)
{
    int lumaBin = colorHistogram ? (binIndex / APL_FFH_COLOR_RATIO_BINS) : binIndex;
    int lowerQ4 = 0;
    int upperQ4 = 0;
    APL_HistogramLumaBinBoundsQ4(lumaBin, lowerQ4, upperQ4);
    float rangeQ4 = float(upperQ4 - lowerQ4);

    if (pixelCount <= 0.0 || rangeQ4 <= 0.0)
        return float(lowerQ4);

    float meanOffset10 = offsetSum10 / pixelCount;
    return float(lowerQ4) + rangeQ4 * (meanOffset10 / float(APL_FFH_PACK_OFFSET_SCALE));
}

int2 APL_HistogramBlockCoord(int binIndex)
{
    return int2(binIndex & 15, binIndex >> 4);
}

int2 APL_RowHistogramCoord(int rowIndex, int binIndex)
{
    int2 binCoord = APL_HistogramBlockCoord(binIndex);
    return int2(binCoord.x, rowIndex * APL_FFH_BLOCK_H + binCoord.y);
}

int2 APL_RowExactMaxLumaCoord(int rowIndex)
{
    return int2(APL_FFH_AUX_X, rowIndex * APL_FFH_BLOCK_H);
}

int2 APL_GlobalExactMaxLumaCoord()
{
    return int2(APL_FFH_AUX_X, 0);
}

int APL_GetRollingPhase()
{
#if APL_FULLFRAME_ROLLING_PHASES > 0
    int safeFrame = max(FrameCount, 0);
    return safeFrame % APL_FULLFRAME_ROLLING_PHASES;
#else
    return 0;
#endif
}

int APL_GetPhysicalRowFromDispatchSlot(int dispatchRowSlot)
{
#if APL_FULLFRAME_ROLLING_PHASES > 0
    return APL_GetRollingPhase() + dispatchRowSlot * APL_FULLFRAME_ROLLING_PHASES;
#else
    return dispatchRowSlot;
#endif
}

int APL_GetExpectedRowPixelCount(int rowIndex)
{
    int rowStartY = rowIndex * APL_FFH_TILE_SIZE;
    int rowHeight = clamp(BUFFER_HEIGHT - rowStartY, 0, APL_FFH_TILE_SIZE);
    return BUFFER_WIDTH * rowHeight;
}

int APL_GetExpectedTilePixelCount(int tileX, int rowIndex)
{
    int tileStartX = tileX * APL_FFH_TILE_SIZE;
    int tileStartY = rowIndex * APL_FFH_TILE_SIZE;
    int tileWidth = clamp(BUFFER_WIDTH - tileStartX, 0, APL_FFH_TILE_SIZE);
    int tileHeight = clamp(BUFFER_HEIGHT - tileStartY, 0, APL_FFH_TILE_SIZE);
    return tileWidth * tileHeight;
}

int APL_GetRollingHistogramSignature(bool colorHistogram)
{
    // SIGNAL_REFERENCE_NITS has integer UI steps and affects scRGB histogram data.
    // Including it prevents rows produced under different runtime interpretations
    // from being mixed after a mode change. Maximum signature remains below 2^12.
    int signalRef = clamp(int(SIGNAL_REFERENCE_NITS + 0.5), 1, 200);
    return clamp(APLInputMode, 0, 1) + (colorHistogram ? 2 : 0) + signalRef * 4;
}

void APL_DecodePackedFullFrameSample(
    int2 pixel,
    bool needsMaxHistogram,
    out int binIndex,
    out int packedContribution,
    out int maxNits2,
    out int lumaQ4,
    out int valid)
{
    binIndex = 0;
    packedContribution = 0;
    maxNits2 = 0;
    lumaQ4 = 0;
    valid = 0;

    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= BUFFER_WIDTH || pixel.y >= BUFFER_HEIGHT)
        return;

    float3 color = tex2Dfetch(ReShade::BackBuffer, pixel).rgb;
    float sceneNits = 0.0;
    float maxChannelNits = 0.0;

    if (APLInputMode == 1)
    {
        float3 channelNits = APL_FastPQToNits(color);
        sceneNits = max(GetLuma2020(channelNits), 0.0);
        if (needsMaxHistogram)
            maxChannelNits = max(max(channelNits.r, channelNits.g), channelNits.b);
    }
    else
    {
        sceneNits = max(GetLuma709(color) * SIGNAL_REFERENCE_NITS, 0.0);
        if (needsMaxHistogram)
        {
            float3 positiveColor = max(color, 0.0.xxx);
            maxChannelNits = max(max(positiveColor.r, positiveColor.g), positiveColor.b) * SIGNAL_REFERENCE_NITS;
        }
    }

    lumaQ4 = APL_QuantizeNitsQ4(sceneNits);
    int maxQ4 = needsMaxHistogram ? APL_QuantizeNitsQ4(maxChannelNits) : 0;
    int lumaBin = APL_HistogramLumaBin64(lumaQ4);
    binIndex = needsMaxHistogram
        ? lumaBin * APL_FFH_COLOR_RATIO_BINS + APL_HistogramColorRatioBin(lumaQ4, maxQ4)
        : lumaBin;

    int offset10 = APL_EncodeWithinBinOffset10(lumaQ4, lumaBin);
    packedContribution = APL_FFH_PACK_ONE_COUNT + offset10;
    maxNits2 = needsMaxHistogram ? APL_QuantizeMaxNits2(maxChannelNits) : 0;
    valid = 1;
}

groupshared int APL_TilePackedCountLuma[APL_FFH_MAX_BINS];
groupshared int APL_TileMaxNits[APL_FFH_MAX_BINS];
groupshared int APL_TileExactMaxLumaQ4;

void APL_AtomicAddMergedTileSample(int binIndex, int packedContribution, int maxNits2, int valid, bool needsMaxHistogram)
{
    if (valid == 0)
        return;

    atomicAdd(APL_TilePackedCountLuma[binIndex], packedContribution);
    if (needsMaxHistogram)
        atomicAdd(APL_TileMaxNits[binIndex], maxNits2);
}

// One 256-thread group clears the row/bin entries selected for this frame.
[shader("compute")]
void CS_ClearAPLFullFrameRowHistograms(uint3 threadID : SV_GroupThreadID)
{
    bool aplActive = NeedsAPLProcessing();
    int localIndex = int(threadID.x);
    int activeBins = APL_ActiveHistogramBins();
    bool needsMaxHistogram = EnableEOTFBoost && EnableColorPreservingBoostMode;

#if APL_FULLFRAME_ROLLING_PHASES > 0
    int rowSlots = APL_FFH_ROLLING_ROW_SLOTS;
    int totalEntries = rowSlots * activeBins;
    int signature = APL_GetRollingHistogramSignature(needsMaxHistogram);
#else
    int rowSlots = APL_FFH_TILE_COUNT_Y;
    int totalEntries = rowSlots * activeBins;
#endif

    if (aplActive)
    {
        [loop]
        for (int entry = localIndex; entry < totalEntries; entry += 256)
        {
            int rowSlot = entry / activeBins;
            int binIndex = entry - rowSlot * activeBins;
            int rowIndex = APL_GetPhysicalRowFromDispatchSlot(rowSlot);
            if (rowIndex < APL_FFH_TILE_COUNT_Y)
            {
                int2 coord = APL_RowHistogramCoord(rowIndex, binIndex);
                tex2Dstore(StorageAPLRowHistCount, coord, 0);
                tex2Dstore(StorageAPLRowHistOffset10, coord, 0);
                if (needsMaxHistogram)
                    tex2Dstore(StorageAPLRowHistMaxNits, coord, 0);
                if (binIndex == 0)
                {
                    tex2Dstore(StorageAPLRowHistCount, APL_RowExactMaxLumaCoord(rowIndex), 0);
#if APL_FULLFRAME_ROLLING_PHASES > 0
                    tex2Dstore(StorageAPLRowMeta, int2(0, rowIndex), signature << APL_FFH_ROW_META_COUNT_BITS);
#endif
                }
            }
        }
    }
}

[shader("compute")]
void CS_BuildAPLFullFrameRowHistogram(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    // All threads follow the same barrier structure.
    bool aplActive = NeedsAPLProcessing();
    uint localIndex = threadID.y * APL_FFH_THREADS_X + threadID.x;
    int activeBins = APL_ActiveHistogramBins();
    bool needsMaxHistogram = EnableEOTFBoost && EnableColorPreservingBoostMode;
    int physicalRow = APL_GetPhysicalRowFromDispatchSlot(int(groupID.y));
    bool validPhysicalRow = physicalRow < APL_FFH_TILE_COUNT_Y;
    bool decodeThisGroup = aplActive && validPhysicalRow;

    if (localIndex < activeBins)
    {
        APL_TilePackedCountLuma[localIndex] = 0;
        APL_TileMaxNits[localIndex] = 0;
    }
    if (localIndex == 0)
        APL_TileExactMaxLumaQ4 = 0;
    barrier();

    if (decodeThisGroup)
    {
        int2 pixelBase = int2(int(groupID.x) * APL_FFH_TILE_SIZE, physicalRow * APL_FFH_TILE_SIZE) + int2(threadID.xy) * 2;

        int b0, p0, m0, l0, v0;
        int b1, p1, m1, l1, v1;
        int b2, p2, m2, l2, v2;
        int b3, p3, m3, l3, v3;
        APL_DecodePackedFullFrameSample(pixelBase + int2(0, 0), needsMaxHistogram, b0, p0, m0, l0, v0);
        APL_DecodePackedFullFrameSample(pixelBase + int2(1, 0), needsMaxHistogram, b1, p1, m1, l1, v1);
        APL_DecodePackedFullFrameSample(pixelBase + int2(0, 1), needsMaxHistogram, b2, p2, m2, l2, v2);
        APL_DecodePackedFullFrameSample(pixelBase + int2(1, 1), needsMaxHistogram, b3, p3, m3, l3, v3);

        atomicMax(APL_TileExactMaxLumaQ4, max(max(l0, l1), max(l2, l3)));

        // Merge equal bins locally before touching group-shared atomics. This is
        // exact because both packed count/offset and max-channel sums are additive.
        if (v1 != 0 && v0 != 0 && b1 == b0) { p0 += p1; m0 += m1; v1 = 0; }
        if (v2 != 0 && v0 != 0 && b2 == b0) { p0 += p2; m0 += m2; v2 = 0; }
        if (v2 != 0 && v1 != 0 && b2 == b1) { p1 += p2; m1 += m2; v2 = 0; }
        if (v3 != 0 && v0 != 0 && b3 == b0) { p0 += p3; m0 += m3; v3 = 0; }
        if (v3 != 0 && v1 != 0 && b3 == b1) { p1 += p3; m1 += m3; v3 = 0; }
        if (v3 != 0 && v2 != 0 && b3 == b2) { p2 += p3; m2 += m3; v3 = 0; }

        APL_AtomicAddMergedTileSample(b0, p0, m0, v0, needsMaxHistogram);
        APL_AtomicAddMergedTileSample(b1, p1, m1, v1, needsMaxHistogram);
        APL_AtomicAddMergedTileSample(b2, p2, m2, v2, needsMaxHistogram);
        APL_AtomicAddMergedTileSample(b3, p3, m3, v3, needsMaxHistogram);
    }
    barrier();

    if (decodeThisGroup && localIndex == 0)
        atomicMax(StorageAPLRowHistCount, APL_RowExactMaxLumaCoord(physicalRow), APL_TileExactMaxLumaQ4);

    if (decodeThisGroup && localIndex < activeBins)
    {
        int packedValue = APL_TilePackedCountLuma[localIndex];
        int count = packedValue >> APL_FFH_PACK_COUNT_SHIFT;
        if (count > 0)
        {
            int offsetSum10 = packedValue & APL_FFH_PACK_OFFSET_MASK;
            int2 rowCoord = APL_RowHistogramCoord(physicalRow, int(localIndex));
            atomicAdd(StorageAPLRowHistCount, rowCoord, count);
            atomicAdd(StorageAPLRowHistOffset10, rowCoord, offsetSum10);
            if (needsMaxHistogram)
                atomicAdd(StorageAPLRowHistMaxNits, rowCoord, APL_TileMaxNits[localIndex]);
        }
    }

#if APL_FULLFRAME_ROLLING_PHASES > 0
    // One metadata atomic per source tile validates that the selected persistent
    // row received all of its pixels before it is included in the global result.
    if (decodeThisGroup && localIndex == 0)
    {
        int tilePixelCount = APL_GetExpectedTilePixelCount(int(groupID.x), physicalRow);
        if (tilePixelCount > 0)
            atomicAdd(StorageAPLRowMeta, int2(0, physicalRow), tilePixelCount);
    }
#endif
}

groupshared int APL_ReduceCount[APL_FFH_REDUCE_THREADS];
groupshared float APL_ReduceOffset10[APL_FFH_REDUCE_THREADS];
groupshared float APL_ReduceMaxNits[APL_FFH_REDUCE_THREADS];

#define APL_REDUCE_ROW_STAGE(S) \
    if (localIndex < S) { \
        APL_ReduceCount[localIndex] += APL_ReduceCount[localIndex + S]; \
        APL_ReduceOffset10[localIndex] += APL_ReduceOffset10[localIndex + S]; \
        if (needsMaxHistogram) APL_ReduceMaxNits[localIndex] += APL_ReduceMaxNits[localIndex + S]; \
    } \
    barrier();

[shader("compute")]
void CS_ReduceAPLFullFrameRowHistogram(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    bool aplActive = NeedsAPLProcessing();
    int localIndex = int(threadID.x);
    int2 localBinCoord = int2(groupID.xy);
    int binIndex = localBinCoord.y * APL_FFH_BLOCK_W + localBinCoord.x;
    int activeBins = APL_ActiveHistogramBins();
    bool validBin = binIndex < activeBins;
    bool needsMaxHistogram = EnableEOTFBoost && EnableColorPreservingBoostMode;

    int countSum = 0;
    float offsetSum10 = 0.0;
    float maxSumNits = 0.0;
    if (aplActive && validBin)
    {
        [loop]
        for (int rowIndex = localIndex; rowIndex < APL_FFH_TILE_COUNT_Y; rowIndex += APL_FFH_REDUCE_THREADS)
        {
            bool includeRow = true;
#if APL_FULLFRAME_ROLLING_PHASES > 0
            int rowMeta = tex2Dfetch(SamplerAPLRowMeta, int2(0, rowIndex));
            int rowCount = rowMeta & APL_FFH_ROW_META_COUNT_MASK;
            int rowSignature = rowMeta >> APL_FFH_ROW_META_COUNT_BITS;
            int expectedSignature = APL_GetRollingHistogramSignature(needsMaxHistogram);
            includeRow = (rowCount == APL_GetExpectedRowPixelCount(rowIndex)) && (rowSignature == expectedSignature);
#endif
            if (includeRow)
            {
                int2 sourceCoord = APL_RowHistogramCoord(rowIndex, binIndex);
                countSum += tex2Dfetch(SamplerAPLRowHistCount, sourceCoord);
                offsetSum10 += float(tex2Dfetch(SamplerAPLRowHistOffset10, sourceCoord));
                if (needsMaxHistogram)
                    maxSumNits += float(tex2Dfetch(SamplerAPLRowHistMaxNits, sourceCoord));
            }
        }
    }

    APL_ReduceCount[localIndex] = countSum;
    APL_ReduceOffset10[localIndex] = offsetSum10;
    APL_ReduceMaxNits[localIndex] = maxSumNits;
    barrier();

    APL_REDUCE_ROW_STAGE(32)
    APL_REDUCE_ROW_STAGE(16)
    APL_REDUCE_ROW_STAGE(8)
    APL_REDUCE_ROW_STAGE(4)
    APL_REDUCE_ROW_STAGE(2)
    APL_REDUCE_ROW_STAGE(1)

    if (aplActive && validBin && localIndex == 0)
    {
        tex2Dstore(StorageAPLGlobalHistCount, localBinCoord, APL_ReduceCount[0]);
        tex2Dstore(StorageAPLGlobalHistOffset10, localBinCoord, APL_ReduceOffset10[0]);
        if (needsMaxHistogram)
            tex2Dstore(StorageAPLGlobalHistMaxNits, localBinCoord, APL_ReduceMaxNits[0]);

        // The first bin group also reduces the exact maximum across all valid
        // persistent rows. This is only a few dozen point reads and adds no pass.
        if (binIndex == 0)
        {
            int exactMaxLumaQ4 = 0;
            [loop]
            for (int rowIndex = 0; rowIndex < APL_FFH_TILE_COUNT_Y; ++rowIndex)
            {
                bool includeRow = true;
#if APL_FULLFRAME_ROLLING_PHASES > 0
                int rowMeta = tex2Dfetch(SamplerAPLRowMeta, int2(0, rowIndex));
                int rowCount = rowMeta & APL_FFH_ROW_META_COUNT_MASK;
                int rowSignature = rowMeta >> APL_FFH_ROW_META_COUNT_BITS;
                int expectedSignature = APL_GetRollingHistogramSignature(needsMaxHistogram);
                includeRow = (rowCount == APL_GetExpectedRowPixelCount(rowIndex)) && (rowSignature == expectedSignature);
#endif
                if (includeRow)
                    exactMaxLumaQ4 = max(exactMaxLumaQ4, tex2Dfetch(SamplerAPLRowHistCount, APL_RowExactMaxLumaCoord(rowIndex)));
            }
            tex2Dstore(StorageAPLGlobalHistCount, APL_GlobalExactMaxLumaCoord(), exactMaxLumaQ4);
        }
    }
}

#undef APL_REDUCE_ROW_STAGE

groupshared float4 APL_EvalWeightedMetrics[APL_FFH_MAX_BINS];
groupshared float APL_EvalPixelCount[APL_FFH_MAX_BINS];

#define APL_REDUCE_EVAL_STAGE(S) \
    if (localIndex < S) { \
        APL_EvalWeightedMetrics[localIndex] += APL_EvalWeightedMetrics[localIndex + S]; \
        APL_EvalPixelCount[localIndex] += APL_EvalPixelCount[localIndex + S]; \
    } \
    barrier();

[shader("compute")]
void CS_EvaluateAPLFullFrameHistogram(uint3 threadID : SV_GroupThreadID)
{
    bool aplActive = NeedsAPLProcessing();
    int localIndex = int(threadID.x);
    bool colorHistogram = EnableEOTFBoost && EnableColorPreservingBoostMode;
    int activeBins = colorHistogram ? APL_FFH_MAX_BINS : APL_FFH_LUMA_BINS;

    float4 weighted = 0.0.xxxx;
    float pixelCount = 0.0;

    if (aplActive && localIndex < activeBins)
    {
        int2 binCoord = APL_HistogramBlockCoord(localIndex);
        int pixelCountI = tex2Dfetch(SamplerAPLGlobalHistCount, binCoord);
        if (pixelCountI > 0)
        {
            pixelCount = float(pixelCountI);
            float offsetSum10 = tex2Dfetch(SamplerAPLGlobalHistOffset10, binCoord);
            float meanLumaQ4 = APL_ReconstructMeanLumaQ4(localIndex, colorHistogram, pixelCount, offsetSum10);
            float sceneNits = max(meanLumaQ4 * 0.25, 0.0);
            float rawMetric = saturate(sceneNits / max(APLReferenceWhiteNits, 1.0));
            float3 candidateMetrics = rawMetric.xxx;

            if (EnableEOTFBoost)
            {
                float maxChannelNits = 0.0;
                if (colorHistogram)
                {
                    float maxSumNits = tex2Dfetch(SamplerAPLGlobalHistMaxNits, binCoord);
                    maxChannelNits = max((maxSumNits / pixelCount) * 2.0, sceneNits);
                }
                candidateMetrics.x = ComputePostBoostMetricForSample(sceneNits, maxChannelNits, LoadCandidateBoostParams(1));
                candidateMetrics.y = ComputePostBoostMetricForSample(sceneNits, maxChannelNits, LoadCandidateBoostParams(2));
                candidateMetrics.z = ComputePostBoostMetricForSample(sceneNits, maxChannelNits, LoadCandidateBoostParams(3));
            }

            weighted = float4(rawMetric, candidateMetrics) * pixelCount;
        }
    }

    APL_EvalWeightedMetrics[localIndex] = weighted;
    APL_EvalPixelCount[localIndex] = pixelCount;
    barrier();

    APL_REDUCE_EVAL_STAGE(128)
    APL_REDUCE_EVAL_STAGE(64)
    APL_REDUCE_EVAL_STAGE(32)
    APL_REDUCE_EVAL_STAGE(16)
    APL_REDUCE_EVAL_STAGE(8)
    APL_REDUCE_EVAL_STAGE(4)
    APL_REDUCE_EVAL_STAGE(2)
    APL_REDUCE_EVAL_STAGE(1)

    if (aplActive && localIndex == 0)
    {
        float totalPixelCount = APL_EvalPixelCount[0];
        if (totalPixelCount <= 0.0)
        {
            tex2Dstore(StorageAPLInstant, int2(0, 0), 0.0.xxxx);
            tex2Dstore(StorageAPLResponse, int2(0, 0), 0.0.xxxx);
        }
        else
        {
            float4 averages = APL_EvalWeightedMetrics[0] / totalPixelCount;
            float exactRollingMaxNits = float(tex2Dfetch(SamplerAPLGlobalHistCount, APL_GlobalExactMaxLumaCoord())) * 0.25;
            tex2Dstore(StorageAPLInstant, int2(0, 0), float4(averages.r, exactRollingMaxNits, 0.0, 1.0));
            tex2Dstore(StorageAPLResponse, int2(0, 0), float4(averages.gba, 1.0));
        }
    }
}

#undef APL_REDUCE_EVAL_STAGE

// --- SHADERS ---

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

    // Closed-loop display-side APL is solved in-frame from the scene's aggregate
    // post-boost response curve reconstructed from the full-frame histogram.
    // Rolloff and the never-darken clamp are evaluated per occupied bin;
    // color-preserving limiting uses each bin's average max-channel headroom.
    float currentDisplayAPL = SolveClosedLoopDisplayAPLFromResponse(rawAPL);
    float smoothedAPL = lerp(currentDisplayAPL, lerp(prevSmoothedAPL, currentDisplayAPL, alpha), hasPrev);

    // Precompute the scene-uniform log-gain once in this 1x1 pass so PS_MainPass
    // does not repeat the APL lookup and log2 chain for every pixel.
    float sceneLogGain = ComputeSceneLogGainFromAPL(smoothedAPL);

    // r/g = smoothed display-side APL used by the boost model and OSD
    // b   = history-valid flag
    // a   = precomputed scene log-gain (uniform across all pixels)
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


float3 DrawStatsOverlay(float2 texcoord, float3 sceneColor, float rawInputAPL, float outputAPL, float maxSceneNits)
{
    float aspect = ReShade::ScreenSize.x / ReShade::ScreenSize.y;
    float invAspect = 1.0 / max(aspect, 1e-6);

    // Smaller compact OSD than previous versions.
    // Row labels:
    //   RAW APL  = raw input APL from the rolling full-frame reconstruction
    //   OUT APL  = smoothed output/display-side APL used by the boost logic
    //   RAW NITS = exact input-scene maximum in the rolling full-frame reconstruction
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

    int nitDisplay = clamp(int(floor(max(maxSceneNits, 0.0) + 0.5)), 0, 99999);

    float bgMask = (texcoord.x >= left - padX && texcoord.x <= right + padX && texcoord.y >= top - padY && texcoord.y <= bottom + padY) ? 1.0 : 0.0;

    float rawLabelMask = DrawOSDLabel8(texcoord, inputLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_R, OSD_CHAR_A, OSD_CHAR_W, OSD_CHAR_SPACE, OSD_CHAR_A, OSD_CHAR_P, OSD_CHAR_L, OSD_CHAR_SPACE);
    float outputLabelMask = DrawOSDLabel8(texcoord, outputLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_O, OSD_CHAR_U, OSD_CHAR_T, OSD_CHAR_SPACE, OSD_CHAR_A, OSD_CHAR_P, OSD_CHAR_L, OSD_CHAR_SPACE);
    float nitsLabelMask = DrawOSDLabel8(texcoord, nitsLabelTopLeft, labelScale, labelStepX, aspect, OSD_CHAR_R, OSD_CHAR_A, OSD_CHAR_W, OSD_CHAR_SPACE, OSD_CHAR_N, OSD_CHAR_I, OSD_CHAR_T, OSD_CHAR_S);

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

// MAIN FULLSCREEN PASS: scene-uniform gain from the measured 1D APL model.

float4 PS_MainPass(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    bool boostEnabled = EnableEOTFBoost;

#if PQHDRLUT_ENABLE
    bool lutEnabled = EnablePQHDRLUT;
#endif

    // EOTF Boost OFF = clean passthrough or the original fixed LUT-only path.
    if (!boostEnabled)
    {
#if PQHDRLUT_ENABLE
        if (lutEnabled)
            color = PQHDRLUT_Apply(color);
#endif
#if ENABLE_APL_GRAPH
        if (ShowAPLGraph)
            color = DrawAPLGraphOverlay(texcoord, color);
#endif
        return float4(color, 1.0);
    }

    float4 aplData = tex2Dlod(SamplerAPL, float4(0.5, 0.5, 0.0, 0.0));
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

    // Avoid a full-screen PQ/scRGB decode-and-reencode round trip when the solved
    // gain is identity but LUT, OSD or graph processing still keeps this pass active.
    if (sceneLogGain > 1e-6)
    {
        float4 boostParams = tex2Dlod(SamplerBoostParams, float4(0.5, 0.5, 0.0, 0.0));
        finalColor = ApplyBoostPreserveColorFromPrecomputedParams(color, boostParams);
    }

#if PQHDRLUT_ENABLE
    // The optional fixed calibration LUT is applied after boost.
    if (lutEnabled)
        finalColor = PQHDRLUT_Apply(finalColor);
#endif

    if (ShowOSD)
    {
        float4 instantData = tex2Dlod(SamplerAPLInstant, float4(0.5, 0.5, 0.0, 0.0));
        float rawInputAPL = saturate(instantData.r);
        float currentMaxSceneNits = max(instantData.g, 0.0);
        finalColor = DrawStatsOverlay(
            texcoord,
            finalColor,
            rawInputAPL,
            saturate(aplData.g),
            currentMaxSceneNits
        );
    }

#if ENABLE_APL_GRAPH
    // Graph overlay is drawn directly in the main pass; the uniform branch skips it when hidden.
    // This removes the full-resolution TexBoosted intermediate and the extra fullscreen
    // Debug_Overlay pass that previously cost ~120 MB/frame of bandwidth even with the
    // graph toggled off. Drawn after the OSD to preserve the original overlay order.
    if (ShowAPLGraph)
        finalColor = DrawAPLGraphOverlay(texcoord, finalColor);
#endif

    return float4(finalColor, 1.0);
}

#if ENABLE_APL_GRAPH
// Live-equivalent fixed-APL operating point for the standard APL graph.
// Graph APL (%) is treated as a uniform representative scene and solved with the
// same candidate parameters and bracketed gain solver used by the live fallback.
// The plotted probe is intentionally excluded from this scene-level operating point,
// so sweeping the x-axis cannot redistribute APL or create graph-only feedback.
float ComputeAPLGraphLiveOperatingPoint(float rawAPLPercent)
{
    return SolveUniformSceneDisplayAPL(saturate(rawAPLPercent * 0.01));
}

float4 PS_CalcGraphParams(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!ShowAPLGraph)
        return 0.0.xxxx;

    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);
    float graphAxisMaxPQ = GraphUsePQSpace ? max(NitsToPQ(graphAxisMaxNits), 1e-6) : 0.0;

    if (GraphUseFullFieldWindowProjection)
    {
        return float4(0.0, 0.0, graphAxisMaxPQ, 0.0);
    }

    float graphRawAPLPercent = clamp(GraphAPLIndex, 0.0, 100.0);
    float graphClosedLoopAPL = ComputeAPLGraphLiveOperatingPoint(graphRawAPLPercent);
    float graphClosedLoopAPLPercent = graphClosedLoopAPL * 100.0;
    float maxMeasuredNits = GetAPLMaxMeasuredNits(graphClosedLoopAPLPercent);

    // r = solved closed-loop APL %, g = max measured nits, b = axis max PQ, a = unused.
    return float4(graphClosedLoopAPLPercent, maxMeasuredNits, graphAxisMaxPQ, 0.0);
}

// Apply the exact live per-pixel boost path at an already solved fixed-APL operating
// point, then project that remapped signal through the measured 2D APL x nits model.
// Returns: .r remapped signal nits, .g predicted measured output nits,
//          .b solved display-side APL, .a valid.
float4 ComputeAPLGraphLiveProjection(float displayAPL, float maxMeasuredNits, float inputNits)
{
    float safeInputNits = max(inputNits, 0.0);
    float remappedSignalNits = safeInputNits;

    if (EnableEOTFBoost && displayAPL > 1e-6)
    {
        float sceneLogGain = ComputeSceneLogGainFromAPL(displayAPL);
        float4 liveBoostParams = ComputeBoostRolloffParamsFromSceneLogGain(sceneLogGain);
        remappedSignalNits = ComputePostBoostNitsForSample(
            safeInputNits,
            safeInputNits,
            liveBoostParams
        );
    }

    float displayAPLPercent = displayAPL * 100.0;
    float predictedMeasuredOutputNits = SampleCorrectedOutputNitsForAPL(
        displayAPLPercent,
        remappedSignalNits,
        maxMeasuredNits
    );

    return float4(remappedSignalNits, predictedMeasuredOutputNits, displayAPL, 1.0);
}

// GRAPH PASS 1a: Live-equivalent projection, one thread per graph x sample.
// Heavy graph-only per-point math is isolated here instead of being duplicated for
// both ends of every segment in PS_CalcGraphCurves. The standard APL mode reuses the
// single fixed operating point solved by Graph_Params; window mode solves per x sample.
float4 PS_CalcGraphProjection(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!ShowAPLGraph)
        return 0.0.xxxx;

    int sampleIndex = min(int(vpos.x), GRAPH_CURVE_SAMPLES - 1);
    float t = float(sampleIndex) / float(GRAPH_CURVE_SAMPLES - 1);

    float graphAxisMaxNits = clamp(GraphAxisMaxNits, 1.0, 10000.0);
    float4 graphParams = tex2Dlod(SamplerGraphParams, float4(0.5, 0.5, 0.0, 0.0));
    float graphAxisMaxPQ = GraphUsePQSpace ? max(graphParams.b, 1e-6) : 0.0;
    float inputNits = GraphSampleNitsFromFraction(t, graphAxisMaxNits, graphAxisMaxPQ);
    float limitedInputNits = ApplyGraphInputSignalLimitNits(inputNits);

    if (GraphUseFullFieldWindowProjection)
        return ComputeFullFieldLiveProjectionByMode(GraphProjectionWindowSize, limitedInputNits);

    float graphClosedLoopAPL = saturate(graphParams.r * 0.01);
    float graphMaxMeasuredNits = max(graphParams.g, 0.0);
    return ComputeAPLGraphLiveProjection(graphClosedLoopAPL, graphMaxMeasuredNits, limitedInputNits);
}

// GRAPH PASS 1b: Precompute grid/tick/reference endpoints in a tiny 32-pixel pass.
// Eliminates ~200 NitsToPQ/pow calls per inGraphCore pixel in the fullscreen draw pass.
float4 PS_CalcGraphLines(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    if (!ShowAPLGraph)
        return float4(-1.0, -1.0, -1.0, -1.0);

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

// GRAPH PASS 2: Precompute curve segment endpoints in a 64 x 4 graph-only pass.
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

    if (!ShowAPLGraph)
        return SENTINEL;

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

    // Projection values are computed once per x sample in the dedicated 64x1 pass.
    // Adjacent segments share those samples instead of running the full response
    // solve twice per endpoint here.
    static const float invGraphSamples = 1.0 / float(GRAPH_CURVE_SAMPLES);
    float4 graphProjection0 = 0.0.xxxx;
    float4 graphProjection1 = 0.0.xxxx;

    if (row <= GCURVE_CORRECTED)
    {
        graphProjection0 = tex2Dlod(
            SamplerGraphProjection,
            float4((float(s) + 0.5) * invGraphSamples, 0.5, 0.0, 0.0)
        );
        graphProjection1 = tex2Dlod(
            SamplerGraphProjection,
            float4((float(s + 1) + 0.5) * invGraphSamples, 0.5, 0.0, 0.0)
        );
    }

    if (row == GCURVE_REMAPPED)
    {
        // Green curve: exact signal produced by the selected graph model.
        float y0 = graphProjection0.r;
        float y1 = graphProjection1.r;

        if (graphProjection0.a > 0.0 && graphProjection1.a > 0.0)
        {
            float2 a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0);
            float2 b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1);
            result = float4(a, b);
        }
    }
    else if (row == GCURVE_CORRECTED)
    {
        // Gray curve: predicted measured output from the live-equivalent projection pass.
        float y0;
        float y1;

        y0 = graphProjection0.g;
        y1 = graphProjection1.g;

        if (graphProjection0.a > 0.0 && graphProjection1.a > 0.0)
        {
            float2 a = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x0, y0);
            float2 b = ToGraphPointWithPQMax(graphPos, graphSize, graphAxisMaxNits, graphAxisMaxPQ, x1, y1);
            result = float4(a, b);
        }
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
    // Three-pixel uniform setup for the response-curve candidate parameter sets.
    pass Candidate_Boost_Params
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcCandidateBoostParams;
        RenderTarget = TexCandidateBoostParams;
    }

    // 129 exact PQ nodes; hardware filtering turns them into the fast full-frame
    // decoder without dynamic-array indexing or FXC unroll failures.
    pass APL_Fast_PQ_LUT
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BuildAPLFastPQLUT;
        RenderTarget = TexAPLFastPQNits;
    }

    // Clear all row histograms or only the current rolling phase before rebuilding.
    pass APL_FullFrame_RowHistogramClear
    {
        ComputeShader = CS_ClearAPLFullFrameRowHistograms<256, 1, 1>;
        DispatchSizeX = 1;
        DispatchSizeY = 1;
        DispatchSizeZ = 1;
        GenerateMipMaps = false;
    }

    // Packed row-histogram build. Rolling mode maps the compact Y dispatch onto
    // one interleaved 32-pixel-high row phase.
    pass APL_FullFrame_RowHistogramBuild
    {
        ComputeShader = CS_BuildAPLFullFrameRowHistogram<16, 16, 1>;
        DispatchSizeX = APL_FFH_TILE_COUNT_X;
        DispatchSizeY = APL_FFH_ROLLING_ROW_SLOTS;
        DispatchSizeZ = 1;
        GenerateMipMaps = false;
    }

    // Rebuild the global histogram every rendered frame from the latest valid
    // persistent row histograms, including newly refreshed rolling rows.
    pass APL_FullFrame_RowHistogramReduce
    {
        ComputeShader = CS_ReduceAPLFullFrameRowHistogram<64, 1, 1>;
        DispatchSizeX = APL_FFH_BLOCK_W;
        DispatchSizeY = APL_FFH_BLOCK_H;
        DispatchSizeZ = 1;
        GenerateMipMaps = false;
    }

    // One 256-thread group evaluates and reduces all bins into the existing 1x1 APL state.
    pass APL_FullFrame_ResponseEvaluate
    {
        ComputeShader = CS_EvaluateAPLFullFrameHistogram<256, 1, 1>;
        DispatchSizeX = 1;
        DispatchSizeY = 1;
        DispatchSizeZ = 1;
        GenerateMipMaps = false;
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

#if ENABLE_APL_GRAPH
    // Graph-only precompute passes use uniforms, candidate parameters and their own
    // small intermediate textures. They have no dependency on a full-resolution
    // boosted image, so Main_Boost can draw the overlay directly.
    pass Graph_Params
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphParams;
        RenderTarget = TexGraphParams;
    }

    pass Graph_Projection
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_CalcGraphProjection;
        RenderTarget = TexGraphProjection;
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
