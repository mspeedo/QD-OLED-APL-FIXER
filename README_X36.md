# EOTF Boost v9.4 for the MSI MPG 341CQR QD-OLED X36

This ReShade shader compensates for the X36’s APL-dependent HDR dimming. It estimates average picture level (APL) with a rolling full-frame histogram and uses the monitor’s measured brightness response to set the boost. Highlight roll-off fades the added gain to zero at the target luminance (1300 nits by default).

By default the shader measures one histogram phase every effect frame. At a capped measurement rate, it continues smoothing the boost in log-gain (stops) space toward the latest solved APL on every effect frame.

Enable `EOTF_Boost_1D_APL_LUT` in ReShade. Under **APL Input Mode**, select **PQ Decoded Normalized** for HDR10 (default) or **scRGB Normalized** for scRGB. [Download the X36 shader](Shaders/EOTF_Boost_X36.fx).

## Boost and color

| ReShade option | Default | What it does |
| --- | --- | --- |
| Enable EOTF Boost | On | Applies the X36 brightness compensation. You can turn it off and still use saturation and the calibration LUT. |
| APL Input Mode | PQ Decoded Normalized | Select **PQ Decoded Normalized** for HDR10 PQ / Rec.2020 or **scRGB Normalized** for linear scRGB / Rec.709. This controls how the shader reads the signal for APL, boost and saturation. |
| APL Reference White (nits) | 1300 | Defines 100% APL. At 1300 nits, a frame averaging 130 nits has 10% raw APL. |
| Compensation Freeze APL % | 50% | Holds the measured compensation factor at the selected APL for all higher solved APL values. Set to 0 to disable the freeze. |
| Global APL Boost Strength | 0.5 | Scales the measured correction. 0 gives unity gain, 1 applies the full correction, and 2 applies the measured gain squared before roll-off. |
| Boost Roll-Off Target (nits) | 1300 | Fades added boost to zero at the target so brighter source highlights keep their luminance. Set to 0 to disable roll-off. |
| Boost Roll-Off Shape | 1.25 | Higher values start easing off the boost earlier. Lower values keep the full boost farther into the highlights. |
| APL Measurement Rate (Hz) | 0 | Targets 1–240 histogram phase updates per second, up to one per effect execution. Set to 0 for the v9.1 every-execution cadence. At 60 Hz the default six phases cover the screen in about 100 ms. At 120 Hz they take about 50 ms if the effect runs that often. |
| Boost Smoothing Time (s) | 0.15 | Exponential smoothing time constant in log-gain (stops) space, updated every effect frame from the wall-clock timer (63% of the stop gap in one time constant). The first measurement initializes it directly. Set to 0 to apply each measured target immediately. |
| Drop Smoothing Scale | 0.65 | Scales **Boost Smoothing Time (s)** when boost is decreasing (bright-to-dark transitions). `1.0` gives symmetric rise and fall; lower values release boost faster to prevent dark-scene flash when OLED ABL releases. |
| Preserve Color by Reducing Boost | Off | Reduces brightness gain near a channel limit so saturated highlights can retain their RGB ratios. |
| HDR Saturation (%) | 100% | Adjusts color after boost and before the LUT while preserving signal luminance. 0% gives grayscale, 100% bypasses the adjustment, and increases up to 150% taper near channel limits. |
| scRGB Signal Reference (nits) | 80 | Sets the luminance represented by a linear scRGB value of 1.0 for APL, boost and saturation. The optional calibration LUT uses the standard 80-nit scRGB conversion. |
| Show APL Stats | Off | With boost enabled, displays raw input APL, estimated output APL and Max Raw Nits. Max Raw Nits is the highest input pixel luminance before boost across the rows in the rolling histogram. |
| OSD Brightness | 0.5 | Adjusts the brightness of the stats overlay and debug graph. |

## Optional calibration LUT

Set `PQHDRLUT_ENABLE=1` to expose these controls. Place your Rec.2020 PQ calibration file, `PQHDRLUT.cube`, in a ReShade texture search path.

| ReShade option | Default | What it does |
| --- | --- | --- |
| Enable PQ HDR LUT | On | Applies the calibration LUT after brightness and saturation adjustments. |
| LUT Input Color Space | Auto | Auto reads ReShade’s reported buffer color space and falls back to HDR10 PQ. Select **HDR10 PQ / Rec.2020** or **scRGB / linear Rec.709** to override detection for the LUT. |

## Optional per-APL boost strength

Set `PER_APL_BOOST_STRENGTH_ENABLE=1` to expose **Enable Per-APL Boost Strength** (default **On**). When enabled, each slider sets the boost strength at the named APL. The shader interpolates between adjacent settings and uses the nearest endpoint outside the 3–50% range. Turn the option off to return to **Global APL Boost Strength**.

| ReShade option | Default | ReShade option | Default |
| --- | --- | --- | --- |
| APL 3% Boost Strength | 0.40 | APL 18% Boost Strength | 0.54 |
| APL 5% Boost Strength | 0.80 | APL 22% Boost Strength | 0.50 |
| APL 7% Boost Strength | 0.90 | APL 25% Boost Strength | 0.50 |
| APL 10% Boost Strength | 0.80 | APL 35% Boost Strength | 0.50 |
| APL 14% Boost Strength | 0.65 | APL 50% Boost Strength | 0.50 |

## Optional EOTF debug graph

Set `ENABLE_APL_GRAPH=1` to expose these controls. They affect the diagnostic overlay only.

| ReShade option | Default | What it does |
| --- | --- | --- |
| Show APL EOTF Debug Graph | On | Plots the reference, measured panel response, remapped signal and predicted output curves. |
| Graph Show BT.2390-Style Reference | Off | Adds a BT.2390-style roll-off curve for comparison. |
| Graph Use Window Projection | On | Uses measured test-window data to predict the response across input levels. Turn off to examine one fixed raw APL. |
| Graph Projection Window Size | 100% Window | Selects the measured window used for projection. Choices are 100%, 50%, 25%, 15% and 10% of the screen. |
| Graph APL (%) | 50% | Sets the raw APL for the fixed-APL graph when window projection is off. |
| Graph Input Signal Limit (nits) | 0 | Caps the input signal used in graph calculations. Set to 0 to remove the cap. |
| Graph Axis Max (nits) | 1300 | Sets the upper luminance shown on both axes. |
| Graph Opacity | 0.5 | Controls how strongly the graph is drawn over the image. |
| Graph PQ-Encoded Axes | On | Uses PQ spacing to give darker values more room on the graph. Turn off for linear spacing. Labels stay in nits. |

## ReShade preprocessor definitions

| Definition | Default | What it does |
| --- | --- | --- |
| `PQHDRLUT_ENABLE` | 0 | Set to 1 to compile calibration LUT support. |
| `PQHDRLUT_SIZE` | 65 | Sets the LUT grid size per axis when LUT support is enabled. Match this to the file’s `LUT_3D_SIZE`. |
| `PER_APL_BOOST_STRENGTH_ENABLE` | 0 | Set to 1 to expose the ten per-APL strength sliders. |
| `ENABLE_APL_GRAPH` | 0 | Set to 1 to compile the debug graph and its rendering passes. |
| `ENABLE_UI_TOOLTIPS` | 0 | Set to 1 to show descriptions when hovering over controls. |
| `APL_FULLFRAME_ROLLING_PHASES` | 6 | Refreshes one of six interleaved groups of image rows per APL measurement. Set to 0 to refresh the entire histogram at every measurement. More phases spread the work over more executions. |

## Measured PQ EOTF

X36 in Peak 1300 nits mode with default shader settings and a 100% window.

<img width="914" height="739" alt="Measured PQ EOTF tracking on the X36 with a 100% window" src="https://github.com/user-attachments/assets/250e1214-2ff3-4aea-88d1-1a9f51f97249" />
