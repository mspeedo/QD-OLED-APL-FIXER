# EOTF Boost v9.0 for the MSI MPG 341CQR QD-OLED X36

This ReShade shader compensates for the X36’s APL-dependent HDR dimming. It estimates average picture level (APL) with a rolling full-frame histogram and uses the monitor’s measured brightness response to set the boost. Highlight roll-off fades the added gain to zero at the target luminance (1300 nits by default).

Enable `EOTF_Boost_1D_APL_LUT` in ReShade. Under **APL Input Mode**, select **PQ Decoded Normalized** for HDR10 (default) or **scRGB Normalized** for scRGB. [Download the X36 shader](Shaders/EOTF_Boost_X36.fx).

**HDR Saturation (%)** adjusts color after the boost while preserving signal luminance. The default **100%** bypasses the adjustment. **0%** gives grayscale, and values up to **150%** increase saturation, tapering as individual colors approach channel limits. To apply a calibration LUT after saturation, set `PQHDRLUT_ENABLE=1` and supply `PQHDRLUT.cube` (65³ by default).

## Measured PQ EOTF

X36 in Peak 1300 nits mode with default shader settings and a 100% window.

<img width="914" height="739" alt="Measured PQ EOTF tracking on the X36 with a 100% window" src="https://github.com/user-attachments/assets/250e1214-2ff3-4aea-88d1-1a9f51f97249" />
