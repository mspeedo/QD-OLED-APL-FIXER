QD-OLED EOTF Boost Automated GPU Benchmark Suite
================================================

What this does
--------------
Runs deterministic GPU timestamp benchmarks (D3D11 hardware timestamp queries)
against BOTH `shaders_main\EOTF_Boost_X36.fx` (production main) and
`shaders_new\EOTF_Boost_X36.fx` (refactored test branch) across:
  - HDR10 PQ  (BUFFER_COLOR_SPACE = 3, 13 combinations)
  - HDR scRGB (BUFFER_COLOR_SPACE = 2,  3 combinations)
  - SDR sRGB  (BUFFER_COLOR_SPACE = 1,  4 combinations)

Then prints and saves a side-by-side `comparison_report.md` (median ms, p95 ms,
delta ms, delta %, and compile status for every case).

Quick Start (3 steps)
---------------------
1. Download 64-bit ReShade WITH FULL ADD-ON SUPPORT from https://reshade.me/
   Extract or install it and copy its 64-bit `ReShade64.dll` into THIS folder
   renamed as:
       dxgi.dll
2. Make sure Windows HDR is ON (Settings -> System -> Display -> Use HDR) so
   HDR10 PQ and scRGB swapchains advertise real HDR color spaces.
3. Double-click `run_compare.bat` (default 3440x1440), or pass a custom
   resolution from CMD:
       run_compare.bat 3840 2160

Customizing / Comparing Other Shader Versions
---------------------------------------------
`ShaderBench.addon64` is 100% generic (works with ANY version of the shader
without recompiling). To compare any two shader versions:
  - Replace `shaders_main\EOTF_Boost_X36.fx` and `shaders_new\EOTF_Boost_X36.fx`
  - Edit `bench_cases_pq.txt`, `bench_cases_scrgb.txt`, or `bench_cases_sdr.txt`
  - Run `run_compare.bat` again!

Notes
-----
- You can also drop `ShaderBench.addon64` + a `bench_config.txt` into any D3D11
  or D3D12 game folder with ReShade (full add-on support) to benchmark in-game.
- Press ESC inside the test window at any time to abort.
- If a case fails to compile, the add-on logs `COMPILE_FAIL` cleanly and
  continues to the next case. Check `ReShade.log` for exact compiler diagnostics.
