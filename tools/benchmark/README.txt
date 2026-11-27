QD-OLED EOTF Boost GPU Shader Benchmark
========================================

Tests whatever shader is placed in `Shaders\EOTF_Boost_X36.fx` across the
8 combinations of:
  - EnableEOTFBoost:               On / Off
  - HDRSaturationPercent:          Off (100) / On (115)
  - EnableHighlightDetailRecovery: On / Off

Waits automatically for ReShade to finish compiling the shader before starting
measurement, then records exact D3D11 hardware GPU timestamps (median, mean,
p95, min, max in ms) to `results.csv` and prints a summary table.

Quick Start
-----------
1. Copy 64-bit ReShade (FULL ADD-ON SUPPORT build from https://reshade.me/)
   into this folder renamed as `dxgi.dll`.
2. (Optional) Double-click `BenchmarkHost.exe` to open the interactive 120 FPS
   window, press `Home` to verify/configure ReShade, then close the window.
3. Double-click `run_bench.bat` to benchmark `Shaders\EOTF_Boost_X36.fx` in
   HDR10 PQ at 3440x1440 @ 120 FPS (make sure Windows HDR is enabled).

Optional CLI Arguments for run_bench.bat
----------------------------------------
  run_bench.bat [mode] [width] [height] [fps]

Examples:
  run_bench.bat pq 3440 1440 120
  run_bench.bat scrgb 3440 1440 120
  run_bench.bat sdr 3440 1440 120
  run_bench.bat pq 3840 2160 120

Testing Another Shader Version
------------------------------
Just replace `Shaders\EOTF_Boost_X36.fx` with any version you want to test
(e.g. `main` or a test branch) and run `run_bench.bat` again!
