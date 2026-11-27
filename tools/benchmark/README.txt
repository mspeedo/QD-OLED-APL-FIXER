QD-OLED EOTF Boost GPU Shader Benchmark
========================================

Tests whatever shader is placed in `Shaders\EOTF_Boost_X36.fx` across the
8 combinations of:
  - EnableEOTFBoost:               On / Off
  - HDRSaturationPercent:          Off (100) / On (115)
  - EnableHighlightDetailRecovery: On / Off

Features
--------
- Auto-calibrated realistic GPU load (`--load-ms 5` default in bench mode):
  Runs offscreen ALU + full-resolution RGBA16F VRAM bandwidth passes before
  the scene to lock the GPU into full P0 boost clocks and exercise L2/VRAM,
  while `ShaderBench.addon64` timestamps ONLY ReShade effect execution.
- Waits automatically for ReShade to finish compiling the shader before
  measuring.
- Outputs `results.csv` using your Windows locale list & decimal separators.

Quick Start
-----------
1. Copy 64-bit ReShade (FULL ADD-ON SUPPORT build from https://reshade.me/)
   into this folder renamed as `dxgi.dll`.
2. (Optional) Double-click `BenchmarkHost.exe` to open the interactive 120 FPS
   window (GPU load off by default), press `Home` to configure ReShade, then
   close the window.
3. Double-click `run_bench.bat` to benchmark `Shaders\EOTF_Boost_X36.fx` in
   HDR10 PQ at 3440x1440 @ 120 FPS with ~5 ms/frame GPU load.

Optional CLI Arguments for run_bench.bat
----------------------------------------
  run_bench.bat [mode] [width] [height] [fps] [load_ms]

Examples:
  run_bench.bat pq 3440 1440 120 5
  run_bench.bat scrgb 3440 1440 120 5
  run_bench.bat sdr 3440 1440 120 5
  run_bench.bat pq 3840 2160 120 6
  run_bench.bat pq 3440 1440 120 0     (disable synthetic GPU load)

Testing Another Shader Version
------------------------------
Just replace `Shaders\EOTF_Boost_X36.fx` with any version you want to test
(e.g. `main` or a test branch) and run `run_bench.bat` again!
