@echo off
setlocal

cd /d "%~dp0"

if not exist "dxgi.dll" (
    echo [ERROR] ReShade dxgi.dll not found in this folder!
    echo Please copy 64-bit ReShade dxgi.dll ^(FULL ADD-ON SUPPORT build from reshade.me^)
    echo into: %~dp0
    echo Then re-run run_bench.bat.
    pause
    exit /b 1
)

if not exist "Shaders\EOTF_Boost_X36.fx" (
    echo [ERROR] Shaders\EOTF_Boost_X36.fx not found!
    pause
    exit /b 1
)

set MODE=pq
set WIDTH=3440
set HEIGHT=1440
set FPS=120

if not "%~1"=="" set MODE=%~1
if not "%~2"=="" set WIDTH=%~2
if not "%~3"=="" set HEIGHT=%~3
if not "%~4"=="" set FPS=%~4

echo ===================================================================
echo  QD-OLED EOTF Boost Shader Benchmark
echo  Shader:     Shaders\EOTF_Boost_X36.fx
echo  Mode:       %MODE%  ^(pq ^| scrgb ^| sdr^)
echo  Resolution: %WIDTH%x%HEIGHT% @ %FPS% FPS
echo  Press ESC in the test window at any time to abort.
echo ===================================================================
echo.

del /Q "results.csv" 2>nul
del /Q "bench_done.flag" 2>nul

BenchmarkHost.exe --bench --mode %MODE% --width %WIDTH% --height %HEIGHT% --fps %FPS% --done-file bench_done.flag --timeout 240
if errorlevel 1 (
    echo.
    echo [ERROR] Benchmark aborted or failed. Check ReShade.log for details.
    pause
    exit /b 1
)

echo.
if exist "results.csv" (
    echo ====================== RESULTS ^(results.csv^) ======================
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Import-Csv .\results.csv | Format-Table case,status,color_space,timer,samples,median_ms,mean_ms,p95_ms -AutoSize"
    echo ===================================================================
)
pause
exit /b 0
