@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist "dxgi.dll" (
    echo [ERROR] ReShade dxgi.dll not found in this folder!
    echo Please copy 64-bit ReShade dxgi.dll ^(FULL ADD-ON SUPPORT build from reshade.me^)
    echo into: %~dp0
    echo Then re-run run_compare.bat.
    pause
    exit /b 1
)

if not exist "shaders_main\EOTF_Boost_X36.fx" (
    echo [ERROR] shaders_main\EOTF_Boost_X36.fx not found!
    pause
    exit /b 1
)

if not exist "shaders_new\EOTF_Boost_X36.fx" (
    echo [ERROR] shaders_new\EOTF_Boost_X36.fx not found!
    pause
    exit /b 1
)

set WIDTH=3440
set HEIGHT=1440
if not "%~1"=="" set WIDTH=%~1
if not "%~2"=="" set HEIGHT=%~2

echo ===================================================================
echo  QD-OLED EOTF Boost Automated GPU Benchmark ^(main vs refactored^)
echo  Resolution: %WIDTH%x%HEIGHT%
echo  NOTE: For accurate HDR PQ/scRGB runs, enable Windows HDR first.
echo        Press ESC in the test window at any time to abort.
echo ===================================================================
echo.

mkdir out_main 2>nul
mkdir out_new 2>nul

call :RunSuite "main" "shaders_main\EOTF_Boost_X36.fx" "out_main"
if errorlevel 1 goto :fail

call :RunSuite "new" "shaders_new\EOTF_Boost_X36.fx" "out_new"
if errorlevel 1 goto :fail

powershell -ExecutionPolicy Bypass -File compare_results.ps1 -MainDir ".\out_main" -NewDir ".\out_new" -ReportPath ".\comparison_report.md"
echo.
echo Done! Full report saved to comparison_report.md
pause
exit /b 0

:RunSuite
set LABEL=%~1
set SRC_FX=%~2
set OUT_DIR=%~3

echo -------------------------------------------------------------------
echo  Benchmarking: %LABEL%  ^(%SRC_FX%^)
echo -------------------------------------------------------------------
copy /Y "%SRC_FX%" "Shaders\EOTF_Boost_X36.fx" >nul

for %%M in (pq scrgb sdr) do (
    echo   [%LABEL%] Running mode=%%M ...
    copy /Y "bench_cases_%%M.txt" "bench_config.txt" >nul
    del /Q "results_%%M.csv" 2>nul
    del /Q "bench_done.flag" 2>nul
    del /Q "BenchPreset.ini" 2>nul
    BenchmarkHost.exe --bench --mode %%M --width %WIDTH% --height %HEIGHT% --done-file bench_done.flag --timeout 240
    if errorlevel 1 (
        echo   [ERROR] BenchmarkHost exited with error during %LABEL% mode=%%M.
        exit /b 1
    )
    if exist "results_%%M.csv" (
        move /Y "results_%%M.csv" "%OUT_DIR%\results_%%M.csv" >nul
    )
)
exit /b 0

:fail
echo.
echo Benchmark aborted or failed. Check ReShade.log for details.
pause
exit /b 1
