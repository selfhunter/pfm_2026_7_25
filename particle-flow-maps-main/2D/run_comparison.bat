@echo off
REM ============================================================
REM PFM Optimization Comparison Runner
REM Run this in your terminal (cmd or Git Bash):
REM   run_comparison.bat
REM Or step by step:
REM   1. run_comparison.bat baseline
REM   2. run_comparison.bat adaptive
REM   3. run_comparison.bat plot
REM ============================================================

set PYTHON=C:\Users\go\.conda\envs\pfm\python.exe
set DIR=e:\项目\pfm_2026_7_25优化\particle-flow-maps-main\2D

cd /d "%DIR%"

if "%1"=="baseline" goto baseline
if "%1"=="adaptive" goto adaptive
if "%1"=="plot" goto plot
if "%1"=="all" goto all
goto usage

:baseline
echo ============================================================
echo Running BASELINE PFM (200 steps)...
echo ============================================================
%PYTHON% run_2D_baseline.py --steps 200
goto end

:adaptive
echo ============================================================
echo Running ADAPTIVE PFM (200 steps)...
echo ============================================================
%PYTHON% run_2D_adaptive.py --steps 200
goto end

:plot
echo ============================================================
echo Generating comparison charts...
echo ============================================================
%PYTHON% plot_comparison.py
goto end

:all
echo ============================================================
echo Step 1/3: Running BASELINE PFM...
echo ============================================================
%PYTHON% run_2D_baseline.py --steps 200
if errorlevel 1 goto error

echo ============================================================
echo Step 2/3: Running ADAPTIVE PFM...
echo ============================================================
%PYTHON% run_2D_adaptive.py --steps 200
if errorlevel 1 goto error

echo ============================================================
echo Step 3/3: Generating comparison charts...
echo ============================================================
%PYTHON% plot_comparison.py
goto end

:usage
echo Usage:
echo   run_comparison.bat baseline   - Run baseline simulation
echo   run_comparison.bat adaptive   - Run adaptive simulation
echo   run_comparison.bat plot       - Generate comparison charts
echo   run_comparison.bat all        - Run all three steps
goto end

:error
echo ERROR: Simulation failed!
pause
exit /b 1

:end
echo Done. Charts in: %DIR%\comparison_plots\
pause
