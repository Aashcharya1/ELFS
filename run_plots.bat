@echo off
setlocal
REM Generate figures from training logs, cluster metrics, TensorBoard, data-score pickles, and embedding k-NN JSON.
REM Run from repo root (this script lives there). Edit defaults via: python scripts\run_all_plots.py --help

cd /d "%~dp0"

python scripts\run_all_plots.py %*
set "RC=%ERRORLEVEL%"

if %RC% NEQ 0 (
    echo.
    echo run_all_plots exited with code %RC%.
    exit /b %RC%
)

echo.
echo Figures are under: data-model\CIFAR10\figures and task-specific figures subfolders (see script output^).
exit /b 0
