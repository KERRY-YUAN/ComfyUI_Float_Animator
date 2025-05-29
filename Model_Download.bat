@echo off
setlocal

:: Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"

:: Navigate to the script's directory
cd "%SCRIPT_DIR%"

:: Find the ComfyUI root directory by looking for a parent folder named "ComfyUI"
set "COMFYUI_ROOT="
for %%I in ("%SCRIPT_DIR%") do (
    set "CURRENT_DIR=%%~fI"
    :loop_parent_dir
    if not "%CURRENT_DIR%"=="" (
        if exist "%CURRENT_DIR%\ComfyUI\" (
            set "COMFYUI_ROOT=%%~dpCURRENT_DIR%%\ComfyUI\"
            goto :found_comfyui_root
        )
        for /f "delims=" %%J in ("%CURRENT_DIR%\..") do set "CURRENT_DIR=%%~fJ"
        if "%CURRENT_DIR%" == "%SCRIPT_DIR%" goto :end_loop_parent_dir_fallback_current_dir
        if "%CURRENT_DIR%" == "%CD%\" goto :end_loop_parent_dir_fallback_current_dir
        goto :loop_parent_dir
    )
)
:end_loop_parent_dir_fallback_current_dir
:: If ComfyUI root not found, assume script is directly in ComfyUI or similar setup.
:: This is a fallback and might not be reliable for all setups.
set "COMFYUI_ROOT=%CD%\"

:found_comfyui_root
if not defined COMFYUI_ROOT (
    echo Error: Could not find ComfyUI root directory. Please ensure this script is inside custom_nodes/ComfyUI_Float_Animator.
    pause
    exit /b 1
)

echo Detected ComfyUI Root: "%COMFYUI_ROOT%"

:: Define the path to ComfyUI's embedded Python executable (Windows specific)
set "PYTHON_EXE=%COMFYUI_ROOT%python_embeded\python.exe"

:: Check if the embedded Python exists
if not exist "%PYTHON_EXE%" (
    echo Warning: ComfyUI embedded Python not found at "%PYTHON_EXE%".
    echo Attempting to use system Python. This might lead to package conflicts.
    set "PYTHON_EXE=python.exe"
    where %PYTHON_EXE% >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: python.exe not found in PATH. Please install Python or set the path to your Python interpreter.
        pause
        exit /b 1
    )
)

echo Using Python: "%PYTHON_EXE%"

:: Run the model_download.py script
echo Starting model download...
"%PYTHON_EXE%" "%SCRIPT_DIR%model_download\model_download.py"

if %errorlevel% neq 0 (
    echo Model download failed. Please check the output above for errors.
    pause
    exit /b 1
)

echo Model download completed successfully.
echo You can now restart ComfyUI to load the models.
pause
endlocal