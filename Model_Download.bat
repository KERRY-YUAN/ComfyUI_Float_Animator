@echo off
setlocal enableDelayedExpansion

:: Change console code page to UTF-8 for proper Chinese display
chcp 65001 >nul

:: Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"

:: Navigate to the script's directory
cd "%SCRIPT_DIR%"

echo.
echo ====================================================================
echo Node Data Download Script / 节点支持数据下载脚本
echo ====================================================================
echo.

echo 待下载的模型列表 (model_list.json):
echo Models to be downloaded (model_list.json):
echo --------------------------------------------------------------------
type "%SCRIPT_DIR%model_download\model_list.json"
echo --------------------------------------------------------------------
echo.

:: --- Step 1: Find ComfyUI Root Directory ---
set "COMFYUI_ROOT="
set "CURRENT_SEARCH_DIR=%SCRIPT_DIR%"
set "MAX_SEARCH_DEPTH=6"
set "SEARCH_DEPTH=0"

:find_comfyui_root_loop
if !SEARCH_DEPTH! geq !MAX_SEARCH_DEPTH! goto :end_find_comfyui_root_loop

:: Check if current directory is ComfyUI root (has main.py or models/ and custom_nodes/)
if exist "!CURRENT_SEARCH_DIR!\main.py" (
    set "COMFYUI_ROOT=!CURRENT_SEARCH_DIR!"
    goto :found_comfyui_root
)
if exist "!CURRENT_SEARCH_DIR!\models\" if exist "!CURRENT_SEARCH_DIR!\custom_nodes\" (
    set "COMFYUI_ROOT=!CURRENT_SEARCH_DIR!"
    goto :found_comfyui_root
)

:: Move up one directory
for /f "delims=" %%I in ("!CURRENT_SEARCH_DIR!\..") do set "PARENT_DIR=%%~fI"
if "!PARENT_DIR!" == "!CURRENT_SEARCH_DIR!" goto :end_find_comfyui_root_loop :: Prevent infinite loop at root
set "CURRENT_SEARCH_DIR=!PARENT_DIR!"
set /a SEARCH_DEPTH+=1
goto :find_comfyui_root_loop

:end_find_comfyui_root_loop
:: Fallback if ComfyUI root not found in specified depth
if not defined COMFYUI_ROOT (
    echo 警告: 未能通过向上溯源找到 ComfyUI 根目录。脚本可能无法找到正确的 Python 环境。
    echo Warning: Could not find ComfyUI root by tracing up. Script might not find the correct Python environment.
    :: For now, proceed, but Python search might be less accurate.
)

:found_comfyui_root
if defined COMFYUI_ROOT (
    echo 检测到 ComfyUI 根目录: "%COMFYUI_ROOT%"
    echo Detected ComfyUI Root: "%COMFYUI_ROOT%"
) else (
    echo 无法确定 ComfyUI 根目录。Python 路径查找将从节点目录开始。
    echo Unable to determine ComfyUI root. Python path search will start from node directory.
    set "COMFYUI_ROOT=%SCRIPT_DIR%"
)
echo.

:: --- Step 2: Find Python Executable Based on Priority ---
set "PYTHON_EXE="
set "COMFYUI_PARENT_DIR="
if defined COMFYUI_ROOT for /f "delims=" %%I in ("%COMFYUI_ROOT%\..") do set "COMFYUI_PARENT_DIR=%%~fI"

:: Priority 1: .venv inside ComfyUI root
if exist "%COMFYUI_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT%\.venv\Scripts\python.exe"
    goto :found_python
)

:: Priority 2: python_embeded or python folder inside ComfyUI root
if exist "%COMFYUI_ROOT%\python_embeded\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT%\python_embeded\python.exe"
    goto :found_python
)
if exist "%COMFYUI_ROOT%\python\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT%\python\python.exe"
    goto :found_python
)

:: Priority 3: python_embeded or python folder in ComfyUI's parent directory
if defined COMFYUI_PARENT_DIR (
    if exist "%COMFYUI_PARENT_DIR%\python_embeded\python.exe" (
        set "PYTHON_EXE=%COMFYUI_PARENT_DIR%\python_embeded\python.exe"
        goto :found_python
    )
    if exist "%COMFYUI_PARENT_DIR%\python\python.exe" (
        set "PYTHON_EXE=%COMFYUI_PARENT_DIR%\python\python.exe"
        goto :found_python
    )
)

:: Priority 4: System-wide Python in PATH
where python.exe >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python.exe"
    goto :found_python
)

:: --- Fallback: Prompt user for Python path ---
:prompt_for_python_path
echo 错误: 未能在自动查找路径中找到 Python 可执行文件。
echo Error: Python executable not found in automatic search paths.
echo.
echo 请手动输入您的 ComfyUI Python 环境路径 (例如: D:\Program\ComfyUI_Program\ComfyUI\.venv):
echo Please manually enter your ComfyUI Python environment path (e.g., D:\Program\ComfyUI_Program\ComfyUI\.venv):
set "USER_PYTHON_ENV_PATH="
set /p "USER_PYTHON_ENV_PATH="

:: Check if user input is empty
if "!USER_PYTHON_ENV_PATH!"=="" (
    echo.
    echo 警告: 未输入路径。请重新输入。
    echo Warning: No path entered. Please try again.
    echo.
    goto :prompt_for_python_path
)

:: Construct the full python.exe path based on user input
set "TEMP_PYTHON_EXE_USER="
if exist "!USER_PYTHON_ENV_PATH!\Scripts\python.exe" (
    set "TEMP_PYTHON_EXE_USER=!USER_PYTHON_ENV_PATH!\Scripts\python.exe"
) else if exist "!USER_PYTHON_ENV_PATH!\python.exe" (
    set "TEMP_PYTHON_EXE_USER=!USER_PYTHON_ENV_PATH!\python.exe"
)

if not defined TEMP_PYTHON_EXE_USER (
    echo.
    echo 无效路径: "%USER_PYTHON_ENV_PATH%"。未能找到 python.exe。请重试。
    echo Invalid path: "%USER_PYTHON_ENV_PATH%". python.exe not found. Please try again.
    echo.
    goto :prompt_for_python_path
) else (
    set "PYTHON_EXE=!TEMP_PYTHON_EXE_USER!"
    echo.
    echo 已接受的用户提供的 Python 路径: "%PYTHON_EXE%"
    echo Accepted user-provided Python path: "%PYTHON_EXE%"
    goto :found_python_execution
)

:found_python
echo 使用 Python: "%PYTHON_EXE%"
echo Using Python: "%PYTHON_EXE%"
echo.

:found_python_execution
:: --- Step 3: Run the model_download.py script ---
echo 启动模型下载...
echo Starting model download...
"%PYTHON_EXE%" "%SCRIPT_DIR%model_download\model_download.py"

if %errorlevel% neq 0 (
    echo.
    echo 模型下载失败。请检查上方输出中的错误信息。
    echo Model download failed. Please check the output above for errors.
    pause
    exit /b 1
)

echo.
echo 模型下载完成。
echo Model download completed successfully.
echo.
echo 您现在可以重启 ComfyUI 以加载模型。
echo You can now restart ComfyUI to load the models.
pause
endlocal