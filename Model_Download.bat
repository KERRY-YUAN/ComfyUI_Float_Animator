@echo off
setlocal enabledelayedexpansion

REM 设置控制台编码为 UTF-8，以正确显示中文
chcp 65001 > nul

echo.
echo ===================================================
echo "ComfyUI_Float_Animator 模型下载器"
echo ===================================================
echo.
echo "此脚本将自动下载 ComfyUI_Float_Animator 节点所需的所有 FLOAT 模型。"
echo "模型将存储到以下目录："
echo.
echo "   [ComfyUI 根目录]\models\Float\"
echo.
echo "!!! 例如: D:\Program\ComfyUI_Program\ComfyUI\custom_nodes\ComfyUI_Float_Animator"
echo.
echo "请确保您已运行 'pip install -r requirements.txt' 安装了所有必要的Python依赖。"
echo "如果遇到下载问题，请检查网络连接或手动安装依赖。"
echo.
set /p confirm="按 Enter 键开始下载，按 Ctrl+C 取消。"

echo.
echo "--- 检查 Python 环境 ---"
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo "错误：未找到 Python。请安装 Python 并将其添加到系统 PATH 环境变量。"
    echo "脚本将暂停。"
    pause
    goto :eof
)
echo "Python 环境检查通过。"

echo.
echo "--- 检查必要的 Python 库 ---"
REM 尝试导入 huggingface_hub 和 gdown，如果失败则提示安装
python -c "import sys; try: import huggingface_hub, gdown; print('Python libs ok') except ImportError: sys.exit(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo "错误：缺少必要的 Python 库 (huggingface_hub 或 gdown)。"
    echo "请运行以下命令安装它们（通常是在您的 ComfyUI Python 环境中）："
    echo "   [您的 ComfyUI 根目录]\python_embeded\python.exe -m pip install -r requirements.txt"
    echo "或者对于系统级/虚拟环境 Python: pip install -r requirements.txt"
    echo "脚本将暂停。"
    pause
    goto :eof
)
echo "Python 库检查通过 (huggingface_hub, gdown)。"

echo.
echo "--- 确定 ComfyUI 根目录 ---"
REM 获取当前批处理文件所在的目录路径
set "NODE_INSTALL_DIR=%~dp0"
REM 去掉末尾的反斜杠
set "NODE_INSTALL_DIR=%NODE_INSTALL_DIR:~0,-1%"

REM 尝试自动检测 ComfyUI 根目录
set "COMFYUI_ROOT_DIR="
set "CURRENT_CHECK_DIR=%NODE_INSTALL_DIR%"

:find_comfyui_root_loop
    REM 检查当前目录是否包含 main.py 或 models 文件夹
    if exist "%CURRENT_CHECK_DIR%\main.py" (
        set "COMFYUI_ROOT_DIR=%CURRENT_CHECK_DIR%"
        goto found_comfyui_root
    )
    if exist "%CURRENT_CHECK_DIR%\models" (
        set "COMFYUI_ROOT_DIR=%CURRENT_CHECK_DIR%"
        goto found_comfyui_root
    )

    REM 如果已经是驱动器根目录 (例如 C:\)，则停止
    REM 检查是否是 "X:\" 格式的根目录
    for %%D in (C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
        if "%CURRENT_CHECK_DIR%\"=="%%D:\" (
            goto not_found_comfyui_root_auto
        )
    )

    REM 获取父目录
    for %%F in ("%CURRENT_CHECK_DIR%\..") do set "PARENT_DIR=%%~fF"

    REM 如果父目录和当前目录相同，说明到达了顶层，停止
    if "%PARENT_DIR%"=="%CURRENT_CHECK_DIR%" (
        goto not_found_comfyui_root_auto
    )

    set "CURRENT_CHECK_DIR=%PARENT_DIR%"
    goto find_comfyui_root_loop

:not_found_comfyui_root_auto
    echo.
    echo 警告：无法自动检测到 ComfyUI 根目录。
    REM 回退：假设 ComfyUI 根目录是 custom_nodes 的父目录
    for %%F in ("%NODE_INSTALL_DIR%\..\..") do set "COMFYUI_ROOT_DIR=%%~fF"
    echo 将使用此目录作为 ComfyUI 根目录: %COMFYUI_ROOT_DIR%
    echo 请确认此路径是否正确。如果不是，请手动编辑此脚本中的 COMFYUI_ROOT_DIR 变量。
    echo.
    goto continue_with_paths

:found_comfyui_root
    echo ComfyUI 根目录已检测到: %COMFYUI_ROOT_DIR%
    echo.

:continue_with_paths
    if not defined COMFYUI_ROOT_DIR (
        echo 错误：无法确定 ComfyUI 根目录。脚本已退出。
        pause
        exit /b 1
    )

REM --- 定义模型目标目录 ---
set "FLOAT_MODELS_TARGET_DIR=%COMFYUI_ROOT_DIR%\models\Float"
if not exist "%FLOAT_MODELS_TARGET_DIR%" (
    echo 创建模型目标目录: %FLOAT_MODELS_TARGET_DIR%
    mkdir "%FLOAT_MODELS_TARGET_DIR%"
)

echo.
echo "--- 开始模型下载 ---"
echo "所有模型将下载到: %FLOAT_MODELS_TARGET_DIR%"

REM 将 Python 代码写入临时文件，用于执行下载任务
set "TEMP_PYTHON_SCRIPT=%NODE_INSTALL_DIR%\temp_float_download_script.py"

(
    echo import os
    echo from pathlib import Path
    echo import sys
    echo import shutil
    echo sys.stdout.reconfigure(encoding='utf-8')
    echo sys.stderr.reconfigure(encoding='utf-8')
    echo 
    echo # 从批处理文件传递过来的目标模型目录路径
    echo float_models_target_dir_str = sys.argv[1]
    echo float_models_target_dir = Path(float_models_target_dir_str).resolve()
    echo 
    echo try:
    echo     from huggingface_hub import snapshot_download
    echo     import gdown
    echo except ImportError:
    echo     sys.stderr.write('错误：缺少必要的 Python 库 (huggingface_hub 或 gdown)。请运行 \"pip install -r requirements.txt\"。\n')
    echo     sys.exit(1)
    echo 
    echo # 定义各个模型文件的完整路径
    echo float_pth_path = float_models_target_dir / "float.pth"
    echo wav2vec2_dir = float_models_target_dir / "wav2vec2-base-960h"
    echo emotion_rec_dir = float_models_target_dir / "wav2vec-english-speech-emotion-recognition"
    echo 
    echo # 下载 float.pth (主模型)
    echo if not float_pth_path.exists():
    echo     sys.stdout.write(f"正在下载 float.pth (主模型)...\n")
    echo     sys.stdout.flush()
    echo     try:
    echo         gdown.download(id="1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0", output=str(float_pth_path), quiet=False)
    echo         sys.stdout.write("float.pth 下载完成。\n")
    echo         sys.stdout.flush()
    echo     except Exception as e:
    echo         sys.stderr.write(f"下载 float.pth 时出错: {e}\n")
    echo         sys.exit(1)
    echo else:
    echo     sys.stdout.write("float.pth 已存在，跳过下载。\n")
    echo     sys.stdout.flush()
    echo 
    echo # 下载 wav2vec2-base-960h 文件夹 (音频编码器)
    echo if not wav2vec2_dir.is_dir() or not any(wav2vec2_dir.iterdir()):
    echo     # 检查目录是否存在但为空的情况，避免重复下载
    echo     if wav2vec2_dir.is_dir() and not any(wav2vec2_dir.iterdir()):
    echo         sys.stdout.write(f"目录 '{wav2vec2_dir.name}' 存在但为空，正在重新下载...\n")
    echo         sys.stdout.flush()
    echo         shutil.rmtree(wav2vec2_dir) # 移除空目录
    echo     else:
    echo         sys.stdout.write(f"正在下载 wav2vec2-base-960h (音频编码器) 文件夹...\n")
    echo         sys.stdout.flush()
    echo     try:
    echo         snapshot_download(repo_id="facebook/wav2vec2-base-960h", local_dir=str(wav2vec2_dir), local_dir_use_symlinks=False)
    echo         sys.stdout.write("wav2vec2-base-960h 下载完成。\n")
    echo         sys.stdout.flush()
    echo     except Exception as e:
    echo         sys.stderr.write(f"下载 wav2vec2-base-960h 时出错: {e}\n")
    echo         sys.exit(1)
    echo else:
    echo     sys.stdout.write("wav2vec2-base-960h 已存在且非空，跳过下载。\n")
    echo     sys.stdout.flush()
    echo 
    echo # 下载 wav2vec-english-speech-emotion-recognition 文件夹 (情感编码器)
    echo if not emotion_rec_dir.is_dir() or not any(emotion_rec_dir.iterdir()):
    echo     # 检查目录是否存在但为空的情况
    echo     if emotion_rec_dir.is_dir() and not any(emotion_rec_dir.iterdir()):
    echo         sys.stdout.write(f"目录 '{emotion_rec_dir.name}' 存在但为空，正在重新下载...\n")
    echo         sys.stdout.flush()
    echo         shutil.rmtree(emotion_rec_dir) # 移除空目录
    echo     else:
    echo         sys.stdout.write(f"正在下载 wav2vec-english-speech-emotion-recognition (情感编码器) 文件夹...\n")
    echo         sys.stdout.flush()
    echo     try:
    echo         snapshot_download(repo_id="r-f/wav2vec-english-speech-emotion-recognition", local_dir=str(emotion_rec_dir), local_dir_use_symlinks=False)
    echo         sys.stdout.write("wav2vec-english-speech-emotion-recognition 下载完成。\n")
    echo         sys.stdout.flush()
    echo     except Exception as e:
    echo         sys.stderr.write(f"下载 wav2vec-english-speech-emotion-recognition 时出错: {e}\n")
    echo         sys.exit(1)
    echo else:
    echo     sys.stdout.write("wav2vec-english-speech-emotion-recognition 已存在且非空，跳过下载。\n")
    echo     sys.stdout.flush()
    echo 
    echo sys.stdout.write("\n所有模型下载/检查完成。\n")
    echo sys.stdout.flush()
) > "%TEMP_PYTHON_SCRIPT%"

REM 执行临时 Python 脚本
REM 将 %FLOAT_MODELS_TARGET_DIR% 作为命令行参数传递给 Python 脚本
python "%TEMP_PYTHON_SCRIPT%" "%FLOAT_MODELS_TARGET_DIR%"
set "PYTHON_EXIT_CODE=%errorlevel%"

REM 清理临时文件
if exist "%TEMP_PYTHON_SCRIPT%" del "%TEMP_PYTHON_SCRIPT%"

if %PYTHON_EXIT_CODE% neq 0 (
    echo.
    echo "模型下载失败。请检查上方错误信息以了解详细原因。"
) else (
    echo.
    echo "模型下载完成。您可以关闭此窗口并重新启动 ComfyUI 来加载节点。"
)
echo.
pause
endlocal