import os
import sys
import json
import subprocess
from pathlib import Path
import shutil
import re # <-- 新增：导入正则表达式模块

# 确定 ComfyUI 嵌入式 Python 的可执行文件路径
def get_python_exe():
    """
    尝试找到 ComfyUI 的嵌入式 Python 可执行文件。
    优先检查 ComfyUI/python_embeded/python.exe (Windows)
    然后检查 ComfyUI/venv/bin/python (Linux/macOS)
    最后回退到当前脚本的 sys.executable。
    """
    current_script_dir = Path(__file__).resolve().parent
    comfyui_root_dir = None
    for parent in current_script_dir.parents:
        if parent.name == "ComfyUI":
            comfyui_root_dir = parent
            break
    
    if comfyui_root_dir:
        # 检查 Windows 嵌入式 Python
        embedded_python_windows = comfyui_root_dir / "python_embeded" / "python.exe"
        if embedded_python_windows.exists():
            return str(embedded_python_windows)
        # 检查 Linux/macOS venv 或标准安装
        embedded_python_linux_mac = comfyui_root_dir / "venv" / "bin" / "python"
        if embedded_python_linux_mac.exists():
            return str(embedded_python_linux_mac)

    # 回退到当前 sys.executable (例如，如果 ComfyUI 是从虚拟环境运行的)
    return sys.executable

PYTHON_EXE = get_python_exe()

def install_package(package):
    """
    使用 ComfyUI 的 Python 环境安装指定的 pip 包。
    """
    print(f"Installing {package}...")
    try:
        # 使用 check_call 确保子进程成功完成
        subprocess.check_call([PYTHON_EXE, "-m", "pip", "install", package])
        print(f"Successfully installed {package}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        print(f"Please try to manually install: {PYTHON_EXE} -m pip install {package}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during installation of {package}: {e}")
        return False

# 在尝试导入之前，确保所有必要的下载包已安装。
# 这样可以在脚本执行时动态安装，避免用户手动操作。
required_download_packages = ["gdown", "huggingface_hub", "GitPython"]
for pkg in required_download_packages:
    try:
        if pkg == "GitPython":
            import git # GitPython
        elif pkg == "huggingface_hub":
            from huggingface_hub import snapshot_download, HfFileSystem # huggingface_hub
        else:
            __import__(pkg) # gdown
    except ImportError:
        if not install_package(pkg):
            print(f"Required package {pkg} could not be installed. Please install it manually.")
            # sys.exit(1) # 如果在节点运行时，不应该退出整个ComfyUI进程，而是让节点报告错误
            # 这里改为 raise Exception，让 Node.py 捕获并处理
            raise Exception(f"Required package {pkg} could not be installed automatically.")

# 重新导入以确保新安装的包可用
try:
    import gdown
    from huggingface_hub import snapshot_download, HfFileSystem
    import git
except ImportError as e:
    # 这种情况通常只会在安装失败后，代码继续执行时发生
    print(f"Error: Essential download packages are missing after installation attempt: {e}")
    raise Exception(f"Error: Essential download packages are missing after installation attempt: {e}")


def get_comfyui_root_dir():
    """
    辅助函数：找到 ComfyUI 的根目录。
    """
    current_dir = Path(__file__).resolve().parent
    for parent in current_dir.parents:
        if parent.name == "ComfyUI":
            return parent
    return None

def download_model(model_info):
    """
    根据模型信息下载模型文件或文件夹。
    """
    model_name = model_info['Model']
    address = model_info['Address']
    # 'To' 路径是相对于 ComfyUI 根目录的路径
    relative_to_path_str = model_info['To']

    comfyui_root = get_comfyui_root_dir()
    if not comfyui_root:
        print(f"Error: Could not find ComfyUI root directory for model {model_name}.")
        return False

    # 构建最终的目标路径
    is_file_download = Path(model_name).suffix != ''

    target_base_dir = comfyui_root / Path(relative_to_path_str)
    
    if is_file_download:
        final_target_path = target_base_dir / model_name
    else:
        final_target_path = target_base_dir / model_name # 这将是目标文件夹的路径
    
    # 检查目标是否已经存在且非空
    if final_target_path.exists():
        if is_file_download and final_target_path.is_file():
            # 对于文件，检查大小是否合理（至少大于某个阈值，例如 1MB，防止小尺寸的错误文件）
            # 或者更严谨地，如果知道目标文件大小，进行比对
            # 这里简单判断是否是明显的小文件（例如小于1MB，假设所有模型都大于此）
            if final_target_path.stat().st_size > 1024 * 1024: # 假设文件大于1MB是正常的
                print(f"Model '{model_name}' already exists at {final_target_path} and seems complete. Skipping download.")
                return True
            else:
                print(f"Model '{model_name}' exists at {final_target_path} but is too small ({final_target_path.stat().st_size / 1024:.2f} KB). Re-downloading.")
                try:
                    os.remove(final_target_path)
                except OSError as e:
                    print(f"Error removing small file {final_target_path}: {e}")
                    return False

        elif not is_file_download and final_target_path.is_dir() and list(final_target_path.iterdir()):
            print(f"Model directory '{model_name}' already exists and is not empty at {final_target_path}. Skipping download.")
            return True
        # 如果是空文件夹或者文件不存在但路径存在 (比如之前下载失败的空文件)，则尝试清理后重新下载
        elif not is_file_download and final_target_path.is_dir() and not list(final_target_path.iterdir()):
            print(f"Existing empty directory {final_target_path} found. Removing to redownload.")
            try:
                shutil.rmtree(final_target_path)
            except OSError as e:
                print(f"Error removing empty directory {final_target_path}: {e}")
                return False
        elif is_file_download and final_target_path.is_file() and final_target_path.stat().st_size == 0:
            print(f"Existing empty file {final_target_path} found. Removing to redownload.")
            try:
                os.remove(final_target_path)
            except OSError as e:
                print(f"Error removing empty file {final_target_path}: {e}")
                return False
    
    # 确保目标文件夹存在
    os.makedirs(final_target_path.parent if is_file_download else final_target_path, exist_ok=True)

    print(f"Downloading '{model_name}' from {address} to {final_target_path}...")

    try:
        if "drive.google.com" in address:
            # Google Drive download
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', address)
            if match:
                file_id = match.group(1)
                gdown.download(id=file_id, output=str(final_target_path), quiet=False)
                print(f"Downloaded {model_name} from Google Drive using ID: {file_id}.")
            else:
                print(f"Error: Could not extract Google Drive file ID from URL: {address}")
                return False
        elif "huggingface.co" in address:
            # Hugging Face download
            repo_id = address.replace("https://huggingface.co/", "")
            
            fs = HfFileSystem()
            if "/" in repo_id and fs.isfile(repo_id): # Example: "org/model/file.safetensors"
                 # 直接下载文件
                if not is_file_download: # 如果 model_name 指定为文件夹，但 address 是文件，给出警告
                    print(f"Warning: Model '{model_name}' specified as folder but '{address}' points to a single file. Downloading to {final_target_path.parent / Path(repo_id).name}.")
                    # 调整 final_target_path 为实际的文件名
                    final_target_path = final_target_path.parent / Path(repo_id).name
                HfFileSystem().get_file(address, str(final_target_path))
                print(f"Downloaded {model_name} from Hugging Face.")
            else:
                # 假设是 Hugging Face 仓库，使用 snapshot_download
                print(f"Downloading repository {repo_id} to {final_target_path}...")
                snapshot_download(repo_id=repo_id, local_dir=str(final_target_path), local_dir_use_symlinks=False)
                print(f"Downloaded {model_name} repository from Hugging Face.")
        elif "github.com" in address:
            # GitHub download (clone repository)
            # 对于 Git 克隆，先确保目录不存在或为空，再进行克隆
            if final_target_path.exists():
                print(f"Removing existing directory {final_target_path} for fresh clone.")
                shutil.rmtree(final_target_path) 
            git.Repo.clone_from(address, str(final_target_path))
            print(f"Cloned {model_name} from GitHub.")
        else:
            print(f"Error: Unsupported download address type for {model_name}: {address}")
            return False
        
        print(f"Successfully downloaded '{model_name}' to {final_target_path}")
        return True
    except Exception as e:
        print(f"Failed to download '{model_name}'. Error: {e}")
        # 如果下载失败，清理可能创建的空文件或空目录
        if final_target_path.exists():
            if final_target_path.is_file():
                os.remove(final_target_path)
            elif final_target_path.is_dir() and not list(final_target_path.iterdir()):
                shutil.rmtree(final_target_path)
        return False

def main():
    """
    主函数：读取模型列表并执行下载。
    """
    script_dir = Path(__file__).resolve().parent
    # 更改为 model_list.json
    model_list_path = script_dir / "model_list.json" 

    if not model_list_path.exists():
        print(f"Error: model_list.json not found at {model_list_path}")
        # sys.exit(1) # 不在 ComfyUI 节点运行时直接退出
        raise FileNotFoundError(f"model_list.json not found at {model_list_path}")

    try:
        # 读取内容并尝试解析为 JSON。
        # 兼容处理非严格 JSON 格式 (例如末尾逗号，未被[]包裹)。
        with open(model_list_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 简单处理末尾逗号 (在手动编辑的 JSON-like 文件中常见)
        # 这一段在标准的 model_list.json 格式下可能不再需要，但保留兼容性更好
        if content.endswith(","):
            content = content[:-1]
        
        # 如果内容不是以数组开头，则包裹成数组
        if not content.startswith("["):
            content = f"[{content}]"

        model_list = json.loads(content)

    except json.JSONDecodeError as e:
        print(f"Error parsing model_list.json as JSON: {e}")
        print(f"Content that caused error:\n---\n{content}\n---")
        print("Please ensure model_list.json is a valid JSON array of objects.")
        # sys.exit(1) # 不在 ComfyUI 节点运行时直接退出
        raise ValueError(f"Error parsing model_list.json: {e}")

    all_downloads_successful = True
    for model_info in model_list:
        # 确保每个 model_info 字典都包含所有必需的键
        if not all(k in model_info for k in ['Model', 'Address', 'To']):
            print(f"Warning: Skipping malformed model entry: {model_info}. Missing 'Model', 'Address', or 'To' key.")
            all_downloads_successful = False
            continue

        if not download_model(model_info):
            all_downloads_successful = False
            # 在这里我们选择继续下载其他模型，而不是立即中断
            # break # Uncomment this line to stop at the first failure.
            
    if all_downloads_successful:
        print("\nAll specified models processed successfully!")
    else:
        print("\nSome models failed to download. Please check the logs above.")
        # sys.exit(1) # 不在 ComfyUI 节点运行时直接退出
        raise Exception("\nSome models failed to download. Please check the logs above.")

if __name__ == "__main__":
    main()