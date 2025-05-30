# ComfyUI_Float_Animator/Node.py

# This file is part of the ComfyUI_Float_Animator project.
#
# ComfyUI_Float_Animator is a custom node for ComfyUI, integrating the FLOAT project (https://github.com/deepbrainai-research/float).
# The underlying FLOAT model and its core inference code are licensed under the Creative Commons
# Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0).
#
# As this project integrates and relies heavily on the FLOAT core, the use of this software,
# including this wrapper code, is subject to the terms of the CC BY-NC-ND 4.0 license.
# You may not use this work for commercial purposes. No adaptations are permitted beyond
# necessary technical integration for framework compatibility.
# For full license details, please refer to the LICENSE.md file in the root of this repository
# and the original FLOAT project repository.

# 1. Standard library imports
import os
import sys
import time
from pathlib import Path
import math
import random
import subprocess
import datetime
import tempfile
import json 

# 2. Third-party library imports
import torch
import numpy as np
import cv2 
import librosa 
import torchvision 
import torchaudio 
import torchvision.utils as vutils
import face_alignment 
import albumentations as A 
import albumentations.pytorch.transforms as A_pytorch 
from transformers import Wav2Vec2FeatureExtractor 

# ComfyUI specific imports
import comfy.model_management as mm

# --- Dynamic Path Setup for Internal Module Imports ---
current_node_dir = Path(__file__).resolve().parent
if str(current_node_dir) not in sys.path:
    sys.path.insert(0, str(current_node_dir))

# --- Internal Module Imports ---
try:
    from models.float.FLOAT import FLOAT
    from models.wav2vec2 import Wav2VecModel
    from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification
    from models import BaseModel
    from options.__init__ import BaseOptionsJson
    from model_download import model_download # 导入模型下载模块
except ImportError as e:
    print(f"[ComfyUI_Float_Animator] 错误：无法导入核心 FLOAT 模块或模型下载模块。请检查 custom_nodes/ComfyUI_Float_Animator 目录结构和文件导入。错误信息: {e}")
    # Define dummy classes for graceful degradation if imports fail
    class FLOAT: pass
    class BaseModel: pass
    class BaseOptionsJson: pass
    class Wav2VecModel: pass
    class Wav2Vec2ForSpeechClassification: pass
    class DataProcessor: pass
    class FloatInferenceWrapper: pass
    # Dummy model_download for graceful degradation if module not found
    class model_download:
        @staticmethod
        def main():
            print("[ComfyUI_Float_Animator] 警告: 模型下载模块未加载。无法自动下载模型。")
        @staticmethod
        def get_comfyui_root_dir():
            return None


# --- Global Flags for UI Status (inspired by NodeSparkTTS.py) ---
_FLOAT_MODELS_PRESENT_FOR_UI_STATUS: bool = False
_FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION: bool = False

# --- Helper function: Find ComfyUI root directory ---
def _get_comfyui_root_dir():
    try:
        import folder_paths
        if hasattr(folder_paths, 'base_path'):
            return Path(folder_paths.base_path)
    except (ImportError, AttributeError):
        pass

    current_dir = Path(__file__).parent
    for parent in current_dir.parents:
        if parent.name == "ComfyUI":
            return parent
    return None

# --- Global Function for UI Status Check and Initial Download Trigger (inspired by NodeSparkTTS.py) ---
def _perform_initial_float_model_check_for_ui_status():
    global _FLOAT_MODELS_PRESENT_FOR_UI_STATUS, _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION
    
    comfyui_root = _get_comfyui_root_dir()
    if not comfyui_root:
        print("[ComfyUI_Float_Animator] 警告: 未能找到 ComfyUI 根目录。UI 状态可能不准确。")
        _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = False
        return

    # Paths for core models to check their existence for UI status
    float_models_base_dir_for_check = comfyui_root / "models" / "Float"
    float_main_model_path_for_check = float_models_base_dir_for_check / "float.pth"
    wav2vec2_model_dir_for_check = float_models_base_dir_for_check / "wav2vec2-base-960h"
    wav2vec_emotion_model_dir_for_check = float_models_base_dir_for_check / "wav2vec-english-speech-emotion-recognition"

    # Perform a check similar to _check_required_models_exist, but simplified for UI status
    current_models_exist = (
        float_main_model_path_for_check.is_file() and float_main_model_path_for_check.stat().st_size > 1024 * 1024 and # Check file exists and is reasonably large (e.g. >1MB)
        wav2vec2_model_dir_for_check.is_dir() and any(wav2vec2_model_dir_for_check.iterdir()) and # Check dir exists and is not empty
        wav2vec_emotion_model_dir_for_check.is_dir() and any(wav2vec_emotion_model_dir_for_check.iterdir())
    )
    
    if not current_models_exist and not _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION:
        print("[ComfyUI_Float_Animator] UI Status: 核心模型缺失。尝试在后台自动下载...")
        if model_download and hasattr(model_download, 'main'):
            try:
                model_download.main() # Trigger the download
                _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION = True
                # Re-check status after download attempt
                current_models_exist_after_download = (
                    float_main_model_path_for_check.is_file() and float_main_model_path_for_check.stat().st_size > 1024 * 1024 and
                    wav2vec2_model_dir_for_check.is_dir() and any(wav2vec2_model_dir_for_check.iterdir()) and
                    wav2vec_emotion_model_dir_for_check.is_dir() and any(wav2vec_emotion_model_dir_for_check.iterdir())
                )
                _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = current_models_exist_after_download
                if _FLOAT_MODELS_PRESENT_FOR_UI_STATUS:
                    print("[ComfyUI_Float_Animator] UI Status: 自动下载成功完成。请刷新 ComfyUI 页面 (F5) 以加载模型。")
                else:
                    print("[ComfyUI_Float_Animator] UI Status: 已启动自动下载，但模型似乎仍缺失。请检查日志。")
            except Exception as e:
                print(f"[ComfyUI_Float_Animator] UI Status: 自动模型下载在 UI 初始化期间失败: {e}")
                _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION = True # Mark as attempted regardless of success
                _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = False # Ensure UI reflects missing state
        else:
            print("[ComfyUI_Float_Animator] UI Status: 模型下载模块不可用。跳过自动下载。")
            _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION = True
            _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = False
    elif not current_models_exist and _FLOAT_DOWNLOAD_ATTEMPTED_THIS_SESSION:
        print("[ComfyUI_Float_Animator] UI Status: 本次会话中，模型在上次尝试下载后仍缺失。刷新 ComfyUI 页面以重新检查。")
        _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = False
    else: # Models are present
        _FLOAT_MODELS_PRESENT_FOR_UI_STATUS = True

# Call this function once when the Node.py module is loaded (on ComfyUI start/refresh)
_perform_initial_float_model_check_for_ui_status()


# --- DataProcessor Class ---
class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # 检查 wav2vec_model_path 是否存在且为目录
        if not Path(opt.wav2vec_model_path).is_dir():
            raise FileNotFoundError(f"wav2vec2 模型路径无效或未找到: {opt.wav2vec_model_path}")
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

        self.transform = A.Compose([
                A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                A_pytorch.ToTensorV2(),
            ])

    @torch.no_grad()
    def process_img(self, img:np.ndarray) -> np.ndarray:
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        mult = 360. / img.shape[0]
        resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        
        if not bboxes:
            print("[ComfyUI_Float_Animator] 警告: 未在参考图像中检测到人脸进行裁剪。将直接缩放图像。")
            return cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        high_conf_bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        if not high_conf_bboxes:
            print("[ComfyUI_Float_Animator] 警告: 未在参考图像中检测到高置信度人脸进行裁剪。将直接缩放图像。")
            return cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            
        bboxes = sorted(high_conf_bboxes, key=lambda x: x[4], reverse=True)[0]

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my  = int((bboxes[1] + bboxes[3]) / 2)
        mx  = int((bboxes[0] + bboxes[2]) / 2)
        
        bs = int(max(bsy, bsx) * 1.6)

        y_start = my - bs
        y_end = my + bs
        x_start = mx - bs
        x_end = mx + bs

        pad_t = max(0, -y_start)
        pad_b = max(0, y_end - img.shape[0])
        pad_l = max(0, -x_start)
        pad_r = max(0, x_end - img.shape[1])

        y_start_actual = max(0, y_start)
        y_end_actual = min(img.shape[0], y_end)
        x_start_actual = max(0, x_start)
        x_end_actual = min(img.shape[1], img.shape[0]) # Adjusted x_end_actual to be min(img.shape[1], x_end) as well
        
        crop_img = img[y_start_actual:y_end_actual, x_start_actual:x_end_actual]
        
        crop_img = cv2.copyMakeBorder(crop_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

        crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img

    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"图像文件未找到: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array = None
        sampling_rate = self.sampling_rate

        try:
            audio_tensor, orig_sampling_rate = torchaudio.load(path)
            
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            if orig_sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_sampling_rate, self.sampling_rate)
                audio_tensor = resampler(audio_tensor)
            
            speech_array = audio_tensor.squeeze(0).numpy()
            
        except Exception as e:
            print(f"[ComfyUI_Float_Animator] 警告: 使用 torchaudio 加载音频 '{path}' 失败 ({e})，尝试使用 librosa。")
            speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)

        if speech_array is None:
            raise RuntimeError(f"无法加载音频文件: {path}")

        return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]


    def preprocess(self, ref_path:str, audio_path:str, no_crop:bool) -> dict:
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(image=s)['image'].unsqueeze(0)
        a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {'s': s, 'a': a, 'p': None, 'e': None}


# --- FloatInferenceWrapper Class ---
class FloatInferenceWrapper:
    def __init__(self, opt: BaseOptionsJson):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank 
        
        self.G = None
        self.data_processor = None

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: torch.device) -> None:
        if self.G is None:
            self.load_model()
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name: 
                    pass
                else:
                    print(f"[ComfyUI_Float_Animator] 警告: 模型参数 '{model_name}' 未在 checkpoint 中找到。")
        del state_dict

    def _prepare_inference_components(self):
        if self.G is None:
            self.load_model()
            self.load_weight(self.opt.ckpt_path, self.opt.rank)
            self.G.to(self.opt.rank)
            self.G.eval()

        if self.data_processor is None:
            self.data_processor = DataProcessor(self.opt)

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str = None,
        ref_path: str = None,
        audio_path: str = None,
        a_cfg_scale: float	= 2.0,
        r_cfg_scale: float	= 1.0,
        e_cfg_scale: float	= 1.0,
        emo: str 			= None,
        nfe: int			= 10,
        no_crop: bool 		= False,
        seed: int			= 25,
        verbose: bool 		= False
    ) -> torch.Tensor:
        self._prepare_inference_components()

        self.G.to(self.opt.rank) 

        self.opt.nfe = nfe # Ensure nfe is updated from node input
        self.opt.seed = seed # Ensure seed is updated from node input

        data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)
        if verbose: print(f"[ComfyUI_Float_Animator] 数据预处理完成。")

        output_dict = self.G.inference( 
            data 		= data,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            emo 		= emo,
            nfe			= nfe,
            seed		= seed
        )
        
        d_hat = output_dict['d_hat']

        animated_frames_bhwc = d_hat.squeeze(0).permute(0, 2, 3, 1)
        animated_frames_bhwc = animated_frames_bhwc.detach().clamp(-1, 1).cpu()
        animated_frames_bhwc = (animated_frames_bhwc + 1) / 2

        if verbose: print(f"[ComfyUI_Float_Animator] 推理完成。输出帧的最终形状: {animated_frames_bhwc.shape}")
        return animated_frames_bhwc


# --- ComfyUI Node Class ---
class Float_Animator:
    # 定义一个常量，用于表示“下载模型”的选项
    DOWNLOAD_PLACEHOLDER = "Run to download model / 运行以下载模型"
    # 定义默认模型名称，这个值会作为节点内部实际加载模型的尝试，即使UI显示的是占位符
    DEFAULT_FLOAT_MODEL_NAME = "float.pth"

    def __init__(self):
        self.inference_core = None
        self.model_ready = False
        self.comfyui_root = _get_comfyui_root_dir()

        # Paths will be set by _initialize_float_core based on run-time check
        self.float_models_base_dir = self.comfyui_root / "models" / "Float" if self.comfyui_root else None
        self.float_main_model_path = None 
        self.wav2vec2_model_dir = None 
        self.wav2vec_emotion_model_dir = None 

    def _check_required_models_exist(self):
        # 读取 model_list.json 以获取所有需要检查的模型路径
        model_list_path = current_node_dir / "model_download" / "model_list.json" 
        if not model_list_path.exists():
            print(f"[ComfyUI_Float_Animator] 错误: {model_list_path.name} 文件未找到于 {model_list_path}。")
            return False

        try:
            with open(model_list_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 兼容处理非严格 JSON 格式 (例如末尾逗号，未被[]包裹)
                if content.endswith(","):
                    content = content[:-1]
                if not content.startswith("["):
                    content = f"[{content}]"
                model_list = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[ComfyUI_Float_Animator] 错误: 解析 {model_list_path.name} 失败: {e}")
            print(f"请检查 {model_list_path.name} 文件内容是否为有效 JSON 格式。")
            return False

        # 只检查 FLOAT 节点所需的特定模型
        required_model_names = [self.DEFAULT_FLOAT_MODEL_NAME, "wav2vec2-base-960h", "wav2vec-english-speech-emotion-recognition"]
        required_models_info = [info for info in model_list if info['Model'] in required_model_names]

        all_exist = True
        for model_info in required_models_info:
            model_name = model_info['Model']
            # 从 model_info['To'] 获取的路径是相对于 ComfyUI 根目录的
            relative_to_comfyui_path = Path(model_info['To'])
            
            if not self.comfyui_root: # If comfyui_root is not found, cannot check
                all_exist = False
                break
            
            # 判断 model_name 是文件还是文件夹
            is_file_download = Path(model_name).suffix != ''

            if is_file_download: # 如果是文件，目标路径是 ComfyUI/models/Float/float.pth
                final_check_path = self.comfyui_root / relative_to_comfyui_path / model_name
            else: # 如果是文件夹，目标路径是 ComfyUI/models/Float/wav2vec2-base-960h/
                final_check_path = self.comfyui_root / relative_to_comfyui_path / model_name 

            # 检查文件或非空文件夹是否存在。
            if not final_check_path.exists():
                print(f"[ComfyUI_Float_Animator] 缺失模型文件或目录: {final_check_path}")
                all_exist = False
                break 
            
            if not is_file_download and final_check_path.is_dir() and not list(final_check_path.iterdir()):
                print(f"[ComfyUI_Float_Animator] 目录 '{final_check_path}' 存在但为空。")
                all_exist = False
                break
            
            # 对于文件，检查大小是否合理（例如大于1MB）
            if is_file_download and final_check_path.is_file() and final_check_path.stat().st_size < 1024 * 1024: 
                print(f"[ComfyUI_Float_Animator] 文件 '{final_check_path}' 存在但大小异常（{final_check_path.stat().st_size / 1024:.2f} KB）。")
                all_exist = False
                break

        return all_exist

    @classmethod
    def INPUT_TYPES(cls): # Using 'cls' for class method
        
        # Use the global status flag directly for UI display
        required_models_exist = _FLOAT_MODELS_PRESENT_FOR_UI_STATUS
        
        node_instance = cls() # Create a temporary instance to access _get_available_float_models
        available_models = node_instance._get_available_float_models()
        
        # Default tooltip for the model dropdown
        model_tooltip = "Data will be auto_downloaded for the first time. After completed, refresh the page to reload the list / 首次运行节点会自动下载数据，下载完成后刷新页面以加载列表"

        if not required_models_exist:
            # If models are missing, show download placeholder and make it default
            available_models = [cls.DOWNLOAD_PLACEHOLDER]
            default_model = cls.DOWNLOAD_PLACEHOLDER
        else:
            # If models are present, prioritize DEFAULT_FLOAT_MODEL_NAME
            default_model = cls.DEFAULT_FLOAT_MODEL_NAME
            if default_model not in available_models and available_models:
                default_model = available_models[0]
            elif not available_models: # Edge case: models/Float dir exists but no .pth files
                available_models = ["(No models found, check models/Float directory) / (未找到模型，请检查 models/Float 目录)"]
                default_model = available_models[0]

        return {
            "required": {
                "ref_image": ("IMAGE", {"image_upload": True, "tooltip": "The still portrait image to animate. / 待动画化的静态肖像图像"}),
                "audio": ("AUDIO", {"tooltip": "The driving audio for animation. / 驱动动画的音频"}),
                "seed": ("INT", {"default": 15, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility. / 结果可复现的随机种子", "widget": "random_seed"}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none", "tooltip": "Target emotion style. 'none' infers from audio. / 目标情感风格。'none'从音频推断。"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 1.0, "round": 0.01, "tooltip": "Frames per second for output animation. / 输出动画的每秒帧数"}),
                "aud_cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for audio control. / 音频控制的无分类器引导尺度"}),
                "ref_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for reference image control. / 参考图像控制的无分类器引导尺度"}),
                "emo_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for emotion control. / 情感控制的无分类器引导尺度"}),
                "model": (available_models, {"default": default_model, "tooltip": model_tooltip}), 
                "auto_crop": ("BOOLEAN", {"default": False, "tooltip": "Automatically crop face in reference image. / 自动裁剪参考图像中的人脸"}),
                # "nfe": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of Function Evaluations (NFEs) for ODE solver. / ODE 求解器的函数评估次数。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT",)
    RETURN_NAMES = ("animated_frames (IMAGE) / 动画帧 (图像)", 
                    "audio (AUDIO) / 音频 (音频)", 
                    "fps (FLOAT) / 每秒帧数 (浮点数)")
    FUNCTION = "animate_portrait"
    CATEGORY = "Animator"
    DESCRIPTION = "Generates speaking portrait video frames from an image and audio using the FLOAT model. / 使用 FLOAT 模型从图像和音频生成说话肖像视频帧。"

    def _get_available_float_models(self):
        # 确保 float_models_base_dir 已正确设置
        if not self.float_models_base_dir or not self.float_models_base_dir.is_dir():
            return []
        
        # 只查找 .pth 文件作为可选择的主模型
        models = [f.name for f in self.float_models_base_dir.iterdir() if f.suffix == '.pth' and f.is_file()]
        return sorted(models)

    def _initialize_float_core(self, selected_model_option: str):
        # 根据 selected_model_option 确定实际要加载的模型文件名
        if selected_model_option == self.DOWNLOAD_PLACEHOLDER:
            actual_model_name = self.DEFAULT_FLOAT_MODEL_NAME
            print(f"[ComfyUI_Float_Animator] 提示: 选择了模型下载选项，将尝试加载默认模型 '{actual_model_name}'。")
        else:
            actual_model_name = selected_model_option
            print(f"[ComfyUI_Float_Animator] 尝试加载模型: '{actual_model_name}'。")


        # 如果模型已加载且是当前选择（或默认）的模型，则无需重新加载
        if self.model_ready and self.inference_core and self.float_main_model_path and self.float_main_model_path.name == actual_model_name:
            print(f"[ComfyUI_Float_Animator] FLOAT 模型核心已加载且为 '{actual_model_name}'，跳过重新加载。")
            return

        if not self.comfyui_root:
            raise Exception("[ComfyUI_Float_Animator] 错误：未能找到 ComfyUI 根目录。请确保自定义节点放置在 ComfyUI/custom_nodes/ 目录下。")

        # 更新模型路径以确保它们是最新的，基于实际要加载的模型名
        self.float_models_base_dir = self.comfyui_root / "models" / "Float" # Ensure base dir is set correctly here
        self.float_main_model_path = self.float_models_base_dir / actual_model_name
        self.wav2vec2_model_dir = self.float_models_base_dir / "wav2vec2-base-960h"
        self.wav2vec_emotion_model_dir = self.float_models_base_dir / "wav2vec-english-speech-emotion-recognition"

        # 检查所有必需的模型文件是否存在
        required_models_check_passed = self._check_required_models_exist()
        
        if not required_models_check_passed:
            print(f"[ComfyUI_Float_Animator] 警告: 核心模型文件缺失。尝试自动下载模型...")
            try:
                # 调用 model_download.py 的 main 函数进行下载
                model_download.main()
                print("[ComfyUI_Float_Animator] 模型下载尝试完成。重新检查模型完整性。")
                required_models_check_passed = self._check_required_models_exist() # 再次检查
            except Exception as e:
                print(f"[ComfyUI_Float_Animator] 自动下载模型失败: {e}")
                # 在自动下载失败时，仍然抛出错误以中断工作流
                raise FileNotFoundError(f"[ComfyUI_Float_Animator] 模型文件仍缺失。请检查网络连接或手动运行 ComfyUI/custom_nodes/ComfyUI_Float_Animator/Model_Download.bat 下载所需模型。错误: {e}")
        
        # 如果下载后模型仍然缺失
        if not required_models_check_passed:
             raise FileNotFoundError(f"[ComfyUI_Float_Animator] 自动下载后，所需核心模型仍缺失。请手动检查并确保以下文件存在: {self.float_main_model_path}, {self.wav2vec2_model_dir}, {self.wav2vec_emotion_model_dir}")


        # 确保 BaseOptionsJson 和 FloatInferenceWrapper 已正确导入
        if BaseOptionsJson is None or FloatInferenceWrapper is None:
            raise Exception("[ComfyUI_Float_Animator] 内部 FLOAT 模块加载失败。请检查 custom_nodes/ComfyUI_Float_Animator 目录结构及其导入。")

        opt_config = BaseOptionsJson() 
        opt_config.rank = mm.get_torch_device()
        opt_config.ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        opt_config.ckpt_path = str(self.float_main_model_path)
        opt_config.pretrained_dir = str(self.float_models_base_dir) # pretrained_dir is typically the parent of checkpoints
        opt_config.wav2vec_model_path = str(self.wav2vec2_model_dir)
        opt_config.audio2emotion_path = str(self.wav2vec_emotion_model_dir)
        
        self.inference_core = FloatInferenceWrapper(opt_config)
        self.model_ready = True
        print(f"[ComfyUI_Float_Animator] FLOAT 模型核心已加载并准备就绪 (使用模型: {actual_model_name})。")


    def animate_portrait(self, ref_image, audio, model, fps, 
                         aud_cfg_scale, ref_cfg_scale, emo_cfg_scale,
                         emotion, auto_crop, seed):
        
        # 尝试初始化 FLOAT 核心。
        # 如果模型缺失或选择了下载占位符，_initialize_float_core 会触发下载或抛出异常。
        self._initialize_float_core(model) 

        if not self.inference_core:
            raise Exception("[ComfyUI_Float_Animator] FLOAT 推理核心未初始化。请检查日志以获取更多信息。")

        # 更新推理参数
        self.inference_core.opt.fps = fps
        self.inference_core.opt.a_cfg_scale = aud_cfg_scale
        self.inference_core.opt.r_cfg_scale = ref_cfg_scale
        self.inference_core.opt.e_cfg_scale = emo_cfg_scale
        self.inference_core.opt.seed = seed
        self.inference_core.opt.no_crop = not auto_crop
        # If 'nfe' was added to INPUT_TYPES: self.inference_core.opt.nfe = nfe

        temp_working_dir = self.comfyui_root / "temp" / "float_animator_tmp"
        os.makedirs(temp_working_dir, exist_ok=True)
        
        timestamp_pid = f"{int(time.time())}_{os.getpid()}_{random.randint(0, 9999)}"
        audio_temp_path = temp_working_dir / f"input_audio_{timestamp_pid}.wav"
        image_temp_path = temp_working_dir / f"reference_image_{timestamp_pid}.png"

        try:
            # Ensure audio waveform is mono and float32
            if audio['waveform'].dim() == 3: # (Batch, Channels, Samples)
                audio_waveform_to_save = audio['waveform'].squeeze(0) # Remove batch dim
            elif audio['waveform'].dim() == 2: # (Channels, Samples)
                audio_waveform_to_save = audio['waveform']
            else:
                raise ValueError("[ComfyUI_Float_Animator] 不支持的音频波形维度。Expected (Batch, Channels, Samples) or (Channels, Samples).")
            
            # If multi-channel, convert to mono (by averaging)
            if audio_waveform_to_save.shape[0] > 1:
                audio_waveform_to_save = torch.mean(audio_waveform_to_save, dim=0, keepdim=True)
            
            if audio_waveform_to_save.dtype != torch.float32:
                 audio_waveform_to_save = audio_waveform_to_save.to(torch.float32)
            
            # Normalize audio to [-1, 1]
            if audio_waveform_to_save.max() > 1.0 or audio_waveform_to_save.min() < -1.0:
                audio_waveform_to_save = audio_waveform_to_save / max(audio_waveform_to_save.abs().max().item(), 1.0)

            torchaudio.save(str(audio_temp_path), audio_waveform_to_save, audio["sample_rate"])
            print(f"[ComfyUI_Float_Animator] 音频已保存至: {audio_temp_path.name}")

            if ref_image.shape[0] != 1:
                raise ValueError("[ComfyUI_Float_Animator] 仅支持单张参考图像 (batch size 必须为 1)。")
            
            # ComfyUI image is (B, H, W, C), vutils expects (C, H, W)
            ref_image_chw = ref_image[0].permute(2, 0, 1)
            vutils.save_image(ref_image_chw, str(image_temp_path))
            print(f"[ComfyUI_Float_Animator] 图像已保存至: {image_temp_path.name}")

            print(f"[ComfyUI_Float_Animator] 开始 FLOAT 推理...")
            animated_output_frames = self.inference_core.run_inference(
                res_video_path=None, # Not used in ComfyUI node, frames returned directly
                ref_path=str(image_temp_path),
                audio_path=str(audio_temp_path),
                a_cfg_scale=aud_cfg_scale,
                r_cfg_scale=ref_cfg_scale,
                e_cfg_scale=emo_cfg_scale,
                emo=None if emotion == "none" else emotion,
                nfe=10, # Using BaseOptionsJson default
                no_crop=not auto_crop,
                seed=seed,
                verbose=True
            )
            print(f"[ComfyUI_Float_Animator] FLOAT 推理完成。输出帧形状: {animated_output_frames.shape}")

        finally:
            # Clean up temporary files
            if os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
            if os.path.exists(image_temp_path):
                os.remove(image_temp_path)

            # Offload model to CPU to free VRAM
            if self.inference_core and self.inference_core.G:
                self.inference_core.G.to(mm.unet_offload_device())
                mm.soft_empty_cache()

        return (animated_output_frames, audio, fps,)


# 将节点类添加到 ComfyUI 的映射中
NODE_CLASS_MAPPINGS = {
    "Float_Animator": Float_Animator,
}