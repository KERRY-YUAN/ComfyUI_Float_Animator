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
import json # 用于解析 model_list.json

# 2. Third-party library imports
import torch
import numpy as np
import cv2 # OpenCV for image processing
import librosa # Audio processing, used as fallback
import torchvision # For saving images (e.g., vutils.save_image)
import torchaudio # Audio processing, preferred over librosa for robustness
import torchvision.utils as vutils
import face_alignment # For face detection and landmarks
import albumentations as A # Image augmentation/transforms
import albumentations.pytorch.transforms as A_pytorch # Albumentations PyTorch transforms
from transformers import Wav2Vec2FeatureExtractor # Audio feature extraction

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
    # Define dummy classes to prevent hard crashes during import, but functional errors will occur
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
        x_end_actual = min(img.shape[1], x_end)

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

        if not self.comfyui_root:
            print("[ComfyUI_Float_Animator] 警告: 未能找到 ComfyUI 根目录。请确保自定义节点放置在 ComfyUI/custom_nodes/ 目录下。")
            self.float_models_base_dir = None
            self.float_main_model_path = None
            self.wav2vec2_model_dir = None
            self.wav2vec_emotion_model_dir = None
            return

        self.float_models_base_dir = self.comfyui_root / "models" / "Float"
        # 初始化路径，这些路径在 _initialize_float_core 中会被更新
        self.float_main_model_path = self.float_models_base_dir / self.DEFAULT_FLOAT_MODEL_NAME
        self.wav2vec2_model_dir = self.float_models_base_dir / "wav2vec2-base-960h"
        self.wav2vec_emotion_model_dir = self.float_models_base_dir / "wav2vec-english-speech-emotion-recognition"

    def _check_required_models_exist(self):
        # 读取 model_list.json 以获取所有需要检查的模型路径
        model_list_path = current_node_dir / "model_download" / "model_list.json" # <--- 使用 .json
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
            
            # 完整的目标路径： ComfyUI根目录 / model_info['To'] / model_name （如果是文件）
            # 或者 ComfyUI根目录 / model_info['To'] / model_name （如果是文件夹）
            
            # 判断 model_name 是文件还是文件夹
            is_file_download = Path(model_name).suffix != ''

            if is_file_download: # 如果是文件，目标路径是 ComfyUI/models/Float/float.pth
                final_check_path = self.comfyui_root / relative_to_comfyui_path / model_name
            else: # 如果是文件夹，目标路径是 ComfyUI/models/Float/wav2vec2-base-960h/
                final_check_path = self.comfyui_root / relative_to_comfyui_path / model_name 

            # 检查文件或非空文件夹是否存在。这里可以复用 model_download.py 中的检查逻辑
            # 为了简化，我们只检查是否存在且非空
            if not final_check_path.exists():
                print(f"[ComfyUI_Float_Animator] 缺失模型文件或目录: {final_check_path}")
                all_exist = False
                break 
            
            if not is_file_download and final_check_path.is_dir() and not list(final_check_path.iterdir()):
                print(f"[ComfyUI_Float_Animator] 目录 '{final_check_path}' 存在但为空。")
                all_exist = False
                break
            
            if is_file_download and final_check_path.is_file() and final_check_path.stat().st_size < 1024 * 1024: # 假设文件小于1MB是异常
                print(f"[ComfyUI_Float_Animator] 文件 '{final_check_path}' 存在但大小异常（{final_check_path.stat().st_size / 1024:.2f} KB）。")
                all_exist = False
                break

        return all_exist

    @classmethod
    def INPUT_TYPES(s):
        node_instance = s() # 创建实例以便调用方法
        
        # 检查核心模型是否存在，影响下拉菜单的显示
        required_models_exist = node_instance._check_required_models_exist()
        
        available_models = node_instance._get_available_float_models()
        
        # 定义通用 tooltip
        model_tooltip = "Select to run; the model will be automatically downloaded for the first time. Please wait. After completed, refresh the page to load the model."

        if not required_models_exist:
            # 如果模型缺失，只显示下载提示，并将其作为默认值
            available_models = [s.DOWNLOAD_PLACEHOLDER]
            default_model = s.DOWNLOAD_PLACEHOLDER
        else:
            # 如果模型存在，优先选择 DEFAULT_FLOAT_MODEL_NAME，如果不存在则选第一个
            default_model = s.DEFAULT_FLOAT_MODEL_NAME
            if default_model not in available_models and available_models:
                default_model = available_models[0]
            elif not available_models: # 极端情况：models/Float 目录存在但没有 .pth 文件
                available_models = ["(未找到可用模型，请检查 models/Float 目录)"]
                default_model = available_models[0]

        return {
            "required": {
                "ref_image": ("IMAGE", {"image_upload": True, "tooltip": "The still portrait image to animate."}),
                "audio": ("AUDIO", {"tooltip": "The driving audio to synchronize with the portrait."}),
                "seed": ("INT", {"default": 15, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility.", "widget": "random_seed"}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none", "tooltip": "Specify a target emotion. 'none' uses emotion from audio."}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 1.0, "round": 0.01, "tooltip": "Frames per second for the output animation."}),
                "aud_cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for audio control."}),
                "ref_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for reference image control."}),
                "emo_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale for emotion control."}),
                # 修改 model 参数，将 tooltip 移到此处
                "model": (available_models, {"default": default_model, "tooltip": model_tooltip}), 
                "auto_crop": ("BOOLEAN", {"default": False, "tooltip": "Automatically crop the face in the reference image for optimal results."}),
                # TODO: 如果需要用户控制 NFE，可以在这里添加一个 INT 参数
                # "nfe": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of Function Evaluations (NFEs) for ODE solver."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT",)
    RETURN_NAMES = ("animated_frames", "audio", "fps",)
    FUNCTION = "animate_portrait"
    CATEGORY = "Animator"
    DESCRIPTION = "Generates speaking portrait video frames from an image and audio using the FLOAT model."

    def _get_available_float_models(self):
        # 确保 float_models_base_dir 已正确设置
        if not self.float_models_base_dir or not self.float_models_base_dir.is_dir():
            return []
        
        # 只查找 .pth 文件作为可选择的主模型
        models = [f.name for f in self.float_models_base_dir.iterdir() if f.suffix == '.pth' and f.is_file()]
        return sorted(models)

    def _initialize_float_core(self, selected_model_option: str):
        # 根据 selected_model_option 确定实际要加载的模型文件名
        # 如果用户选择了下载提示项，我们仍然尝试加载 DEFAULT_FLOAT_MODEL_NAME
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
        
        # 如果下载后模型仍然缺失，或者下载成功但选中的模型不是 float.pth，并且 float.pth 仍然缺失，需要特别处理
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
        # 传入的 `model` 参数是 UI 选择的字符串，它可能是真实模型名，也可能是下载占位符。
        # 在 _initialize_float_core 内部会处理这个字符串，并总是尝试加载 DEFAULT_FLOAT_MODEL_NAME。
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
        # 如果在 INPUT_TYPES 中添加了 nfe 参数，这里也需要更新：
        # self.inference_core.opt.nfe = nfe_from_input 
        # 否则，使用 BaseOptionsJson 默认的 nfe=10

        temp_working_dir = self.comfyui_root / "temp" / "float_animator_tmp"
        os.makedirs(temp_working_dir, exist_ok=True)
        
        timestamp_pid = f"{int(time.time())}_{os.getpid()}_{random.randint(0, 9999)}"
        audio_temp_path = temp_working_dir / f"input_audio_{timestamp_pid}.wav"
        image_temp_path = temp_working_dir / f"reference_image_{timestamp_pid}.png"

        try:
            # 确保音频波形是单声道且浮点类型
            if audio['waveform'].dim() == 3: # (Batch, Channels, Samples)
                audio_waveform_to_save = audio['waveform'].squeeze(0) # Remove batch dim
            elif audio['waveform'].dim() == 2: # (Channels, Samples)
                audio_waveform_to_save = audio['waveform']
            else:
                raise ValueError("[ComfyUI_Float_Animator] 不支持的音频波形维度。Expected (Batch, Channels, Samples) or (Channels, Samples).")
            
            # 如果是多声道，转为单声道（取平均）
            if audio_waveform_to_save.shape[0] > 1:
                audio_waveform_to_save = torch.mean(audio_waveform_to_save, dim=0, keepdim=True)
            
            if audio_waveform_to_save.dtype != torch.float32:
                 audio_waveform_to_save = audio_waveform_to_save.to(torch.float32)
            
            # 归一化音频到 [-1, 1]
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
                nfe=10, # 使用 BaseOptionsJson 默认值，如果需要用户控制请在 INPUT_TYPES 中添加参数
                no_crop=not auto_crop,
                seed=seed,
                verbose=True
            )
            print(f"[ComfyUI_Float_Animator] FLOAT 推理完成。输出帧形状: {animated_output_frames.shape}")

        finally:
            # 清理临时文件
            if os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
            if os.path.exists(image_temp_path):
                os.remove(image_temp_path)

            # 模型卸载到CPU以释放VRAM
            if self.inference_core and self.inference_core.G:
                self.inference_core.G.to(mm.unet_offload_device())
                mm.soft_empty_cache()

        return (animated_output_frames, audio, fps,)


# 将节点类添加到 ComfyUI 的映射中
NODE_CLASS_MAPPINGS = {
    "Float_Animator": Float_Animator,
}