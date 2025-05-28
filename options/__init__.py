# ComfyUI_Float_Animator/options/__init__.py

from types import SimpleNamespace

# BaseOptionsJson will act as a SimpleNamespace to hold model options,
# replacing the argparse.Namespace used in the original generate.py.
class BaseOptionsJson(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize with default values from official base_options.py
        # These are the default values if no specific argument is passed
        self.pretrained_dir = './checkpoints'
        self.seed = 15
        self.fix_noise_seed = False # Original default is False, not an action_store_true

        # video
        self.input_size = 512
        self.input_nc = 3       
        self.fps = 25.0

        # audio
        self.sampling_rate = 16000
        self.audio_marcing = 2        
        self.wav2vec_sec = 2.0
        self.wav2vec_model_path = './checkpoints/wav2vec2-base-960h' # These will be overwritten by Node.py
        self.audio2emotion_path = './checkpoints/wav2vec-english-speech-emotion-recognition' # These will be overwritten by Node.py
        self.attention_window = 2

        self.only_last_features = False # Original default is False
        self.average_emotion = False # Original default is False

        # dropout
        self.audio_dropout_prob = 0.1
        self.ref_dropout_prob = 0.1
        self.emotion_dropout_prob = 0.1

        # model Hyper Parameters
        self.style_dim = 512
        self.dim_a = 512
        self.dim_w = 512
        self.dim_h = 1024
        self.dim_m = 20
        self.dim_e = 7

        # option for FMT
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False # Original default is False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0

        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.nfe = 10
        self.torchdiffeq_ode_method = 'euler'
        self.a_cfg_scale = 2.0
        self.e_cfg_scale = 1.0       
        self.r_cfg_scale = 1.0        

        # option for Diffusion (ablation)
        self.n_diff_steps = 500
        self.diff_schedule = 'cosine'
        self.diffusion_mode = 'sample'

        # Apply any kwargs to override defaults
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Additional ComfyUI specific parameters (will be set by Node.py)
        self.rank = None # Will be set by ComfyUI's model management
        self.ngpus = 1 # Will be set by ComfyUI's model management
        self.ckpt_path = None # Will be set by Node.py
        self.res_dir = "./results" # Default output dir, though Node.py won't use it directly for saving