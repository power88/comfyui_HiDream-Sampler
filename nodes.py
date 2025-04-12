# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29c (NF4/FP8/BNB Support, Final Call Arg Fix)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 models: optimum, accelerate, auto-gptq (`pip install optimum accelerate auto-gptq`)
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.
import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
import gc
import os # For checking paths if needed
import importlib.util
from pathlib import Path
import json

# --- Optional Dependency Handling ---
accelerate_available = True
autogptq_available = True
optimum_available = True
bnb_available = True
TransformersBitsAndBytesConfig = None
DiffusersBitsAndBytesConfig = None

if not importlib.util.find_spec("accelerate"):
    print("Warning: accelerate not installed. device_map='auto' for GPTQ models will not be available.")
    accelerate_available = False

if not importlib.util.find_spec("auto_gptq") or importlib.util.find_spec("optimum"):
    print("Warning: auto_gptq and optimum is not installed. Please install it to use quantization module.")
    autogptq_available = False
    optimum_available = False

if not importlib.util.find_spec("BitsAndBytesConfig") or importlib.util.find_spec("BitsAndBytesConfig"):
    bnb_available = False
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None


# --- Core Imports ---
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    LlamaForCausalLM,
    AutoTokenizer
)
from diffusers.models.autoencoders import AutoencoderKL
from comfy.utils import load_torch_file, pil2tensor
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    hidream_classes_loaded = False
    raise ModuleNotFoundError(f"Could not import local hi_diffusers: {e}")
    

# --- Directories ---
extension_directory = os.path.abspath(__file__)
models_directory = os.path.join(os.path.dirname(os.path.dirname(extension_directory)), 'models')
text_encoder_dir = os.path.abspath(os.path.join(models_directory, 'text_encoders'))
clip_dir = os.path.abspath(os.path.join(models_directory, 'clip'))
diffusion_models_dir = os.path.abspath(os.path.join(models_directory, 'diffusion_models'))
vae_dir = os.path.abspath(os.path.join(models_directory, 'vae'))

# --- Configurations ---
TEXT_ENCODER_CONFIGS = {
    "original": {
        "name": "unsloth/Meta-Llama-3.1-8B", # Unsloth's fork of Original Llama-3.1-8B (text generation only, no instruct)
    },
    "uncensored": {
        "name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", # Using the same model as NF4 since it's less censored.
    },
    "nf4": {
        "name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", 
    },
    "nf4-uncensored": {
        "name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", 
    }
}

MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "name": "azaneko/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "dev-nf4": {
        "name": "azaneko/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "fast-nf4": {
        "name": "azaneko/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    # --- Original/BNB Models ---
    "full": {
        "name": "HiDream-ai/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "name": "HiDream-ai/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "name": "HiDream-ai/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}

available_schedulers = {
    "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
    "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler
}

RESOLUTION_OPTIONS = [ 
    "1024 x 1024 (Square)",
    "768 x 1360 (Portrait)",
    "1360 x 768 (Landscape)",
    "880 x 1168 (Portrait)",
    "1168 x 880 (Landscape)",
    "1248 x 832 (Landscape)",
    "832 x 1248 (Portrait)"
]

default_model_dtype = torch.bfloat16 if (torch.cuda.is_avaliable() and torch.cuda.is_bf16_supported()) else torch.float16
bnb_llm_config = None
bnb_transformer_4bit_config = None
if bnb_available:
    bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)



class HiDreamDMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusion_model_name": (list(MODEL_CONFIGS.keys()), {"default": 'full-nf4'}),
            },
            "optional":{
                "dtype": (["fp16", "fp8_e4m3fn", "fp8_e5m2", "INT4"]),
            }
        }
    RETURN_TYPES = ("HiDreamDiT",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "HiDream"
    def load_diffusion_model(self, diffusion_model_name, dtype='INT4'):

        if not MODEL_CONFIGS[diffusion_model_name]["exist"]:
            print("Downloading diffusion model via huggingface...")
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ModuleNotFoundError('Please install huggingface_hub')
            snapshot_download(
                repo_id=MODEL_CONFIGS[diffusion_model_name]['name'],
                local_dir=MODEL_CONFIGS[diffusion_model_name]['path'],
                local_dir_use_symlinks=False
            )
        device = mm.get_torch_device()
        free_memory = mm.get_free_memory()
        if free_memory < 16 * 1024 * 1024 * 1024:
            print('Cleaning cache...')
            mm.unload_all_models()
            mm.soft_empty_cache()

        print('Loading diffusion model...')

        transformer_load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": default_model_dtype,
            "low_cpu_mem_usage": True,
        }
        is_nf4 = ('nf4' in diffusion_model_name)

        if is_nf4:
            print("Type: NF4")
        elif dtype == 'INT4':  # Default BNB case
            print("Type: Standard (Applying 4-bit BNB quantization)")
            if bnb_transformer_4bit_config:
                transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
            else:
                raise ImportError("BNB config required for transformer but unavailable.")
        elif dtype == 'fp8_e4m3fn':
            transformer_load_kwargs["torch_dtype"] = torch.float8_e4m3fn
        elif dtype == 'fp8_e5m2':
            transformer_load_kwargs["torch_dtype"] = torch.float8_e5m2
        else:
            if mm.get_free_memory() < 34211466368: # Value from I1-Full
                print("Warning: You are trying to load full checkpoint in fp16 weight which your free VRAM is not big enough.")
            
        
        print("Loading Transformer...")
        transformer = HiDreamImageTransformer2DModel.from_pretrained(MODEL_CONFIGS[diffusion_model_name]['path'], **transformer_load_kwargs)
        print("Moving Transformer to main device...")
        transformer.to(device)

        # scheduler        
        scheduler_name = MODEL_CONFIGS[diffusion_model_name]["scheduler_class"]
        shift_value = MODEL_CONFIGS[diffusion_model_name]["shift"]

        scheduler_class = available_schedulers.get(scheduler_name)
        if scheduler_class is None:
            raise ValueError(f"Scheduler class '{scheduler_name}' not found...")
        
        scheduler = scheduler_class(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
        pipe_config = {
            "diffusion_model": transformer,
            "diffusion_path": MODEL_CONFIGS[diffusion_model_name]['path'],
            "shift": MODEL_CONFIGS[diffusion_model_name]['shift'],
            "scheduler": scheduler
        }
        
        return (pipe_config,)

class HiDreamTELoader:
    @classmethod
    def INPUT_TYPES(s):
        text_encoder_names = list(TEXT_ENCODER_CONFIGS.keys()).append('custom')
        clip_names = list(os.path.basename(clip_file) for clip_file in os.listdir(clip_dir))
        return {
            "required": {
                "llama_text_encoder_name": (text_encoder_names, {"default": 'nf4'}),
                "clip_g": (clip_names, {"default": 'clip_g'}),
                "clip_l": (clip_names, {"default": 'clip_l'}),
                "t5_model": (clip_names, )
            },
            "optional":{
                "custom_text_encoder": ("STRING"),
                "dtype": (["fp16", "INT8", "INT4"])
            }
        }
    RETURN_TYPES = ("HiDreamTE",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "HiDream"

    def load_text_encoder(self, llama_text_encoder_name, clip_g, clip_l, t5_model, custom_text_encoder=None, dtype='INT4'):
        if llama_text_encoder_name == 'custom':
            if custom_text_encoder == None:
                raise ValueError("Please choose at least one text encoder.")
            if len(custom_text_encoder.split('/')) < 2:
                raise ValueError("Please provide the custom_text_encoder in huggingface format. eg.'meta-llama/Llama-3.1-8B'")
            if not os.path.exists(os.path.join(llama_text_encoder, str(custom_text_encoder.split("/")[1]))):
                print("Downloading text encoder via huggingface...")
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    raise ModuleNotFoundError('Please install huggingface_hub')
                snapshot_download(
                    repo_id=custom_text_encoder,
                    local_dir=os.path.join(llama_text_encoder, str(custom_text_encoder.split("/")[1])),
                    local_dir_use_symlinks=False
                )
            use_custom_text_encoder = True

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()


        if not TEXT_ENCODER_CONFIGS[llama_text_encoder_name]["exist"]:
            print("Downloading text encoder via huggingface...")
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ModuleNotFoundError('Please install huggingface_hub')
            snapshot_download(
                repo_id=TEXT_ENCODER_CONFIGS[llama_text_encoder_name]['name'],
                local_dir=TEXT_ENCODER_CONFIGS[llama_text_encoder_name]['path'],
                local_dir_use_symlinks=False
            )
        
        print('Loading text encoder...')

        text_encoder_load_kwargs = {
            "output_hidden_states": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": default_model_dtype,
        }
        is_nf4 = ('nf4' in llama_text_encoder_name)
        if is_nf4:
            if accelerate_available:
                text_encoder_load_kwargs["device_map"] = "auto"
            else:
                print("Warning: accelerate not found, attempting manual placement.")
        elif dtype == 'INT4':
            if bnb_llm_config:
                text_encoder_load_kwargs["quantization_config"] = TransformersBitsAndBytesConfig(load_in_4bit=True)
                print("Using 4-bit BNB.")
            else:
                raise ImportError("BNB config required for standard LLM.")
        elif dtype == 'INT8':
            if bnb_llm_config:
                text_encoder_load_kwargs["quantization_config"] = TransformersBitsAndBytesConfig(load_in_8bit=True)
                print("Using 8-bit BNB.")
            else:
                raise ImportError("BNB config required for standard LLM.")
        else:
            if accelerate_available:
                text_encoder_load_kwargs["device_map"] = "auto"
            else:
                print("Warning: accelerate not found, attempting manual placement.")
            
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
        
        if use_custom_text_encoder:
            llama_text_encoder_name = custom_text_encoder
            text_encoder_path = os.path.join(llama_text_encoder, str(custom_text_encoder.split("/")[1]))
        else:
            llama_text_encoder_name = TEXT_ENCODER_CONFIGS[llama_text_encoder_name]["name"]
            text_encoder_path = TEXT_ENCODER_CONFIGS[llama_text_encoder_name]["path"]
        
        print(f"Loading Tokenizer and Text Encoder: {llama_text_encoder_name}...")

        llama_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)
        llama_text_encoder = LlamaForCausalLM.from_pretrained(text_encoder_path, **text_encoder_load_kwargs)
        
        if "device_map" not in text_encoder_load_kwargs:
            print("Moving text encoder to offload device...")
            llama_text_encoder.to(offload_device)
        
        print("Loading CLIP and T5...")
        clip_paths = list(os.path.abspath(clip_file) for clip_file in os.listdir(clip_dir))
        clips = {}
        for clip_file in clip_paths:
            clips[os.path.basename(clip_file)] = clip_file
        
        clip_g_path = clips[clip_g]
        clip_l_path = clips[clip_l]
        t5_path = clips[t5_model]

        with open(os.path.join(extension_directory, "configs", "clip_g_config.json")) as f:
            clip_g_config = json.load(f)
        with open(os.path.join(extension_directory, "configs", "clip_l_config.json")) as f:
            clip_l_config = json.load(f)
        with open(os.path.join(extension_directory, "configs", "t5_config.json")) as f:
            t5_config = json.load(f)
        
        clip_g_sd = load_torch_file(clip_g_path)
        clip_l_sd = load_torch_file(clip_l_path)
        t5_sd = load_torch_file(t5_path)

        clip_g_te = CLIPTextModelWithProjection.from_config(clip_g_config)
        clip_l_te = CLIPTextModelWithProjection.from_config(clip_l_config)
        t5_te = T5EncoderModel.from_config(t5_config)

        clip_g_te.load_state_dict(clip_g_sd).eval()
        clip_l_te.load_state_dict(clip_l_sd).eval()
        t5_te.load_state_dict(t5_sd).eval()

        text_encoders = {
            "llama_te": llama_text_encoder,
            "llama_tokenizer": llama_tokenizer,
            "clip_g_te": clip_g_te,
            "clip_l_te": clip_l_te,
            "t5_te":t5_te,
        }

        return(text_encoders,)

class HiDreamVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        vae_names = list(os.path.basename(vae_file) for vae_file in os.listdir(vae_dir))
        return {
            "required": {
                "vae_name": (vae_names, {"tooltip": "Flux vae only'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            }
        }
    RETURN_TYPES = ("HiDreamVAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "HiDream"

    def load_vae(self, vae_name, precision):
        vae_paths = list(os.path.abspath(vae_file) for vae_file in os.listdir(vae_dir))
        vaes = {}
        for vae_file in vae_paths:
            vaes[os.path.basename(vae_file)] = vae_file
        
        vae_model_path = vaes[vae_name]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        with open(os.path.join(extension_directory, "configs", "vae_config.json")) as f:
            vae_config = json.load(f)

        vae_sd = load_torch_file(vae_model_path)
        vae = AutoencoderKL.from_config(vae_config).to(dtype).to(offload_device)

        vae.load_state_dict(vae_sd).requires_grad_(False).eval()

        return(vae,)

class HiDreamClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "CLIP": ("HiDreamTE", ),
                "positive_prompt": ("STRING", {"multiline": True, "default": "positive prompt"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "negative prompt"})
            },
        }
    RETURN_TYPES = ("HiDreamTextEmbedding",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "prepare_prompt"
    CATEGORY = "HiDream"

    def prepare_prompt(self, CLIP, positive_prompt, negative_prompt):
        prompt = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "llama_te": CLIP['llama_te'],
            "llama_tokenizer": CLIP['llama_tokenizer'],
            "clip_g_te": CLIP['clip_g_te'],
            "clip_l_te": CLIP['clip_l_te'],
            "t5_te": CLIP['t5_te'],
        }

        return (prompt,)

class HiDreamKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "MODEL":("HiDreamDiT",),
                "CONDITIONING":("HiDreamTextEmbedding",),
                "VAE": ("HiDreamVAE",),
                "LATENT": ("LATENT", {"tooltip": "Now LATENT can be only used for get image size. It's not supported in img2img."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "scheduler": (["default", "UniPC", "Euler", "Euler Karras", "Euler Exponential"], {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "run_pipe"
    CATEGORY = "HiDream"

    def run_pipe(self, MODEL, CONDITIONING, VAE, LATENT, seed, steps, cfg, scheduler):
        # Pipeline setup.
        # TODO: Use __call__ only, no from_pretrained in the future.
        pipe = HiDreamImagePipeline.from_pretrained(
            MODEL['diffusion_path'],
            scheduler=MODEL['scheduler'],
            text_encoder_1=CONDITIONING['clip_l_te'],
            text_encoder_2=CONDITIONING['clip_g_te'],
            text_encoder_3=CONDITIONING['t5_te'],
            tokenizer_4=CONDITIONING['llama_tokenizer'],
            text_encoder_4=CONDITIONING['llama_te'],
            vae=VAE,
            transformer=None,
            torch_dtype=torch.bfloat16 if (torch.cuda.is_avaliable() and torch.cuda.is_bf16_supported()) else torch.float16,
            low_cpu_mem_usage=True
        )
        # Patch scheduler
        original_shift = MODEL['shift']
        if scheduler != "Default":
            if scheduler == "UniPC":
                new_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler Karras":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000, 
                    shift=original_shift, 
                    use_dynamic_shifting=False,
                    use_karras_sigmas=True
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler Exponential":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000, 
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_exponential_sigmas=True
                )
                pipe.scheduler = new_scheduler
        # Patch transformer
        inference_device = mm.get_torch_device()
        pipe.transformer = MODEL['diffusion_model']

        pipe.to(inference_device)
        # TODO: Calculate Sigma to support denosing length.

        # Seed
        generator = torch.Generator(device=inference_device).manual_seed(seed)

        # Try process latent.(The latent has been preprocessed by VAE Encode or EmptyLatentImage)
        # ComfyUI provides (B, C, H, W). No need to convert.
        width, height = LATENT.shape[2], LATENT.shape[1]

        # Almost done. Let's run pipe!
        result_image = pipe(
            prompt=CONDITIONING["positive_prompt"],
            negative_prompt=CONDITIONING["negative_prompt"],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            latents=LATENT,
            output_type="pil"
        )
        tensor = pil2tensor(result_image)
        return(tensor,)


        """
        def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        max_sequence_length_clip_l: Optional[int] = None,
        max_sequence_length_openclip: Optional[int] = None,
        max_sequence_length_t5: Optional[int] = None,
        max_sequence_length_llama: Optional[int] = None,
    ):
        """
