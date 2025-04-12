from .hidreamsampler import HiDreamSampler, HiDreamSamplerAdvanced
from .nodes import HiDreamKSampler, HiDreamClipTextEncode, HiDreamVAELoader, HiDreamTELoader, HiDreamDMLoader
"""
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler",
    "HiDreamSamplerAdvanced": "HiDream Sampler (Advanced)"
}
"""
NODE_CLASS_MAPPINGS = {
    "HiDreamKSampler": HiDreamKSampler,
    "HiDreamClipTextEncode": HiDreamClipTextEncode,
    "HiDreamVAELoader": HiDreamVAELoader,
    "HiDreamTELoader": HiDreamTELoader,
    "HiDreamDMLoader": HiDreamDMLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamKSampler": "HiDream KSampler",
    "HiDreamClipTextEncode": "HiDream Clip Text Encode",
    "HiDreamVAELoader": "HiDream VAE Loader",
    "HiDreamTELoader": "HiDream Load CLIP",
    "HiDreamDMLoader": "HiDream Load diffusion model"
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
