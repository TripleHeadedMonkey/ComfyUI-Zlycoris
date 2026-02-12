import torch
import folder_paths
import comfy.sd
import comfy.ops
import comfy.model_management
from .loader import gguf_sd_loader, gguf_clip_loader
from .ops import GGMLOps, manual_resize_tensor
from .nodes2 import GGUFModelPatcher
from .Universal_LoRA import ZImageRawWrapper

# ==============================================================================
# 1. STANDALONE GGUF LOADER (Reader + Fixer)
# ==============================================================================
class ZImageGGUFStandaloneLoader:
    @classmethod
    def INPUT_TYPES(s):
        models = folder_paths.get_filename_list("unet_gguf")
        clips = folder_paths.get_filename_list("clip_gguf")
        return {
            "required": {
                "transformer_name": (["None"] + models, ),
                "text_encoder_name": (["None"] + clips, ),
                "type": (
                    [
                        "stable_diffusion", 
                        "stable_diffusion_xl", 
                        "sd3", 
                        "flux", 
                        "qwen_image", 
                        "gemma", 
                        "hunyuan_di_t"
                    ], 
                ),
            }
        }

    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP")
    FUNCTION = "load_gguf_raw"
    CATEGORY = "Z-Image/Loaders"

    def _fix_tensor_padding(self, sd):
        """
        Scans state dict for specific GGUF padding sizes and trims them.
        2720 -> 2560 (Qwen/LLM Hidden Size)
        4352 -> 4096 (Gemma/LLM Hidden Size)
        """
        new_sd = {}
        for k, v in sd.items():
            current_shape = list(v.shape)
            needs_resize = False
            
            # Iterate through all dimensions to find padding
            for i, dim in enumerate(current_shape):
                if dim == 2720:
                    current_shape[i] = 2560
                    needs_resize = True
                elif dim == 4352:
                    current_shape[i] = 4096
                    needs_resize = True
            
            if needs_resize:
                new_v = manual_resize_tensor(v, torch.Size(current_shape))
                new_sd[k] = new_v
            else:
                new_sd[k] = v
        return new_sd

    def load_gguf_raw(self, transformer_name, text_encoder_name, type="stable_diffusion"):
        raw_model = None
        raw_clip = None

        # --- 1. LOAD MODEL (UNET) & FIX SHAPES ---
        if transformer_name != "None":
            path = folder_paths.get_full_path("unet_gguf", transformer_name)
            sd = gguf_sd_loader(path) 
            
            # Apply padding fix if the type warrants it
            if type in ["qwen_image", "gemma", "stable_diffusion"]:
                sd = self._fix_tensor_padding(sd)

            raw_model = ZImageRawWrapper(sd, path)

        # --- 2. LOAD CLIP & FIX SHAPES ---
        if text_encoder_name != "None":
            path = folder_paths.get_full_path("clip_gguf", text_encoder_name)
            sd = gguf_clip_loader(path)

            # Apply padding fix (handles 2720 AND 4352)
            if type in ["qwen_image", "gemma", "stable_diffusion"]:
                sd = self._fix_tensor_padding(sd)

            raw_clip = ZImageRawWrapper(sd, path)

        return (raw_model, raw_clip)

# ==============================================================================
# 2. GGUF INJECTOR (Builder)
# ==============================================================================
class ZImageGGUFInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_model": ("RAW_MODEL",),
                "raw_clip": ("RAW_CLIP",),
                "clip_type": (
                    [
                        "stable_diffusion", 
                        "stable_diffusion_xl", 
                        "sd3", 
                        "flux", 
                        "qwen_image", 
                        "gemma",
                        "hunyuan_di_t"
                    ], 
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "inject_gguf"
    CATEGORY = "Z-Image/Injectors"

    def inject_gguf(self, raw_model, raw_clip, clip_type):
        model = None
        clip = None

        if raw_model is not None:
            sd = raw_model.sd
            
            # We still keep GGMLOps here for the specialized Linear layers
            ops = GGMLOps()

            model_obj = comfy.sd.load_diffusion_model_state_dict(
                sd, 
                model_options={"custom_operations": ops}
            )
            
            model = GGUFModelPatcher.clone(model_obj)

        if raw_clip is not None:
            try:
                clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper())
            except AttributeError:
                clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
            
            clip = comfy.sd.load_text_encoder_state_dicts(
                [raw_clip.sd],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_enum,
                model_options = {
                    "custom_operations": GGMLOps,
                    "initial_device": comfy.model_management.text_encoder_offload_device()
                }
            )
            clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        return (model, clip)

# ==============================================================================
# MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "ZImageGGUFStandaloneLoader": ZImageGGUFStandaloneLoader,
    "ZImageGGUFInjector": ZImageGGUFInjector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageGGUFStandaloneLoader": "Z-Image GGUF Standalone Loader",
    "ZImageGGUFInjector": "Z-Image GGUF Injector"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']