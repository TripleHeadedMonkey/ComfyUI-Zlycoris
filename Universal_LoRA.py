import torch
import folder_paths
from safetensors.torch import load_file
import logging
import os
import re

# Setup logging
file_handler = logging.FileHandler('z_image_universal.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger = logging.getLogger("ZImageUniversal")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

# ==============================================================================
# 1. HELPER: RAW IMAGE WRAPPER
# ==============================================================================
class ZImageRawWrapper:
    def __init__(self, sd, path):
        self.sd = sd
        self.path = path

# ==============================================================================
# 2. MATH HELPERS
# ==============================================================================
def robust_matmul(a, b):
    if len(a.shape) == 4 and a.shape[2] == 1: a = a.squeeze(3).squeeze(2)
    if len(b.shape) == 4 and b.shape[2] == 1: b = b.squeeze(3).squeeze(2)
    
    if a.shape[-1] == b.shape[0]: return a @ b
    elif b.shape[-1] == a.shape[0]: return b @ a
    elif a.shape[0] == b.shape[0]: return a.T @ b
    elif a.shape[-1] == b.shape[-1]: return a @ b.T
    return None

def make_lora(wa, wb, scale):
    res = robust_matmul(wa, wb)
    if res is None: return None
    return res * scale

# ==============================================================================
# 3. UNIVERSAL PATCHING LOGIC
# ==============================================================================
def normalize_key(k):
    """
    Standardizes keys for comparison.
    """
    k = k.replace("model.diffusion_model.", "")
    k = k.replace("diffusion_model.", "").replace("transformer.", "")
    k = k.replace("text_model.", "").replace("model.", "")
    k = k.replace("lora_unet_", "").replace("lora_te_", "")
    k = k.replace("lycoris_all_", "").replace("lycoris_", "")
    
    # Clean Artifacts
    k = k.replace("_2-1", "").replace(".2-1", "")
    k = re.sub(r'(\d+)_\d+', r'\1', k) # Fix "blocks_1_2" -> "blocks_1"
    
    # FIX: Standardize Attention Output
    # Converts "to_out_0", "to_out.0", "to_out" all to "out"
    k = k.replace("to_out_0", "out").replace("to_out.0", "out").replace("to_out", "out")
    
    # Standardize QKV
    k = k.replace("to_k", "k").replace("to_q", "q").replace("to_v", "v")
    k = k.replace("k_proj", "k").replace("q_proj", "q").replace("v_proj", "v")
    k = k.replace("qkv_proj", "qkv")

    # Strip separators
    k = k.replace(".", "").replace("_", "").replace("-", "")
    return k.lower()

def apply_universal_lora(target_dict, lora_path, strength):
    if strength == 0 or target_dict is None: return 0
    
    filename = os.path.basename(lora_path)
    logger.info(f"Applying Universal Patch: {filename}")
    
    lora_sd = load_file(lora_path)
    
    # 1. Group Keys
    modules = {}
    for k, v in lora_sd.items():
        if "lora_up" in k:
            base = k.replace(".lora_up.weight", "")
            param = "up"
        elif "lora_down" in k:
            base = k.replace(".lora_down.weight", "")
            param = "down"
        elif "lora_A" in k:
            base = k.replace(".lora_A.weight", "")
            param = "down" 
        elif "lora_B" in k:
            base = k.replace(".lora_B.weight", "")
            param = "up"   
        elif "alpha" in k:
            base = k.replace(".alpha", "")
            param = "alpha"
        else:
            continue
        if base not in modules: modules[base] = {}
        modules[base][param] = v

    # 2. Map Target Dict
    normalized_target_map = {}
    for k in target_dict.keys():
        if not k.endswith(".weight"): continue
        bare = k[:-7]
        norm = normalize_key(bare)
        normalized_target_map[norm] = k

    patch_count = 0
    skipped_modules = []

    # 3. Patching Loop
    for lora_prefix, params in modules.items():
        if "up" not in params or "down" not in params: continue
        
        up, down = params["up"].float(), params["down"].float()
        alpha = params.get("alpha", None)
        dim = down.shape[0]
        scale = (float(alpha) / dim) * strength if alpha is not None else strength
            
        diff = make_lora(up, down, scale)
        if diff is None: continue

        # --- MATCHING STRATEGY ---
        norm_lora = normalize_key(lora_prefix)
        target_key = normalized_target_map.get(norm_lora)
        
        # Strategy B: QKV Fusion Patching
        slice_idx = None
        
        if not target_key:
            if norm_lora.endswith("q") or norm_lora.endswith("k") or norm_lora.endswith("v"):
                base_qkv = norm_lora[:-1] + "qkv"
                parent_key = normalized_target_map.get(base_qkv)
                if parent_key:
                    target_key = parent_key
                    if norm_lora.endswith("q"): slice_idx = 0
                    elif norm_lora.endswith("k"): slice_idx = 1
                    elif norm_lora.endswith("v"): slice_idx = 2

        if not target_key: 
            skipped_modules.append(lora_prefix)
            continue

        # --- APPLY PATCH ---
        w = target_dict[target_key]
        try:
            # Case 1: Standard Patch
            if w.shape == diff.shape:
                target_dict[target_key] = w + diff.to(w.dtype).to(w.device)
                patch_count += 1
            
            # Case 2: QKV Sliced Patch
            elif slice_idx is not None:
                chunk_size = w.shape[0] // 3
                start = slice_idx * chunk_size
                end = start + chunk_size
                if diff.shape[0] == chunk_size and diff.shape[1] == w.shape[1]:
                    target_dict[target_key][start:end, :] += diff.to(w.dtype).to(w.device)
                    patch_count += 1

            # Case 3: Reshape
            elif len(w.shape) == 4 and diff.dim() == 2:
                 diff_reshaped = diff.reshape(w.shape)
                 target_dict[target_key] = w + diff_reshaped.to(w.dtype).to(w.device)
                 patch_count += 1

        except Exception as e:
            logger.error(f"Patch error {target_key}: {e}")

    logger.info(f"Total Patched: {patch_count} / {len(modules)}")
    
    if len(skipped_modules) > 0:
        logger.info("--- Skipped Modules ---")
        for s in skipped_modules[:5]: 
            logger.info(f"Skipped: {s} (Norm: {normalize_key(s)})")

    return patch_count

# ==============================================================================
# 4. NODE DEFINITION
# ==============================================================================
class ZImageUniversalLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        models = ["None"] + folder_paths.get_filename_list("diffusion_models")
        tes = ["None"] + folder_paths.get_filename_list("text_encoders")
        loras = ["None"] + folder_paths.get_filename_list("loras") 
        return {
            "required": {
                "transformer_name": (models, ),
                "text_encoder_name": (tes, ), 
                "lora_name": (loras, ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP")
    FUNCTION = "load_universal"
    CATEGORY = "Z-Image/Loaders"

    def load_universal(self, transformer_name, text_encoder_name, lora_name, strength_model, strength_clip):
        unet_sd = None
        clip_sd = None
        unet_path = None
        clip_path = None

        if transformer_name != "None":
            unet_path = folder_paths.get_full_path("diffusion_models", transformer_name)
            logger.info(f"Loading Model: {os.path.basename(unet_path)}")
            unet_sd = load_file(unet_path)
        
        if text_encoder_name != "None":
            clip_path = folder_paths.get_full_path("text_encoders", text_encoder_name)
            if not clip_path: clip_path = folder_paths.get_full_path("checkpoints", text_encoder_name)
            logger.info(f"Loading CLIP: {os.path.basename(clip_path)}")
            clip_sd = load_file(clip_path)

        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path:
                if strength_model != 0 and unet_sd:
                    c = apply_universal_lora(unet_sd, lora_path, strength_model)
                    logging.info(f"Universal Model Patch: {c}")
                if strength_clip != 0 and clip_sd:
                    c = apply_universal_lora(clip_sd, lora_path, strength_clip)
                    logging.info(f"Universal CLIP Patch: {c}")

        raw_model = ZImageRawWrapper(unet_sd, unet_path) if unet_sd else None
        raw_clip = ZImageRawWrapper(clip_sd, clip_path) if clip_sd else None
        
        return (raw_model, raw_clip)

NODE_CLASS_MAPPINGS = {
    "ZImageUniversalLoRALoader": ZImageUniversalLoRALoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageUniversalLoRALoader": "Z-Image Universal LoRA Loader"
}