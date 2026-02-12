import torch
import folder_paths
import comfy.sd
import comfy.utils
from safetensors.torch import load_file, save_file
import logging
import os
import sys
import re
import glob
from unittest.mock import patch

# ==============================================================================
# 1. MATH HELPERS
# ==============================================================================

def robust_matmul(a, b):
    if len(a.shape) == 4 and a.shape[2] == 1 and a.shape[3] == 1: a = a.squeeze(3).squeeze(2)
    if len(b.shape) == 4 and b.shape[2] == 1 and b.shape[3] == 1: b = b.squeeze(3).squeeze(2)

    if a.shape[-1] == b.shape[0]: return a @ b
    elif b.shape[-1] == a.shape[0]: return b @ a
    elif a.shape[0] == b.shape[0]: return a.T @ b
    elif a.shape[-1] == b.shape[-1]: return a @ b.T
    else: return None

def make_kron(w1, w2, scale):
    if len(w2.shape) == 4: w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2) * scale

def make_hada(w1a, w1b, w2a, w2b, scale):
    w1 = robust_matmul(w1a, w1b)
    w2 = robust_matmul(w2a, w2b)
    if w1 is None or w2 is None: return None
    return (w1 * w2) * scale

def make_lora(wa, wb, scale):
    res = robust_matmul(wa, wb)
    if res is None: return None
    return res * scale

# ==============================================================================
# 2. PATCHING ENGINES
# ==============================================================================

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def group_lora_keys(lora_sd):
    modules = {}
    suffixes = [
        ".lora_A.weight", ".lora_B.weight",
        ".lora_up.weight", ".lora_down.weight",
        ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
        ".lokr_w1", ".lokr_w2",
        ".alpha"
    ]

    for key, value in lora_sd.items():
        base = key
        param_name = None
        for s in suffixes:
            if key.endswith(s):
                base = key[:-len(s)]
                param_name = s.strip(".")
                break
        
        if param_name is None:
            if key.endswith(".weight"):
                base = key[:-7]
                param_name = "weight"
            else:
                continue

        if base not in modules:
            modules[base] = {}
        modules[base][param_name] = value
        
    return modules

def apply_lycoris_to_dict(target_dict, lora_path, strength, is_clip=False):
    if strength == 0 or target_dict is None: return 0
    filename = os.path.basename(lora_path)
    logging.info(f"Applying LyCORIS Patch: {filename}")
    lora_sd = load_file(lora_path)
    
    modules = group_lora_keys(lora_sd)
    sorted_prefixes = sorted(modules.keys(), key=natural_sort_key)
    
    target_linear_keys = []
    for k in sorted(target_dict.keys(), key=natural_sort_key):
        v = target_dict[k]
        if k.endswith(".weight") and len(v.shape) >= 2:
            target_linear_keys.append(k)

    patch_count = 0
    used_targets = set()

    for i, prefix in enumerate(sorted_prefixes):
        params = modules[prefix]
        diff = None
        alpha = params.get("alpha", None)
        
        try:
            if "hada_w1_a" in params:
                w1a, w1b = params["hada_w1_a"].float(), params["hada_w1_b"].float()
                w2a, w2b = params["hada_w2_a"].float(), params["hada_w2_b"].float()
                alpha_val = float(alpha) if alpha else float(min(w1a.shape))
                base_scale = alpha_val / min(w1a.shape)
                diff = make_hada(w1a, w1b, w2a, w2b, base_scale * strength)
            elif "lokr_w1" in params:
                w1, w2 = params["lokr_w1"].float(), params["lokr_w2"].float()
                dim = w1.shape[0] * w2.shape[0]
                alpha_val = float(alpha) if alpha else float(dim)
                base_scale = alpha_val / dim
                diff = make_kron(w1, w2, base_scale * strength)
            elif "lora_up.weight" in params or "lora_A.weight" in params:
                up_key = "lora_up.weight" if "lora_up.weight" in params else "lora_A.weight"
                down_key = "lora_down.weight" if "lora_down.weight" in params else "lora_B.weight"
                
                if "lora_A.weight" in params and "lora_B.weight" in params:
                    down = params["lora_A.weight"].float()
                    up = params["lora_B.weight"].float()
                else:
                    up = params[up_key].float()
                    down = params[down_key].float()

                alpha_val = float(alpha) if alpha else float(down.shape[0])
                base_scale = alpha_val / down.shape[0]
                diff = make_lora(up, down, base_scale * strength)
        except Exception:
            continue

        if diff is None: continue

        applied = False
        clean_prefix = prefix.replace("lora_unet_", "").replace("lora_te_", "").replace("transformer.", "")
        
        candidates = []
        candidates.append(f"{clean_prefix}.weight")
        
        if is_clip:
            candidates.append(f"text_encoder.{clean_prefix}.weight")
            candidates.append(f"model.{clean_prefix}.weight")
            candidates.append(f"transformer.{clean_prefix}.weight")
            if "text_model" in clean_prefix:
                candidates.append(f"{clean_prefix.replace('text_model.', '')}.weight")
        else:
            candidates.append(f"diffusion_model.{clean_prefix}.weight")
            candidates.append(f"model.{clean_prefix}.weight")
            candidates.append(f"transformer.{clean_prefix}.weight")
            candidates.append(f"model.diffusion_model.{clean_prefix}.weight")
            if "blocks" in clean_prefix:
                swapped = clean_prefix.replace("blocks", "layers")
                candidates.append(f"{swapped}.weight")
                candidates.append(f"model.{swapped}.weight")
                candidates.append(f"transformer.{swapped}.weight")

        for target_key in candidates:
            if target_key in target_dict:
                target_param = target_dict[target_key]
                if target_param.shape == diff.shape:
                    target_dict[target_key] = target_param + diff.to(target_param.dtype)
                    patch_count += 1
                    applied = True
                    used_targets.add(target_key)
                    break
        
        if not applied and ("modules" in prefix or "blocks" in prefix):
            for t_key in target_linear_keys:
                if t_key in used_targets: continue
                target_param = target_dict[t_key]
                if target_param.shape == diff.shape or target_param.shape == diff.T.shape:
                    if target_param.shape == diff.T.shape:
                        diff = diff.T
                    target_dict[t_key] = target_param + diff.to(target_param.dtype)
                    patch_count += 1
                    applied = True
                    used_targets.add(t_key)
                    break

    return patch_count

def apply_aitk_to_dict(target_dict, lora_path, strength, is_clip=False):
    if strength == 0 or target_dict is None: return 0
    filename = os.path.basename(lora_path)
    logging.info(f"Applying AITK Patch: {filename}")
    lora_sd = load_file(lora_path)
    
    modules = group_lora_keys(lora_sd)
    patch_count = 0
    
    for prefix, params in modules.items():
        try:
            if "lora_A.weight" in params and "lora_B.weight" in params:
                down = params["lora_A.weight"].float()
                up = params["lora_B.weight"].float()
            elif "lora_up.weight" in params and "lora_down.weight" in params:
                up = params["lora_up.weight"].float()
                down = params["lora_down.weight"].float()
            else:
                continue

            alpha = params.get("alpha", None)
            alpha_val = float(alpha) if alpha else float(down.shape[0])
            scale = (alpha_val / down.shape[0]) * strength
            
            diff = make_lora(up, down, scale)
            if diff is None: continue
        except Exception:
            continue

        target_keys = []
        if is_clip:
            if prefix.startswith("lora_te."):
                bare = prefix.replace("lora_te.", "")
                target_keys.append(f"transformer.{bare}.weight")
                target_keys.append(f"{bare}.weight")
        else:
            if "lora_unet" in prefix:
                bare = prefix.replace("lora_unet_", "").replace("lora_unet.", "")
                target_keys.append(f"{bare}.weight")
                target_keys.append(f"diffusion_model.{bare}.weight")
                if "blocks" in bare:
                    swapped = bare.replace("blocks", "layers")
                    target_keys.append(f"{swapped}.weight")
                    target_keys.append(f"diffusion_model.{swapped}.weight")

        for t_key in target_keys:
            if t_key in target_dict:
                w = target_dict[t_key]
                if w.shape == diff.shape:
                    target_dict[t_key] = w + diff.to(w.dtype)
                    patch_count += 1
                    break
    
    return patch_count

def merge_raw_state_dicts(sd_a, sd_b, strength):
    m = {}
    keys = set(sd_a.keys()) | set(sd_b.keys())
    for k in keys:
        if k in sd_a and k in sd_b:
            wa = sd_a[k]
            wb = sd_b[k]
            if wa.shape == wb.shape:
                res = wa.to(dtype=torch.float32) * (1.0 - strength) + wb.to(dtype=torch.float32) * strength
                m[k] = res.to(dtype=wa.dtype)
            else:
                logging.warning(f"Merge shape mismatch for {k}. Keeping A.")
                m[k] = wa
        elif k in sd_a:
            m[k] = sd_a[k]
        else:
            m[k] = sd_b[k]
    return m

# ==============================================================================
# 3. NODES
# ==============================================================================

class ZImageRawWrapper:
    def __init__(self, sd, path):
        self.sd = sd
        self.path = path

class ZImageAITKLoRALoader:
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
            },
            "optional": {
                "lora_stack": ("LYCORIS_STACK", )
            }
        }

    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP")
    FUNCTION = "load_and_patch_aitk"
    CATEGORY = "Z-Image/Loaders"

    def load_and_patch_aitk(self, transformer_name, text_encoder_name, lora_name, strength_model, strength_clip, lora_stack=None):
        unet_sd = None
        clip_sd = None
        unet_path = None
        clip_path = None

        # -------------------------------------------------
        # Load Base Model / CLIP
        # -------------------------------------------------
        if transformer_name != "None":
            unet_path = folder_paths.get_full_path("diffusion_models", transformer_name)
            logging.info(f"Loading Model: {os.path.basename(unet_path)}")
            unet_sd = load_file(unet_path)
        
        if text_encoder_name != "None":
            clip_path = folder_paths.get_full_path("text_encoders", text_encoder_name)
            if not clip_path:
                clip_path = folder_paths.get_full_path("checkpoints", text_encoder_name)
            logging.info(f"Loading CLIP: {os.path.basename(clip_path)}")
            clip_sd = load_file(clip_path)

        # -------------------------------------------------
        # Build Patch Job List (STACK SUPPORT ADDED HERE)
        # -------------------------------------------------
        jobs = []

        # Add stacked LoRAs first (if provided)
        if lora_stack:
            jobs.extend(lora_stack)

        # Add single LoRA input
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path:
                jobs.append({
                    "path": lora_path,
                    "str_model": strength_model,
                    "str_clip": strength_clip
                })

        # -------------------------------------------------
        # Apply All AITK LoRAs Sequentially
        # -------------------------------------------------
        if jobs:
            logging.info(f"Applying {len(jobs)} AITK LoRA Patches...")

            for job in jobs:
                lora_path = job["path"]

                if job["str_model"] != 0 and unet_sd:
                    c = apply_aitk_to_dict(unet_sd, lora_path, job["str_model"], is_clip=False)
                    logging.info(f"AITK Model Patched ({os.path.basename(lora_path)}): {c} layers")

                if job["str_clip"] != 0 and clip_sd:
                    c = apply_aitk_to_dict(clip_sd, lora_path, job["str_clip"], is_clip=True)
                    logging.info(f"AITK CLIP Patched ({os.path.basename(lora_path)}): {c} layers")

        raw_model = ZImageRawWrapper(unet_sd, unet_path) if unet_sd else None
        raw_clip = ZImageRawWrapper(clip_sd, clip_path) if clip_sd else None
        
        return (raw_model, raw_clip)

class ZImageDiffusersLoader:
    """
    Robust loader for Diffusers folders with automatic key conversion.
    Matches z_image_convert_original_to_comfy.py logic internally.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "load_transformer": ("BOOLEAN", {"default": True}),
                "load_text_encoder": ("BOOLEAN", {"default": True}),
                "load_vae": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP", "VAE")
    FUNCTION = "load_diffusers"
    CATEGORY = "Z-Image/Loaders"

    def _load_sharded_or_single(self, folder):
        index_files = glob.glob(os.path.join(folder, "*.index.json"))
        if index_files:
            logging.info(f"Z-Image: Detected sharded model in {folder}")
            combined_sd = {}
            shards = glob.glob(os.path.join(folder, "*.safetensors"))
            if not shards: shards = glob.glob(os.path.join(folder, "*.bin"))
            
            for shard in shards:
                if shard.endswith(".safetensors"):
                    sd = load_file(shard)
                else:
                    sd = torch.load(shard, map_location="cpu")
                combined_sd.update(sd)
            return combined_sd, shards[0]

        candidates = [
            "model.safetensors", "model.fp16.safetensors", 
            "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors",
            "diffusion_pytorch_model.bin", "model.bin"
        ]
        for fn in candidates:
            p = os.path.join(folder, fn)
            if os.path.exists(p):
                logging.info(f"Z-Image: Found weights {fn}")
                if fn.endswith(".safetensors"):
                    return load_file(p), p
                else:
                    return torch.load(p, map_location="cpu"), p
        return None, None

    def _convert_diffusers_to_comfy(self, sd):
        """
        Converts keys using a robust two-pass method to ensure fusion works
        regardless of dictionary key order.
        """
        new_sd = {}
        handled_keys = set()
        
        # Mappings from user's script + standard maps
        replace_keys = {
            "all_final_layer.2-1.": "final_layer.",
            "all_x_embedder.2-1.": "x_embedder.",
            ".attention.to_out.0.bias": ".attention.out.bias",
            ".attention.norm_k.weight": ".attention.k_norm.weight",
            ".attention.norm_q.weight": ".attention.q_norm.weight",
            ".attention.to_out.0.weight": ".attention.out.weight"
        }

        # --- Pass 1: QKV Fusion (Scan for to_q) ---
        for k in sd.keys():
            if ".attention.to_q.weight" in k:
                k_q = k
                k_k = k.replace(".attention.to_q.weight", ".attention.to_k.weight")
                k_v = k.replace(".attention.to_q.weight", ".attention.to_v.weight")
                
                if k_k in sd and k_v in sd:
                    try:
                        # Weights
                        w_q, w_k, w_v = sd[k_q], sd[k_k], sd[k_v]
                        fused_w = torch.cat([w_q, w_k, w_v], dim=0)
                        
                        # Generate output key
                        # 1. Apply replacements to base key (k_q)
                        clean_k = k_q
                        for r, rr in replace_keys.items():
                            clean_k = clean_k.replace(r, rr)
                        # 2. Strip prefixes
                        for p in ["transformer.", "unet.", "diffusion_model."]:
                            if clean_k.startswith(p): clean_k = clean_k[len(p):]
                        # 3. Swap suffix
                        final_key_w = clean_k.replace(".attention.to_q.weight", ".attention.qkv.weight")
                        
                        new_sd[final_key_w] = fused_w
                        handled_keys.update([k_q, k_k, k_v])
                        
                        # Biases (Optional)
                        k_q_b = k_q.replace("weight", "bias")
                        k_k_b = k_k.replace("weight", "bias")
                        k_v_b = k_v.replace("weight", "bias")
                        
                        if k_q_b in sd and k_k_b in sd and k_v_b in sd:
                            b_q, b_k, b_v = sd[k_q_b], sd[k_k_b], sd[k_v_b]
                            fused_b = torch.cat([b_q, b_k, b_v], dim=0)
                            final_key_b = final_key_w.replace("weight", "bias")
                            new_sd[final_key_b] = fused_b
                            handled_keys.update([k_q_b, k_k_b, k_v_b])
                            
                    except Exception as e:
                        logging.warning(f"Z-Image: QKV Fusion failed for {k}: {e}")

        # --- Pass 2: Process Remaining Keys ---
        for k in sd.keys():
            if k in handled_keys: continue
            
            # Apply replacements
            clean_k = k
            for r, rr in replace_keys.items():
                clean_k = clean_k.replace(r, rr)
            
            # Regex Cleaning (artifacts)
            clean_k = re.sub(r'^all_', '', clean_k)
            clean_k = re.sub(r'\.\d+-\d+', '', clean_k) # .2-1
            
            # Prefix Stripping
            for p in ["transformer.", "unet.", "diffusion_model."]:
                if clean_k.startswith(p): 
                    clean_k = clean_k[len(p):]
                    break
            
            # Text Encoder
            if clean_k.startswith("text_model."): clean_k = clean_k.replace("text_model.", "")
                
            new_sd[clean_k] = sd[k]
            
        return new_sd

    def load_diffusers(self, folder_path, load_transformer, load_text_encoder, load_vae):
        folder_path = folder_path.strip().strip('"').strip("'")
        
        raw_model = None
        raw_clip = None
        vae = None

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Diffusers folder not found: {folder_path}")

        # --- 1. Transformer / UNet ---
        if load_transformer:
            sd = None
            path = None
            search_dirs = [os.path.join(folder_path, "transformer"), os.path.join(folder_path, "unet"), folder_path]
            for d in search_dirs:
                if os.path.isdir(d):
                    sd, path = self._load_sharded_or_single(d)
                    if sd: break
            
            if sd:
                sd = self._convert_diffusers_to_comfy(sd)
                raw_model = ZImageRawWrapper(sd, path)
            else:
                raise FileNotFoundError(f"Z-Image: Could not find Transformer/UNet weights in {folder_path}. Checked: {search_dirs}")

        # --- 2. Text Encoder ---
        if load_text_encoder:
            sd = None
            path = None
            search_dirs = [os.path.join(folder_path, "text_encoder"), os.path.join(folder_path, "text_encoder_2"), folder_path]
            for d in search_dirs:
                if os.path.isdir(d):
                    sd, path = self._load_sharded_or_single(d)
                    if sd: break
            
            if sd:
                new_sd = {}
                for k, v in sd.items():
                    new_k = k
                    if new_k.startswith("text_model."): new_k = new_k.replace("text_model.", "")
                    new_sd[new_k] = v
                raw_clip = ZImageRawWrapper(new_sd, path)
            else:
                raise FileNotFoundError(f"Z-Image: Could not find Text Encoder weights in {folder_path}. Checked: {search_dirs}")

        # --- 3. VAE ---
        if load_vae:
            path = None
            search_dirs = [os.path.join(folder_path, "vae"), folder_path]
            for d in search_dirs:
                if os.path.isdir(d):
                    _, path = self._load_sharded_or_single(d) 
                    if path: break
            
            if path:
                vae = comfy.sd.load_vae(path)
            else:
                raise FileNotFoundError(f"Z-Image: Could not find VAE weights in {folder_path}")

        return (raw_model, raw_clip, vae)

class ZImageLoaderAndPatcher:
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
            },
            "optional": {"lora_stack": ("LYCORIS_STACK", )}
        }
    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP")
    FUNCTION = "load_and_patch_raw"
    CATEGORY = "Z-Image/Loaders"

    def load_and_patch_raw(self, transformer_name, text_encoder_name, lora_name, strength_model, strength_clip, lora_stack=None):
        unet_sd = None
        clip_sd = None
        unet_path = None
        clip_path = None

        if transformer_name != "None":
            unet_path = folder_paths.get_full_path("diffusion_models", transformer_name)
            logging.info(f"Loading Model: {os.path.basename(unet_path)}")
            unet_sd = load_file(unet_path)
        
        if text_encoder_name != "None":
            clip_path = folder_paths.get_full_path("text_encoders", text_encoder_name)
            if not clip_path: clip_path = folder_paths.get_full_path("checkpoints", text_encoder_name)
            logging.info(f"Loading CLIP: {os.path.basename(clip_path)}")
            clip_sd = load_file(clip_path)

        jobs = []
        if lora_stack: jobs.extend(lora_stack)
        if lora_name != "None":
            path = folder_paths.get_full_path("loras", lora_name)
            if path: jobs.append({"path": path, "str_model": strength_model, "str_clip": strength_clip})

        if jobs:
            logging.info(f"Applying {len(jobs)} Manual Patches...")
            for job in jobs:
                if job["str_model"] != 0 and unet_sd:
                    c = apply_lycoris_to_dict(unet_sd, job["path"], job["str_model"], is_clip=False)
                    logging.info(f"Model Layers Patched: {c}")
                if job["str_clip"] != 0 and clip_sd:
                    c = apply_lycoris_to_dict(clip_sd, job["path"], job["str_clip"], is_clip=True)
                    logging.info(f"CLIP Layers Patched: {c}")

        raw_model = ZImageRawWrapper(unet_sd, unet_path) if unet_sd else None
        raw_clip = ZImageRawWrapper(clip_sd, clip_path) if clip_sd else None
        
        return (raw_model, raw_clip)

class ZImageLycorisStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "input_stack": ("LYCORIS_STACK", ),
            }
        }
    RETURN_TYPES = ("LYCORIS_STACK",)
    FUNCTION = "stack_lora"
    CATEGORY = "Z-Image/Loaders"

    def stack_lora(self, lora_name, strength_model, strength_clip, input_stack=None):
        stack = []
        if input_stack: stack.extend(input_stack)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            stack.append({"path": lora_path, "str_model": strength_model, "str_clip": strength_clip})
        return (stack,)

class ZImageRawModelMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_a": ("RAW_MODEL",),
                "model_b": ("RAW_MODEL",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("RAW_MODEL",)
    FUNCTION = "merge"
    CATEGORY = "Z-Image/Loaders"

    def merge(self, model_a, model_b, strength):
        if not model_a or not model_b:
            return (model_a if model_a else model_b,)
        
        logging.info(f"Merging models with strength {strength}")
        merged_sd = merge_raw_state_dicts(model_a.sd, model_b.sd, strength)
        return (ZImageRawWrapper(merged_sd, "merged_model.safetensors"),)

class ZImageRawClipMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_a": ("RAW_CLIP",),
                "clip_b": ("RAW_CLIP",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("RAW_CLIP",)
    FUNCTION = "merge"
    CATEGORY = "Z-Image/Loaders"

    def merge(self, clip_a, clip_b, strength):
        if not clip_a or not clip_b:
            return (clip_a if clip_a else clip_b,)
            
        logging.info(f"Merging CLIPs with strength {strength}")
        merged_sd = merge_raw_state_dicts(clip_a.sd, clip_b.sd, strength)
        return (ZImageRawWrapper(merged_sd, "merged_clip.safetensors"),)

class ZImageComfyInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "raw_model": ("RAW_MODEL",),
                "raw_clip": ("RAW_CLIP",),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "inject"
    CATEGORY = "Z-Image/Loaders"

    def inject(self, raw_model=None, raw_clip=None):
        m, c = None, None
        
        if raw_model:
            logging.info("Injecting Diffusion Model...")
            with patch('comfy.utils.load_torch_file', return_value=(raw_model.sd, None)):
                m = comfy.sd.load_diffusion_model(raw_model.path, model_options={})
        
        if raw_clip:
            logging.info("Injecting CLIP...")
            with patch('comfy.utils.load_torch_file', return_value=(raw_clip.sd, None)):
                c = comfy.sd.load_clip(ckpt_paths=[raw_clip.path], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        return (m, c)

class ZImageComfyUninjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }
    RETURN_TYPES = ("RAW_MODEL", "RAW_CLIP")
    FUNCTION = "uninject"
    CATEGORY = "Z-Image/Loaders"

    def bake_patches(self, patcher, sd):
        out_sd = {}
        for key, weight in sd.items():
            if key in patcher.patches:
                patches = patcher.patches[key]
                w = weight.to("cpu", copy=True)
                try:
                    for p in patches:
                        patch_weights = p[1]
                        patch_strength = p[2]
                        if hasattr(patch_weights, "shape") and patch_weights.shape == w.shape:
                             w += patch_weights.to("cpu") * patch_strength
                        elif isinstance(patch_weights, dict) and "diff" in patch_weights:
                             w += patch_weights["diff"].to("cpu") * patch_strength
                except Exception:
                    pass
                out_sd[key] = w
            else:
                out_sd[key] = weight
        return out_sd

    def sanitize_keys(self, sd, is_clip=False):
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if is_clip:
                new_k = new_k.replace("cond_stage_model.", "") # Unwrap
            else:
                new_k = new_k.replace("model.diffusion_model.", "")
            new_sd[new_k] = v
        return new_sd

    def uninject(self, model=None, clip=None):
        raw_model = None
        raw_clip = None

        if model:
            try:
                sd = model.model.state_dict()
                if hasattr(model, "patcher"): sd = self.bake_patches(model.patcher, sd)
                sanitized_sd = self.sanitize_keys(sd, is_clip=False)
                raw_model = ZImageRawWrapper(sanitized_sd, "uninject_model.safetensors")
            except Exception as e:
                logging.error(f"Z-Image Uninjector: Failed Model: {e}")

        if clip:
            try:
                sd = clip.cond_stage_model.state_dict() if hasattr(clip, "cond_stage_model") else clip.state_dict()
                if hasattr(clip, "patcher"): sd = self.bake_patches(clip.patcher, sd)
                sanitized_sd = self.sanitize_keys(sd, is_clip=True)
                raw_clip = ZImageRawWrapper(sanitized_sd, "uninject_clip.safetensors")
            except Exception as e:
                logging.error(f"Z-Image Uninjector: Failed CLIP: {e}")

        return (raw_model, raw_clip)

class ZImageSaveTransformer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"raw_model": ("RAW_MODEL",), "filename_prefix": ("STRING", {"default": "z_image_patched"})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Z-Image/Saving"
    def save(self, raw_model, filename_prefix):
        if not raw_model: return {}
        path = os.path.join(folder_paths.get_output_directory(), "diffusion_models", f"{filename_prefix}.safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(raw_model.sd, path)
        return {}

class ZImageSaveTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"raw_clip": ("RAW_CLIP",), "filename_prefix": ("STRING", {"default": "qwen_patched"})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Z-Image/Saving"
    def save(self, raw_clip, filename_prefix):
        if not raw_clip: return {}
        path = os.path.join(folder_paths.get_output_directory(), "text_encoders", f"{filename_prefix}.safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(raw_clip.sd, path)
        return {}

NODE_CLASS_MAPPINGS = {
    "ZImageLoaderAndPatcher": ZImageLoaderAndPatcher,
    "ZImageAITKLoRALoader": ZImageAITKLoRALoader,
    "ZImageComfyInjector": ZImageComfyInjector,
    "ZImageComfyUninjector": ZImageComfyUninjector,
    "ZImageLycorisStacker": ZImageLycorisStacker,
    "ZImageDiffusersLoader": ZImageDiffusersLoader,
    "ZImageRawModelMerge": ZImageRawModelMerge,
    "ZImageRawClipMerge": ZImageRawClipMerge,
    "ZImageSaveTransformer": ZImageSaveTransformer,
    "ZImageSaveTextEncoder": ZImageSaveTextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoaderAndPatcher": "Z-Image Loader & Patcher (LyCORIS/LoHA)",
    "ZImageAITKLoRALoader": "Z-Image AITK LoRA Loader (Standard)",
    "ZImageComfyInjector": "Z-Image Comfy Injector",
    "ZImageComfyUninjector": "Z-Image Comfy Uninjector",
    "ZImageLycorisStacker": "Z-Image Stacker",
    "ZImageDiffusersLoader": "Z-Image Diffusers Loader",
    "ZImageRawModelMerge": "Z-Image Raw Model Merge",
    "ZImageRawClipMerge": "Z-Image Raw CLIP Merge",
    "ZImageSaveTransformer": "Z-Image Save Transformer",
    "ZImageSaveTextEncoder": "Z-Image Save Text Encoder"
}