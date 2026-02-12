import os
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora

import logging


def _split_lora_state_dict(sd: dict):
    te_sd = {}
    rest_sd = {}
    for k, v in sd.items():
        if k.startswith("lora_te."):
            te_sd[k] = v
        else:
            rest_sd[k] = v
    return te_sd, rest_sd


def _get_cond_stage_model(clip):
    # In your build, clip has cond_stage_model and that is where the Qwen weights live.
    if hasattr(clip, "cond_stage_model"):
        return clip.cond_stage_model
    return clip


def _infer_qwen_prefix_from_state_dict_keys(keys):
    """
    We saw keys like:
      qwen3_4b.transformer.model.layers.0.self_attn.q_proj.weight

    We want to discover the prefix:
      qwen3_4b.transformer.

    Return something like "qwen3_4b.transformer."
    """
    for k in keys:
        if k.endswith(".weight") and ".transformer.model.layers." in k:
            # everything before "model.layers"
            # e.g. "qwen3_4b.transformer.model.layers..." -> "qwen3_4b.transformer."
            idx = k.find("model.layers.")
            return k[:idx]  # includes trailing dot
    return None


def _build_qwen_te_key_map(cond_stage_model):
    """
    Map ai-toolkit TE LoRA prefixes to actual TE weight keys.

    ai-toolkit LoRA prefix: lora_te.<path_without_.weight>
      e.g. lora_te.model.layers.0.self_attn.q_proj

    actual weight key: <QWEN_PREFIX>model.layers.0.self_attn.q_proj.weight
      e.g. qwen3_4b.transformer.model.layers.0.self_attn.q_proj.weight
    """
    sd = cond_stage_model.state_dict()
    keys = list(sd.keys())

    qwen_prefix = _infer_qwen_prefix_from_state_dict_keys(keys)
    if qwen_prefix is None:
        raise RuntimeError("Could not infer Qwen prefix. Expected keys containing '.transformer.model.layers.'")

    key_map = {}

    # Build mapping for every .weight key under qwen_prefix + "model."
    for k in keys:
        if not k.endswith(".weight"):
            continue
        if not k.startswith(qwen_prefix):
            continue

        # Strip ".weight" and strip leading prefix => path like "model.layers.0.self_attn.q_proj"
        bare = k[:-len(".weight")]
        if not bare.startswith(qwen_prefix):
            continue
        bare_no_prefix = bare[len(qwen_prefix):]  # "model.layers...."

        lora_key = "lora_te." + bare_no_prefix
        key_map[lora_key] = k

    return key_map


class ZImageQwenTELoRALoader:
    """
    Like Load LoRA, but applies ai-toolkit TE LoRA keys (lora_te.model.layers...)
    onto ComfyUI's Qwen TE keyspace (qwen3_4b.transformer.model.layers...).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_name}")

        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        te_sd, rest_sd = _split_lora_state_dict(lora_sd)

        # 1) Apply non-TE LoRA weights normally (diffusion/transformer part)
        model_out, clip_out = comfy.sd.load_lora_for_models(model, clip, rest_sd, strength_model, 0.0)

        # 2) Apply TE LoRA directly to the Qwen cond_stage_model
        if te_sd and abs(strength_clip) > 1e-8:
            try:
                clip_out = clip_out.clone() if hasattr(clip_out, "clone") else clip_out

                cond = _get_cond_stage_model(clip_out)
                key_map = _build_qwen_te_key_map(cond)

                # NOTE: lm_head won't exist in your cond_stage_model state_dict, so it will be skipped naturally.
                patches = comfy.lora.load_lora(te_sd, key_map)

                # Add patches to the CLIP object (patcher exists on clip wrapper)
                # FIXED: strength_model must be 1.0 to preserve the original base weights.
                # Only strength_patch should use the slider value (strength_clip).
                if hasattr(clip_out, "add_patches"):
                    clip_out.add_patches(patches, strength_patch=strength_clip, strength_model=1.0)
                elif hasattr(clip_out, "patcher") and hasattr(clip_out.patcher, "add_patches"):
                    clip_out.patcher.add_patches(patches, strength_patch=strength_clip, strength_model=1.0)
                else:
                    raise RuntimeError("Could not find add_patches on CLIP wrapper")

            except Exception:
                logging.exception("Failed to apply Z-Image Qwen TE LoRA")
                # keep diffusion LoRA applied even if TE fails
                pass

        return (model_out, clip_out)


NODE_CLASS_MAPPINGS = {
    "ZImageQwenTELoRALoader": ZImageQwenTELoRALoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageQwenTELoRALoader": "Load LoRA (Z-Image Qwen TE)"
}