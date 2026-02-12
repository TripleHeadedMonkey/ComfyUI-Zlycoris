import torch
import folder_paths
import comfy.utils
import comfy.lora
import logging

class ZImageDiffSynthLoader:
    """
    A specialized LoRA loader for Z-Image Turbo that automatically fixes
    key mismatches from DiffSynth-trained LoRAs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "Z-Image/Loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # --- KEY MAPPING LOGIC ---
        # DiffSynth keys are "flat" (e.g. "layers.0.attention...")
        # ComfyUI Z-Image expects "diffusion_model." prefix for these.
        
        fixed_sd = {}
        for k, v in lora_sd.items():
            new_key = k
            
            # 1. Handle specialized Z-Image Refiners
            if k.startswith("context_refiner."):
                new_key = f"diffusion_model.{k}"
            elif k.startswith("noise_refiner."):
                new_key = f"diffusion_model.{k}"
                
            # 2. Handle standard Llama-3 Backbone Layers
            # If it starts with "layers." it belongs to the DiT backbone
            elif k.startswith("layers."):
                new_key = f"diffusion_model.{k}"
            
            # 3. Handle Input/Output blocks if they are bare
            elif k.startswith("final_layer.") or k.startswith("label_emb."):
                 new_key = f"diffusion_model.{k}"

            fixed_sd[new_key] = v

        # --- APPLY PATCHES ---
        # We pass the fixed dictionary to ComfyUI's standard loader
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, fixed_sd, strength_model, strength_clip
        )
        
        return (model_lora, clip_lora)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "ZImageDiffSynthLoader": ZImageDiffSynthLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageDiffSynthLoader": "Z-Image DiffSynth LoRA Loader"
}