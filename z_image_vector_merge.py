import torch
import copy
from comfy.model_patcher import ModelPatcher

class ZImageVectorMerge:
    """
    Implements Task Arithmetic Merging:
    Result = Base + Strength * (Turbo - Base)
    
    This treats the difference between Turbo and Base as a 'Task Vector'
    and injects that vector into the Base model.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_turbo": ("MODEL",),
                "strength": ("FLOAT", {
                    "default": 0.3, 
                    "min": -2.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "apply_vector_merge"
    CATEGORY = "Experimental"

    def apply_vector_merge(self, model_base, model_turbo, strength):
        print(f"Applying Vector Merge with strength: {strength}")
        
        # Clone the base model structure so we don't corrupt the loaded checkpoint
        # We use ModelPatcher.clone() if available, otherwise manual copy
        if isinstance(model_base, ModelPatcher):
            new_model_patcher = model_base.clone()
            base_model_obj = new_model_patcher.model
        else:
            # Fallback for raw model objects
            base_model_obj = model_base
            new_model_patcher = copy.deepcopy(model_base)
            
        # Get the underlying state dicts
        # Note: We access the diffusion_model directly to avoid VAE/TextEncoder noise
        base_sd = base_model_obj.diffusion_model.state_dict()
        
        # specific handling for getting the turbo state dict
        if isinstance(model_turbo, ModelPatcher):
            turbo_sd = model_turbo.model.diffusion_model.state_dict()
        else:
            turbo_sd = model_turbo.diffusion_model.state_dict()

        # Prepare the new state dict
        merged_sd = {}
        
        keys_processed = 0
        
        for key in base_sd.keys():
            if key in turbo_sd:
                # Get weights
                w_base = base_sd[key]
                w_turbo = turbo_sd[key]
                
                # Check for shape mismatch (safety)
                if w_base.shape != w_turbo.shape:
                    print(f"Warning: Shape mismatch for key {key}. Skipping. Base: {w_base.shape}, Turbo: {w_turbo.shape}")
                    merged_sd[key] = w_base
                    continue
                
                # METHOD 1 MATH:
                # Vector = (Turbo - Base)
                # New = Base + Strength * Vector
                # This simplifies to: New = Base + Strength * Turbo - Strength * Base
                # Or: New = (1 - Strength) * Base + Strength * Turbo
                
                # We perform operation on correct device to save VRAM/Time
                # Using float32 for precision during merge is recommended
                w_base_f = w_base.to(dtype=torch.float32)
                w_turbo_f = w_turbo.to(dtype=torch.float32)
                
                # Calculate the vector difference
                task_vector = w_turbo_f - w_base_f
                
                # Apply vector
                merged_weight = w_base_f + (strength * task_vector)
                
                # Cast back to original dtype (usually float16 or bfloat16)
                merged_sd[key] = merged_weight.to(w_base.dtype)
                keys_processed += 1
            else:
                # If key missing in Turbo, keep Base
                merged_sd[key] = base_sd[key]

        print(f"Merge complete. Processed {keys_processed} keys.")

        # Load the new weights into our cloned model
        # We use strict=False just in case, but keys should match based on your check
        base_model_obj.diffusion_model.load_state_dict(merged_sd, strict=False)
        
        return (new_model_patcher,)

# Node Mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ZImageVectorMerge": ZImageVectorMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageVectorMerge": "Z-Image Vector Merge (Method 1)"
}