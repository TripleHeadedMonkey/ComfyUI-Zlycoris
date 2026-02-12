import torch
import copy
from comfy.model_patcher import ModelPatcher

class ZImageTIESMerge:
    """
    Implements a simplified TIES Merging (Trim-Only for Single Pair):
    1. Calculate Delta = Turbo - Base
    2. Trim: Zero out the bottom (1 - density)% of values by magnitude.
    3. Merge: Base + Strength * (Trimmed_Delta)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_turbo": ("MODEL",),
                "density": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.01, 
                    "max": 1.0, 
                    "step": 0.05,
                    "display": "number",
                    "tooltip": "Fraction of weights to keep (0.2 = keep top 20% of changes)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "apply_ties_merge"
    CATEGORY = "Experimental"

    def apply_ties_merge(self, model_base, model_turbo, density, strength):
        print(f"Applying TIES Merge (Density: {density}, Strength: {strength})")
        
        # 1. Clone Base Model (Target)
        if isinstance(model_base, ModelPatcher):
            new_model_patcher = model_base.clone()
            base_model_obj = new_model_patcher.model
        else:
            base_model_obj = model_base
            new_model_patcher = copy.deepcopy(model_base)
            
        base_sd = base_model_obj.diffusion_model.state_dict()
        
        # 2. Get Turbo State Dict
        if isinstance(model_turbo, ModelPatcher):
            turbo_sd = model_turbo.model.diffusion_model.state_dict()
        else:
            turbo_sd = model_turbo.diffusion_model.state_dict()

        merged_sd = {}
        keys_processed = 0
        
        # We perform operations on CPU to avoid "Tensor on different device" errors 
        # and to avoid OOM on GPU during the heavy sort/top-k operations.
        calculation_device = torch.device("cpu")

        for key in base_sd.keys():
            if key in turbo_sd:
                # 3. Load tensors and force to same device/dtype
                w_base = base_sd[key].to(device=calculation_device, dtype=torch.float32)
                w_turbo = turbo_sd[key].to(device=calculation_device, dtype=torch.float32)
                
                if w_base.shape != w_turbo.shape:
                    print(f"Skipping {key}: Shape mismatch.")
                    merged_sd[key] = base_sd[key]
                    continue
                
                # 4. Calculate Task Vector (Delta)
                delta = w_turbo - w_base
                
                # 5. TIES-TRIM: Filter out small values (noise)
                # We only want the top 'density' (e.g. 20%) of changes
                if density < 1.0:
                    # Flatten to find the global threshold for this layer
                    flat_delta = delta.abs().view(-1)
                    k = int(flat_delta.numel() * density)
                    
                    if k > 0:
                        # Find the k-th largest value
                        top_k_value, _ = torch.kthvalue(flat_delta, flat_delta.numel() - k + 1)
                        threshold = top_k_value.item()
                        
                        # Zero out elements below threshold
                        mask = delta.abs() >= threshold
                        delta = delta * mask
                    else:
                        delta.zero_()

                # 6. Apply Scaled Delta
                merged_weight = w_base + (strength * delta)
                
                # Cast back to original dtype/device of the base model logic (handled by Comfy loading)
                # We store it back to CPU dict to be safe
                merged_sd[key] = merged_weight.to(base_sd[key].dtype)
                keys_processed += 1
            else:
                merged_sd[key] = base_sd[key]

        print(f"TIES Merge complete. Processed {keys_processed} keys.")
        
        # Load weights back
        base_model_obj.diffusion_model.load_state_dict(merged_sd, strict=False)
        
        return (new_model_patcher,)

NODE_CLASS_MAPPINGS = {
    "ZImageTIESMerge": ZImageTIESMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTIESMerge": "Z-Image TIES Merge (Method 2)"
}