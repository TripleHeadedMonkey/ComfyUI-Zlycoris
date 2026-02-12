import torch

class ZImageAdvancedConditioning:
    """
    FINAL PRODUCTION VERSION.
    - Fixes 'Drift' bug (No more in-place modification of cached inputs).
    - Removes debug prints for speed.
    - Safely handles Z-Image/Qwen/Llama architectures.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING", ),
                "conditioning_from": ("CONDITIONING", ),
                "operation": (["mix_slerp", "purge_ortho", "add_perpendicular"], ),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "RES4LYF/conditioning"

    def process(self, conditioning_to, conditioning_from, operation, strength):
        results = []
        
        # Match list lengths
        min_len = min(len(conditioning_to), len(conditioning_from))
        
        for i in range(min_len):
            # --- 1. PROCESS MAIN TENSOR (t[0]) ---
            t0_input = conditioning_to[i][0]
            t0_ref   = conditioning_from[i][0]
            
            # Create a SAFE COPY of the main tensor to avoid modifying the input cache
            t0_output = t0_input.clone()
            
            # Handle Shape Mismatch (Slice to common length)
            common_len = min(t0_input.shape[1], t0_ref.shape[1])
            
            # Extract working slices
            v0 = t0_input[:, :common_len, :].clone()
            v1 = t0_ref[:, :common_len, :].clone()

            # Apply Math
            v_processed = self.apply_math(v0, v1, operation, strength)
            
            # Write into the NEW output tensor (not the input!)
            t0_output[:, :common_len, :] = v_processed
            
            # --- 2. PROCESS DICTIONARY ---
            # Create a shallow copy of the dict, but we will replace values with new tensors
            new_dict = conditioning_to[i][1].copy()
            ref_dict = conditioning_from[i][1]
            
            target_keys = ["conditioning_llama3", "llama_embeds", "pooled_output"]
            
            for key in new_dict.keys():
                if key in target_keys and key in ref_dict:
                    val_to = new_dict[key]
                    val_from = ref_dict[key]
                    
                    if val_to is not None and val_from is not None and isinstance(val_to, torch.Tensor):
                        # Ensure we only process if shapes align
                        if val_to.shape == val_from.shape:
                            # Apply Math
                            # apply_math returns a new tensor, so this is safe
                            new_dict[key] = self.apply_math(val_to, val_from, operation, strength)

            results.append([t0_output, new_dict])

        return (results,)

    def apply_math(self, v0, v1, operation, strength):
        """Helper for vector operations"""
        # Epsilon for stability
        eps = 1e-8

        if operation == "mix_slerp":
            # Normalize
            v0_n = v0 / (v0.norm(dim=-1, keepdim=True) + eps)
            v1_n = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
            
            dot = (v0_n * v1_n).sum(dim=-1, keepdim=True)
            dot = torch.clamp(dot, -0.9995, 0.9995)
            
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta) + eps
            
            w0 = torch.sin((1.0 - strength) * theta) / sin_theta
            w1 = torch.sin(strength * theta) / sin_theta
            
            return (w0 * v0 + w1 * v1).contiguous()

        elif operation == "purge_ortho":
            # Project v0 onto v1
            dot_v0_v1 = (v0 * v1).sum(dim=-1, keepdim=True)
            dot_v1_v1 = (v1 * v1).sum(dim=-1, keepdim=True)
            
            proj = (dot_v0_v1 / (dot_v1_v1 + eps)) * v1
            return (v0 - (proj * strength)).contiguous()

        elif operation == "add_perpendicular":
            # Find part of v1 orthogonal to v0
            dot_v1_v0 = (v1 * v0).sum(dim=-1, keepdim=True)
            dot_v0_v0 = (v0 * v0).sum(dim=-1, keepdim=True)
            
            proj = (dot_v1_v0 / (dot_v0_v0 + eps)) * v0
            ortho_v1 = v1 - proj
            
            return (v0 + (ortho_v1 * strength)).contiguous()
        
        return v0.contiguous()

# Register
NODE_CLASS_MAPPINGS = {
    "ZImageAdvancedConditioning": ZImageAdvancedConditioning
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageAdvancedConditioning": "Z-Image Advanced Mixing"
}