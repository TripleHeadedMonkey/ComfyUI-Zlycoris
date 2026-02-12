import torch

class ZImageConditioningContrast:
    """
    Applies Non-Linear Contrast (Power Scaling) to Z-Image/Qwen embeddings.
    
    Why this works where Multiplication fails:
    Z-Image applies Layer Normalization to inputs, which mathematically cancels out 
    any linear multiplication (1.2 * x -> normalized -> x).
    
    This node applies a Power Function (x ^ exponent), which changes the 
    distribution shape (Kurtosis) of the embedding. Normalization cannot 
    revert this, allowing you to effectively 'sharpen' or 'soften' the prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_contrast"
    CATEGORY = "RES4LYF/conditioning"
    
    def apply_contrast(self, conditioning, contrast):
        c = []
        for t in conditioning:
            # 1. Handle Standard CLIP Embeddings (Less important for Z-Image, but good practice)
            # Formula: sign(x) * |x|^contrast
            # We use sign() to preserve negative values, as x^2 would lose them.
            original_emb = t[0]
            new_emb = original_emb.sign() * original_emb.abs().pow(contrast)
            
            # 2. Handle Metadata Dictionary (The Critical Qwen Part)
            original_dict = t[1]
            new_dict = original_dict.copy()
            
            # Keys known to hold Qwen/Llama embeddings
            target_keys = ["conditioning_llama3", "llama_embeds", "pooled_output"]
            
            for key, value in new_dict.items():
                if isinstance(value, torch.Tensor):
                    # Skip masks to prevent crashes
                    if "mask" in key or "ids" in key or "size" in key:
                        continue
                        
                    # Target specific embedding keys OR keys that look like high-dim embeddings
                    if key in target_keys or (value.dim() >= 3 and value.shape[-1] > 64):
                        # Apply Power Scaling
                        # .contiguous() is strictly required for Flash Attention in Turbo models
                        new_dict[key] = (value.sign() * value.abs().pow(contrast)).contiguous()

            c.append([new_emb, new_dict])
            
        return (c,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "ZImageConditioningContrast": ZImageConditioningContrast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageConditioningContrast": "Z-Image Conditioning Contrast"
}