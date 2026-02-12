import math
import torch

MAX_RESOLUTION = 8192

class GetImageSizePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "calculate_from_target": ("BOOLEAN", {"default": False}), 
                "multiple_of": (["8", "16", "32", "64", "None"], {"default": "8"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "essentials_mb/image utils"

    def execute(self, image, target_width, target_height, calculate_from_target, multiple_of):
        # 1. Get current dimensions
        orig_width = image.shape[2]
        orig_height = image.shape[1]
        count = image.shape[0]

        # 2. If toggle is off, return original dimensions
        if not calculate_from_target:
            return (orig_width, orig_height, count)

        # 3. Calculate Target Pixel Count (Area)
        target_area = target_width * target_height
        
        # 4. Calculate Aspect Ratio
        aspect_ratio = orig_width / orig_height

        # 5. Calculate new dimensions
        new_height = math.sqrt(target_area / aspect_ratio)
        new_width = new_height * aspect_ratio

        # 6. Snap to nearest multiple (e.g., 8)
        if multiple_of == "None":
            final_width = int(round(new_width))
            final_height = int(round(new_height))
        else:
            divisor = int(multiple_of)
            final_width = int(round(new_width / divisor) * divisor)
            final_height = int(round(new_height / divisor) * divisor)

        return (final_width, final_height, count)

NODE_CLASS_MAPPINGS = {
    "GetImageSizePlus+": GetImageSizePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageSizePlus+": "🔧 Get Image Size Plus"
}