# primitive_widget_to_string.py

class PrimitiveWidgetToString:
    """
    Takes a widget-driven value (e.g. Era Styler's 'era' or a 'folder' socket)
    and simply outputs that value as a string for use elsewhere in the graph.
    Uses STRING input to avoid ALL "Value not in list" validation errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "passthrough"
    CATEGORY = "utils"

    def passthrough(self, folder: str):
        # Just return the input value unchanged
        return (folder,)


NODE_CLASS_MAPPINGS = {
    "PrimitiveWidgetToString": PrimitiveWidgetToString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveWidgetToString": "Primitive Widget → String",
}
