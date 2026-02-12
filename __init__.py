# =====================================================
# Z-Image Core Nodes
# =====================================================

from .nodes import NODE_CLASS_MAPPINGS as NODES_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_NAME_MAPPINGS

from .ZCondAdv import NODE_CLASS_MAPPINGS as ZCOND_CLASS_MAPPINGS
from .ZCondAdv import NODE_DISPLAY_NAME_MAPPINGS as ZCOND_DISPLAY_NAME_MAPPINGS

from .CondMul import NODE_CLASS_MAPPINGS as CONDMUL_CLASS_MAPPINGS
from .CondMul import NODE_DISPLAY_NAME_MAPPINGS as CONDMUL_DISPLAY_NAME_MAPPINGS

from .diffz import NODE_CLASS_MAPPINGS as DIFFZ_CLASS_MAPPINGS
from .diffz import NODE_DISPLAY_NAME_MAPPINGS as DIFFZ_DISPLAY_NAME_MAPPINGS

from .Universal_LoRA import NODE_CLASS_MAPPINGS as UNIVERSAL_CLASS_MAPPINGS
from .Universal_LoRA import NODE_DISPLAY_NAME_MAPPINGS as UNIVERSAL_DISPLAY_NAME_MAPPINGS

from .Qwen_Lora import NODE_CLASS_MAPPINGS as QWEN_CLASS_MAPPINGS
from .Qwen_Lora import NODE_DISPLAY_NAME_MAPPINGS as QWEN_DISPLAY_NAME_MAPPINGS


# =====================================================
# Experimental Merge Nodes
# =====================================================

from .z_image_vector_merge import NODE_CLASS_MAPPINGS as VECTOR_CLASS_MAPPINGS
from .z_image_vector_merge import NODE_DISPLAY_NAME_MAPPINGS as VECTOR_DISPLAY_NAME_MAPPINGS

from .TIES import NODE_CLASS_MAPPINGS as TIES_CLASS_MAPPINGS
from .TIES import NODE_DISPLAY_NAME_MAPPINGS as TIES_DISPLAY_NAME_MAPPINGS


# =====================================================
# Utility / Logic Nodes
# =====================================================

from .getimagesizeplus import NODE_CLASS_MAPPINGS as SIZE_CLASS_MAPPINGS
from .getimagesizeplus import NODE_DISPLAY_NAME_MAPPINGS as SIZE_DISPLAY_NAME_MAPPINGS

from .keyword_match_gate import NODE_CLASS_MAPPINGS as GATE_CLASS_MAPPINGS
from .keyword_match_gate import NODE_DISPLAY_NAME_MAPPINGS as GATE_DISPLAY_NAME_MAPPINGS

from .primitive_widget_to_string import NODE_CLASS_MAPPINGS as PRIMITIVE_CLASS_MAPPINGS
from .primitive_widget_to_string import NODE_DISPLAY_NAME_MAPPINGS as PRIMITIVE_DISPLAY_NAME_MAPPINGS


# =====================================================
# GGUF RAW System (Root + Components)
# =====================================================

# Root GGUF initializer
from .GGUF_RAW import NODE_CLASS_MAPPINGS as GGUF_ROOT_CLASS_MAPPINGS
from .GGUF_RAW import NODE_DISPLAY_NAME_MAPPINGS as GGUF_ROOT_DISPLAY_NAME_MAPPINGS


# =====================================================
# Merge All Mappings
# =====================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

ALL_CLASS_MAPPINGS = [
    NODES_CLASS_MAPPINGS,
    ZCOND_CLASS_MAPPINGS,
    CONDMUL_CLASS_MAPPINGS,
    DIFFZ_CLASS_MAPPINGS,
    UNIVERSAL_CLASS_MAPPINGS,
    QWEN_CLASS_MAPPINGS,
    VECTOR_CLASS_MAPPINGS,
    TIES_CLASS_MAPPINGS,
    SIZE_CLASS_MAPPINGS,
    GATE_CLASS_MAPPINGS,
    PRIMITIVE_CLASS_MAPPINGS,   # <-- ADDED
    GGUF_ROOT_CLASS_MAPPINGS,
]

ALL_DISPLAY_MAPPINGS = [
    NODES_DISPLAY_NAME_MAPPINGS,
    ZCOND_DISPLAY_NAME_MAPPINGS,
    CONDMUL_DISPLAY_NAME_MAPPINGS,
    DIFFZ_DISPLAY_NAME_MAPPINGS,
    UNIVERSAL_DISPLAY_NAME_MAPPINGS,
    QWEN_DISPLAY_NAME_MAPPINGS,
    VECTOR_DISPLAY_NAME_MAPPINGS,
    TIES_DISPLAY_NAME_MAPPINGS,
    SIZE_DISPLAY_NAME_MAPPINGS,
    GATE_DISPLAY_NAME_MAPPINGS,
    PRIMITIVE_DISPLAY_NAME_MAPPINGS,  # <-- ADDED
    GGUF_ROOT_DISPLAY_NAME_MAPPINGS,
]

for mapping in ALL_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in ALL_DISPLAY_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]