import re

class KeywordMatchGate:
    """
    Keyword Match Gate Node

    Acts as a string-based AND gate:
    - If the keyword matches (with the chosen options), returns a string.
    - If not, returns the literal string "no match".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "match_keyword": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": False,
                }),
                "match_whole_word": ("BOOLEAN", {
                    "default": True,
                }),
                "echo_input": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gated_text",)
    FUNCTION = "execute"
    CATEGORY = "Logic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Pure function: change when inputs change
        return float("nan")

    def _match(
        self,
        input_text: str,
        match_keyword: str,
        case_sensitive: bool,
        match_whole_word: bool,
    ) -> bool:
        # Empty values are treated as "no match"
        if not input_text or not match_keyword:
            return False

        # Case normalization
        if not case_sensitive:
            norm_input = input_text.lower()
            norm_keyword = match_keyword.lower()
        else:
            norm_input = input_text
            norm_keyword = match_keyword

        try:
            if match_whole_word:
                # Whole-word regex using word boundaries
                escaped = re.escape(norm_keyword)
                pattern = rf"\b{escaped}\b"
                return re.search(pattern, norm_input) is not None
            else:
                # Simple substring check
                return norm_keyword in norm_input
        except re.error:
            # Fail-safe: treat as no match
            return False

    def execute(
        self,
        input_text,
        match_keyword,
        case_sensitive,
        match_whole_word,
        echo_input,
    ):
        match_found = self._match(
            input_text=input_text,
            match_keyword=match_keyword,
            case_sensitive=case_sensitive,
            match_whole_word=match_whole_word,
        )

        if match_found:
            # Gate open: return either the keyword or the original input
            return (input_text if echo_input else match_keyword,)
        else:
            # Gate closed: explicit "no match" so it can be mapped to 0 later
            return ("no match",)


NODE_CLASS_MAPPINGS = {
    "KeywordMatchGate": KeywordMatchGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeywordMatchGate": "🔹 Keyword Match Gate",
}
