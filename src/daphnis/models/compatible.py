"""Generic Chat Completions mappings; no inferred thinking protocol."""

from .compatibility import Compatibility

COMPATIBILITY = Compatibility(field_mappings={"output_limit": "max_tokens"})
