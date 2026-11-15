"""MiMo Chat Completions request mappings."""

from saturnus import Compatibility

COMPATIBILITY = Compatibility(
    field_mappings={
        "output_limit": "max_completion_tokens",
        "thinking_mode": "thinking",
    },
    value_mappings={
        "thinking_mode": {
            "off": {"type": "disabled"},
            "on": {"type": "enabled"},
        },
    },
    content_part_mappings={"image_url": {"type": "image_url"}},
)
