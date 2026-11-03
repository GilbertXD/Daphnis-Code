"""DeepSeek Chat Completions request mappings."""

from saturnus import Compatibility

COMPATIBILITY = Compatibility(
    field_mappings={
        "output_limit": "max_tokens",
        "thinking_mode": "reasoning_effort",
    },
    # Official API: none disables thinking; low/high/max enable it.
    # "on" selects the documented default effort, high.
    value_mappings={"thinking_mode": {"off": "none", "on": "high"}},
)
