"""Capability checks and request options for a specific model."""

from typing import Any

from .models import ModelInfo

_MANAGED_FIELDS = {
    "model",
    "messages",
    "stream",
    "tools",
    "thinking_mode",
    "output_limit",
    "thinking",
    "reasoning_effort",
}


class CapabilityError(ValueError):
    """A requested capability is unknown or unsupported for this model."""


def require_image_input(model: ModelInfo) -> None:
    if (model.provider, model.id) not in {
        ("mimo", "mimo-v2.6-flash"), ("mimo", "mimo-v2.6-pro"), ("deepseek", "deepseek-flash"),
    }:
        raise CapabilityError(f"当前模型 {model.id!r} 不支持 Daphnis 图片输入。")


def require_tools(model: ModelInfo) -> None:
    if model.supports_tools is not True:
        reason = "unknown" if model.supports_tools is None else "unsupported"
        raise CapabilityError(
            f"Tool calling is {reason} for {(model.provider, model.id)!r}."
        )


def prepare_request_options(
    *,
    model: ModelInfo,
    thinking_mode: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    output_limit: int | None = None,
    extra_body: dict[str, Any] | None = None,
    custom_compatibility: bool = False,
) -> dict[str, Any]:
    """Prepare ChatClient extra_body before applying the connection's mappings.

    Set custom_compatibility when the client uses explicit compatibility rules.
    """
    if extra_body is not None and not isinstance(extra_body, dict):
        raise TypeError("extra_body must be a dictionary or None.")
    body = {} if extra_body is None else extra_body.copy()
    conflicts = _MANAGED_FIELDS.intersection(body)
    if conflicts:
        raise ValueError(
            f"Use explicit request arguments instead of extra_body for: "
            f"{', '.join(sorted(conflicts))}."
        )
    if tools is not None:
        body["tools"] = tools
    if tools is not None or body.get("tool_choice") not in (None, "none"):
        require_tools(model)
    if thinking_mode is not None:
        modes = model.thinking_modes
        if modes is None:
            raise CapabilityError(f"Thinking options are unknown for {model.id!r}.")
        if thinking_mode not in modes:
            raise CapabilityError(
                f"Thinking mode {thinking_mode!r} is unsupported for "
                f"{model.id!r}; available options: {modes!r}."
            )
        if model.provider == "compatible" and not custom_compatibility:
            raise ValueError(
                "Generic thinking requires an explicit Compatibility mapping "
                "for thinking_mode."
            )
        body["thinking_mode"] = thinking_mode
    if output_limit is not None:
        if type(output_limit) is not int or output_limit <= 0:
            raise ValueError("output_limit must be a positive integer.")
        if model.max_output_tokens is not None and output_limit > model.max_output_tokens:
            raise ValueError(f"output_limit exceeds the model maximum of {model.max_output_tokens}.")
        body["output_limit"] = output_limit
    return body
