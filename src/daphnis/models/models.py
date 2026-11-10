"""Manually maintained model capabilities, independent of provider adapters."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..token_budget import DEFAULT_COUNTER, TokenCounter


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    context_window: int | None = None
    supports_tools: bool | None = None
    thinking_modes: tuple[str, ...] | None = None
    token_counter: TokenCounter = DEFAULT_COUNTER

    def __post_init__(self) -> None:
        for name in ("id", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.context_window is not None and (
            type(self.context_window) is not int or self.context_window <= 0
        ):
            raise ValueError("context_window must be a positive integer or None.")
        if self.supports_tools is not None and type(self.supports_tools) is not bool:
            raise ValueError("supports_tools must be True, False, or None.")
        if self.thinking_modes is not None:
            if isinstance(self.thinking_modes, str):
                raise ValueError("thinking_modes must be a sequence of mode names.")
            modes = tuple(self.thinking_modes)
            if any(not isinstance(mode, str) or not mode.strip() for mode in modes):
                raise ValueError("Thinking mode names must be non-empty strings.")
            if len(set(modes)) != len(modes):
                raise ValueError("Thinking mode names must be unique.")
            object.__setattr__(self, "thinking_modes", modes)


# Verified 2026-11-04 against official DeepSeek and MiMo API documentation.
# DeepSeek's /models example gives an exact token count. MiMo documents "1M"
# without an exact integer, so its context_window remains unknown here.
DEEPSEEK_MODELS = (
    ModelInfo(
        id="deepseek-flash",
        provider="deepseek",
        context_window=1_048_576,
        supports_tools=True,
        thinking_modes=("off", "on", "low", "high", "max"),
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        context_window=1_048_576,
        supports_tools=True,
        thinking_modes=("off", "on", "low", "high", "max"),
    ),
)

MIMO_MODELS = (
    ModelInfo(
        id="mimo-v2.6-pro",
        provider="mimo",
        supports_tools=True,
        thinking_modes=("off", "on"),
    ),
    ModelInfo(
        id="mimo-v2.6-flash",
        provider="mimo",
        supports_tools=True,
        thinking_modes=("off", "on"),
    ),
)

MODELS = (*DEEPSEEK_MODELS, *MIMO_MODELS)


def validate_catalog(catalog: Sequence[ModelInfo]) -> None:
    if len({(model.provider, model.id) for model in catalog}) != len(catalog):
        raise ValueError("Model catalog contains duplicate (provider, id) keys.")


def list_models(
    provider: str | None = None, *, catalog: Sequence[ModelInfo] = MODELS
) -> tuple[ModelInfo, ...]:
    return tuple(
        model for model in catalog if provider is None or model.provider == provider
    )


def get_model(
    provider: str, model_id: str, *, catalog: Sequence[ModelInfo] = MODELS
) -> ModelInfo:
    matches = [
        model for model in catalog if (model.provider, model.id) == (provider, model_id)
    ]
    if not matches:
        raise KeyError(f"Model is not registered: {(provider, model_id)!r}.")
    if len(matches) > 1:
        raise ValueError(f"Duplicate model registration: {(provider, model_id)!r}.")
    return matches[0]


def resolve_model(
    provider: str, model_id: str, *, catalog: Sequence[ModelInfo] = MODELS
) -> ModelInfo:
    """Resolve model capabilities, allowing unregistered generic model IDs."""
    try:
        return get_model(provider, model_id, catalog=catalog)
    except KeyError:
        if provider == "compatible":
            return ModelInfo(id=model_id, provider=provider)
        raise
