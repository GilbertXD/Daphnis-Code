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
    max_output_tokens: int | None = None
    default_output_tokens: int | None = None
    supports_image: bool | None = None

    def __post_init__(self) -> None:
        for name in ("id", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in ("context_window", "max_output_tokens", "default_output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")
        if (
            self.default_output_tokens is not None
            and self.max_output_tokens is not None
            and self.default_output_tokens > self.max_output_tokens
        ):
            raise ValueError("default_output_tokens cannot exceed max_output_tokens.")
        for name in ("supports_tools", "supports_image"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be True, False, or None.")
        if self.thinking_modes is not None:
            if isinstance(self.thinking_modes, str):
                raise ValueError("thinking_modes must be a sequence of mode names.")
            modes = tuple(self.thinking_modes)
            if any(not isinstance(mode, str) or not mode.strip() for mode in modes):
                raise ValueError("Thinking mode names must be non-empty strings.")
            if len(set(modes)) != len(modes):
                raise ValueError("Thinking mode names must be unique.")
            object.__setattr__(self, "thinking_modes", modes)


# Exact limits come from DeepSeek's /models example and MiMo's OpenCode guide.
# MiMo's default is documented for Chat Completions.
DEEPSEEK_MODELS = (
    ModelInfo(
        id="deepseek-flash",
        provider="deepseek",
        context_window=1_048_576,
        max_output_tokens=393_216,
        supports_tools=True,
        supports_image=True,
        thinking_modes=("off", "on", "low", "high", "max"),
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        context_window=1_048_576,
        max_output_tokens=393_216,
        supports_tools=True,
        thinking_modes=("off", "on", "low", "high", "max"),
    ),
)

MIMO_MODELS = (
    ModelInfo(
        id="mimo-v2.6-pro",
        provider="mimo",
        context_window=1_048_576,
        max_output_tokens=131_072,
        default_output_tokens=131_072,
        supports_tools=True,
        supports_image=True,
        thinking_modes=("off", "on"),
    ),
    ModelInfo(
        id="mimo-v2.6-flash",
        provider="mimo",
        context_window=1_048_576,
        max_output_tokens=131_072,
        default_output_tokens=131_072,
        supports_tools=True,
        supports_image=True,
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
