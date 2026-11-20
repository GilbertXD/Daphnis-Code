"""Manually maintained model capabilities, independent of provider adapters."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..token_budget import DEFAULT_COUNTER, TokenCounter
from .image_budget import (
    conservative_image_token_reserve,
    deepseek_image_token_reserve,
    mimo_image_token_reserve,
)


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
    supports_document: bool | None = None
    image_token_estimator: Callable[[int, int], int] = conservative_image_token_reserve
    thinking_output_tokens: tuple[tuple[str, int], ...] = ()

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
        for name in ("supports_tools", "supports_image", "supports_document"):
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
        if not callable(self.image_token_estimator):
            raise TypeError("image_token_estimator must be callable.")
        if not isinstance(self.thinking_output_tokens, tuple):
            raise TypeError("thinking_output_tokens must be a tuple of (mode, tokens) pairs.")
        configured_modes = set()
        for pair in self.thinking_output_tokens:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("thinking_output_tokens must contain (mode, tokens) pairs.")
            mode, tokens = pair
            if not isinstance(mode, str) or mode not in (self.thinking_modes or ()) or mode in configured_modes:
                raise ValueError("thinking_output_tokens must use unique, supported thinking modes.")
            if type(tokens) is not int or tokens <= 0:
                raise ValueError("thinking_output_tokens values must be positive integers.")
            if self.max_output_tokens is not None and tokens > self.max_output_tokens:
                raise ValueError("thinking_output_tokens values cannot exceed max_output_tokens.")
            configured_modes.add(mode)

    def output_reserve_for(self, thinking_mode: str | None) -> int | None:
        for mode, tokens in self.thinking_output_tokens:
            if mode == thinking_mode:
                return tokens
        return self.default_output_tokens


# Exact limits come from DeepSeek's /models example and MiMo's OpenCode guide.
# Default reserves follow Chat Completions defaults, including DeepSeek's effort overrides.
_DEEPSEEK_THINKING_OUTPUT_TOKENS = (("off", 8192), ("max", 131_072))
DEEPSEEK_MODELS = (
    ModelInfo(
        id="deepseek-flash",
        provider="deepseek",
        context_window=1_048_576,
        max_output_tokens=393_216,
        default_output_tokens=65_536,
        supports_tools=True,
        supports_image=True,
        supports_document=True,
        thinking_modes=("off", "on", "low", "high", "max"),
        image_token_estimator=deepseek_image_token_reserve,
        thinking_output_tokens=_DEEPSEEK_THINKING_OUTPUT_TOKENS,
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        context_window=1_048_576,
        max_output_tokens=393_216,
        default_output_tokens=65_536,
        supports_tools=True,
        thinking_modes=("off", "on", "low", "high", "max"),
        image_token_estimator=deepseek_image_token_reserve,
        thinking_output_tokens=_DEEPSEEK_THINKING_OUTPUT_TOKENS,
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
        image_token_estimator=mimo_image_token_reserve,
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
        image_token_estimator=mimo_image_token_reserve,
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
