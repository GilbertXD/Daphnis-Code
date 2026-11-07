"""Default request mappings for provider connections."""

from saturnus import Compatibility

from . import compatible, deepseek, mimo

_ADAPTERS = {
    "deepseek": deepseek.COMPATIBILITY,
    "mimo": mimo.COMPATIBILITY,
    "compatible": compatible.COMPATIBILITY,
}


def provider_compatibility(
    provider: str, *, override: Compatibility | None = None
) -> Compatibility:
    """Select the compatibility rules for a provider's ChatClient."""
    if provider not in _ADAPTERS:
        raise ValueError(f"Unknown provider: {provider!r}.")
    return _ADAPTERS[provider] if override is None else override
