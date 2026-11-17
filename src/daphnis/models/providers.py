"""Default request mappings for provider connections."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import compatible, deepseek, mimo
from .compatibility import Compatibility
from .documents import DocumentAdapter

_ADAPTERS = {
    "deepseek": deepseek.COMPATIBILITY,
    "mimo": mimo.COMPATIBILITY,
    "compatible": compatible.COMPATIBILITY,
}
_DOCUMENT_ADAPTERS = {"deepseek": deepseek.document_adapter}


def provider_compatibility(
    provider: str, *, override: Compatibility | None = None
) -> Compatibility:
    """Select the compatibility rules for a provider's ChatClient."""
    if provider not in _ADAPTERS:
        raise ValueError(f"Unknown provider: {provider!r}.")
    return _ADAPTERS[provider] if override is None else override


def provider_documents(provider: str, connection: Mapping[str, Any], cache_directory: Path) -> DocumentAdapter | None:
    adapter = _DOCUMENT_ADAPTERS.get(provider)
    return adapter(connection, cache_directory) if adapter else None
