"""Model catalog, provider routing, and capability-aware requests."""

from .models import MODELS, ModelInfo, get_model, list_models, resolve_model
from .providers import provider_compatibility
from .requests import CapabilityError, prepare_request_options, require_tools

__all__ = [
    "MODELS",
    "CapabilityError",
    "ModelInfo",
    "get_model",
    "list_models",
    "prepare_request_options",
    "provider_compatibility",
    "require_tools",
    "resolve_model",
]
