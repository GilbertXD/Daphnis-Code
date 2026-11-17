"""Strict JSON helpers for Daphnis data."""

import json
from typing import Any

from .errors import ValidationError


def validate_json(value: Any) -> None:
    """Check JSON serializability without implicit type conversions."""
    try:
        json.dumps(value, allow_nan=False)
        _check_json_structure(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValidationError("Request data must be valid JSON.") from exc


def _check_json_structure(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("JSON object keys must be strings.")
            _check_json_structure(item)

    elif isinstance(value, list):
        for item in value:
            _check_json_structure(item)

    elif isinstance(value, tuple):
        raise ValidationError("JSON arrays must use lists, not tuples.")
