"""Read environment variables into Application connection options."""

import json
import os
from pathlib import Path
from typing import Any


def environment_connections() -> tuple[str, str, dict[str, dict[str, Any]]]:
    """Read the current environment; provider profiles never share keys."""
    provider = os.getenv("LLM_PROVIDER", "")
    model_id = os.getenv("LLM_MODEL", "")
    connections = {}
    base_url = os.getenv("LLM_BASE_URL")
    if provider and base_url:
        connections[provider] = {
            "base_url": base_url,
            "api_key": os.getenv("LLM_API_KEY") or None,
        }
    for name in ("deepseek", "mimo", "compatible"):
        prefix = name.upper()
        address = os.getenv(f"{prefix}_BASE_URL")
        if address:
            connections[name] = {
                "base_url": address,
                "api_key": os.getenv(f"{prefix}_API_KEY") or None,
            }
    if not provider or not model_id or provider not in connections:
        raise ValueError("请在 .env 配置 LLM_PROVIDER、LLM_MODEL 和对应的 Base URL。")
    return provider, model_id, connections


def load_mcp_config(path: Path) -> tuple[str, str, list[str]]:
    """Read one explicitly selected local stdio server configuration."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict) or not {"name", "command"} <= config.keys():
        raise ValueError("MCP 配置需要 name 和 command。")
    if config.keys() - {"name", "command", "args"}:
        raise ValueError("MCP 配置仅支持 name、command 和 args。")
    return config["name"], config["command"], config.get("args", [])
