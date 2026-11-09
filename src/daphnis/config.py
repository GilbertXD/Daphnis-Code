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


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("MCP 配置包含重复的键。")
        result[key] = value
    return result


def load_mcp_config(path: Path) -> dict[str, dict[str, Any]]:
    """Read explicit server options. Relative cwd is based on the config file."""
    try:
        with path.open(encoding="utf-8") as source:
            config = json.load(source, object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MCP 配置不是有效 JSON（第 {exc.lineno} 行）。") from None
    if not isinstance(config, dict) or config.keys() != {"servers"}:
        raise ValueError("MCP 配置需要一个 servers 对象。")
    servers = config["servers"]
    if not isinstance(servers, dict):
        raise TypeError("MCP servers 必须是以名称为键的对象。")
    for name, options in servers.items():
        if not isinstance(options, dict):
            raise TypeError(f"MCP Server {name!r} 的配置必须是对象。")
        match options.get("transport"):
            case "stdio":
                if "command" not in options or options.keys() - {
                    "transport",
                    "command",
                    "args",
                    "cwd",
                    "env",
                }:
                    raise ValueError(f"MCP Server {name!r} 需要有效的 stdio 配置。")
                if isinstance(options.get("cwd"), str) and options["cwd"]:
                    options["cwd"] = path.resolve().parent / options["cwd"]
            case "streamable_http":
                if "url" not in options or options.keys() - {
                    "transport",
                    "url",
                    "auth",
                }:
                    raise ValueError(
                        f"MCP Server {name!r} 需要有效的 Streamable HTTP 配置。"
                    )
            case _:
                raise ValueError(
                    f"MCP Server {name!r} 的 transport 须为 stdio 或 streamable_http。"
                )
    return servers
