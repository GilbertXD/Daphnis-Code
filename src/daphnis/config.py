"""Read server-side settings into Application options."""

import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from .compaction_settings import CompactionSettings
from .models import MODELS, ModelInfo
from .models.models import validate_catalog
from .models.tokenizer import verified_tokenizer
from .tools.retrieval import environment_retrieval

_DEEPSEEK_V4_PRO_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"


def environment_catalog() -> tuple[ModelInfo, ...]:
    path = os.getenv("DEEPSEEK_V4_PRO_TOKENIZER")
    if not path:
        return MODELS
    counter = verified_tokenizer(Path(path).expanduser(), _DEEPSEEK_V4_PRO_TOKENIZER_SHA256)
    return tuple(
        replace(model, token_counter=counter)
        if (model.provider, model.id) == ("deepseek", "deepseek-v4-pro") else model
        for model in MODELS
    )


def _table(value: Any, label: str, *, allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 TOML 表。")
    if allowed is not None and (unknown := value.keys() - allowed):
        raise ValueError(f"{label} 有未知配置项：{', '.join(sorted(unknown))}。")
    return value


def _compaction_settings(
    value: Any, label: str, base: CompactionSettings,
) -> CompactionSettings:
    fields = _table(value, label, allowed={
        "auto_enabled", "threshold_mode", "threshold_value", "recent_tokens",
    })
    if fields.get("threshold_mode", base.threshold_mode) != base.threshold_mode and (
        "threshold_value" not in fields
    ):
        raise ValueError(f"{label} 切换阈值模式时须同时设置 threshold_value。")
    try:
        return replace(base, **fields)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None


def load_context_options(
    path: Path, *, catalog: tuple[ModelInfo, ...] = MODELS,
) -> dict[str, Any]:
    """Read compaction preferences and deployment limits from an explicit TOML file."""
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("DAPHNIS_CONFIG 不是有效的 UTF-8 TOML。") from None
    _table(data, "DAPHNIS_CONFIG", allowed={"compaction", "models"})
    settings = _compaction_settings(data.get("compaction", {}), "compaction", CompactionSettings())
    validate_catalog(catalog)
    models = {(model.provider, model.id): model for model in catalog}
    overrides: dict[tuple[str, str], CompactionSettings] = {}
    output_limits: dict[tuple[str, str], int] = {}
    for name, options in _table(data.get("models", {}), "models").items():
        label = f"models.{name}"
        provider, _, model_id = name.partition(":")
        if not provider or not model_id or model_id != model_id.strip():
            raise ValueError(f"{label} 需要使用 provider:model_id 格式。")
        key = (provider, model_id)
        fields = _table(options, label, allowed={"context_window", "output_limit", "compaction"})
        model = models.get(key)
        if model is None:
            if provider != "compatible":
                raise ValueError(f"{label} 对应的模型不在模型目录中。")
            model = ModelInfo(id=model_id, provider=provider)
        if "context_window" in fields:
            window = fields["context_window"]
            if type(window) is not int or window <= 0:
                raise ValueError(f"{label}.context_window 必须是正整数。")
            model = replace(model, context_window=window)
        models[key] = model
        if "output_limit" in fields:
            limit = fields["output_limit"]
            if type(limit) is not int or limit <= 0:
                raise ValueError(f"{label}.output_limit 必须是正整数。")
            if model.max_output_tokens is not None and limit > model.max_output_tokens:
                raise ValueError(f"{label}.output_limit 不能超过模型上限 {model.max_output_tokens}。")
            output_limits[key] = limit
        if "compaction" in fields:
            overrides[key] = _compaction_settings(fields["compaction"], f"{label}.compaction", settings)
    return {
        "catalog": tuple(models.values()),
        "compaction_settings": settings,
        "model_compaction_settings": overrides,
        "output_limits": output_limits,
    }


def environment_context_options() -> dict[str, Any]:
    catalog = environment_catalog()
    path = os.getenv("DAPHNIS_CONFIG")
    options = load_context_options(Path(path).expanduser(), catalog=catalog) if path else {"catalog": catalog}
    if (retrieval := environment_retrieval()) is not None:
        options["retrieval"] = retrieval
    return options


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
