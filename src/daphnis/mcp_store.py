"""Application-wide MCP configuration, separate from session history."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx2

from .mcp import MCPConnection

_OPTIONS = {"transport", "command", "args", "cwd", "env", "url", "auth"}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    enabled: bool
    options: dict[str, Any]

    def connection(self) -> MCPConnection:
        if type(self.enabled) is not bool or self.options.keys() - _OPTIONS:
            raise ValueError("MCP 服务配置字段无效。")
        connection = MCPConnection(self.name, **self.options)
        if (url := self.options.get("url")) and b"?" in httpx2.URL(url).raw_path:
            raise ValueError("保存的 MCP URL 不能包含查询参数，请通过环境变量配置认证。")
        return connection

    @classmethod
    def from_json(cls, name: str, value: Any) -> "MCPServerConfig":
        if not isinstance(value, dict) or value.keys() != {"enabled", "options"} or not isinstance(value["options"], dict):
            raise ValueError("MCP 服务配置格式无效。")
        config = cls(name, value["enabled"], value["options"])
        config.connection()
        return config


class MCPStore:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> dict[str, MCPServerConfig]:
        try:
            data = json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("servers"), dict):
            raise ValueError("MCP 配置文件格式无效。")
        return {name: MCPServerConfig.from_json(name, value) for name, value in data["servers"].items()}

    def write(self, configs: dict[str, MCPServerConfig]) -> None:
        data = {"version": 1, "servers": {
            name: {"enabled": config.enabled, "options": config.options}
            for name, config in configs.items()
        }}
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
