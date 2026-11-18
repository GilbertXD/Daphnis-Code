"""Application-wide MCP configuration, separate from session history."""

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx2

from .history import SessionLease
from .mcp import MCPConnection
from .redaction import Redactor

_OPTIONS = {"transport", "command", "args", "cwd", "env", "url", "auth", "public_only"}
_CREDENTIAL_FLAG = re.compile(
    r"--?(?:api[-_]key|token|access[-_]token|bearer[-_]token|password|client[-_]secret|authorization)(?:=|$)",
    re.IGNORECASE,
)
_REDACTOR = Redactor(())


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    state: str
    options: dict[str, Any]
    workspace: Path | None = None
    description: str = ""

    @property
    def scope(self) -> str:
        return "global" if self.workspace is None else "project"

    @property
    def id(self) -> str:
        return f"{self.scope}:{self.name}"

    @property
    def key(self) -> str:
        if self.workspace is None:
            return self.name
        digest = sha256(os.fsencode(self.workspace)).hexdigest()[:16]
        return f"p_{digest}_{self.name}"

    @property
    def enabled(self) -> bool:
        return self.state != "disabled"

    def applies_to(self, workspace: Path) -> bool:
        return self.workspace is None or self.workspace == workspace

    def connection(self, redactor: Redactor = _REDACTOR) -> MCPConnection:
        if (self.state not in ("disabled", "always", "on_demand")
            or not isinstance(self.options, dict) or self.options.keys() - _OPTIONS):
            raise ValueError("MCP 服务配置字段无效。")
        if not isinstance(self.description, str) or len(self.description) > 500:
            raise ValueError("MCP 服务简介须为最多 500 字的文本。")
        if self.workspace is not None and (
            not self.workspace.is_absolute() or self.workspace != Path(os.path.normpath(self.workspace))
        ):
            raise ValueError("MCP 项目工作区须使用规范化的绝对路径。")
        connection = MCPConnection(self.name, key=self.key, **self.options)
        if (url := self.options.get("url")) and b"?" in httpx2.URL(url).raw_path:
            raise ValueError("保存的 MCP URL 不能包含查询参数，请通过环境变量配置认证。")
        args = self.options.get("args", [])
        values = [*args, self.options.get("command"), self.options.get("cwd"), url, self.description]
        if any(_CREDENTIAL_FLAG.match(arg) for arg in args) or any(
            value is not None and redactor.text(str(value)) != str(value) for value in values
        ):
            raise ValueError("请通过环境变量引用提供 MCP 凭据。")
        return connection

    @classmethod
    def from_json(cls, key: str, value: Any, *, version: int) -> "MCPServerConfig":
        if not isinstance(value, dict) or not isinstance(value.get("options"), dict):
            raise TypeError("MCP 服务配置格式无效。")
        if version == 1 and value.keys() == {"enabled", "options"} and type(value["enabled"]) is bool:
            config = cls(key, "always" if value["enabled"] else "disabled", value["options"])
        elif version == 2 and value.keys() == {"name", "state", "options", "workspace", "description"}:
            workspace = value["workspace"]
            if workspace is not None and (not isinstance(workspace, str) or not workspace):
                raise ValueError("MCP 项目工作区无效。")
            config = cls(value["name"], value["state"], value["options"],
                         Path(workspace) if workspace else None, value["description"])
        else:
            raise ValueError("MCP 服务配置格式无效。")
        config.connection()
        if config.key != key:
            raise ValueError("MCP 服务标识与配置不一致。")
        return config


class MCPStore:
    def __init__(self, path: Path):
        self.path = path
        self.attempts_path = path.with_name("mcp-attempts.json")

    def read(self) -> dict[str, MCPServerConfig]:
        try:
            data = json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return {}
        if (not isinstance(data, dict) or type(data.get("version")) is not int
            or data["version"] not in (1, 2) or not isinstance(data.get("servers"), dict)):
            raise ValueError("MCP 配置文件格式无效。")
        return {key: MCPServerConfig.from_json(key, value, version=data["version"])
                for key, value in data["servers"].items()}

    def attempts(self, key: str) -> dict[str, int]:
        return self._read_attempts().get(key, {})

    def _read_attempts(self) -> dict[str, dict[str, int]]:
        try:
            data = json.loads(self.attempts_path.read_bytes())
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or any(
            not isinstance(sessions, dict)
            or any(type(count) is not int or count < 1 for count in sessions.values())
            for sessions in data.values()
        ):
            raise ValueError("MCP 会话重试额度文件格式无效。")
        return data

    def reserve_attempt(self, key: str, session_id: str, limit: int) -> bool:
        with SessionLease(self.attempts_path):
            data = self._read_attempts()
            sessions = data.setdefault(key, {})
            count = sessions.get(session_id, 0)
            if count >= limit:
                return False
            sessions[session_id] = count + 1
            self._write(self.attempts_path, data)
        return True

    def reset_attempts(self, key: str) -> None:
        with SessionLease(self.attempts_path):
            data = self._read_attempts()
            if key in data:
                del data[key]
                self._write(self.attempts_path, data)

    def write(self, configs: dict[str, MCPServerConfig]) -> None:
        data = {"version": 2, "servers": {
            key: {"name": config.name, "state": config.state, "options": config.options,
                  "workspace": str(config.workspace) if config.workspace is not None else None,
                  "description": config.description}
            for key, config in configs.items()
        }}
        self._write(self.path, data)

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
