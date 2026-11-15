"""Public composition entry point for clients using Operations."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .app import Application
from .config import (
    environment_connections,
    environment_context_options,
    load_mcp_config,
)
from .operations import Operations


class MCPStartupError(RuntimeError):
    """An MCP initialization failure with a message safe for server logs."""


async def open_operations(
    workspace: Path,
    *,
    session_id: str | None = None,
    observe_model_io: bool = False,
) -> Operations:
    """Start an interactive runtime from server-side configuration."""
    load_dotenv(Path.cwd() / ".env", override=False)
    provider, model_id, connections = environment_connections()
    servers = {}
    if config := os.getenv("DAPHNIS_MCP_CONFIG"):
        try:
            servers = load_mcp_config(Path(config).expanduser())
        except OSError:
            raise MCPStartupError("无法读取 DAPHNIS_MCP_CONFIG 指定的文件，请检查路径和权限。") from None
        except (RuntimeError, TypeError, ValueError):
            raise MCPStartupError("DAPHNIS_MCP_CONFIG 无效，请检查路径及 MCP JSON 配置格式。") from None
    ops = Operations(Application(
        workspace=workspace,
        provider=provider,
        model_id=model_id,
        connections=connections,
        **environment_context_options(),
        interactive=True,
        observe_model_io=observe_model_io,
    ))
    try:
        if session_id is not None:
            ops.restore_session(session_id)
        for index, (name, options) in enumerate(servers.items(), 1):
            try:
                await ops.connect_mcp(name, **options)
            except (OSError, RuntimeError, TypeError, ValueError):
                raise MCPStartupError(
                    f"无法连接 DAPHNIS_MCP_CONFIG 中的第 {index} 个 MCP 服务，请检查连接配置和服务状态。"
                ) from None
    except BaseException:
        await ops.close()
        raise
    return ops
