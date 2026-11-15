"""Public composition entry point for clients using Operations."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .app import Application
from .config import (
    environment_connections,
    environment_context_options,
)
from .operations import Operations

logger = logging.getLogger(__name__)


async def open_operations(
    workspace: Path,
    *,
    session_id: str | None = None,
    observe_model_io: bool = False,
) -> Operations:
    """Start an interactive runtime from server-side configuration."""
    load_dotenv(Path.cwd() / ".env", override=False)
    provider, model_id, connections = environment_connections()
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
        if config := os.getenv("DAPHNIS_MCP_CONFIG"):
            try:
                ops.import_mcp(Path(config).expanduser())
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.warning("无法导入 DAPHNIS_MCP_CONFIG，请检查路径及 MCP JSON 配置格式。")
        try:
            servers = ops.managed_mcp()
        except (OSError, TypeError, ValueError):
            logger.warning("无法读取已保存的 MCP 配置，请检查 mcp.json。")
            servers = ()
        for server in servers:
            if not server.enabled:
                continue
            try:
                await ops.connect_mcp(server.name, **server.options)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.warning("无法恢复 MCP 服务 %s，可在管理界面重试。", server.name)
    except BaseException:
        await ops.close()
        raise
    return ops
