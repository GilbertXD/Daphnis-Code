"""Public composition entry point for clients using Operations."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .app import Application
from .config import (
    environment_connections,
    environment_context_options,
    environment_embedding,
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
    app = Application(
        workspace=workspace,
        provider=provider,
        model_id=model_id,
        connections=connections,
        embedding=environment_embedding(),
        **environment_context_options(),
        interactive=True,
        observe_model_io=observe_model_io,
    )
    try:
        ops = Operations(app, initial_session_id=session_id)
        if config := os.getenv("DAPHNIS_MCP_CONFIG"):
            try:
                ops.import_mcp(Path(config).expanduser())
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.warning("无法导入 DAPHNIS_MCP_CONFIG，请检查路径及 MCP JSON 配置格式。")
        await ops.start_mcp(session_id=ops.initial_session_id)
    except BaseException:
        await app.close()
        raise
    return ops
