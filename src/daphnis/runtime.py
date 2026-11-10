"""Public composition entry point for clients using Operations."""

from pathlib import Path

from dotenv import load_dotenv

from .app import Application
from .config import environment_catalog, environment_connections
from .operations import Operations


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
        catalog=environment_catalog(),
        interactive=True,
        observe_model_io=observe_model_io,
    ))
    try:
        if session_id is not None:
            ops.restore_session(session_id)
    except BaseException:
        await ops.close()
        raise
    return ops
