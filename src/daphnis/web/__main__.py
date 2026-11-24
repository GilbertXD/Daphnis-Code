"""Run the single-owner Web client on a loopback address."""

import argparse
import os
import secrets
from pathlib import Path

import uvicorn

from ..runtime import open_operations
from .bridge import Bridge
from .redaction import Redactor
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Daphnis Web")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--session")
    parser.add_argument(
        "--assets", type=Path,
        default=Path(__file__).resolve().parents[3] / "web" / "dist",
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port 必须介于 1024 和 65535。")
    if args.session and args.workspace:
        parser.error("--session 使用保存的工作区，请勿同时指定 --workspace。")
    if not (args.assets / "index.html").is_file():
        parser.error("请先运行 npm --prefix web ci 和 npm --prefix web run build。")

    token = secrets.token_urlsafe(32)

    async def open_bridge():
        try:
            ops = await open_operations(
                args.workspace or Path.cwd(), session_id=args.session
            )
        except Exception as exc:  # noqa: BLE001 - Avoid leaking configuration in startup logs.
            raise RuntimeError(f"Daphnis 无法启动（{type(exc).__name__}），请检查服务端配置。") from None
        redactor = Redactor([
            token,
            *(os.getenv(f"{name}_API_KEY", "") for name in ("LLM", "MIMO", "DEEPSEEK", "COMPATIBLE", "EMBEDDING")),
        ])
        print(f"Daphnis Web  http://127.0.0.1:{args.port}\n访问令牌  {token}", flush=True)
        return Bridge(ops, redactor)

    uvicorn.run(
        create_app(open_bridge, token=token, port=args.port, assets=args.assets),
        host="127.0.0.1", port=args.port, access_log=False, proxy_headers=False,
        ws="websockets-sansio", ws_max_size=1024, ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
