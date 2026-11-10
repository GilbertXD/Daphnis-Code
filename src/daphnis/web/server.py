"""Authenticated, same-origin HTTP commands and WebSocket observations."""

import asyncio
import json
import secrets
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

_MAX_BODY_BYTES = 256 * 1024
_FIELDS = {
    "new": {},
    "restore": {"id": str},
    "rename": {"id": str, "title": str},
    "delete": {"id": str},
    "model": {"provider": str, "model_id": str},
    "thinking": {"mode": (str, type(None))},
    "tools": {"names": list},
    "send": {"prompt": str},
    "cancel": {"run_id": str},
    "approve": {"request_id": str, "decision": str},
}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


async def _command(request: Request):
    if request.headers.get("content-type", "").partition(";")[0] != "application/json":
        raise RequestError("需要 application/json 请求。", 415)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_BODY_BYTES:
            raise RequestError("请求过大。", 413)
        body.extend(chunk)
    try:
        value = json.loads(body.decode())
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, RecursionError):
        raise RequestError("请求 JSON 无效。") from None
    if not isinstance(value, dict):
        raise RequestError("需要 JSON 对象。")
    action = value.pop("action", None)
    session = value.pop("session_id", None)
    if not isinstance(action, str) or action not in _FIELDS:
        raise RequestError("未知操作。")
    fields = _FIELDS[action]
    if (
        not isinstance(session, str) or not 0 < len(session) <= 255
        or value.keys() != fields.keys()
        or any(not isinstance(value[key], kind) for key, kind in fields.items())
    ):
        raise RequestError("请求字段无效。")
    if action == "tools" and (
        len(value["names"]) > 1000
        or any(not isinstance(name, str) or not 0 < len(name) <= 255 for name in value["names"])
    ):
        raise RequestError("工具列表无效。")
    if action == "send" and not value["prompt"].strip():
        raise RequestError("请输入消息。")
    if action == "approve" and value["decision"] not in {"deny", "allow_once", "trust_session"}:
        raise RequestError("审批选项无效。")
    return action, session, value


async def _execute(bridge, action, session, args):
    ops = bridge.ops
    if ops.session.id != session:
        raise RequestError("当前会话已切换，请重新操作。", 409)
    if action == "cancel":
        if not bridge.run or bridge.run["id"] != args["run_id"]:
            raise RequestError("该轮已结束。", 409)
        await bridge.cancel()
        return None
    if action == "approve":
        pending = ops.pending_approval
        if (
            pending is None
            or (args["decision"] == "trust_session" and not pending.can_trust)
            or not ops.resolve_approval(args["request_id"], args["decision"])
        ):
            raise RequestError("该审批已失效或不支持所选选项。", 409)
        return None
    if bridge.run is not None or ops.running:
        raise RequestError("请先等待或停止当前轮。", 409)
    match action:
        case "new":
            await ops.new_session()
        case "restore":
            if args["id"] != session:
                ops.restore_session(args["id"])
        case "rename":
            await ops.rename_session(args["id"], args["title"])
        case "delete":
            await ops.delete_session(args["id"])
            bridge.journals.pop(args["id"], None)
        case "model":
            return await ops.set_model(args["provider"], args["model_id"])
        case "thinking":
            await ops.set_thinking(args["mode"])
        case "tools":
            await ops.set_tools(args["names"])
        case "send":
            bridge.start(args["prompt"])
    return None


def create_app(open_bridge, *, token: str, port: int, assets: Path):
    host = f"127.0.0.1:{port}"
    origin = f"http://{host}"

    def trusted(headers):
        return headers.get("host") == host and headers.get("origin") == origin

    def authenticated(candidate):
        return isinstance(candidate, str) and secrets.compare_digest(
            token.encode(), candidate.encode(errors="replace")
        )

    @asynccontextmanager
    async def lifespan(app):
        bridge = await open_bridge()
        app.state.bridge = bridge
        try:
            yield
        finally:
            await bridge.close()

    async def command(request: Request):
        if not trusted(request.headers):
            return JSONResponse({"error": "请求来源不受信任。"}, 403)
        scheme, _, candidate = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not authenticated(candidate):
            return JSONResponse({"error": "访问令牌无效。"}, 401)
        bridge = request.app.state.bridge
        try:
            action, session, args = await _command(request)
            if action in {"approve", "cancel"}:
                notice = await _execute(bridge, action, session, args)
            else:
                if bridge.lock.locked():
                    raise RequestError("正在更新会话，请稍后重试。", 409)
                async with bridge.lock:
                    notice = await _execute(bridge, action, session, args)
            bridge._notify(state=action not in {"send", "cancel", "approve"})
            return JSONResponse(
                {"notice": bridge.redactor.json(notice), "revision": bridge.revision},
                202 if action == "send" else 200,
            )
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, exc.status)
        except FileNotFoundError:
            return JSONResponse({"error": "会话不存在。"}, 404)
        except (RuntimeError, ValueError, TypeError) as exc:
            return JSONResponse({"error": bridge.redactor.text(str(exc))}, 409)
        except Exception as exc:  # noqa: BLE001 - Report a redacted transport failure.
            return JSONResponse({"error": bridge.redactor.text(str(exc))}, 500)

    async def events(socket: WebSocket):
        if not trusted(socket.headers):
            await socket.close(1008)
            return
        await socket.accept()
        reader = watcher = None
        try:
            try:
                candidate = await asyncio.wait_for(socket.receive_text(), timeout=5)
            except (TimeoutError, KeyError):
                candidate = None
            if not authenticated(candidate):
                await socket.close(4401)
                return
            bridge = socket.app.state.bridge
            reader = asyncio.create_task(socket.receive())
            session, revision, cursor = "", -1, 0
            while True:
                changed = bridge.changed
                try:
                    frame = bridge.frame(session, revision, cursor)
                except Exception as exc:  # noqa: BLE001 - Report an unavailable snapshot.
                    await socket.send_json({"error": bridge.redactor.text(str(exc))})
                    await socket.close(1011)
                    return
                await socket.send_json(frame)
                session, revision, cursor = frame["session_id"], frame["revision"], frame["cursor"]
                watcher = asyncio.create_task(changed.wait())
                done, _ = await asyncio.wait((reader, watcher), return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    if reader.result()["type"] != "websocket.disconnect":
                        await socket.close(1008)
                    return
                # Coalesce a burst of model deltas before encoding the next frame.
                await asyncio.sleep(0.04)
        except (WebSocketDisconnect, OSError):
            pass
        finally:
            for task in (reader, watcher):
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/api/command", command, methods=["POST"]),
            WebSocketRoute("/api/events", events),
            Mount("/", app=StaticFiles(directory=assets, html=True)),
        ],
    )

    async def security_headers(request, call_next):
        if request.headers.get("host") != host:
            return JSONResponse({"error": "无效的 Host。"}, 403)
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                f"default-src 'none'; script-src 'self'; style-src 'self'; "
                f"connect-src 'self' ws://{host}; base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        })
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=security_headers)
    return app
