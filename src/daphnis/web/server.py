"""Authenticated, same-origin HTTP commands and WebSocket observations."""

import asyncio
import json
import secrets
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..attachments import MAX_FILES, TextAttachmentInput, TextResourceInput, valid_id
from ..mcp.resources import resource_template, text_resource
from ..mcp_registry import RegistryDiscovery, RegistryError
from ..prompt_groups import MAX_PROMPT_MESSAGES, PROMPT_ROLES, PromptInput
from ..tools.approval import ApprovalMode
from ..turn_state import RecoveryKind

_MAX_BODY_BYTES = 256 * 1024
_ATTACHMENT_PATH = "/api/attachments/send"
_MAX_ATTACHMENT_BODY_BYTES = 2 * 1024 * 1024
_FIELDS = {
    "new": {},
    "project_create": {"name": str, "workspace": (str, type(None))},
    "project_open": {"project_id": (str, type(None))},
    "project_remove": {"project_id": str},
    "restore": {"id": str},
    "rename": {"id": str, "title": str},
    "move": {"id": str, "project_id": (str, type(None)), "workspace": (str, type(None))},
    "delete": {"id": str},
    "model": {"provider": str, "model_id": str},
    "thinking": {"mode": (str, type(None))},
    "system": {"prompt": (str, type(None))},
    "tools": {"names": list},
    "mcp_save": {"name": str, "enabled": bool, "options": dict, "create": bool},
    "mcp_enable": {"name": str, "enabled": bool},
    "mcp_restart": {"name": str},
    "mcp_delete": {"name": str},
    "mcp_list_prompts": {"server": str},
    "mcp_get_prompt": {"server": str, "name": str, "arguments": dict},
    "mcp_run_prompt": {"preview_id": str, "texts": list},
    "mcp_list_resources": {"server": str},
    "mcp_read_resource": {"server": str, "uri": str, "arguments": (dict, type(None))},
    "registry_search": {"query": str, "cursor": (str, type(None))},
    "registry_detail": {"registry_name": str, "version": str},
    "mcp_add_registry": {"name": str, "registry_name": str, "version": str, "url": str, "auth_env": (str, type(None))},
    "set_approval_mode": {"mode": str},
    "confirm_full": {},
    "send": {"prompt": str},
    "send_attachments": {"prompt": str, "attachments": list, "submission_id": str},
    "retry": {"key": str},
    "continue": {"key": str},
    "terminate": {"key": str},
    "cancel": {"run_id": str},
    "approve": {"request_id": str, "decision": str},
}
_RUN_ACTIONS = {"send", "retry", "continue", "terminate"}
_PROMPT_ACTIONS = {"mcp_list_prompts", "mcp_get_prompt"}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


async def _command(request: Request):
    if request.headers.get("content-type", "").partition(";")[0] != "application/json":
        raise RequestError("需要 application/json 请求。", 415)
    body = bytearray()
    upload = request.url.path == _ATTACHMENT_PATH
    limit = _MAX_ATTACHMENT_BODY_BYTES if upload else _MAX_BODY_BYTES
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
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
    if upload != (action == "send_attachments"):
        raise RequestError("请求地址和操作不匹配。")
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
    if action == "send_attachments" and (
        not valid_id(value["submission_id"]) or not 0 < len(value["attachments"]) <= MAX_FILES
        or len(value["prompt"].encode()) > _MAX_BODY_BYTES
        or any(not isinstance(item, dict) or not (
            item.keys() == {"resource_id"} and valid_id(item["resource_id"])
            or item.keys() == {"name", "text"} and all(isinstance(field, str) for field in item.values())
        ) for item in value["attachments"])
    ):
        raise RequestError("附件发送字段无效。")
    if action == "move" and (value["project_id"] is None) != (value["workspace"] is None):
        raise RequestError("目标项目与工作目录必须同时提供。")
    if action == "approve" and value["decision"] not in {"deny", "allow_once", "trust_session"}:
        raise RequestError("审批选项无效。")
    if action == "set_approval_mode":
        if value["mode"] == ApprovalMode.FULL:
            raise RequestError("启用 FULL 前需要确认风险。")
        if value["mode"] not in tuple(ApprovalMode):
            raise RequestError("审批模式无效。")
    return action, session, value


async def _send_attachments(bridge, session, args):
    prompt = args["prompt"]
    submission_id = args["submission_id"]
    entries = tuple(item["resource_id"] if "resource_id" in item else TextAttachmentInput(**item)
                    for item in args["attachments"])
    resource_ids = [item for item in entries if isinstance(item, str)]
    if len(set(resource_ids)) != len(resource_ids):
        raise RequestError("不能重复添加同一份 MCP 资源草稿。")
    async with bridge.lock:
        ops = bridge.ops
        if ops.session.id != session:
            raise RequestError("当前会话已切换，请重新操作。", 409)
        if ops.attachment_submitted(submission_id, prompt, entries):
            return JSONResponse({"notice": None, "revision": bridge.revision}, 202)
        attachments = tuple(bridge.resource_draft(session, entry) if isinstance(entry, str) else entry for entry in entries)
        if bridge.upload and bridge.upload[0] == submission_id:
            _, old_prompt, old_attachments, accepted = bridge.upload
            if (prompt, attachments) != (old_prompt, old_attachments):
                raise RequestError("该附件发送标识已用于不同内容。", 409)
        else:
            if bridge.run is not None or ops.running:
                raise RequestError("请先等待或停止当前轮。", 409)
            if ops.recovery and ops.recovery.kind is RecoveryKind.INTERRUPTED_TURN:
                raise RequestError("上轮任务未完成，请先选择继续或终止。", 409)
            accepted = bridge.start(prompt, attachments=attachments, submission_id=submission_id)
    error = await asyncio.shield(accepted)
    if error is not None:
        raise RequestError(error, 409)
    return JSONResponse({"notice": None, "revision": bridge.revision}, 202)


async def _run_prompt(bridge, session, args):
    preview_id = args["preview_id"]
    texts = tuple(args["texts"])
    if (not valid_id(preview_id) or not 0 < len(texts) <= MAX_PROMPT_MESSAGES
        or any(not isinstance(text, str) for text in texts)):
        raise RequestError("MCP Prompt 运行字段无效。")
    async with bridge.lock:
        ops = bridge.ops
        if ops.session.id != session:
            raise RequestError("当前会话已切换，请重新操作。", 409)
        if ops.prompt_submitted(preview_id, texts):
            return JSONResponse({"notice": None, "revision": bridge.revision}, 202)
        if bridge.prompt_run and bridge.prompt_run[0].id == preview_id:
            previous, accepted = bridge.prompt_run
            if texts != tuple(message["content"] for message in previous.snapshot()):
                raise RequestError("该 MCP Prompt 运行标识已用于不同内容。", 409)
        else:
            if bridge.run is not None or ops.running:
                raise RequestError("请先等待或停止当前轮。", 409)
            if ops.recovery and ops.recovery.kind is RecoveryKind.INTERRUPTED_TURN:
                raise RequestError("上轮任务未完成，请先选择继续或终止。", 409)
            preview = bridge.prompt_previews.get(preview_id)
            if preview is None or preview[0] != session:
                raise RequestError("MCP Prompt 预览已失效，请重新生成。", 409)
            _, server, name, original = preview
            roles = tuple(role for role, _ in original)
            if len(texts) == len(roles) + 1:
                roles = (*roles, "user")
            if len(texts) != len(roles):
                raise RequestError("需保留预览中的全部消息及顺序。")
            if roles[-1] != "user":
                raise RequestError("当前模型接口需要以 User 消息结束。请补充一条 User 消息。")
            if any(bridge.redactor.text(text) != text for text in texts):
                raise RequestError("编辑内容中检测到凭据。请移除凭据并重新确认。")
            values = []
            for index, text in enumerate(texts):
                original_value = original[index][1] if index < len(original) else None
                if isinstance(original_value, TextResourceInput):
                    if text != "":
                        raise RequestError("内嵌资源只读，请保留确认预览的内容。")
                    values.append(original_value)
                else:
                    values.append(text)
            messages = tuple(zip(roles, values, strict=True))
            prompt_input = PromptInput(preview_id, server, name, messages, edited=messages != original)
            accepted = bridge.start(None, prompt_input=prompt_input)
    error = await asyncio.shield(accepted)
    if error is not None:
        raise RequestError(error, 409)
    return JSONResponse({"notice": None, "revision": bridge.revision}, 202)


async def _execute(bridge, action, session, args, discovery):
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
        case "project_create":
            await ops.create_project(args["name"], workspace=args["workspace"])
        case "project_open" | "project_remove":
            try:
                if action == "project_remove":
                    await ops.remove_project(args["project_id"])
                else:
                    await ops.open_project(args["project_id"])
            except FileNotFoundError as exc:
                raise RequestError("项目不存在。", 404) from exc
        case "restore":
            if args["id"] != session:
                ops.restore_session(args["id"])
        case "rename":
            await ops.rename_session(args["id"], args["title"])
        case "move":
            await ops.move_session(args["id"], args["project_id"], expected_workspace=args["workspace"])
        case "delete":
            await ops.delete_session(args["id"])
            bridge.journals.pop(args["id"], None)
        case "model":
            return await ops.set_model(args["provider"], args["model_id"])
        case "thinking":
            await ops.set_thinking(args["mode"])
        case "system":
            if args["prompt"] is None:
                await ops.reset_system_prompt()
            else:
                await ops.set_system_prompt(args["prompt"])
        case "tools":
            await ops.set_tools(args["names"])
        case "mcp_save" | "mcp_enable":
            return await ops.save_mcp(**args)
        case "mcp_restart":
            return await ops.restart_mcp(args["name"])
        case "mcp_delete":
            await ops.delete_mcp(args["name"])
        case "mcp_add_registry":
            server = await discovery.detail(args["registry_name"], args["version"])
            remote = next((item for item in server["remotes"] if (
                item["url"] == args["url"] and item["auth"] != "unsupported"
            )), None)
            if server["status"] != "active" or remote is None:
                raise RequestError("该地址无法通过在线发现添加。")
            auth_env = args["auth_env"]
            if remote["auth"] == "bearer" and not auth_env:
                raise RequestError("请填写 Bearer Token 环境变量名。")
            options = {
                "transport": "streamable_http", "url": remote["url"], "public_only": True,
            }
            if auth_env:
                options["auth"] = {"type": "bearer", "env": auth_env}
            return await ops.save_mcp(args["name"], enabled=True, options=options, create=True)
        case "set_approval_mode":
            ops.set_approval_mode(args["mode"])
        case "confirm_full":
            ops.set_approval_mode(ApprovalMode.FULL)
        case "send":
            if ops.recovery and ops.recovery.kind is RecoveryKind.INTERRUPTED_TURN:
                raise RequestError("上轮任务未完成，请先选择继续或终止。", 409)
            bridge.start(args["prompt"])
        case "retry" | "continue" | "terminate":
            recovery = ops.recovery
            kind = RecoveryKind.FAILED_USER if action == "retry" else RecoveryKind.INTERRUPTED_TURN
            if recovery is None or recovery.kind is not kind or recovery.key != args["key"]:
                raise RequestError("恢复状态已变化，请刷新后再操作。", 409)
            operation = {
                "retry": ops.retry_failed_user,
                "continue": ops.continue_interrupted_turn,
                "terminate": ops.terminate_interrupted_turn,
            }[action]
            bridge.start(None, operation=lambda: operation(recovery.key))
    return None


def _resource_preview(item: TextResourceInput, redactor) -> tuple[TextResourceInput, dict]:
    if redactor.text(item.uri) != item.uri or redactor.text(item.mime_type) != item.mime_type:
        raise ValueError("MCP 资源标识包含敏感内容。")
    visible = replace(item, name=redactor.text(item.name), text=redactor.json(item.text))
    return visible, {
        "name": visible.name, "uri": visible.uri, "mime_type": visible.mime_type,
        "text": visible.text, "size_bytes": len(visible.text.encode("utf-8")),
        "redacted": visible.text != item.text,
    }


async def _resource_response(bridge, action, session, args):
    ops = bridge.ops
    if ops.session.id != session:
        raise RequestError("当前会话已切换，请重新打开 Resources。", 409)
    server = args["server"]
    redactor = bridge.redactor
    if action == "mcp_read_resource":
        resources = await ops.read_mcp_resource(server, args["uri"], args["arguments"])
        if ops.session.id != session:
            raise RequestError("当前会话已切换，请重新打开 Resources。", 409)
        previews = [_resource_preview(item, redactor) for item in resources]
        return JSONResponse({"mcp_resource_preview": [
            {**data, "id": bridge.cache_resource_draft(session, item)} for item, data in previews
        ]})

    response = {"entries": [], "errors": []}
    for templates, load in ((False, ops.list_mcp_resources), (True, ops.list_mcp_resource_templates)):
        try:
            entries = await load(server)
        except (RuntimeError, ValueError) as exc:
            response["errors"].append(redactor.text(str(exc)))
            continue
        for entry in entries:
            uri = entry.uri_template if templates else str(entry.uri)
            if redactor.text(uri) != uri:
                continue
            variables = None
            error = None
            if templates:
                try:
                    variables = list(resource_template(uri).variable_names)
                except ValueError as exc:
                    error = str(exc)
            response["entries"].append({
                "uri": uri, "name": redactor.text(entry.name), "title": redactor.text(entry.title or ""),
                "description": redactor.text(entry.description or ""), "mime_type": redactor.text(entry.mime_type or ""),
                "variables": variables, "error": error,
            })
    if ops.session.id != session:
        raise RequestError("当前会话已切换，请重新打开 Resources。", 409)
    return JSONResponse({"mcp_resource_catalog": response})


async def _prompt_response(bridge, action, session, args):
    if bridge.ops.session.id != session:
        raise RequestError("当前会话已切换，请重新打开模板。", 409)
    redactor = bridge.redactor
    if action == "mcp_list_prompts":
        prompts = await bridge.ops.list_mcp_prompts(args["server"])
        response = {"mcp_prompts": [{
            "name": prompt.name,
            "title": redactor.text(prompt.title or ""),
            "description": redactor.text(prompt.description or ""),
            "arguments": [{
                "name": field.name, "title": redactor.text(field.title or ""),
                "description": redactor.text(field.description or ""), "required": bool(field.required),
            } for field in (prompt.arguments or ())],
        } for prompt in prompts if redactor.text(prompt.name) == prompt.name and all(
            redactor.text(field.name) == field.name for field in (prompt.arguments or ())
        )]}
    else:
        result = await bridge.ops.get_mcp_prompt(args["server"], args["name"], args["arguments"])
        messages = []
        original = []
        redacted = False
        runnable = 0 < len(result.messages) <= MAX_PROMPT_MESSAGES
        for message in result.messages:
            content = {"type": message.content.type}
            value = None
            if message.content.type == "text":
                value = redactor.text(message.content.text)
                content["text"] = value
                redacted |= value != message.content.text
            if message.content.type == "resource":
                try:
                    value, resource = _resource_preview(
                        text_resource(args["server"], message.content.resource, "embedded"), redactor,
                    )
                except (TypeError, ValueError) as exc:
                    content["error"] = str(exc)
                else:
                    content["resource"] = resource
                    redacted |= resource["redacted"]
            runnable &= value is not None and message.role in PROMPT_ROLES
            original.append((message.role, value))
            messages.append({"role": message.role, "content": content})
        if bridge.ops.session.id != session:
            raise RequestError("当前会话已切换，请重新打开 Prompts。", 409)
        preview_id = bridge.cache_prompt_preview(
            session, args["server"], args["name"], tuple(original),
        ) if runnable else None
        response = {"mcp_prompt_result": {
            "description": redactor.text(result.description or ""), "messages": messages, "redacted": redacted,
            "preview_id": preview_id,
        }}
    if bridge.ops.session.id != session:
        raise RequestError("当前会话已切换，请重新打开模板。", 409)
    return JSONResponse(response)


def create_app(open_bridge, *, token: str, port: int, assets: Path):
    host = f"127.0.0.1:{port}"
    origin = f"http://{host}"
    discovery = RegistryDiscovery()

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
            if action == "send_attachments":
                return await _send_attachments(bridge, session, args)
            if action == "mcp_run_prompt":
                return await _run_prompt(bridge, session, args)
            if action in {"mcp_list_resources", "mcp_read_resource"}:
                return await _resource_response(bridge, action, session, args)
            if action in _PROMPT_ACTIONS:
                return await _prompt_response(bridge, action, session, args)
            if action in {"registry_search", "registry_detail"}:
                if bridge.ops.session.id != session:
                    raise RequestError("当前会话已切换，请重新操作。", 409)
                if action == "registry_search":
                    return JSONResponse({"registry_page": await discovery.search(args["query"], args["cursor"])})
                return JSONResponse({"registry_server": await discovery.detail(args["registry_name"], args["version"])})
            if action in {"approve", "cancel"}:
                notice = await _execute(bridge, action, session, args, discovery)
            else:
                if bridge.lock.locked():
                    raise RequestError("正在更新会话，请稍后重试。", 409)
                async with bridge.lock:
                    notice = await _execute(bridge, action, session, args, discovery)
            bridge._notify(state=action not in _RUN_ACTIONS | {"cancel", "approve"})
            return JSONResponse(
                {"notice": bridge.redactor.json(notice), "revision": bridge.revision},
                202 if action in _RUN_ACTIONS else 200,
            )
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, exc.status)
        except RegistryError as exc:
            return JSONResponse({"error": str(exc)}, 502)
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
            Route(_ATTACHMENT_PATH, command, methods=["POST"]),
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
