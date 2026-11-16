"""Authenticated, same-origin HTTP commands and WebSocket observations."""

import asyncio
import json
import secrets
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from python_multipart.exceptions import MultipartParseError
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..attachments import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENTS,
    MAX_FILES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_TOTAL_BYTES,
    MAX_IMAGES,
    DocumentAttachmentInput,
    ImageAttachmentInput,
    TextAttachmentInput,
    TextResourceInput,
    valid_id,
)
from ..mcp.resources import resource_template, text_resource
from ..mcp_registry import RegistryDiscovery, RegistryError
from ..prompt_groups import MAX_PROMPT_MESSAGES, PROMPT_ROLES, PromptInput
from ..tools.approval import ApprovalMode
from ..turn_state import RecoveryKind

_MAX_BODY_BYTES = 256 * 1024
_ATTACHMENT_PATH = "/api/attachments/send"
_IMAGE_PATH = "/api/attachments/image"
_DOCUMENT_PATH = "/api/attachments/document"
_MAX_ATTACHMENT_BODY_BYTES = 2 * 1024 * 1024
_MAX_MULTIPART_BODY_BYTES = MAX_IMAGE_TOTAL_BYTES + MAX_DOCUMENT_BYTES + _MAX_ATTACHMENT_BODY_BYTES
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
    "compact": {},
    "send_attachments": {"prompt": str, "attachments": list, "submission_id": str},
    "retry": {"key": str},
    "continue": {"key": str},
    "terminate": {"key": str},
    "cancel": {"run_id": str},
    "approve": {"request_id": str, "decision": str},
}
_RUN_ACTIONS = {"send", "retry", "continue", "terminate", "compact"}
_PROMPT_ACTIONS = {"mcp_list_prompts", "mcp_get_prompt"}
_SESSION_SWITCHES = {"new", "project_open", "restore"}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


async def _limited_stream(request: Request, limit: int):
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise RequestError("请求过大。", 413)
        yield chunk


async def _command(request: Request):
    if request.headers.get("content-type", "").partition(";")[0] != "application/json":
        raise RequestError("需要 application/json 请求。", 415)
    body = bytearray()
    upload = request.url.path == _ATTACHMENT_PATH
    limit = _MAX_ATTACHMENT_BODY_BYTES if upload else _MAX_BODY_BYTES
    async for chunk in _limited_stream(request, limit):
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
    keep_draft = value.pop("keep_draft_session", False) if action in _SESSION_SWITCHES else False
    fields = _FIELDS[action]
    if (
        not isinstance(session, str) or not 0 < len(session) <= 255
        or not isinstance(keep_draft, bool)
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
    if keep_draft:
        value["keep_draft_session"] = True
    return action, session, value


async def _multipart_attachments(request: Request):
    parser = MultiPartParser(
        request.headers, _limited_stream(request, _MAX_MULTIPART_BODY_BYTES),
        max_files=MAX_IMAGES + MAX_DOCUMENTS, max_fields=4, max_part_size=_MAX_ATTACHMENT_BODY_BYTES,
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        status = 413 if exc.message.startswith("Part exceeded") else 400
        raise RequestError("上传表单无效或超过限制。", status) from None
    except MultipartParseError:
        raise RequestError("上传表单无效。") from None
    try:
        fields = dict(form.multi_items())
        if len(form.multi_items()) != len(fields):
            raise RequestError("上传字段重复。")
        session = fields.pop("session_id", None)
        submission = fields.pop("submission_id", None)
        prompt = fields.pop("prompt", None)
        manifest = fields.pop("manifest", None)
        if (
            not isinstance(session, str) or not 0 < len(session) <= 255 or not valid_id(submission)
            or not isinstance(prompt, str) or len(prompt.encode()) > _MAX_BODY_BYTES
            or not isinstance(manifest, str)
        ):
            raise RequestError("附件发送字段无效。")
        try:
            parts = json.loads(manifest)
            json.dumps(parts, ensure_ascii=False, allow_nan=False).encode()
        except (ValueError, RecursionError):
            raise RequestError("附件清单无效。") from None
        if not isinstance(parts, list) or not 0 < len(parts) <= MAX_FILES + MAX_IMAGES + MAX_DOCUMENTS:
            raise RequestError("附件清单无效。")

        entries = []
        image_count = text_count = document_count = image_size = 0
        for index, item in enumerate(parts):
            if not isinstance(item, dict):
                raise RequestError("附件清单无效。")
            match item:
                case {"kind": "text", "name": str(name), "text": str(text)} if len(item) == 3:
                    entries.append(TextAttachmentInput(name, text))
                    text_count += 1
                case {"kind": "resource", "resource_id": str(resource)} if len(item) == 2 and valid_id(resource):
                    entries.append(resource)
                    text_count += 1
                case {"kind": "image", "name": str(name), "field": str(field)} if len(item) == 3 and field == f"image_{index}":
                    if image_count == MAX_IMAGES:
                        raise RequestError("每轮最多可发送 4 张图片。")
                    upload = fields.pop(field, None)
                    if not isinstance(upload, UploadFile):
                        raise RequestError("图片二进制字段缺失。")
                    if upload.size is not None and upload.size > MAX_IMAGE_BYTES:
                        raise RequestError("每张图片最多 5 MiB。", 413)
                    data = await upload.read(MAX_IMAGE_BYTES + 1)
                    image_size += len(data)
                    if len(data) > MAX_IMAGE_BYTES or image_size > MAX_IMAGE_TOTAL_BYTES:
                        raise RequestError("单张图片最多 5 MiB，图片总计最多 20 MiB。", 413)
                    entries.append(ImageAttachmentInput(name, data, upload.content_type or ""))
                    image_count += 1
                case {"kind": "document", "name": str(name), "field": str(field), "reading_mode": str(mode)} if len(item) == 4 and field == f"document_{index}":
                    if document_count == MAX_DOCUMENTS:
                        raise RequestError("每轮最多可发送一份文档。")
                    upload = fields.pop(field, None)
                    if not isinstance(upload, UploadFile):
                        raise RequestError("文档二进制字段缺失。")
                    if upload.size is not None and upload.size > MAX_DOCUMENT_BYTES:
                        raise RequestError("每份文档最多 10 MiB。", 413)
                    data = await upload.read(MAX_DOCUMENT_BYTES + 1)
                    if len(data) > MAX_DOCUMENT_BYTES:
                        raise RequestError("每份文档最多 10 MiB。", 413)
                    entries.append(DocumentAttachmentInput(name, data, upload.content_type or "", mode))
                    document_count += 1
                case _:
                    raise RequestError("附件清单条目无效。")
        if fields or not (image_count or document_count) or text_count > MAX_FILES:
            raise RequestError("附件清单或数量无效。")
        return session, prompt, submission, tuple(entries)
    finally:
        await form.close()


async def _send_attachments(bridge, session, prompt, submission_id, entries):
    resource_ids = [item for item in entries if isinstance(item, str)]
    if len(set(resource_ids)) != len(resource_ids):
        raise RequestError("不能重复添加同一份 MCP 资源草稿。")
    async with bridge.lock:
        ops = bridge.ops
        if bridge.session_id != session:
            raise RequestError("当前会话已切换，请重新操作。", 409)
        if ops.attachment_submitted(submission_id, prompt, entries, session_id=session):
            return JSONResponse({"notice": None, "revision": bridge.revision}, 202)
        attachments = tuple(bridge.resource_draft(session, entry) if isinstance(entry, str) else entry for entry in entries)
        if bridge.upload and bridge.upload[0] == submission_id:
            _, old_prompt, old_attachments, accepted = bridge.upload
            if (prompt, attachments) != (old_prompt, old_attachments):
                raise RequestError("该附件发送标识已用于不同内容。", 409)
        else:
            if bridge.run is not None or ops.running(session):
                raise RequestError("请先等待或停止当前轮。", 409)
            if ops.recovery(session) and ops.recovery(session).kind is RecoveryKind.INTERRUPTED_TURN:
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
        if bridge.session_id != session:
            raise RequestError("当前会话已切换，请重新操作。", 409)
        if ops.prompt_submitted(preview_id, texts, session_id=session):
            return JSONResponse({"notice": None, "revision": bridge.revision}, 202)
        if bridge.prompt_run and bridge.prompt_run[0].id == preview_id:
            previous, accepted = bridge.prompt_run
            if texts != tuple(message["content"] for message in previous.snapshot()):
                raise RequestError("该 MCP Prompt 运行标识已用于不同内容。", 409)
        else:
            if bridge.run is not None or ops.running(session):
                raise RequestError("请先等待或停止当前轮。", 409)
            if ops.recovery(session) and ops.recovery(session).kind is RecoveryKind.INTERRUPTED_TURN:
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
    if bridge.session_id != session:
        raise RequestError("当前会话已切换，请重新操作。", 409)
    if action == "cancel":
        if not bridge.run or bridge.run["id"] != args["run_id"]:
            raise RequestError("该轮已结束。", 409)
        await bridge.cancel()
        return None
    if action == "approve":
        pending = ops.pending_approval(session)
        if (
            pending is None
            or (args["decision"] == "trust_session" and not pending.can_trust)
            or not ops.resolve_approval(args["request_id"], args["decision"], session_id=session)
        ):
            raise RequestError("该审批已失效或不支持所选选项。", 409)
        return None
    if bridge.run is not None or ops.running(session):
        raise RequestError("请先等待或停止当前轮。", 409)
    if args.get("keep_draft_session") and (action != "restore" or args["id"] != session):
        await ops.remember_session(session)
    match action:
        case "new":
            bridge.session_id = (await ops.new_session(session_id=session)).id
        case "project_create":
            await ops.create_project(args["name"], workspace=args["workspace"], session_id=session)
        case "project_open" | "project_remove":
            try:
                if action == "project_remove":
                    await ops.remove_project(args["project_id"])
                else:
                    bridge.session_id = (await ops.open_project(args["project_id"], session_id=session)).id
            except FileNotFoundError as exc:
                raise RequestError("项目不存在。", 404) from exc
        case "restore":
            if args["id"] != session:
                bridge.session_id = ops.restore_session(args["id"]).id
        case "rename":
            await ops.rename_session(args["id"], args["title"])
        case "move":
            await ops.move_session(args["id"], args["project_id"], expected_workspace=args["workspace"])
        case "delete":
            fresh = await ops.delete_session(args["id"], replacement=args["id"] == session)
            if fresh is not None:
                bridge.session_id = fresh.id
            bridge.journals.pop(args["id"], None)
        case "model":
            return await ops.set_model(args["provider"], args["model_id"], session_id=session)
        case "thinking":
            await ops.set_thinking(args["mode"], session_id=session)
        case "system":
            if args["prompt"] is None:
                await ops.reset_system_prompt(session)
            else:
                await ops.set_system_prompt(args["prompt"], session_id=session)
        case "tools":
            await ops.set_tools(args["names"], session_id=session)
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
            ops.set_approval_mode(args["mode"], session_id=session)
        case "confirm_full":
            ops.set_approval_mode(ApprovalMode.FULL, session_id=session)
        case "send" | "compact":
            if ops.recovery(session) and ops.recovery(session).kind is RecoveryKind.INTERRUPTED_TURN:
                raise RequestError("上轮任务未完成，请先选择继续或终止。", 409)
            if action == "compact":
                bridge.start_compaction()
            else:
                bridge.start(args["prompt"])
        case "retry" | "continue" | "terminate":
            recovery = ops.recovery(session)
            kind = RecoveryKind.FAILED_USER if action == "retry" else RecoveryKind.INTERRUPTED_TURN
            if recovery is None or recovery.kind is not kind or recovery.key != args["key"]:
                raise RequestError("恢复状态已变化，请刷新后再操作。", 409)
            operation = {
                "retry": ops.retry_failed_user,
                "continue": ops.continue_interrupted_turn,
                "terminate": ops.terminate_interrupted_turn,
            }[action]
            bridge.start(None, operation=lambda: operation(recovery.key, session_id=session))
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
    if bridge.session_id != session:
        raise RequestError("当前会话已切换，请重新打开 Resources。", 409)
    server = args["server"]
    redactor = bridge.redactor
    if action == "mcp_read_resource":
        resources = await ops.read_mcp_resource(server, args["uri"], args["arguments"])
        if bridge.session_id != session:
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
    if bridge.session_id != session:
        raise RequestError("当前会话已切换，请重新打开 Resources。", 409)
    return JSONResponse({"mcp_resource_catalog": response})


async def _prompt_response(bridge, action, session, args):
    if bridge.session_id != session:
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
        if bridge.session_id != session:
            raise RequestError("当前会话已切换，请重新打开 Prompts。", 409)
        preview_id = bridge.cache_prompt_preview(
            session, args["server"], args["name"], tuple(original),
        ) if runnable else None
        response = {"mcp_prompt_result": {
            "description": redactor.text(result.description or ""), "messages": messages, "redacted": redacted,
            "preview_id": preview_id,
        }}
    if bridge.session_id != session:
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

    def bearer_authenticated(headers):
        scheme, _, candidate = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and authenticated(candidate)

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
        if not bearer_authenticated(request.headers):
            return JSONResponse({"error": "访问令牌无效。"}, 401)
        bridge = request.app.state.bridge
        try:
            if request.url.path == _ATTACHMENT_PATH and request.headers.get("content-type", "").partition(";")[0].lower() == "multipart/form-data":
                return await _send_attachments(bridge, *await _multipart_attachments(request))
            action, session, args = await _command(request)
            if action == "send_attachments":
                entries = tuple(item["resource_id"] if "resource_id" in item else TextAttachmentInput(**item)
                                for item in args["attachments"])
                return await _send_attachments(bridge, session, args["prompt"], args["submission_id"], entries)
            if action == "mcp_run_prompt":
                return await _run_prompt(bridge, session, args)
            if action in {"mcp_list_resources", "mcp_read_resource"}:
                return await _resource_response(bridge, action, session, args)
            if action in _PROMPT_ACTIONS:
                return await _prompt_response(bridge, action, session, args)
            if action in {"registry_search", "registry_detail"}:
                if bridge.session_id != session:
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

    async def attachment_binary(request: Request):
        document = request.url.path == _DOCUMENT_PATH
        label = "文档" if document else "图片"
        headers = request.headers
        if headers.get("origin", origin) != origin or headers.get("sec-fetch-site", "same-origin") != "same-origin":
            return JSONResponse({"error": "请求来源不受信任。"}, 403)
        if not bearer_authenticated(headers):
            return JSONResponse({"error": "访问令牌无效。"}, 401)
        params = request.query_params
        if (
            len(params.multi_items()) != 3 or params.keys() != {"session_id", "message_index", "attachment_id"}
            or len(params["session_id"]) > 255 or not valid_id(params["attachment_id"])
            or not 0 < len(index := params["message_index"]) <= 10 or not index.isascii() or not index.isdecimal()
        ):
            return JSONResponse({"error": f"{label}引用无效。"}, 400)
        ops = request.app.state.bridge.ops
        args = (params["session_id"], int(index), params["attachment_id"])
        try:
            if document:
                data, mime_type, name = await ops.read_attachment_document(*args)
                return Response(data, media_type=mime_type, headers={
                    "Content-Disposition": "attachment; filename*=UTF-8''" + quote(name, safe=""),
                })
            data, mime_type = await ops.read_attachment_image(*args)
        except FileNotFoundError:
            return JSONResponse({"error": f"{label}不可用。"}, 404)
        return Response(data, media_type=mime_type)

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
            Route(_IMAGE_PATH, attachment_binary, methods=["GET"]),
            Route(_DOCUMENT_PATH, attachment_binary, methods=["GET"]),
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
                f"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
                f"connect-src 'self' ws://{host}; base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        })
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=security_headers)
    return app
