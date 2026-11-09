"""Select and authorize tools before delegating execution to Saturnus."""

import json
from collections.abc import Iterable
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from saturnus import ToolRegistry, ToolResult

from ..events import EventBus
from ..mcp import MCPConnection
from ..session import Session
from .approval import (
    ApprovalCallback,
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    CleanupError,
    Decision,
    Operation,
    Target,
    ToolError,
    failure,
)
from .files import MAX_FILE_BYTES, prepare_list, prepare_patch, prepare_read
from .search import prepare_search
from .shell import prepare_command


@dataclass(frozen=True)
class _Binding:
    workspace: Target
    session_id: str
    turn_id: str
    events: EventBus


@dataclass(frozen=True)
class _Permit:
    name: str
    arguments: dict[str, Any]
    operation: Operation


_PERMIT: ContextVar[_Permit | None] = ContextVar("daphnis_tool_permit", default=None)
_PREPARERS = {
    "read_file": prepare_read,
    "list_directory": prepare_list,
    "search": prepare_search,
    "apply_patch": prepare_patch,
    "run_command": prepare_command,
}


class _ExecutionDecision(Enum):
    DENY = auto()
    EXECUTE = auto()
    PROMPT = auto()


def _execution_decision(
    operation: Operation, mode: ApprovalMode, *, trusted: bool, forced: bool
) -> _ExecutionDecision:
    if operation.decision is Decision.DENY:
        return _ExecutionDecision.DENY
    if forced or mode is ApprovalMode.ASK:
        return _ExecutionDecision.PROMPT
    if mode is ApprovalMode.FULL or operation.decision is Decision.ALLOW:
        return _ExecutionDecision.EXECUTE
    if operation.allow_session_trust and trusted:
        return _ExecutionDecision.EXECUTE
    return _ExecutionDecision.PROMPT


def _guarded_handler(name):
    async def handler(**arguments):
        permit = _PERMIT.get()
        if permit is None or permit.name != name or permit.arguments != arguments:
            return failure(
                "permission_denied", "Use Daphnis ToolExecutor.execute for this tool."
            )
        try:
            permit.operation.verify()
            return await permit.operation.run()
        except CleanupError as exc:
            # Runtime import avoids a dependency cycle with Agent's tools interface.
            from ..agent import FatalToolError

            raise FatalToolError(str(exc)) from exc
        except ToolError as exc:
            return failure(exc.code, str(exc), **exc.details)
        except UnicodeError as exc:
            return failure("invalid_utf8", str(exc))
        except FileNotFoundError as exc:
            return failure("file_not_found", str(exc))
        except PermissionError as exc:
            return failure("permission_denied", str(exc))
        except OSError as exc:
            return failure("io_error", str(exc))

    return handler


class ToolExecutor:
    """A selected registry with per-call authorization and per-Turn binding."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        approval: ApprovalCallback | None = None,
        binding: _Binding | None = None,
        mcp_tools: dict[str, tuple[MCPConnection, str]] | None = None,
        mode: ApprovalMode = ApprovalMode.AUTO,
        trusted: set[tuple[str, str]] | None = None,
    ) -> None:
        self._registry = registry
        self._approval = approval
        self._binding = binding
        self._mcp_tools = mcp_tools if mcp_tools is not None else {}
        self._mode = mode
        self._trusted = trusted if trusted is not None else set()

    def definitions(self) -> list[dict[str, Any]]:
        return self._registry.definitions()

    def select(
        self, names: list[str], *, mode: ApprovalMode | None = None
    ) -> "ToolExecutor":
        return ToolExecutor(
            self._registry.select(names),
            approval=self._approval,
            binding=self._binding,
            mcp_tools=self._mcp_tools,
            mode=self._mode if mode is None else mode,
            trusted=self._trusted,
        )

    def bind(
        self, *, session: Session, events: EventBus, turn_id: str
    ) -> "ToolExecutor":
        workspace = Target.capture(session.workspace, Path.cwd())
        if workspace.identity is None or not workspace.path.is_dir():
            raise ValueError(
                "Built-in tools require an existing Session workspace directory."
            )
        return ToolExecutor(
            self._registry,
            approval=self._approval,
            binding=_Binding(workspace, session.id, turn_id, events),
            mcp_tools=self._mcp_tools,
            mode=self._mode,
            trusted=self._trusted,
        )

    async def execute(
        self, call: dict[str, Any], *, request_number: int = 0
    ) -> ToolResult:
        call = deepcopy(call)
        arguments = self._registry.validate_call(call)
        name = call["function"]["name"]
        binding = self._binding

        def result(value):
            return ToolResult(
                tool_call_id=call["id"],
                value=value,
                content=json.dumps(value, ensure_ascii=False, allow_nan=False),
                arguments=deepcopy(arguments),
            )

        if binding is None:
            return result(
                failure(
                    "permission_denied", "Bind tools to a Session before execution."
                )
            )
        fields = {
            "session_id": binding.session_id,
            "turn_id": binding.turn_id,
            "request_number": request_number,
            "tool_call_id": call["id"],
            "name": name,
        }
        try:
            if name in self._mcp_tools:
                connection, original_name = self._mcp_tools[name]
                if not connection.available:
                    raise ToolError("mcp_unavailable", "MCP 连接不可用。")

                async def run():
                    return await connection.call(original_name, arguments)

                operation = Operation(
                    decision=Decision.ASK,
                    reason=f"调用 MCP Server {connection.name} 的工具需要审批。",
                    targets=(),
                    details={
                        "server": connection.name,
                        "tool": original_name,
                        "arguments": arguments,
                    },
                    run=run,
                )
            else:
                operation = _PREPARERS[name](arguments, binding.workspace.path)
            forced = operation.force_approval or any(
                not target.path.is_relative_to(binding.workspace.path)
                for target in operation.targets
            )
            policy = _execution_decision(
                operation,
                self._mode,
                trusted=(binding.session_id, name) in self._trusted,
                forced=forced,
            )
            if policy is _ExecutionDecision.DENY:
                raise ToolError("command_blocked", operation.reason)
            if policy is _ExecutionDecision.PROMPT:
                can_trust = (
                    self._mode is ApprovalMode.AUTO
                    and not forced
                    and operation.allow_session_trust
                )
                request = ApprovalRequest(
                    session_id=binding.session_id,
                    turn_id=binding.turn_id,
                    tool_call_id=call["id"],
                    tool_name=name,
                    workspace=str(binding.workspace.path),
                    reason=operation.reason,
                    details={
                        **operation.details,
                        "targets": [str(t.path) for t in operation.targets],
                    },
                    can_trust=can_trust,
                )
                await binding.events.emit(
                    "tool.approval_requested", **fields, request=asdict(request)
                )
                decision = ApprovalDecision.DENY
                if self._approval is not None:
                    try:
                        decision = await self._approval(deepcopy(request))
                    except Exception as exc:
                        raise ToolError(
                            "permission_denied",
                            f"Approval callback failed: {type(exc).__name__}",
                        ) from exc
                if decision is not ApprovalDecision.ALLOW_ONCE and not (
                    can_trust and decision is ApprovalDecision.TRUST_SESSION
                ):
                    raise ToolError(
                        "permission_denied",
                        "Approval was refused or no approval callback is available.",
                    )
            # The workspace may gain files, but must not be replaced or retargeted.
            current = Target.capture(
                binding.workspace.requested, binding.workspace.path
            )
            if (
                current.path != binding.workspace.path
                or current.identity is None
                or current.identity[:3] != binding.workspace.identity[:3]
            ):
                raise ToolError(
                    "target_changed", "Session workspace changed; rebind tools."
                )
            operation.verify()
            token = _PERMIT.set(_Permit(name, deepcopy(arguments), operation))
            try:
                return await self._registry.execute(call)
            finally:
                _PERMIT.reset(token)
        except ToolError as exc:
            if exc.code in ("permission_denied", "command_blocked"):
                await binding.events.emit(
                    "tool.blocked",
                    **fields,
                    decision=Decision.DENY.value
                    if exc.code == "command_blocked"
                    else Decision.ASK.value,
                    code=exc.code,
                    reason=str(exc),
                )
            return result(failure(exc.code, str(exc), **exc.details))
        except (OSError, ValueError) as exc:
            return result(failure("invalid_operation", str(exc)))


def create_tools(
    *,
    approval: ApprovalCallback | None = None,
    connections: Iterable[MCPConnection] = (),
    trusted: set[tuple[str, str]] | None = None,
) -> ToolExecutor:
    """Build the tool catalogue; use select() to choose the enabled subset."""
    registry = ToolRegistry()
    text_path = {"type": "string", "minLength": 1}
    definitions = {
        "read_file": (
            "Read bounded UTF-8 text, optionally selecting 1-based inclusive lines.",
            {
                "path": text_path,
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_FILE_BYTES,
                },
            },
            ["path"],
        ),
        "list_directory": (
            "List a directory without following symlink entries.",
            {
                "path": text_path,
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            [],
        ),
        "search": (
            "Bounded filename/content search; skips symlinks and common caches.",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "path": text_path,
                "mode": {"enum": ["filename", "content", "both"]},
                "case_sensitive": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            ["query"],
        ),
        "apply_patch": (
            (
                "Apply plain unified diff: ---/+++ headers (a/ and b/ optional), "
                "numbered @@ hunks, /dev/null for create/delete. Exact original text "
                "required; no rename or Git metadata headers. Requires approval "
                "and reports partial filesystem effects."
            ),
            {"patch": {"type": "string", "minLength": 1, "maxLength": MAX_FILE_BYTES}},
            ["patch"],
        ),
        "run_command": (
            (
                "Execute argv, never a shell string. No pipelines, shell/inline "
                "interpreters or dangerous commands. Scripts/builds/uncertain writes "
                "require approval; output is bounded."
            ),
            {
                "argv": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "items": {"type": "string", "maxLength": 65536},
                },
                "cwd": text_path,
                "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 3600},
                "max_output_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_FILE_BYTES,
                },
            },
            ["argv"],
        ),
    }
    for name, (description, properties, required) in definitions.items():
        registry.register(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            handler=_guarded_handler(name),
        )
    mcp_tools = {}
    for connection in connections:
        for name, tool in connection.tools.items():
            registry.register(
                name=name,
                description=tool.description or "",
                parameters=tool.input_schema,
                handler=_guarded_handler(name),
            )
            mcp_tools[name] = connection, tool.name
    return ToolExecutor(registry, approval=approval, mcp_tools=mcp_tools, trusted=trusted)
