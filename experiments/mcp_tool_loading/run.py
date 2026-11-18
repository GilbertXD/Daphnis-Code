"""Run bounded, sequential MCP loading comparisons with public documentation tasks."""

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.compaction_settings import CompactionSettings
from daphnis.config import environment_connections
from daphnis.tools import ApprovalDecision
from daphnis.usage import _reported_usage

MODELS = {"mimo": "mimo-v2.6-flash", "deepseek": "deepseek-flash"}
SERVER = "https://learn.microsoft.com/api/mcp"
SERVER_ID = "global:learn"
READ_ONLY_TOOLS = {"microsoft_docs_search", "microsoft_docs_fetch", "microsoft_code_sample_search"}
TASKS = {
    "T1": (
        ("Use Microsoft Learn to name the four Azure Blob Storage access tiers and identify which "
        "tier is offline. Cite the Microsoft Learn access tiers article. Keep the answer short."),
        ("hot", "cool", "cold", "archive", "offline"),
        ("/storage/blobs/access-tiers-overview",),
        1,
    ),
    "T2": (
        ("Use Microsoft Learn to compare Azure Storage zone-redundant storage with geo-redundant "
        "storage, including availability zones, where the secondary copy resides and which GRS "
        "variant permits reads from the secondary. Also verify which Azure Blob Storage access "
        "tier is offline. Cite a separate official article for redundancy and access tiers. "
        "Keep the answer under 180 words."),
        ("availability zone", "secondary region", "ra-grs", "archive", "offline"),
        ("/storage/common/storage-redundancy", "/storage/blobs/access-tiers-overview"),
        2,
    ),
}
EVENTS = (
    "model.started", "model.usage", "model.finished", "tool.started", "tool.executing",
    "tool.approval_requested", "tool.blocked", "tool.finished", "compaction.started",
)


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


class Trace:
    """Attach to one sequential chat session and keep only allowlisted telemetry."""

    def __init__(self, provider, model, mode, task_id, stage, repeat):
        self.tags = {"provider": provider, "model": model, "mode": mode, "task_id": task_id,
                         "stage": stage, "repeat": repeat}
        self.requests = {}
        self.current = None
        self.activated = False
        self.used_mcp = False
        self.evidence = []
        self.calls = {}
        self.compactions = 0
        self.directories = []

    def observe(self, event):
        data = event.data
        name = event.name
        if name == "compaction.started":
            self.compactions += 1
            return
        number = data["request_number"]
        key = (event.turn_id, number)
        if name == "model.started":
            purpose = data.get("purpose") or "primary"
            phase = ("after_tool" if self.used_mcp else "after_activation" if self.activated
                     else "before_activation") if purpose == "primary" else purpose
            self.current = self.requests[key] = {
                **self.tags, "kind": "request", "turn_id": event.turn_id,
                "request_number": number, "purpose": purpose, "phase": phase,
                "finished": False, "http_status": None, "sent": False,
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "cached_tokens": None, "cache_reported": False, "uncached_prompt_tokens": None,
                "cache_fields": {},
            }
        elif name == "model.usage" and data.get("usage") is not None:
            raw = data["usage"]
            usage = _reported_usage(raw)
            fields = {}
            if isinstance(raw, dict):
                for field in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    if type(raw.get(field)) in (int, bool):
                        fields[field] = raw[field]
                nested = raw.get("prompt_tokens_details")
                if isinstance(nested, dict) and type(nested.get("cached_tokens")) in (int, bool):
                    fields["prompt_tokens_details"] = {"cached_tokens": nested["cached_tokens"]}
            reported = bool(usage and usage.cache_reported_requests)
            self.requests[key].update(
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                cached_tokens=usage.cached_prompt_tokens if reported else None,
                cache_reported=reported, cache_fields=fields,
                uncached_prompt_tokens=usage.prompt_tokens - usage.cached_prompt_tokens if reported else None,
            )
        elif name == "model.finished":
            self.requests[key]["finished"] = True
        elif name == "tool.started":
            tool = data["name"]
            raw = data.get("arguments", "")
            # Fingerprints are compared in-memory for repeats. Arguments never enter JSONL.
            try:
                args = json.loads(raw)
            except ValueError:
                args = None
            signature = json.dumps(args, sort_keys=True, ensure_ascii=False)
            self.calls[(*key, data["tool_call_id"])] = {
                "tool": tool if tool in {"activate_mcp", "call_mcp_tool"} or
                tool.removeprefix("mcp__learn__") in READ_ONLY_TOOLS else "other",
                "signature": (tool, signature), "request_number": number,
                "approval": "not_requested", "status": "incomplete", "error": None, "mcp_tool": None,
            }
        else:
            call = self.calls[(*key, data["tool_call_id"])]
            if name == "tool.approval_requested":
                call["approval"] = "requested"
                actual = data["request"]["tool_name"].removeprefix("mcp__learn__")
                if actual in READ_ONLY_TOOLS:
                    call["mcp_tool"] = actual
            elif name == "tool.executing" and call["approval"] == "requested":
                call["approval"] = "allowed_once"
            elif name == "tool.blocked":
                call["approval"] = "blocked"
            elif name == "tool.finished":
                call["status"] = data["status"]
                call["error"] = (data.get("error") or {}).get("code")
                if data["status"] == "COMPLETED":
                    if call["tool"] == "activate_mcp":
                        self.activated = True
                        payload = json.loads(data["content"])
                        if "tools" in payload:
                            self.directories.append({
                                "tool_count": len(payload["tools"]),
                                "hash": digest(payload["tools"])[0],
                            })
                    elif call["mcp_tool"] is not None:
                        self.used_mcp = True
                        self.evidence.append((call["signature"], data["content"].lower()))

    async def request(self, request):
        body = json.loads(request.content)
        tools = body.get("tools", [])
        fingerprint, size = digest(tools)
        self.current.update(
            sent=True, tool_count=len(tools), tool_schema_hash=fingerprint, tool_schema_bytes=size,
            tool_names=[item["function"]["name"] for item in tools],
            tool_schema_hashes=[digest(item)[0] for item in tools],
            system_hash=digest(body["messages"][0].get("content"))[0],
            request_options={key: body[key] for key in (
                "model", "stream", "max_tokens", "max_completion_tokens", "thinking", "reasoning_effort"
            ) if key in body},
        )

    async def response(self, response):
        self.current["http_status"] = response.status_code

    def result(self, result, session_usage):
        _, expected_facts, paths, tool_count = TASKS[self.tags["task_id"]]
        answer = ((result.message or {}).get("content") or "").lower()
        facts = {fact: fact in answer for fact in expected_facts}
        citations = {
            path: bool(re.search(r"https://learn\.microsoft\.com/[^\s)>\]]*" + re.escape(path), answer))
            and any(path in content for _, content in self.evidence) for path in paths
        }
        distinct_calls = len({signature for signature, _ in self.evidence})
        success = (result.status is TurnStatus.COMPLETED and all(facts.values())
                   and all(citations.values()) and distinct_calls >= tool_count)
        signatures = Counter(call["signature"] for call in self.calls.values())
        calls = [{k: v for k, v in call.items() if k != "signature"} for call in self.calls.values()]
        return {
            **self.tags, "kind": "trajectory", "turn_id": result.turn_id,
            "success": success, "status": result.status.value, "facts": facts, "citations": citations,
            "primary_requests": result.primary_requests, "maintenance_requests": result.maintenance_requests,
            "tool_calls": calls, "repeat_calls": sum(count - 1 for count in signatures.values()),
            "distinct_successful_mcp_calls": distinct_calls, "compactions": self.compactions,
            "activation_directories": self.directories,
            "session_totals": {
                key: getattr(session_usage, key) for key in (
                    "prompt_tokens", "completion_tokens", "cached_prompt_tokens",
                    "cache_reported_prompt_tokens", "cache_reported_requests", "reported_requests",
                )
            },
        }


async def trajectory(root, connection, provider, mode, task_id, stage, repeat):
    model = MODELS[provider]
    options = {"mcp_tool_loading": mode} if mode == "gateway" else {}
    app = Application(
        workspace=root / "workspace", provider=provider, model_id=model, interactive=True,
        connections={provider: connection}, history=History(root / "state" / "sessions"),
        compaction_settings=CompactionSettings(auto_enabled=False),
        output_limits={(provider, model): 768}, max_primary_requests=6, max_total_requests=7,
        **options,
    )
    ops = Operations(app)
    sid = ops.initial_session_id
    trace = Trace(provider, model, mode, task_id, stage, repeat)
    try:
        if not ops.managed_mcp(sid):
            await ops.save_mcp(
                "learn", state="on_demand", description="Search official Microsoft Learn technical documentation.",
                options={"transport": "streamable_http", "url": SERVER, "public_only": True}, create=True,
            )
        await ops.set_tools(("glob", "grep", "list_directory", "read_file"), session_id=sid)
        await ops.set_thinking("off", session_id=sid)
        ops.set_approval_mode("ask", session_id=sid)
        for name in EVENTS:
            ops.subscribe(name, trace.observe)

        def approve(event):
            tool = event.data["request"]["tool_name"]
            decision = ApprovalDecision.ALLOW_ONCE if tool in {
                f"mcp__learn__{name}" for name in READ_ONLY_TOOLS
            } else ApprovalDecision.DENY
            ops.resolve_approval(event.data["request_id"], decision, session_id=sid)

        ops.subscribe("tool.approval_requested", approve)
        http = app._client(provider)._http  # Test instrumentation only, after compatibility mapping.
        http.event_hooks["request"].append(trace.request)
        http.event_hooks["response"].append(trace.response)
        result = await ops.send(TASKS[task_id][0], session_id=sid, stream=False)
        await app.events.drain()
        summary = trace.result(result, ops.session_usage(sid))
        if summary["status"] == "FAILED" or not all(
            row["sent"] and row["finished"] for row in trace.requests.values() if row["purpose"] == "primary"
        ):
            summary["request_failures"] = sum(
                not row["finished"] for row in trace.requests.values() if row["purpose"] == "primary"
            )
        else:
            summary["request_failures"] = 0
        # The experiment makes no explicit retries. The Agent can retry in-place if
        # a context error is recoverable; annotate failures so repeated phases are visible.
        summary["retries"] = sum(
            not first["finished"] and second["purpose"] == first["purpose"] == "primary"
            for first, second in zip(list(trace.requests.values()), list(trace.requests.values())[1:])
        )
        return [*trace.requests.values(), summary]
    finally:
        await ops.close()


async def run(args):
    load_dotenv(".env")
    _, _, connections = environment_connections()
    missing = set(MODELS).difference(connections)
    if missing:
        raise SystemExit("Missing provider configurations: " + ", ".join(sorted(missing)))
    repeats = 3 if args.stage == "formal" else 1
    tasks = TASKS if args.stage == "formal" else ("T1",)
    providers = (args.provider,) if args.provider else tuple(MODELS)
    failures = 0
    attempts = 0
    with TemporaryDirectory(prefix="daphnis-mcp-ab-") as directory, args.output.open("a", encoding="utf-8") as output:
        root = Path(directory)
        (root / "workspace").mkdir()
        for provider in providers:
            for task in tasks:
                for repeat in range(1, repeats + 1):
                    modes = (args.mode,) if args.mode else (("native", "gateway") if repeat % 2 else ("gateway", "native"))
                    for mode in modes:
                        if attempts + 7 > args.request_budget or failures >= 2:
                            raise SystemExit(f"Stopped at {attempts} requests after {failures} failed trajectories.")
                        rows = await trajectory(root, connections[provider], provider, mode, task, args.stage, repeat)
                        output.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                        output.flush()
                        summary = rows[-1]
                        attempts += len(rows) - 1
                        failures = failures + 1 if summary["request_failures"] or not summary["distinct_successful_mcp_calls"] else 0
                        print(provider, task, repeat, mode, summary["status"], summary["success"], len(rows) - 1,
                              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--mode", choices=("native", "gateway"))
    parser.add_argument("--provider", choices=tuple(MODELS))
    parser.add_argument("--request-budget", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    logging.basicConfig(level=logging.CRITICAL)
    asyncio.run(run(parser.parse_args()))
