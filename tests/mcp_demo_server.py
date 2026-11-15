"""Local stdio fixture with a paginated tool catalogue."""

import asyncio
import os
import sys
from pathlib import Path

from mcp import MCPError
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
)

root = Path(sys.argv[1])
schema = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}
tools = {
    name: Tool(name=name, description="Add two integers.", input_schema=schema)
    for name in ("add", "math.add", "sum")
}


async def list_tools(ctx, params):
    if (root / "invalid").exists():
        return ListToolsResult(tools=[tools["add"]], next_cursor="invalid")
    if (root / "changed").exists():
        return ListToolsResult(tools=[tools["sum"]])
    if params and params.cursor == "second":
        return ListToolsResult(tools=[tools["math.add"]])
    return ListToolsResult(tools=[tools["add"]], next_cursor="second")


async def call_tool(ctx, params):
    a, b = (params.arguments[key] for key in ("a", "b"))
    with (root / "calls").open("a") as log:
        log.write(f"{params.name} {a} {b}\n")
    if a == -1:
        await asyncio.Event().wait()
    if a == -2:
        return CallToolResult(
            content=[TextContent(type="text", text="Demo error")], is_error=True
        )
    if a == -3:
        return CallToolResult(
            content=[TextContent(type="text", text="界" * 30_000)],
            structured_content={"kind": "text"},
        )
    if a == -4:
        return CallToolResult(
            content=[TextContent(type="text", text="summary")],
            structured_content={"rows": ["x" * 4096] * 25},
        )
    return CallToolResult(
        content=[TextContent(type="text", text=str(a + b))],
        structured_content={"sum": a + b},
    )


async def list_prompts(ctx, params):
    if params and params.cursor == "more":
        return ListPromptsResult(prompts=[Prompt(name=name) for name in ("bundle", "slow", "fail", "large", "dialogue", "embedded", "embedded-large", "embedded-oversize")])
    return ListPromptsResult(
        prompts=[Prompt(name="greet", description="Create a greeting.", arguments=[
            PromptArgument(name="who", required=True, description="Name to greet."),
            PromptArgument(name="tone", required=False),
        ])], next_cursor="more",
    )


async def get_prompt(ctx, params):
    if params.name == "slow":
        await asyncio.sleep(60)
    if params.name == "fail":
        raise MCPError(-32602, "fixture upstream error")
    if params.name == "bundle":
        return GetPromptResult(messages=[
            PromptMessage(role="user", content=TextContent(type="text", text="first")),
            PromptMessage(role="assistant", content=ImageContent(type="image", data="AA==", mime_type="image/png")),
        ])
    if params.name.startswith("embedded"):
        size = {"embedded": 0, "embedded-large": 70_000, "embedded-oversize": 131_073}[params.name]
        text = "X" * size if size else (root / "resource-text").read_text()
        return GetPromptResult(messages=[
            PromptMessage(role="user", content=TextContent(type="text", text="Review this resource.")),
            PromptMessage(role="assistant", content=EmbeddedResource(type="resource", resource=TextResourceContents(
                uri="notes://embedded-source", mime_type="text/markdown", text=text,
            ))),
            PromptMessage(role="user", content=TextContent(type="text", text="resource-retry")),
        ])
    if params.name == "dialogue":
        return GetPromptResult(messages=[
            PromptMessage(role="user", content=TextContent(type="text", text="Review the project.")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="I will check tests.")),
        ])
    if params.name == "large":
        text = "界" * 30_000
    else:
        args = params.arguments or {}
        text = f"Hello, {args['who']}! ({args.get('tone', 'neutral')})"
    return GetPromptResult(
        description="A greeting",
        messages=[PromptMessage(role="user", content=TextContent(type="text", text=text))],
    )


async def list_resources(ctx, params):
    if (root / "resource-cursor").exists():
        return ListResourcesResult(resources=[], next_cursor="loop")
    if not params or not params.cursor:
        return ListResourcesResult(resources=[
            Resource(name="Architecture", uri="notes://architecture", mime_type="text/markdown"),
        ], next_cursor="remaining")
    return ListResourcesResult(resources=[
        Resource(name=name, uri=f"notes://{name}") for name in ("bundle", "blob", "slow", "fail", "bad-mime")
    ])


async def list_resource_templates(ctx, params):
    return ListResourceTemplatesResult(resource_templates=[
        ResourceTemplate(name="Note", uri_template="notes://pages/{name}", mime_type="text/plain")
    ])


async def read_resource(ctx, params):
    uri = str(params.uri)
    with (root / "resource-reads").open("a") as log:
        log.write(uri + "\n")
    if uri == "notes://slow":
        await asyncio.sleep(60)
    if uri == "notes://fail":
        raise MCPError(-32602, "fixture upstream error")
    if uri == "notes://blob":
        return ReadResourceResult(contents=[BlobResourceContents(uri=uri, mime_type="image/png", blob="AA==")])
    if uri == "notes://bundle":
        return ReadResourceResult(contents=[
            TextResourceContents(uri="notes://bundle/a", text="first"),
            TextResourceContents(uri="notes://bundle/b", mime_type="application/json", text='{"second":true}'),
        ])
    mime_type = "image/png" if uri == "notes://bad-mime" else None
    text = (root / "resource-text").read_text() if (root / "resource-text").exists() else uri
    return ReadResourceResult(contents=[TextResourceContents(uri=uri, text=text, mime_type=mime_type)])


async def main():
    if any(name in os.environ for name in ("LLM_API_KEY", "MIMO_API_KEY")):
        raise RuntimeError("Parent API keys reached the MCP process.")
    (root / "pid").write_text(str(os.getpid()))
    if (root / "stderr-diagnostics").exists():
        for chunk in (
            b"fixture start \x1b]52;",
            b"c;CLIPBOARD\x07 \x1b[",
            b"31m\x1b[0m " + "\x9b31m\x9dHIDDEN\x9c".encode() + b" diagnostic \n",
        ):
            os.write(2, chunk)
            await asyncio.sleep(0.02)
        sys.stderr.buffer.write(b"x" * (256 * 1024) + b"AFTER-LIMIT")
        sys.stderr.buffer.flush()
        (root / "stderr-written").touch()
    if (root / "stderr-exit").exists():
        return
    if (root / "stderr-stall").exists():
        await asyncio.Event().wait()
    supports_prompts = (root / "prompts").exists()
    supports_resources = (root / "resources").exists()
    tools_enabled = not (root / "prompts-only").exists()
    server = Server(
        "demo",
        on_list_tools=list_tools if tools_enabled else None,
        on_call_tool=call_tool if tools_enabled else None,
        get_tool_input_schema=lambda name: (
            tools[name].input_schema if name in tools else None
        ),
        on_list_resources=list_resources if supports_resources else None,
        on_list_resource_templates=list_resource_templates if supports_resources else None,
        on_read_resource=read_resource if supports_resources else None,
        on_list_prompts=list_prompts if supports_prompts else None,
        on_get_prompt=get_prompt if supports_prompts else None,
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
