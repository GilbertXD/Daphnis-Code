"""Local stdio fixture with a paginated tool catalogue."""

import asyncio
import os
import sys
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

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
    return CallToolResult(
        content=[TextContent(type="text", text=str(a + b))],
        structured_content={"sum": a + b},
    )


async def main():
    if any(name in os.environ for name in ("LLM_API_KEY", "MIMO_API_KEY")):
        raise RuntimeError("Parent API keys reached the MCP process.")
    (root / "pid").write_text(str(os.getpid()))
    server = Server(
        "demo",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        get_tool_input_schema=lambda name: (
            tools[name].input_schema if name in tools else None
        ),
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
