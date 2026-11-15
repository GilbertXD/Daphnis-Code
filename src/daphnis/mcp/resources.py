"""Validate Resource identifiers and text returned by the MCP SDK."""

import re
from urllib.parse import urlsplit

from mcp.types import TextResourceContents
from uritemplate import URITemplate

from ..attachments import (
    MAX_FILES,
    MAX_TOTAL_BYTES,
    TextResourceInput,
    validate_resource_uri,
)

_TEMPLATE_EXPRESSION = re.compile(r"\{[+#./;?&]?(?:[\w.%]+(?::[1-9]\d{0,3}|\*)?)(?:,[\w.%]+(?::[1-9]\d{0,3}|\*)?)*\}", re.ASCII)


def resource_template(value: str) -> URITemplate:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("MCP 资源 URI 模板无效。")
    shape = _TEMPLATE_EXPRESSION.sub("x", value)
    if "{" in shape or "}" in shape or not urlsplit(value.split("{", 1)[0]).scheme:
        raise ValueError("MCP 资源 URI 模板无效。")
    validate_resource_uri(shape)
    template = URITemplate(value)
    if not 0 < len(template.variable_names) <= 16:
        raise ValueError("MCP 资源 URI 模板参数数量无效。")
    return template


def expand_resource_template(template: str, arguments: dict[str, str]) -> str:
    parsed = resource_template(template)
    if (
        not isinstance(arguments, dict) or arguments.keys() != set(parsed.variable_names)
        or any(not isinstance(value, str) or not value.strip() or not value.isprintable()
               or len(value.encode("utf-8")) > 512 for value in arguments.values())
    ):
        raise ValueError("请填写有效的 MCP 资源模板参数。")
    uri = parsed.expand(arguments)
    validate_resource_uri(uri)
    return uri


def text_resource(server: str, content, source: str, *, name="", mime_type=None) -> TextResourceInput:
    if not isinstance(content, TextResourceContents):
        raise TypeError("MCP 资源包含 Blob 或其他非文本内容，暂不支持。")
    uri = str(content.uri)
    name = (name.strip() or uri)[:255]
    return TextResourceInput(name, content.text, server, uri, content.mime_type or mime_type or "text/plain", source)


def validate_resource_batch(resources: tuple[TextResourceInput, ...]) -> None:
    if not 0 < len(resources) <= MAX_FILES:
        raise ValueError("每次最多可以使用 10 份 MCP 资源。")
    if sum(len(item.text.encode("utf-8")) for item in resources) > MAX_TOTAL_BYTES:
        raise ValueError("MCP 资源总大小超过 256 KiB。")
