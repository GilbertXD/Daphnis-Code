import json
import unittest

from daphnis.errors import ToolCallValidationError
from daphnis.tools import ToolRegistry, ToolResult


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_definitions_and_execution_arguments_are_isolated(self):
        def update(items):
            items.append("changed")
            return items

        parameters = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        registry = ToolRegistry()
        for name in ("first", "second"):
            registry.register(
                name=name,
                description="Update items",
                parameters=parameters,
                handler=update,
            )
        selected = registry.select(["second", "first"])
        definitions = selected.definitions()
        self.assertEqual(
            [definition["function"]["name"] for definition in definitions],
            ["first", "second"],
        )
        definitions[0]["function"]["parameters"]["required"].clear()
        self.assertEqual(
            selected.definitions()[0]["function"]["parameters"], parameters
        )

        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "first", "arguments": '{"items": [NaN]}'},
        }
        with self.assertRaises(ToolCallValidationError) as caught:
            selected.validate_call(call)
        self.assertEqual(caught.exception.code, "invalid_arguments_json")

        call["function"]["arguments"] = '{"items": ["original"]}'
        result = await selected.execute(call)
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.arguments, {"items": ["original"]})
        self.assertEqual(result.value, ["original", "changed"])
        self.assertEqual(json.loads(result.content), result.value)
