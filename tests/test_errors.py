import unittest

from daphnis.errors import (
    DaphnisError,
    ToolCallValidationError,
    ToolExecutionError,
    ValidationError,
)


class ErrorsTests(unittest.TestCase):
    def test_tool_validation_error_keeps_context_and_copies_path(self):
        path = ["items", 1]
        with self.assertRaises(ValidationError) as caught:
            raise ToolCallValidationError(
                "Invalid arguments.",
                code="arguments_schema_mismatch",
                tool_call_id="call-1",
                tool_name="read_file",
                argument_path=path,
            )

        path.clear()
        error = caught.exception
        self.assertIsInstance(error, DaphnisError)
        self.assertEqual(str(error), "Invalid arguments.")
        self.assertEqual(error.code, "arguments_schema_mismatch")
        self.assertEqual(error.tool_call_id, "call-1")
        self.assertEqual(error.tool_name, "read_file")
        self.assertEqual(error.argument_path, ["items", 1])

    def test_tool_execution_error_keeps_call_context(self):
        error = ToolExecutionError(
            "Execution failed.", tool_call_id="call-1", tool_name="read_file"
        )
        self.assertIsInstance(error, DaphnisError)
        self.assertNotIsInstance(error, ValidationError)
        self.assertEqual(str(error), "Execution failed.")
        self.assertEqual(error.tool_call_id, "call-1")
        self.assertEqual(error.tool_name, "read_file")
