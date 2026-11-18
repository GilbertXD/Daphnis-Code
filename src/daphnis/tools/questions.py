"""Validate batches of questions and their answers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from .approval import Decision, Operation, ToolError

MAX_ANSWER_LENGTH = 2000
MAX_QUESTIONS = 5
MAX_OPTIONS = 8
QUESTION_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": MAX_QUESTIONS,
    "description": "Independent questions to present in order. Use another call for a follow-up that depends on an answer.",
    "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "question", "type"],
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "description": "Unique answer key in this batch."},
            "question": {"type": "string", "minLength": 1, "maxLength": 500, "description": "Question shown to the user."},
            "type": {"type": "string", "enum": ["single_select", "multi_select", "text"]},
            "options": {
                "type": "array", "minItems": 2, "maxItems": MAX_OPTIONS, "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
                "description": "Required for selection questions and omitted for text. The UI provides a custom answer separately.",
            },
        },
    },
}

Answers = dict[str, str | list[str]]


@dataclass(frozen=True)
class QuestionRequest:
    session_id: str
    turn_id: str
    tool_call_id: str
    questions: list[dict]
    request_id: str = field(default_factory=lambda: uuid4().hex, init=False)

    def validate_answers(self, answers: Answers) -> Answers:
        if not isinstance(answers, dict) or answers.keys() != {item["id"] for item in self.questions}:
            raise ValueError("请为本批次的每个问题提供一个答案。")
        validated: Answers = {}
        for question in self.questions:
            key = question["id"]
            answer = answers[key]
            multi = question["type"] == "multi_select"
            values = answer if multi else [answer]
            if not isinstance(values, list) or not 1 <= len(values) <= MAX_OPTIONS + 1 or (
                not multi and not isinstance(answer, str)
            ):
                raise ValueError("答案类型或选项数量无效。")
            if any(not isinstance(value, str) or not value.strip() or len(value) > MAX_ANSWER_LENGTH for value in values):
                raise ValueError("每个答案必须是 1–2000 字符的非空文本。")
            cleaned = [value.strip() for value in values]
            if len(set(cleaned)) != len(cleaned) or (
                multi and sum(value not in question["options"] for value in cleaned) > 1
            ):
                raise ValueError("多选答案不能重复，且最多包含一条自定义回答。")
            validated[key] = cleaned if multi else cleaned[0]
        return validated


QuestionCallback = Callable[[QuestionRequest], Awaitable[Answers]]


def prepare_question(args: dict, session_id: str, turn_id: str, tool_call_id: str, callback: QuestionCallback | None) -> Operation:
    questions = []
    ids = set()
    for item in args["questions"]:
        key = item["id"]
        question = item["question"].strip()
        is_text = item["type"] == "text"
        if key in ids or not question or is_text == ("options" in item):
            raise ToolError("invalid_questions", "Use unique IDs, nonblank questions, and options only for selection questions.")
        ids.add(key)
        normalized = {**item, "question": question}
        if not is_text:
            options = [option.strip() for option in item["options"]]
            if "" in options or len(set(options)) != len(options):
                raise ToolError("invalid_questions", "Options must be nonblank and distinct.")
            normalized["options"] = options
        questions.append(normalized)
    request = QuestionRequest(session_id, turn_id, tool_call_id, questions)

    async def run():
        if callback is None:
            raise ToolError("question_unavailable", "No interactive recipient is available for this question.")
        return {"ok": True, "answers": await callback(request)}

    return Operation(Decision.ALLOW, "Ask the user for task information.", (), {}, run)
