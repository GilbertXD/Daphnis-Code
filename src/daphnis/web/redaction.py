"""Redact browser-bound JSON and free text before transmission or retention."""

import json
import re
from collections.abc import Iterable
from pathlib import Path

_MASK = "[REDACTED]"
_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"authorization|(?:set[_-]?)?cookie|client[_-]?secret|secret(?:[_-]?key)?|"
    r"password|private[_-]?key|credentials?)(?:$|[_-])",
    re.IGNORECASE,
)
_VALUE = re.compile(
    r"""(?ix)
    (\b(?:[\w-]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|
        authorization|password|client[_-]?secret|private[_-]?key|set-cookie))["']?
        \s*[:=]\s*)
    ("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\r\n]+)
    | \bBearer\s+[\w.~+/=-]+
    | \b(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-)[\w-]{8,}
    | -----BEGIN\ [^-\n]*PRIVATE\ KEY-----[\s\S]*?(?:-----END\ [^-\n]*PRIVATE\ KEY-----|\Z)
    """
)


class Redactor:
    def __init__(self, secrets: Iterable[str]):
        self.secrets = tuple(sorted({secret for secret in secrets if secret}, key=len, reverse=True))

    def text(self, text: str, *, partial: bool = False) -> str:
        if partial:
            # Hold a suffix that could complete a configured secret in a later chunk.
            held = 0
            for secret in self.secrets:
                for length in range(min(len(secret) - 1, len(text)), held, -1):
                    if text.endswith(secret[:length]):
                        held = length
                        break
            if held:
                text = text[:-held]
        for secret in self.secrets:
            text = text.replace(secret, _MASK)
        return _VALUE.sub(lambda match: (match[1] or "") + _MASK, text)

    def json(self, value):
        if isinstance(value, dict):
            return {
                self.text(str(key)): _MASK if _KEY.search(str(key)) else self.json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.json(item) for item in value]
        if isinstance(value, Path):
            return self.text(str(value))
        if isinstance(value, str):
            # Tool arguments/results can be serialized JSON inside the outer JSON.
            if value.startswith(("{", "[")):
                try:
                    embedded = json.loads(value)
                except (ValueError, RecursionError):
                    pass
                else:
                    sanitized = self.json(embedded)
                    if sanitized != embedded:
                        return json.dumps(sanitized, ensure_ascii=False)
            return self.text(value)
        return value
