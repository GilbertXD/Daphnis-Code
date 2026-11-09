"""Shared filtering for untrusted terminal text."""

import re

_TERMINAL_CONTROLS = re.compile(
    r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|\Z)"  # OSC
    r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f]).*?(?:\x1b\\|\x9c|\Z)"  # Control strings
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"  # CSI, including incomplete sequences
    r"|\x1b[ -/]*[0-~]"
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]",
    re.DOTALL,
)


def sanitize_terminal(text: str) -> str:
    """Remove terminal commands while preserving printable text, tabs and LF."""
    return _TERMINAL_CONTROLS.sub("", text)
