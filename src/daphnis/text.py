"""Text operations shared by workspace files and attachment snapshots."""

import codecs
from collections.abc import Iterable, Iterator


def decode_text(data: bytes, truncated: bool = False) -> str:
    if not truncated:
        return data.decode("utf-8")
    return codecs.getincrementaldecoder("utf-8")().decode(data, final=False)


def line_range(lines: list[str], start: int, end: int | None) -> list[str]:
    """Select an inclusive, one-based range using the caller's line splitting."""
    return lines[start - 1:end]


def literal_matches(
    lines: Iterable[str], needle: str, *, case_sensitive: bool = True,
) -> Iterator[tuple[int, int, str]]:
    """Yield the first match per line, using a case-folded needle when requested."""
    for number, line in enumerate(lines, 1):
        position = (line if case_sensitive else line.casefold()).find(needle)
        if position >= 0:
            yield number, position, line
