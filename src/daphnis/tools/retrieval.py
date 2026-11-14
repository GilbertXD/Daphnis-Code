"""Rank redacted, session-local history and return exact offsets in its readable view."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from saturnus import Messages

from ..redaction import Redactor
from ..token_budget import DEFAULT_COUNTER
from .approval import Decision, Operation, ToolError
from .history import HISTORY_NOTICE, visible_history_text

MAX_RESULTS = 10
MAX_RESULT_TOKENS = 1300
CHUNK_CHARS = 560
OVERLAP = 120
_TERMS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]|[^\W\d_]+", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    message_id: int
    role: str
    start: int
    text: str


@dataclass
class HistoryIndex:
    key: tuple[str, bytes]
    chunks: list[Chunk]
    terms: list[Counter[str]]
    df: Counter[str]
    average_length: float

    @classmethod
    def build(cls, key: tuple[str, bytes], visible: list[tuple[int, str, str]]) -> "HistoryIndex":
        chunks = []
        for index, role, content in visible:
            for start in range(0, len(content), CHUNK_CHARS - OVERLAP):
                chunks.append(Chunk(index, role, start, content[start:start + CHUNK_CHARS]))
                if start + CHUNK_CHARS >= len(content):
                    break
        counts = [Counter(_TERMS.findall(chunk.text.casefold())) for chunk in chunks]
        df = Counter(term for count in counts for term in count)
        average = sum(sum(count.values()) for count in counts) / max(1, len(chunks))
        return cls(key, chunks, counts, df, average or 1)

    def bm25(self, query: str) -> dict[int, float]:
        query_terms = set(_TERMS.findall(query.casefold())) & self.df.keys()
        weights = {term: math.log1p((len(self.chunks) - self.df[term] + .5) / (self.df[term] + .5))
                   for term in sorted(query_terms)}
        scores = {}
        for index, terms in enumerate(self.terms):
            norm = 1.2 * (.25 + .75 * sum(terms.values()) / self.average_length)
            score = sum(weight * terms[term] * 2.2 / (terms[term] + norm) for term, weight in weights.items())
            if score:
                scores[index] = score
        return scores


class RetrievalEngine:
    def __init__(self) -> None:
        self._index: HistoryIndex | None = None

    def index(self, session_id: str, messages: Messages, redactor: Redactor) -> HistoryIndex:
        current = next((i for i in range(len(messages) - 1, -1, -1) if messages.get(i)["role"] == "user"), len(messages))
        visible = [(i, *entry) for i in range(len(messages))
                   if (i < current or messages.get(i)["role"] == "tool")
                   and (entry := visible_history_text(messages, i, redactor)) is not None]
        digest = hashlib.sha256(json.dumps(visible, ensure_ascii=True).encode()).digest()
        key = (session_id, digest)
        if self._index is None or self._index.key != key:
            self._index = HistoryIndex.build(key, visible)
        return self._index

    def search(self, index: HistoryIndex, session_id: str, query: str, limit: int) -> dict:
        if not query.strip():
            raise ToolError("invalid_query", "Retrieval requires a non-empty query.")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ToolError("invalid_limit", f"Retrieval limit must be 1–{MAX_RESULTS}.")
        if index.key[0] != session_id:
            raise ValueError("Retrieval index belongs to a different session")
        scores = index.bm25(query)
        result = {
            "ok": True, "notice": HISTORY_NOTICE, "source": "history",
            "session_id": session_id, "matches": [],
        }
        seen = set()
        for i in self._rank(index, scores):
            chunk = index.chunks[i]
            if chunk.message_id in seen:
                continue
            end = chunk.start + len(chunk.text)
            match = {"message_id": chunk.message_id, "role": chunk.role, "start": chunk.start, "end": end,
                     "excerpt": chunk.text, "rank": len(result["matches"]) + 1, "score": round(scores[i], 7),
                     "history_read": {"message_id": chunk.message_id, "start": chunk.start, "end": end}}
            if DEFAULT_COUNTER.count({**result, "matches": [*result["matches"], match]}) > MAX_RESULT_TOKENS:
                continue
            result["matches"].append(match)
            seen.add(chunk.message_id)
            if len(seen) == limit:
                break
        return result

    @staticmethod
    def _rank(index: HistoryIndex, scores: dict[int, float]) -> list[int]:
        return sorted(scores, key=lambda i: (-scores[i], -index.chunks[i].message_id, index.chunks[i].start))


def prepare_retrieval(engine: RetrievalEngine, session_id: str, arguments: dict, messages: Messages,
                      redactor: Redactor) -> Operation:
    async def run():
        if arguments.get("source", "history") != "history":
            raise ToolError("invalid_source", "Only the history source is available.")
        index = engine.index(session_id, messages, redactor)
        return engine.search(index, session_id, redactor.text(arguments["query"]), arguments.get("limit", 5))

    return Operation(decision=Decision.ALLOW, reason="Retrieve exposed text from the current Session.",
                     targets=(), details=arguments, run=run)
