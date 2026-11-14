"""Rank redacted, session-local history and return exact offsets in its readable view."""

import asyncio
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import httpx2 as httpx
from saturnus import Messages

from ..redaction import Redactor
from ..token_budget import DEFAULT_COUNTER
from .approval import Decision, Operation, ToolError
from .history import HISTORY_NOTICE, visible_history_text

MAX_RESULTS = 10
MAX_RESULT_TOKENS = 1300
CHUNK_CHARS = 560
OVERLAP = 120
INDEX_VERSION = 1
STRATEGIES = ("bm25", "semantic", "hybrid")
_TERMS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]|[^\W\d_]+", re.IGNORECASE)


class Embedder(Protocol):
    model: str
    dimensions: int | None

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class HttpEmbedder:
    url: str
    model: str
    dimensions: int | None = None
    api_key: str | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        rows = []
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            for start in range(0, len(texts), 24):
                batch = texts[start:start + 24]
                response = await client.post(self.url, headers=headers, json={"model": self.model, "input": batch})
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list) or any(
                    not isinstance(item, dict) or type(item.get("index")) is not int
                    or not isinstance(item.get("embedding"), list) for item in data
                ):
                    raise ValueError("Invalid embedding response rows")
                data = sorted(data, key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ValueError("Invalid embedding row indexes")
                rows.extend(item["embedding"] for item in data)
        return rows


def environment_retrieval() -> "RetrievalEngine | None":
    strategy = os.getenv("DAPHNIS_RETRIEVAL", "").strip()
    if not strategy:
        return None
    if strategy == "bm25":
        return RetrievalEngine(strategy)
    url = os.getenv("DAPHNIS_EMBEDDING_URL", "").strip()
    model = os.getenv("DAPHNIS_EMBEDDING_MODEL", "").strip()
    dimension = os.getenv("DAPHNIS_EMBEDDING_DIMENSIONS")
    embedder = None
    if url and model:
        dimensions = int(dimension) if dimension else None
        if dimensions is not None and dimensions <= 0:
            raise ValueError("DAPHNIS_EMBEDDING_DIMENSIONS must be a positive integer")
        embedder = HttpEmbedder(url, model, dimensions, os.getenv("DAPHNIS_EMBEDDING_API_KEY"))
    return RetrievalEngine(strategy, embedder)


@dataclass(frozen=True)
class Chunk:
    message_id: int
    role: str
    start: int
    text: str


@dataclass
class HistoryIndex:
    key: tuple
    chunks: list[Chunk]
    terms: list[Counter]
    df: Counter
    average_length: float
    vectors: list[tuple[float, ...]] | None = None

    @classmethod
    def build(cls, key: tuple, visible: list[tuple[int, str, str]]) -> "HistoryIndex":
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


def unit_vectors(rows: list[list[float]], count: int, dimension: int | None) -> list[tuple[float, ...]]:
    if len(rows) != count:
        raise ValueError("Unexpected embedding count")
    result = []
    for row in rows:
        dimension = len(row) if dimension is None else dimension
        if not dimension or len(row) != dimension or any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
            raise ValueError("Invalid embedding dimensions or values")
        norm = math.hypot(*row)
        if not norm or not math.isfinite(norm):
            raise ValueError("Invalid embedding norm")
        result.append(tuple(x / norm for x in row))
    return result


class RetrievalEngine:
    def __init__(self, strategy: str = "bm25", embedder: Embedder | None = None):
        if strategy not in STRATEGIES:
            raise ValueError("DAPHNIS_RETRIEVAL must be bm25, semantic or hybrid")
        self.strategy = strategy
        self.embedder = embedder
        self._index: HistoryIndex | None = None

    def index(self, session_id: str, messages: Messages, redactor: Redactor, *, active: bool = True) -> HistoryIndex:
        current = next((i for i in range(len(messages) - 1, -1, -1) if messages.get(i)["role"] == "user"), len(messages))
        visible = [(i, *entry) for i in range(len(messages))
                   if (not active or i < current or messages.get(i)["role"] == "tool")
                   and (entry := visible_history_text(messages, i, redactor)) is not None]
        digest = hashlib.sha256(json.dumps(visible, ensure_ascii=True).encode()).digest()
        embedding = (self.embedder.model, self.embedder.dimensions) if self.embedder else None
        key = (session_id, INDEX_VERSION, digest, embedding)
        if self._index is None or self._index.key != key:
            self._index = HistoryIndex.build(key, visible)
        return self._index

    async def search(self, index: HistoryIndex, session_id: str, query: str, limit: int) -> dict:
        if not query.strip():
            raise ToolError("invalid_query", "Retrieval requires a non-empty query.")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ToolError("invalid_limit", f"Retrieval limit must be 1–{MAX_RESULTS}.")
        if index.key[0] != session_id:
            raise ValueError("Retrieval index belongs to a different session")
        strategy = self.strategy
        fallback = None
        embedding_calls = embedding_inputs = 0
        embedding_index_ms = embedding_query_ms = 0.0
        scores = index.bm25(query)
        if strategy != "bm25":
            if self.embedder is None:
                fallback = "embedding_not_configured"
            elif len(index.chunks) > 4096:
                fallback = "embedding_index_too_large"
            elif index.chunks:
                try:
                    async with asyncio.timeout(40):
                        if index.vectors is None:
                            texts = [chunk.text for chunk in index.chunks]
                            embedding_calls += 1
                            embedding_inputs += len(texts)
                            started = perf_counter()
                            index.vectors = unit_vectors(await self.embedder.embed(texts), len(texts), self.embedder.dimensions)
                            embedding_index_ms = (perf_counter() - started) * 1000
                        embedding_calls += 1
                        embedding_inputs += 1
                        started = perf_counter()
                        vector, = unit_vectors(await self.embedder.embed([query]), 1, len(index.vectors[0]))
                        embedding_query_ms = (perf_counter() - started) * 1000
                    semantic = {i: sum(a*b for a, b in zip(vector, row, strict=True))
                                for i, row in enumerate(index.vectors)}
                    if strategy == "semantic":
                        scores = semantic
                    else:
                        fused = Counter()
                        for ranking in (scores, semantic):
                            for rank, i in enumerate(self._rank(index, ranking), 1):
                                fused[i] += 1 / (60 + rank)
                        scores = fused
                except (OSError, ValueError, TypeError, KeyError, httpx.HTTPError, httpx.InvalidURL):
                    fallback = "embedding_unavailable"
        result = {"ok": True, "notice": HISTORY_NOTICE, "source": "history", "session_id": session_id,
                  "requested_strategy": strategy, "strategy": "bm25" if fallback else strategy, "fallback": fallback,
                  "embedding_calls": embedding_calls, "embedding_inputs": embedding_inputs,
                  "embedding_index_ms": round(embedding_index_ms, 2), "embedding_query_ms": round(embedding_query_ms, 2),
                  "matches": []}
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
        return await engine.search(index, session_id, redactor.text(arguments["query"]), arguments.get("limit", 5))

    return Operation(decision=Decision.ALLOW, reason="Retrieve exposed text from the current Session.",
                     targets=(), details=arguments, run=run)
