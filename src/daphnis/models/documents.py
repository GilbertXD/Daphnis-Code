"""The document preparation boundary used by the Agent."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..attachments import DocumentAttachment


class DocumentInputError(ValueError):
    """A document could not be prepared without exposing service details."""


@dataclass(frozen=True)
class PreparedDocuments:
    messages: list[dict[str, Any]]
    private_values: tuple[str, ...]


class DocumentAdapter(Protocol):
    async def prepare(
        self, messages: list[dict[str, Any]], documents: dict[int, tuple[DocumentAttachment, ...]],
        read: Callable[[DocumentAttachment], bytes],
    ) -> PreparedDocuments: ...

    async def invalidate_missing(self, response: httpx.Response, prepared: PreparedDocuments) -> bool: ...

    async def aclose(self) -> None: ...
