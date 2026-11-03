"""Model selection and capability checks on top of Saturnus."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

from saturnus import ChatClient, Compatibility

from . import compatible, deepseek, mimo
from .models import MODELS, ModelInfo, get_model

_ADAPTERS = {
    "deepseek": deepseek.COMPATIBILITY,
    "mimo": mimo.COMPATIBILITY,
    "compatible": compatible.COMPATIBILITY,
}
_MANAGED_FIELDS = {
    "model",
    "messages",
    "stream",
    "tools",
    "thinking_mode",
    "output_limit",
    "thinking",
    "reasoning_effort",
}


class CapabilityError(ValueError):
    """A requested capability is unknown or unsupported for this model."""


class LLM:
    """One provider connection with a selectable model and per-request options.

    Create another LLM to switch providers, so credentials and compatibility
    rules stay attached to their connection. No message history is stored here.
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        catalog: Sequence[ModelInfo] = MODELS,
        compatibility: Compatibility | None = None,
        **client_options: Any,
    ) -> None:
        if provider not in _ADAPTERS:
            raise ValueError(f"Unknown provider: {provider!r}.")
        self._provider = provider
        self._catalog = tuple(catalog)
        keys = [(model.provider, model.id) for model in self._catalog]
        if len(set(keys)) != len(keys):
            raise ValueError("Model catalog contains duplicate (provider, id) keys.")
        self._model = self._resolve_model(model_id)
        self._custom_compatibility = compatibility is not None
        self._client = ChatClient(
            base_url=base_url,
            api_key=api_key,
            compatibility=(
                _ADAPTERS[provider] if compatibility is None else compatibility
            ),
            **client_options,
        )

    def _resolve_model(self, model_id: str) -> ModelInfo:
        try:
            return get_model(self._provider, model_id, catalog=self._catalog)
        except KeyError:
            if self._provider == "compatible":
                return ModelInfo(id=model_id, provider="compatible")
            raise

    @property
    def model(self) -> ModelInfo:
        """Capabilities for UI and callers; None means unknown."""
        return self._model

    def select_model(self, model_id: str) -> ModelInfo:
        """Select a model on this provider; each request rechecks its options."""
        self._model = self._resolve_model(model_id)
        return self._model

    def require_tools(self) -> None:
        if self.model.supports_tools is not True:
            reason = "unknown" if self.model.supports_tools is None else "unsupported"
            raise CapabilityError(
                f"Tool calling is {reason} for "
                f"{(self.model.provider, self.model.id)!r}."
            )

    def _prepare_options(
        self,
        *,
        thinking_mode: str | None,
        tools: list[dict[str, Any]] | None,
        output_limit: int | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dictionary or None.")
        body = {} if extra_body is None else extra_body.copy()
        conflicts = _MANAGED_FIELDS.intersection(body)
        if conflicts:
            raise ValueError(
                f"Use explicit request arguments instead of extra_body for: "
                f"{', '.join(sorted(conflicts))}."
            )
        if tools is not None:
            self.require_tools()
            body["tools"] = tools
        if body.get("tool_choice") not in (None, "none"):
            self.require_tools()
        if thinking_mode is not None:
            modes = self.model.thinking_modes
            if modes is None:
                raise CapabilityError(
                    f"Thinking options are unknown for {self.model.id!r}."
                )
            if thinking_mode not in modes:
                raise CapabilityError(
                    f"Thinking mode {thinking_mode!r} is unsupported for "
                    f"{self.model.id!r}; available options: {modes!r}."
                )
            if self._provider == "compatible" and not self._custom_compatibility:
                raise ValueError(
                    "Generic thinking requires an explicit Compatibility mapping "
                    "for thinking_mode."
                )
            body["thinking_mode"] = thinking_mode
        if output_limit is not None:
            if type(output_limit) is not int or output_limit <= 0:
                raise ValueError("output_limit must be a positive integer.")
            body["output_limit"] = output_limit
        return body

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking_mode: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        output_limit: int | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = self._prepare_options(
            thinking_mode=thinking_mode,
            tools=tools,
            output_limit=output_limit,
            extra_body=extra_body,
        )
        return await self._client.create(
            model=self.model.id,
            messages=messages,
            extra_body=body,
            extra_headers=extra_headers,
        )

    @asynccontextmanager
    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking_mode: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        output_limit: int | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        body = self._prepare_options(
            thinking_mode=thinking_mode,
            tools=tools,
            output_limit=output_limit,
            extra_body=extra_body,
        )
        async with self._client.stream(
            model=self.model.id,
            messages=messages,
            extra_body=body,
            extra_headers=extra_headers,
        ) as chunks:
            yield chunks

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, traceback)

    async def aclose(self) -> None:
        await self._client.aclose()
