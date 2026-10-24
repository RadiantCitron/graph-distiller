"""Embedder / LlmChat default Adapters — OpenRouter via Settings (ADR-002)."""

from __future__ import annotations

from typing import Any, List, Optional

from graph_distiller.config import Settings, get_settings
from graph_distiller.ports import Embedder, LlmChat


class OpenRouterEmbedder:
    """
    Default Embedder Adapter.

    Wraps Settings.build_embeddings(); delegates LangChain methods so
    SemanticChunker can consume this Adapter directly.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._inner = self._settings.build_embeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class OpenRouterLlmChat:
    """
    Default LlmChat Adapter.

    Model slug comes from Settings (or explicit ``model``) — never hardcoded.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._inner = self._settings.build_chat_model(model)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        # Avoid passing config=None — mock LLMs often lack the LangChain signature.
        if config is not None:
            kwargs["config"] = config
        return self._inner.invoke(input, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.with_structured_output(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def default_embedder(settings: Settings | None = None) -> Embedder:
    return OpenRouterEmbedder(settings)


def default_llm_chat(
    settings: Settings | None = None,
    model: str | None = None,
) -> LlmChat:
    return OpenRouterLlmChat(settings, model=model)
