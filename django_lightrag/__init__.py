from typing import Any

from .config import get_lightrag_core_settings
from .serialization import to_serializable
from .types import QueryParam


def run_update(
    content: str,
    metadata: dict[str, Any],
    track_id: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> dict[str, Any]:
    """
    Ingest text content into LightRAG programmatically.

    Args:
        content: The text content to ingest
        metadata: Dictionary of metadata to associate with the document
        track_id: Optional string identifier for tracking the document
        llm_model: Optional LLM model override
        llm_temperature: Optional LLM temperature override

    Returns:
        Dictionary containing document_id or error information
    """
    from .core import LightRAGCore

    config = get_lightrag_core_settings()

    try:
        core = LightRAGCore(
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            embedding_base_url=config.embedding_base_url,
            llm_model=llm_model or config.llm_model,
            llm_temperature=llm_temperature if llm_temperature is not None else 0.0,
        )
        try:
            document_id = core.ingest_document(
                content=content,
                metadata=metadata,
                track_id=track_id or "",
            )
            return {
                "document_id": document_id,
                "message": "Document ingested successfully",
            }
        finally:
            core.close()
    except Exception as e:
        return {"error": "ingestion_failed", "message": str(e)}


def run_query(
    query: str,
    param: QueryParam | dict[str, Any] | None = None,
    *,
    _core_factory: Any = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> dict[str, Any]:
    """
    Query the LightRAG graph programmatically.

    Args:
        query: The user query text
        param: Optional query parameters as a QueryParam or plain dict
        _core_factory: Optional override for core instantiation
        llm_model: Optional LLM model override
        llm_temperature: Optional LLM temperature override

    Returns:
        Dictionary containing the query result or error information
    """
    try:
        config = get_lightrag_core_settings()
        query_param = QueryParam(**param) if isinstance(param, dict) else param
        real_llm_model = llm_model or config.llm_model
        real_llm_temperature = llm_temperature if llm_temperature is not None else 0.0

        if _core_factory is not None:
            core = _core_factory(
                embedding_model=config.embedding_model,
                embedding_provider=config.embedding_provider,
                embedding_base_url=config.embedding_base_url,
                llm_model=real_llm_model,
            )
        else:
            from .core import LightRAGCore

            core = LightRAGCore(
                embedding_model=config.embedding_model,
                embedding_provider=config.embedding_provider,
                embedding_base_url=config.embedding_base_url,
                llm_model=real_llm_model,
                llm_temperature=real_llm_temperature,
            )
        try:
            return to_serializable(core.query(query, query_param or QueryParam()))
        finally:
            core.close()
    except Exception as e:
        return {"error": "query_failed", "message": str(e)}


__all__ = ["run_query", "run_update"]
