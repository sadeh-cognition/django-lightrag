from typing import Any

from .config import get_lightrag_core_settings
from .serialization import to_serializable
from .types import QueryParam


def run_update(
    content: str, metadata: dict[str, Any], track_id: str | None = None
) -> dict[str, Any]:
    """
    Ingest text content into LightRAG programmatically.

    Args:
        content: The text content to ingest
        metadata: Dictionary of metadata to associate with the document
        track_id: Optional string identifier for tracking the document

    Returns:
        Dictionary containing document_id or error information
    """
    from .core import LightRAGCore

    config = get_lightrag_core_settings()

    try:
        core = LightRAGCore(
            embedding_model=config["EMBEDDING_MODEL"],
            embedding_provider=config["EMBEDDING_PROVIDER"],
            embedding_base_url=config["EMBEDDING_BASE_URL"],
            llm_model=config["LLM_MODEL"],
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
    query: str, param: QueryParam | dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Query the LightRAG graph programmatically.

    Args:
        query: The user query text
        param: Optional query parameters as a QueryParam or plain dict

    Returns:
        Dictionary containing the query result or error information
    """
    from .core import LightRAGCore

    try:
        config = get_lightrag_core_settings()
        query_param = QueryParam(**param) if isinstance(param, dict) else param
        core = LightRAGCore(
            embedding_model=config["EMBEDDING_MODEL"],
            embedding_provider=config["EMBEDDING_PROVIDER"],
            embedding_base_url=config["EMBEDDING_BASE_URL"],
            llm_model=config["LLM_MODEL"],
        )
        try:
            return to_serializable(core.query(query, query_param or QueryParam()))
        finally:
            core.close()
    except Exception as e:
        return {"error": "query_failed", "message": str(e)}


__all__ = ["run_query", "run_update"]
