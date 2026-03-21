from typing import Any

from .config import get_lightrag_core_settings


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
            )
            return {
                "document_id": document_id,
                "message": "Document ingested successfully",
            }
        finally:
            core.close()
    except Exception as e:
        return {"error": "ingestion_failed", "message": str(e)}


__all__ = ["run_update"]
