from typing import Any, Optional


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
    from django.conf import settings

    from .core import LightRAGCore

    config = getattr(settings, "LIGHTRAG", {})
    embedding_model = config.get(
        "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
    )
    embedding_provider = config.get("EMBEDDING_PROVIDER", "LMStudio")
    embedding_base_url = config.get("EMBEDDING_BASE_URL", "http://localhost:1234")
    llm_model = config.get("LLM_MODEL", "gpt-4o-mini")

    try:
        core = LightRAGCore(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
            llm_model=llm_model,
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
