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
    from .core import LightRAGCore

    try:
        core = LightRAGCore()
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
