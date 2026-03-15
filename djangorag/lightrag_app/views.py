"""
Django API views for LightRAG using django-ninja.
"""

from typing import List, Dict, Optional
from ninja import Router, File, Form
from ninja.files import UploadedFile

from .core import LightRAGCore, QueryParam
from .schemas import (
    DocumentIngestSchema,
    DocumentSchema,
    DocumentStatusSchema,
    QueryRequestSchema,
    QueryResultSchema,
    EntitySchema,
    RelationSchema,
    ErrorResponseSchema,
    SuccessResponseSchema,
)

router = Router()


@router.post(
    "/documents/ingest", response={201: Dict[str, str], 400: ErrorResponseSchema}
)
def ingest_document(request, data: DocumentIngestSchema):
    """Ingest a document into the system"""
    try:
        core = LightRAGCore()
        try:
            document_id = core.ingest_document(
                content=data.content,
                title=data.title,
                file_path=data.file_path,
                metadata=data.metadata,
                track_id=data.track_id,
            )
            return 201, {
                "document_id": document_id,
                "message": "Document ingested successfully",
            }
        finally:
            core.close()
    except Exception as e:
        return 400, {"error": "ingestion_failed", "message": str(e)}


@router.post(
    "/documents/ingest-file", response={201: Dict[str, str], 400: ErrorResponseSchema}
)
def ingest_document_file(
    request,
    file: UploadedFile = File(...),
    title: str = Form(""),
    track_id: str = Form(""),
    metadata: str = Form("{}"),
):
    """Ingest a file into the system"""
    try:
        # Read file content
        content = file.read().decode("utf-8")

        # Parse metadata
        try:
            import json

            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {}

        # Use filename as title if not provided
        if not title:
            title = file.name

        core = LightRAGCore()
        try:
            document_id = core.ingest_document(
                content=content,
                title=title,
                file_path=file.name,
                metadata=metadata_dict,
                track_id=track_id,
            )
            return 201, {
                "document_id": document_id,
                "message": "File ingested successfully",
            }
        finally:
            core.close()

    except Exception as e:
        return 400, {"error": "ingestion_failed", "message": str(e)}


@router.get(
    "/documents", response={200: List[DocumentSchema], 400: ErrorResponseSchema}
)
def list_documents(request, status: Optional[str] = None):
    """List documents in the system"""
    try:
        core = LightRAGCore()
        try:
            documents = core.list_documents(status)
            return [DocumentSchema(**doc) for doc in documents]
        finally:
            core.close()
    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.get(
    "/documents/{document_id}/status",
    response={200: DocumentStatusSchema, 404: ErrorResponseSchema},
)
def get_document_status(request, document_id: str):
    """Get document processing status"""
    try:
        core = LightRAGCore()
        try:
            status = core.get_document_status(document_id)
            if not status:
                return 404, {
                    "error": "document_not_found",
                    "message": f"Document '{document_id}' not found",
                }
            return DocumentStatusSchema(**status)
        finally:
            core.close()
    except Exception as e:
        return 400, {"error": "status_failed", "message": str(e)}


@router.post("/query", response={200: QueryResultSchema, 400: ErrorResponseSchema})
def query_rag(request, data: QueryRequestSchema):
    """Query the RAG system"""
    try:
        # Create query parameters
        param_data = data.param.dict() if data.param else {}
        param = QueryParam(**param_data)

        core = LightRAGCore()
        try:
            result = core.query(data.query, param)
            return QueryResultSchema(
                response=result.response,
                sources=result.sources,
                context=result.context,
                query_time=result.query_time,
                tokens_used=result.tokens_used,
            )
        finally:
            core.close()

    except Exception as e:
        return 400, {"error": "query_failed", "message": str(e)}


@router.delete(
    "/documents/{document_id}",
    response={
        200: SuccessResponseSchema,
        404: ErrorResponseSchema,
        400: ErrorResponseSchema,
    },
)
def delete_document(request, document_id: str):
    """Delete a document"""
    try:
        core = LightRAGCore()
        try:
            success = core.delete_document(document_id)
            if not success:
                return 404, {
                    "error": "document_not_found",
                    "message": f"Document '{document_id}' not found",
                }
            return 200, {"success": True, "message": "Document deleted successfully"}
        finally:
            core.close()

    except Exception as e:
        return 400, {"error": "deletion_failed", "message": str(e)}


@router.get("/entities", response={200: List[EntitySchema], 400: ErrorResponseSchema})
def list_entities(request, limit: Optional[int] = None):
    """List entities in the system"""
    try:
        core = LightRAGCore()
        try:
            # Get entities from graph storage
            entities = core.graph_storage.get_all_entities(limit)
            return [EntitySchema(**entity) for entity in entities]
        finally:
            core.graph_storage.close()

    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.get(
    "/relations", response={200: List[RelationSchema], 400: ErrorResponseSchema}
)
def list_relations(request, limit: Optional[int] = None):
    """List relations in the system"""
    try:
        core = LightRAGCore()
        try:
            # Get relations from graph storage
            relations = core.graph_storage.get_all_relations(limit)
            return [RelationSchema(**relation) for relation in relations]
        finally:
            core.graph_storage.close()

    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.get("/health", response=Dict[str, str])
def health_check(request):
    """Health check endpoint"""
    return {"status": "healthy", "service": "lightrag-django"}
