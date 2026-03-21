"""Django API views for LightRAG using django-ninja."""

from django.conf import settings
from django.utils.module_loading import import_string
from ninja import Router

from .core import LightRAGCore, QueryParam
from .models import Entity, Relation
from .schemas import (
    DocumentIngestSchema,
    DocumentSchema,
    EntitySchema,
    ErrorResponseSchema,
    QueryRequestSchema,
    QueryResultSchema,
    RelationSchema,
    SuccessResponseSchema,
)

router = Router()


def build_lightrag_core(
    *,
    embedding_model: str,
    embedding_provider: str,
    embedding_base_url: str,
    llm_model: str,
):
    return LightRAGCore(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        llm_model=llm_model,
    )


def create_lightrag_core():
    config = getattr(settings, "LIGHTRAG", {})
    embedding_model = config.get(
        "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
    )
    embedding_provider = config.get("EMBEDDING_PROVIDER", "LMStudio")
    embedding_base_url = config.get("EMBEDDING_BASE_URL", "http://localhost:1234")
    llm_model = config.get("LLM_MODEL", "gpt-4o-mini")
    factory_path = config.get("CORE_FACTORY")
    factory = import_string(factory_path) if factory_path else build_lightrag_core
    return factory(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        llm_model=llm_model,
    )


@router.post(
    "/documents/ingest", response={201: dict[str, str], 400: ErrorResponseSchema}
)
def ingest_document(request, data: DocumentIngestSchema):
    """Ingest a document into the system"""
    try:
        core = create_lightrag_core()
        try:
            document_id = core.ingest_document(
                content=data.content,
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


@router.get(
    "/documents", response={200: list[DocumentSchema], 400: ErrorResponseSchema}
)
def list_documents(request):
    """List documents in the system"""
    try:
        core = create_lightrag_core()
        try:
            documents = core.list_documents()
            return [DocumentSchema(**doc) for doc in documents]
        finally:
            core.close()
    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.post("/query", response={200: QueryResultSchema, 400: ErrorResponseSchema})
def query_rag(request, data: QueryRequestSchema):
    """Query the RAG system"""
    try:
        # Create query parameters
        param_data = data.param.dict() if data.param else {}
        param = QueryParam(**param_data)
        core = create_lightrag_core()
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
        core = create_lightrag_core()
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


@router.get("/entities", response={200: list[EntitySchema], 400: ErrorResponseSchema})
def list_entities(request, limit: int | None = None):
    """List entities in the system"""
    try:
        queryset = Entity.objects.order_by("created_at")
        if limit is not None:
            queryset = queryset[:limit]
        return [
            EntitySchema(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_ids=entity.source_ids,
                metadata=entity.metadata,
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
            )
            for entity in queryset
        ]

    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.get(
    "/relations", response={200: list[RelationSchema], 400: ErrorResponseSchema}
)
def list_relations(request, limit: int | None = None):
    """List relations in the system"""
    try:
        queryset = Relation.objects.select_related(
            "source_entity", "target_entity"
        ).order_by("created_at")
        if limit is not None:
            queryset = queryset[:limit]
        return [
            RelationSchema(
                id=relation.id,
                source_entity=relation.source_entity_id,
                target_entity=relation.target_entity_id,
                relation_type=relation.relation_type,
                description=relation.description,
                source_ids=relation.source_ids,
                weight=relation.weight,
                metadata=relation.metadata,
                created_at=relation.created_at.isoformat(),
                updated_at=relation.updated_at.isoformat(),
            )
            for relation in queryset
        ]

    except Exception as e:
        return 400, {"error": "list_failed", "message": str(e)}


@router.get("/health", response=dict[str, str])
def health_check(request):
    """Health check endpoint"""
    return {"status": "healthy", "service": "lightrag-django"}
