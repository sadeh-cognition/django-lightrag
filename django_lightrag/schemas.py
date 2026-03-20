"""
Pydantic schemas for LightRAG API using django-ninja.
"""

from typing import List, Dict, Any, Optional
from ninja import Schema


class DocumentIngestSchema(Schema):
    content: str
    track_id: str = ""
    metadata: Dict[str, Any] = {}


class DocumentSchema(Schema):
    id: str
    track_id: str
    created_at: str
    updated_at: str


class QueryParamSchema(Schema):
    mode: str = "hybrid"
    top_k: int = 10
    max_tokens: int = 4000
    temperature: float = 0.1
    stream: bool = False


class QueryRequestSchema(Schema):
    query: str
    param: Optional[QueryParamSchema] = None


class SourceSchema(Schema):
    type: str
    id: str
    name: Optional[str] = None
    content: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    entity_type: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    relation_type: Optional[str] = None
    target: Optional[str] = None


class QueryResultSchema(Schema):
    response: str
    sources: List[SourceSchema]
    context: Dict[str, Any]
    query_time: float
    tokens_used: int


class EntitySchema(Schema):
    id: str
    name: str
    entity_type: str
    description: str
    source_ids: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class RelationSchema(Schema):
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    description: str
    source_ids: List[str]
    weight: float
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class ErrorResponseSchema(Schema):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponseSchema(Schema):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
