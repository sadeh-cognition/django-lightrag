"""Pydantic schemas for LightRAG API using django-ninja."""

from typing import Any

from ninja import Schema
from pydantic import Field


class DocumentIngestSchema(Schema):
    content: str
    track_id: str = ""
    metadata: dict[str, Any] = {}


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
    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)
    one_hop_enabled: bool = False
    one_hop_max_entities: int = 10
    one_hop_max_relations: int = 10


class QueryRequestSchema(Schema):
    query: str
    param: QueryParamSchema | None = None


class SourceSchema(Schema):
    type: str
    id: str
    name: str | None = None
    content: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    entity_type: str | None = None
    description: str | None = None
    source: str | None = None
    relation_type: str | None = None
    target: str | None = None


class QueryResultSchema(Schema):
    response: str
    sources: list[SourceSchema]
    context: dict[str, Any]
    query_time: float
    tokens_used: int


class EntitySchema(Schema):
    id: str
    name: str
    entity_type: str
    description: str
    source_ids: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class RelationSchema(Schema):
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    description: str
    source_ids: list[str]
    weight: float
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class ErrorResponseSchema(Schema):
    error: str
    message: str
    details: dict[str, Any] | None = None


class SuccessResponseSchema(Schema):
    success: bool
    message: str
    data: dict[str, Any] | None = None
