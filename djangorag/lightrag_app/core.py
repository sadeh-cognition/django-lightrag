import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import uuid

import tiktoken
from django.conf import settings
from django.utils import timezone

from .models import (
    Document,
    DocumentStatus,
    TextChunk,
    Entity,
    Relation,
    EntityChunk,
    RelationChunk,
    VectorEmbedding,
    CacheEntry,
    ProcessingJob,
)
from .storage import LadybugGraphStorage, ChromaVectorStorage


@dataclass
class QueryParam:
    """Parameters for RAG queries"""

    mode: str = "hybrid"  # "local", "global", "hybrid" # TODO: make this an Enum
    top_k: int = 10
    max_tokens: int = 4000
    temperature: float = 0.1
    stream: bool = False  # TODO: remove this option


@dataclass
class QueryResult:
    """Result of a RAG query"""

    response: str
    sources: List[Dict[str, Any]]
    context: Dict[str, Any]
    query_time: float
    tokens_used: int


class Tokenizer:
    """Tokenizer using tiktoken"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])


class LightRAGCore:
    """Core LightRAG functionality integrated with Django"""

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.graph_storage = LadybugGraphStorage()
        self.vector_storage = ChromaVectorStorage()

        # Load configuration from settings
        self.config = getattr(settings, "LIGHTRAG", {})
        self.chunk_size = self.config.get("CHUNK_SIZE", 1200)
        self.chunk_overlap = self.config.get("CHUNK_OVERLAP_SIZE", 100)
        self.top_k = self.config.get("TOP_K", 10)
        self.max_entity_tokens = self.config.get("MAX_ENTITY_TOKENS", 8000)
        self.max_relation_tokens = self.config.get("MAX_RELATION_TOKENS", 4000)
        self.max_total_tokens = self.config.get("MAX_TOTAL_TOKENS", 12000)
        self.cosine_threshold = self.config.get("COSINE_THRESHOLD", 0.2)

    def _generate_id(self, content: str) -> str:
        """Generate a consistent ID from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def _chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Split text into chunks"""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                {
                    "id": self._generate_id(f"{text}_{chunk_index}"),
                    "content": chunk_text,
                    "tokens": len(chunk_tokens),
                    "chunk_order_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                }
            )

            if end >= len(tokens):
                break

            start = end - chunk_overlap
            chunk_index += 1

        return chunks

    def ingest_document(
        self,
        content: str,
        title: str = "",
        file_path: str = "",
        metadata: Dict[str, Any] = None,
        track_id: str = "",
    ) -> str:
        """Ingest a document into the RAG system"""
        document_id = self._generate_id(content)
        metadata = metadata or {}

        # Create document
        document = Document.objects.create(
            id=document_id,
            title=title,
            content=content,
            file_path=file_path,
            metadata=metadata,
            track_id=track_id,
        )

        # Update status to processing
        doc_status = document.status
        doc_status.status = "processing"
        doc_status.save()

        try:
            # Chunk the document
            chunks = self._chunk_text(content)

            # Save chunks to database
            saved_chunks = []
            for chunk_data in chunks:
                chunk = TextChunk.objects.create(
                    id=chunk_data["id"],
                    document=document,
                    content=chunk_data["content"],
                    tokens=chunk_data["tokens"],
                    chunk_order_index=chunk_data["chunk_order_index"],
                    metadata=chunk_data,
                )
                saved_chunks.append(chunk)

            # Extract entities and relations (simplified version)
            self._extract_knowledge_graph(saved_chunks)

            # Generate embeddings for chunks
            self._generate_chunk_embeddings(saved_chunks)

            # Update status to completed
            doc_status.status = "processed"
            doc_status.chunks_count = len(chunks)
            doc_status.chunks_list = [chunk.id for chunk in saved_chunks]
            doc_status.save()

            return document_id

        except Exception as e:
            # Update status to failed
            doc_status.status = "failed"
            doc_status.error_message = str(e)
            doc_status.save()
            raise

    def _extract_knowledge_graph(self, chunks: List[TextChunk]):
        """Extract entities and relations from chunks"""
        # This is a simplified version - in practice, you'd use LLM for extraction
        for chunk in chunks:
            # Simple entity extraction (placeholder)
            entities = self._extract_entities_from_chunk(chunk)

            # Save entities
            for entity_data in entities:
                entity, created = Entity.objects.get_or_create(
                    id=entity_data["id"],
                    defaults={
                        "name": entity_data["name"],
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data.get("description", ""),
                        "source_ids": [chunk.id],
                        "metadata": entity_data.get("metadata", {}),
                    },
                )

                if not created:
                    # Update existing entity
                    if chunk.id not in entity.source_ids:
                        entity.source_ids.append(chunk.id)
                        entity.save()

                # Add to graph storage
                self.graph_storage.add_entity(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "metadata": entity.metadata,
                    }
                )

                # Create entity-chunk mapping
                EntityChunk.objects.get_or_create(
                    id=f"{entity.id}_{chunk.id}", entity=entity, chunk=chunk
                )

            # Simple relation extraction (placeholder)
            relations = self._extract_relations_from_chunk(chunk, entities)

            # Save relations
            for relation_data in relations:
                relation, created = Relation.objects.get_or_create(
                    id=relation_data["id"],
                    defaults={
                        "source_entity": Entity.objects.get(
                            id=relation_data["source_entity"]
                        ),
                        "target_entity": Entity.objects.get(
                            id=relation_data["target_entity"]
                        ),
                        "relation_type": relation_data["relation_type"],
                        "description": relation_data.get("description", ""),
                        "source_ids": [chunk.id],
                        "metadata": relation_data.get("metadata", {}),
                    },
                )

                if not created:
                    # Update existing relation
                    if chunk.id not in relation.source_ids:
                        relation.source_ids.append(chunk.id)
                        relation.save()

                # Add to graph storage
                self.graph_storage.add_relation(
                    {
                        "id": relation.id,
                        "source_entity": relation.source_entity.id,
                        "target_entity": relation.target_entity.id,
                        "relation_type": relation.relation_type,
                        "description": relation.description,
                        "metadata": relation.metadata,
                    }
                )

                # Create relation-chunk mapping
                RelationChunk.objects.get_or_create(
                    id=f"{relation.id}_{chunk.id}", relation=relation, chunk=chunk
                )

    def _extract_entities_from_chunk(self, chunk: TextChunk) -> List[Dict[str, Any]]:
        """Extract entities from a text chunk (simplified placeholder)"""
        # In practice, this would use an LLM to extract entities
        # For now, return empty list as placeholder
        return []

    def _extract_relations_from_chunk(
        self, chunk: TextChunk, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from a text chunk (simplified placeholder)"""
        # In practice, this would use an LLM to extract relations
        # For now, return empty list as placeholder
        return []

    def _generate_chunk_embeddings(self, chunks: List[TextChunk]):
        """Generate embeddings for text chunks"""
        # This is a placeholder - in practice, you'd use an embedding model
        for chunk in chunks:
            # Generate dummy embedding for demonstration
            embedding_dim = 1536  # Typical embedding dimension
            embedding = [0.0] * embedding_dim

            # Save to database
            VectorEmbedding.objects.update_or_create(
                id=f"chunk_{chunk.id}",
                vector_type="chunk",
                content_id=chunk.id,
                defaults={"embedding": embedding},
            )

            # Save to vector storage - skip for now to avoid ChromaDB issues
            # self.vector_storage.add_embedding(
            #     'chunk', chunk.id, embedding,
            #     metadata={
            #         'content': chunk.content[:500],  # First 500 chars
            #         'document_id': chunk.document.id,
            #         'chunk_order_index': chunk.chunk_order_index
            #     }
            # )

    def query(self, query_text: str, param: QueryParam = None) -> QueryResult:
        """Query the RAG system"""
        if param is None:
            param = QueryParam()

        start_time = time.time()

        # Generate query embedding
        query_embedding = self._get_query_embedding(query_text)

        # Retrieve relevant chunks
        relevant_chunks = self._retrieve_chunks(query_embedding, param.top_k)

        # Retrieve relevant entities and relations
        relevant_entities, relevant_relations = self._retrieve_knowledge_graph(
            query_text, param.top_k
        )

        # Build context
        context = self._build_context(
            relevant_chunks, relevant_entities, relevant_relations, param
        )

        # Generate response (placeholder)
        response = self._generate_response(query_text, context, param)

        query_time = time.time() - start_time

        return QueryResult(
            response=response,
            sources=self._format_sources(
                relevant_chunks, relevant_entities, relevant_relations
            ),
            context=context,
            query_time=query_time,
            tokens_used=self.tokenizer.count_tokens(response),
        )

    def _get_query_embedding(self, query_text: str) -> List[float]:
        """Get embedding for query text"""
        # Placeholder - in practice, use embedding model
        embedding_dim = 1536
        return [0.0] * embedding_dim

    def _retrieve_chunks(
        self, query_embedding: List[float], top_k: int
    ) -> List[TextChunk]:
        """Retrieve relevant chunks using vector similarity"""
        # For now, just return recent chunks to avoid vector storage issues
        chunks = list(TextChunk.objects.all().order_by("-created_at")[:top_k])
        return chunks

    def _retrieve_knowledge_graph(
        self, query_text: str, top_k: int
    ) -> Tuple[List[Entity], List[Relation]]:
        """Retrieve relevant entities and relations"""
        # Placeholder implementation - in practice, use graph traversal or entity matching
        entities = list(Entity.objects.all()[:top_k])
        relations = list(Relation.objects.all()[:top_k])

        return entities, relations

    def _build_context(
        self,
        chunks: List[TextChunk],
        entities: List[Entity],
        relations: List[Relation],
        param: QueryParam,
    ) -> Dict[str, Any]:
        """Build context for response generation"""
        context = {"chunks": [], "entities": [], "relations": [], "total_tokens": 0}

        # Add chunks
        for chunk in chunks:
            chunk_text = chunk.content
            if (
                context["total_tokens"] + self.tokenizer.count_tokens(chunk_text)
                > param.max_tokens
            ):
                chunk_text = self.tokenizer.truncate_by_tokens(
                    chunk_text, param.max_tokens - context["total_tokens"]
                )

            context["chunks"].append(
                {
                    "content": chunk_text,
                    "document_id": chunk.document.id,
                    "chunk_order_index": chunk.chunk_order_index,
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(chunk_text)

        # Add entities
        for entity in entities:
            entity_text = f"{entity.name} ({entity.entity_type}): {entity.description}"
            if (
                context["total_tokens"] + self.tokenizer.count_tokens(entity_text)
                > param.max_tokens
            ):
                break

            context["entities"].append(
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(entity_text)

        # Add relations
        for relation in relations:
            relation_text = f"{relation.source_entity.name} -> {relation.relation_type} -> {relation.target_entity.name}"
            if (
                context["total_tokens"] + self.tokenizer.count_tokens(relation_text)
                > param.max_tokens
            ):
                break

            context["relations"].append(
                {
                    "source": relation.source_entity.name,
                    "relation_type": relation.relation_type,
                    "target": relation.target_entity.name,
                    "description": relation.description,
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(relation_text)

        return context

    def _generate_response(
        self, query_text: str, context: Dict[str, Any], param: QueryParam
    ) -> str:
        """Generate response based on context"""
        # Placeholder - in practice, use LLM to generate response
        context_text = json.dumps(context, indent=2)

        response = f"""Based on the provided context, here's my response to your query "{query_text}":

This is a placeholder response. In a real implementation, this would be generated by an LLM using the retrieved context.

Context summary:
- {len(context["chunks"])} relevant text chunks
- {len(context["entities"])} relevant entities
- {len(context["relations"])} relevant relations
- Total context tokens: {context["total_tokens"]}

The actual implementation would use the context to provide a detailed, relevant answer to your query.
"""
        return response

    def _format_sources(
        self, chunks: List[TextChunk], entities: List[Entity], relations: List[Relation]
    ) -> List[Dict[str, Any]]:
        """Format sources for the response"""
        sources = []

        for chunk in chunks:
            sources.append(
                {
                    "type": "chunk",
                    "id": chunk.id,
                    "content": chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    "document_id": chunk.document.id,
                    "document_title": chunk.document.title or chunk.document.id[:50],
                    "chunk_order_index": chunk.chunk_order_index,
                }
            )

        for entity in entities:
            sources.append(
                {
                    "type": "entity",
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description[:200] + "..."
                    if len(entity.description) > 200
                    else entity.description,
                }
            )

        for relation in relations:
            sources.append(
                {
                    "type": "relation",
                    "id": relation.id,
                    "source": relation.source_entity.name,
                    "relation_type": relation.relation_type,
                    "target": relation.target_entity.name,
                    "description": relation.description[:200] + "..."
                    if len(relation.description) > 200
                    else relation.description,
                }
            )

        return sources

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its related data"""
        try:
            document = Document.objects.get(id=document_id)

            # Get chunks to delete from vector storage (skipped for now)
            # chunks = await sync_to_async(list)(TextChunk.objects.filter(document=document))
            # for chunk in chunks:
            #     await self.vector_storage.delete_embedding('chunk', chunk.id)

            # Get entities and relations to delete from graph storage (skipped for now)
            # entities = await sync_to_async(list)(Entity.objects.all()))
            # for entity in entities:
            #     if document_id in [doc.id for doc in entity.source_ids if doc]:
            #         await self.graph_storage.delete_entity(entity.id)

            # Delete document (cascade will handle related records)
            document.delete()

            return True
        except Document.DoesNotExist:
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to delete document: {e}")

    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status"""
        try:
            document = Document.objects.get(id=document_id)
            status = document.status

            return {
                "document_id": document.id,
                "title": document.title,
                "status": status.status,
                "chunks_count": status.chunks_count,
                "chunks_list": status.chunks_list,
                "error_message": status.error_message,
                "started_at": status.started_at.isoformat()
                if status.started_at
                else None,
                "completed_at": status.completed_at.isoformat()
                if status.completed_at
                else None,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
            }
        except Document.DoesNotExist:
            return None

    def list_documents(self, status_filter: str = None) -> List[Dict[str, Any]]:
        """List documents in the system"""

        documents = Document.objects.all().select_related("status")
        if status_filter:
            documents = documents.filter(status__status=status_filter)
        documents = list(documents)

        result = []
        for doc in documents:
            result.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "status": doc.status.status,
                    "chunks_count": doc.status.chunks_count,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )

        return result

    def close(self):
        """Close storage connections"""
        self.graph_storage.close()
        self.vector_storage.close()
