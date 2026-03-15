import hashlib
import json
import time
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

import tiktoken
from django.conf import settings

from embed_gen.generator import generate_embeddings

from .models import (
    Document,
    Entity,
    Relation,
    VectorEmbedding,
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
        self.top_k = self.config.get("TOP_K", 10)
        self.max_entity_tokens = self.config.get("MAX_ENTITY_TOKENS", 8000)
        self.max_relation_tokens = self.config.get("MAX_RELATION_TOKENS", 4000)
        self.max_total_tokens = self.config.get("MAX_TOTAL_TOKENS", 12000)
        self.cosine_threshold = self.config.get("COSINE_THRESHOLD", 0.2)
        self.embedding_provider = self.config.get("EMBEDDING_PROVIDER", "LMStudio")
        self.embedding_model = self.config.get(
            "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
        )
        self.embedding_base_url = self.config.get(
            "EMBEDDING_BASE_URL", "http://localhost:1234"
        )

    def _generate_id(self, content: str) -> str:
        """Generate a consistent ID from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def ingest_document(
        self,
        content: str,
        title: str = "",
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
            metadata=metadata,
            track_id=track_id,
        )

        try:
            # Extract entities and relations (simplified version)
            self._extract_knowledge_graph_from_document(document)

            # Generate embeddings for document
            self._generate_document_embeddings(document)

            return document_id

        except Exception as e:
            raise

    def _extract_knowledge_graph_from_document(self, document: Document):
        """Extract entities and relations from a document"""
        # This is a simplified version - in practice, you'd use LLM for extraction
        # Simple entity extraction (placeholder)
        entities = self._extract_entities_from_document(document)

        # Save entities
        for entity_data in entities:
            entity, created = Entity.objects.get_or_create(
                id=entity_data["id"],
                defaults={
                    "name": entity_data["name"],
                    "entity_type": entity_data["entity_type"],
                    "description": entity_data.get("description", ""),
                    "source_ids": [document.id],
                    "metadata": entity_data.get("metadata", {}),
                },
            )

            if not created:
                # Update existing entity
                updated = False
                if document.id not in entity.source_ids:
                    entity.source_ids.append(document.id)
                    updated = True
                if updated:
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

        # Simple relation extraction (placeholder)
        relations = self._extract_relations_from_document(document, entities)

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
                    "source_ids": [document.id],
                    "metadata": relation_data.get("metadata", {}),
                },
            )

            if not created:
                # Update existing relation
                updated = False
                if document.id not in relation.source_ids:
                    relation.source_ids.append(document.id)
                    updated = True
                if updated:
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

    def _extract_entities_from_document(
        self, document: Document
    ) -> List[Dict[str, Any]]:
        """Extract entities from a document (simplified placeholder)"""
        # In practice, this would use an LLM to extract entities
        # For now, return empty list as placeholder
        return []

    def _extract_relations_from_document(
        self, document: Document, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from a document (simplified placeholder)"""
        # In practice, this would use an LLM to extract relations
        # For now, return empty list as placeholder
        return []

    def _generate_document_embeddings(self, document: Document):
        """Generate embeddings for a document"""
        embedding = self._get_embeddings([document.content])[0]

        # Save to database
        VectorEmbedding.objects.update_or_create(
            id=f"document_{document.id}",
            vector_type="document",
            content_id=document.id,
            defaults={"embedding": embedding},
        )

        # Save to vector storage
        self.vector_storage.add_embedding(
            "document",
            document.id,
            embedding,
            metadata={
                "content": document.content[:500],  # First 500 chars
                "document_id": document.id,
                "document_title": document.title,
            },
        )

    def query(self, query_text: str, param: QueryParam = None) -> QueryResult:
        """Query the RAG system"""
        if param is None:
            param = QueryParam()

        start_time = time.time()

        # Generate query embedding
        query_embedding = self._get_query_embedding(query_text)

        # Retrieve relevant documents
        relevant_documents = self._retrieve_documents(query_embedding, param.top_k)

        # Retrieve relevant entities and relations
        relevant_entities, relevant_relations = self._retrieve_knowledge_graph(
            query_text, param.top_k
        )

        # Build context
        context = self._build_context(
            relevant_documents, relevant_entities, relevant_relations, param
        )

        # Generate response (placeholder)
        response = self._generate_response(query_text, context, param)

        query_time = time.time() - start_time

        return QueryResult(
            response=response,
            sources=self._format_sources(
                relevant_documents, relevant_entities, relevant_relations
            ),
            context=context,
            query_time=query_time,
            tokens_used=self.tokenizer.count_tokens(response),
        )

    def _get_query_embedding(self, query_text: str) -> List[float]:
        """Get embedding for query text"""
        return self._get_embeddings([query_text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            raise ValueError("No texts provided for embedding generation.")

        try:
            return generate_embeddings(
                texts=texts,
                model_name=self.embedding_model,
                provider=self.embedding_provider,
                base_url=self.embedding_base_url,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    def _retrieve_documents(
        self, query_embedding: List[float], top_k: int
    ) -> List[Document]:
        """Retrieve relevant documents using vector similarity"""
        results = self.vector_storage.search_similar(
            "document", query_embedding, top_k=top_k
        )
        if not results:
            return list(Document.objects.all().order_by("-created_at")[:top_k])

        doc_ids = [item["id"] for item in results]
        documents_by_id = {
            doc.id: doc for doc in Document.objects.filter(id__in=doc_ids)
        }
        return [
            documents_by_id[doc_id] for doc_id in doc_ids if doc_id in documents_by_id
        ]

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
        documents: List[Document],
        entities: List[Entity],
        relations: List[Relation],
        param: QueryParam,
    ) -> Dict[str, Any]:
        """Build context for response generation"""
        context = {"documents": [], "entities": [], "relations": [], "total_tokens": 0}

        # Add documents
        for document in documents:
            document_text = document.content
            if (
                context["total_tokens"] + self.tokenizer.count_tokens(document_text)
                > param.max_tokens
            ):
                document_text = self.tokenizer.truncate_by_tokens(
                    document_text, param.max_tokens - context["total_tokens"]
                )

            context["documents"].append(
                {
                    "content": document_text,
                    "document_id": document.id,
                    "document_title": document.title or document.id[:50],
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(document_text)

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
- {len(context["documents"])} relevant documents
- {len(context["entities"])} relevant entities
- {len(context["relations"])} relevant relations
- Total context tokens: {context["total_tokens"]}

The actual implementation would use the context to provide a detailed, relevant answer to your query.
"""
        return response

    def _format_sources(
        self,
        documents: List[Document],
        entities: List[Entity],
        relations: List[Relation],
    ) -> List[Dict[str, Any]]:
        """Format sources for the response"""
        sources = []

        for document in documents:
            sources.append(
                {
                    "type": "document",
                    "id": document.id,
                    "content": document.content[:200] + "..."
                    if len(document.content) > 200
                    else document.content,
                    "document_id": document.id,
                    "document_title": document.title or document.id[:50],
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

            # Get document embedding to delete from vector storage (skipped for now)
            self.vector_storage.delete_embedding("document", document.id)

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

    def list_documents(self) -> List[Dict[str, Any]]:
        """List documents in the system"""

        documents = list(Document.objects.all())

        result = []
        for doc in documents:
            result.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "track_id": doc.track_id,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )

        return result

    def close(self):
        """Close storage connections"""
        self.graph_storage.close()
        self.vector_storage.close()
