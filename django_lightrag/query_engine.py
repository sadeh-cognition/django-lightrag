from typing import Any

from .llm import LLMService
from .models import Document, Entity, Relation
from .storage import ChromaVectorStorage
from .types import QueryParam, QueryResult


class QueryEngine:
    """Handles RAG retrieval and response generation"""

    def __init__(
        self,
        llm_service: LLMService,
        vector_storage: ChromaVectorStorage,
        tokenizer: Any,
    ):
        self.llm_service = llm_service
        self.vector_storage = vector_storage
        self.tokenizer = tokenizer

    def query(self, query_text: str, param: QueryParam) -> QueryResult:
        """Query the RAG system"""
        raise NotImplementedError("Use LightRAGCore.query() instead.")

    def retrieve_documents(
        self, query_embedding: list[float], top_k: int
    ) -> list[Document]:
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

    def retrieve_entities(
        self, query_embedding: list[float], top_k: int
    ) -> list[Entity]:
        entity_results = self.vector_storage.search_similar(
            "entity", query_embedding, top_k=top_k
        )
        entity_ids = [item["id"] for item in entity_results]
        entities_by_id = {
            entity.id: entity for entity in Entity.objects.filter(id__in=entity_ids)
        }
        return [
            entities_by_id[entity_id]
            for entity_id in entity_ids
            if entity_id in entities_by_id
        ]

    def retrieve_relations(
        self, query_embedding: list[float], top_k: int
    ) -> list[Relation]:
        relation_results = self.vector_storage.search_similar(
            "relation", query_embedding, top_k=top_k
        )
        relation_ids = [item["id"] for item in relation_results]
        relations_by_id = {
            relation.id: relation
            for relation in Relation.objects.select_related(
                "source_entity", "target_entity"
            ).filter(id__in=relation_ids)
        }
        return [
            relations_by_id[relation_id]
            for relation_id in relation_ids
            if relation_id in relations_by_id
        ]

    def merge_unique_records(self, records: list[Any]) -> list[Any]:
        unique_records: list[Any] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.id in seen_ids:
                continue
            seen_ids.add(record.id)
            unique_records.append(record)
        return unique_records

    def build_context(
        self,
        documents: list[Document],
        entities: list[Entity],
        relations: list[Relation],
        param: QueryParam,
    ) -> dict[str, Any]:
        """Build context for response generation"""
        context = {
            "documents": [],
            "entities": [],
            "relations": [],
            "query_keywords": {
                "low_level_keywords": list(param.low_level_keywords),
                "high_level_keywords": list(param.high_level_keywords),
            },
            "total_tokens": 0,
        }

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
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(document_text)

        # Add entities
        for entity in entities:
            profile_text = entity.profile_value or entity.description
            entity_text = f"{entity.name} ({entity.entity_type}): {profile_text}"
            if (
                context["total_tokens"] + self.tokenizer.count_tokens(entity_text)
                > param.max_tokens
            ):
                break

            context["entities"].append(
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": profile_text,
                    "profile_key": entity.profile_key,
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(entity_text)

        # Add relations
        for relation in relations:
            profile_text = relation.profile_value or relation.description
            relation_text = (
                f"{relation.source_entity.name} -> {relation.relation_type} -> "
                f"{relation.target_entity.name}: {profile_text}"
            )
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
                    "description": profile_text,
                    "profile_key": relation.profile_key,
                }
            )
            context["total_tokens"] += self.tokenizer.count_tokens(relation_text)

        return context

    def generate_response(
        self, query_text: str, context: dict[str, Any], param: QueryParam
    ) -> str:
        """Generate response based on context"""
        # Placeholder - in practice, use LLM to generate response
        # context_text = json.dumps(context, indent=2)

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

    def format_sources(
        self,
        documents: list[Document],
        entities: list[Entity],
        relations: list[Relation],
    ) -> list[dict[str, Any]]:
        """Format sources for the response"""
        sources = []

        for document in documents:
            sources.append(
                {
                    "type": "document",
                    "id": document.id,
                    "content": (
                        document.content[:200] + "..."
                        if len(document.content) > 200
                        else document.content
                    ),
                    "document_id": document.id,
                }
            )

        for entity in entities:
            profile_text = entity.profile_value or entity.description
            sources.append(
                {
                    "type": "entity",
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": (
                        profile_text[:200] + "..."
                        if len(profile_text) > 200
                        else profile_text
                    ),
                }
            )

        for relation in relations:
            profile_text = relation.profile_value or relation.description
            sources.append(
                {
                    "type": "relation",
                    "id": relation.id,
                    "source": relation.source_entity.name,
                    "relation_type": relation.relation_type,
                    "target": relation.target_entity.name,
                    "description": (
                        profile_text[:200] + "..."
                        if len(profile_text) > 200
                        else profile_text
                    ),
                }
            )

        return sources
