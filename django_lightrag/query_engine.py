from typing import Any

from django.db.models import Q

from .llm import LLMService
from .models import Document, Entity, Relation
from .storage import ChromaVectorStorage
from .types import (
    QueryContext,
    QueryContextDocument,
    QueryContextEntity,
    QueryContextRelation,
    QueryKeywordsResult,
    QueryParam,
    QueryResult,
    QuerySource,
)


class QueryEngine:
    """Handles RAG retrieval and response generation"""

    GROUNDED_FALLBACK_RESPONSE = (
        "I don't have enough information in the provided context to answer your query."
    )

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

    def search_document_vectors(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """Retrieve relevant document vectors using vector similarity"""
        return self.vector_storage.search_similar(
            "document", query_embedding, top_k=top_k
        )

    def hydrate_documents(
        self, vector_results: list[dict], fallback_top_k: int
    ) -> list[Document]:
        """Hydrate vector search results into Document ORM objects"""
        if not vector_results:
            return list(Document.objects.all().order_by("-created_at")[:fallback_top_k])

        doc_ids = [item["id"] for item in vector_results]
        documents_by_id = {
            doc.id: doc for doc in Document.objects.filter(id__in=doc_ids)
        }
        seen = set()
        hydrated = []
        for doc_id in doc_ids:
            if doc_id in documents_by_id and doc_id not in seen:
                seen.add(doc_id)
                hydrated.append(documents_by_id[doc_id])
        return hydrated

    def search_entity_vectors(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """Retrieve relevant entity vectors using vector similarity"""
        return self.vector_storage.search_similar(
            "entity", query_embedding, top_k=top_k
        )

    def hydrate_entities(self, vector_results: list[dict]) -> list[Entity]:
        """Hydrate vector search results into Entity ORM objects, deduplicating by entity ID while preserving vector rank."""
        entity_ids = []
        seen = set()
        for item in vector_results:
            eid = item["metadata"].get("entity_id", item["id"])
            if eid and eid not in seen:
                seen.add(eid)
                entity_ids.append(eid)

        entities_by_id = {
            entity.id: entity for entity in Entity.objects.filter(id__in=entity_ids)
        }
        return [entities_by_id[eid] for eid in entity_ids if eid in entities_by_id]

    def search_relation_vectors(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """Retrieve relevant relation vectors using vector similarity"""
        return self.vector_storage.search_similar(
            "relation", query_embedding, top_k=top_k
        )

    def hydrate_relations(self, vector_results: list[dict]) -> list[Relation]:
        """Hydrate vector search results into Relation ORM objects, deduplicating by relation ID while preserving vector rank."""
        relation_ids = []
        seen = set()
        for item in vector_results:
            rid = item["metadata"].get("relation_id", item["id"])
            if rid and rid not in seen:
                seen.add(rid)
                relation_ids.append(rid)

        relations_by_id = {
            relation.id: relation
            for relation in Relation.objects.select_related(
                "source_entity", "target_entity"
            ).filter(id__in=relation_ids)
        }
        return [relations_by_id[rid] for rid in relation_ids if rid in relations_by_id]

    def merge_unique_records(self, records: list[Any]) -> list[Any]:
        unique_records: list[Any] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.id in seen_ids:
                continue
            seen_ids.add(record.id)
            unique_records.append(record)
        return unique_records

    def expand_one_hop_neighborhood(
        self,
        relevant_entities: list[Entity],
        relevant_relations: list[Relation],
        max_entities: int,
        max_relations: int,
    ) -> tuple[list[Entity], list[Relation]]:
        """Expand the context by finding one-hop neighboring entities and relations.
        Excludes the seeds that were already found via vector matching.
        """
        seed_entity_ids = {e.id for e in relevant_entities}
        for rel in relevant_relations:
            seed_entity_ids.add(rel.source_entity_id)
            seed_entity_ids.add(rel.target_entity_id)

        if not seed_entity_ids:
            return [], []

        # Find one-hop incident relations
        neighbor_relations = list(
            Relation.objects.filter(
                Q(source_entity_id__in=seed_entity_ids)
                | Q(target_entity_id__in=seed_entity_ids)
            )
            .select_related("source_entity", "target_entity")
            .order_by("-weight", "-updated_at")
        )

        # Exclude relations that are already seeds
        seed_relation_ids = {r.id for r in relevant_relations}
        new_relations = [
            r for r in neighbor_relations if r.id not in seed_relation_ids
        ][:max_relations]

        # Extract resulting new entities from the endpoints of the relations
        new_entities_dict = {}
        for r in new_relations:
            if r.source_entity_id not in seed_entity_ids:
                new_entities_dict[r.source_entity_id] = r.source_entity
            if r.target_entity_id not in seed_entity_ids:
                new_entities_dict[r.target_entity_id] = r.target_entity

        new_entities = list(new_entities_dict.values())[:max_entities]

        return new_entities, new_relations

    def build_context(
        self,
        documents: list[Document],
        entities: list[Entity],
        relations: list[Relation],
        param: QueryParam,
    ) -> QueryContext:
        """Build context for response generation"""
        context = QueryContext(
            query_keywords=QueryKeywordsResult(
                low_level_keywords=list(param.low_level_keywords),
                high_level_keywords=list(param.high_level_keywords),
            )
        )

        aggregated_chunks = []

        # Add entities first (Graph-first strategy)
        for entity in entities:
            profile_text = entity.profile_value or entity.description
            entity_chunk = "\n".join(
                [
                    "Entity",
                    f"Name: {entity.name}",
                    f"Type: {entity.entity_type}",
                    f"Summary: {profile_text}",
                ]
            )
            tokens = self.tokenizer.count_tokens(entity_chunk)
            if context.total_tokens + tokens > param.max_tokens:
                break

            context.entities.append(
                QueryContextEntity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=profile_text,
                    profile_key=entity.profile_key,
                )
            )
            context.total_tokens += tokens
            aggregated_chunks.append(entity_chunk)

        # Add relations next
        for relation in relations:
            profile_text = relation.profile_value or relation.description
            relation_chunk = "\n".join(
                [
                    "Relation",
                    f"Source: {relation.source_entity.name}",
                    f"Type: {relation.relation_type}",
                    f"Target: {relation.target_entity.name}",
                    f"Summary: {profile_text}",
                ]
            )
            tokens = self.tokenizer.count_tokens(relation_chunk)
            if context.total_tokens + tokens > param.max_tokens:
                break

            context.relations.append(
                QueryContextRelation(
                    source=relation.source_entity.name,
                    relation_type=relation.relation_type,
                    target=relation.target_entity.name,
                    description=profile_text,
                    profile_key=relation.profile_key,
                )
            )
            context.total_tokens += tokens
            aggregated_chunks.append(relation_chunk)

        # Add documents last
        for document in documents:
            document_text = document.content
            document_chunk = "\n".join(
                [
                    "Document",
                    f"Document ID: {document.id}",
                    f"Excerpt: {document_text}",
                ]
            )
            tokens = self.tokenizer.count_tokens(document_chunk)
            if context.total_tokens + tokens > param.max_tokens:
                remaining_tokens = param.max_tokens - context.total_tokens
                if remaining_tokens <= 0:
                    break
                document_text = self.tokenizer.truncate_by_tokens(
                    document_text, remaining_tokens
                ).strip()
                if not document_text:
                    break
                document_chunk = "\n".join(
                    [
                        "Document",
                        f"Document ID: {document.id}",
                        f"Excerpt: {document_text}",
                    ]
                )
                tokens = self.tokenizer.count_tokens(document_chunk)
                if context.total_tokens + tokens > param.max_tokens:
                    break

            context.documents.append(
                QueryContextDocument(
                    content=document_text,
                    document_id=document.id,
                )
            )
            context.total_tokens += tokens
            aggregated_chunks.append(document_chunk)

        context.aggregated_context = "\n\n".join(aggregated_chunks)
        return context

    generate_system_prompt = """You are a retrieval-augmented assistant.

Answer the user using only the provided context.
Do not invent, assume, or import outside knowledge.
Preserve the user's language.
If the context does not contain enough information, reply that you do not have enough information from the provided context.
Do not add a references section or cite sources inline.

Context:
{context}
"""

    def generate_response(
        self, query_text: str, context: QueryContext, param: QueryParam
    ) -> str:
        """Generate response based on context using LLM"""
        aggregated_context = context.aggregated_context.strip()

        if not aggregated_context:
            return self.GROUNDED_FALLBACK_RESPONSE

        system_prompt = self.generate_system_prompt.format(context=aggregated_context)

        try:
            return self.llm_service.call_llm(
                user_prompt=query_text,
                system_prompt=system_prompt,
                temperature=param.temperature,
            )
        except Exception:
            return self.GROUNDED_FALLBACK_RESPONSE

    def format_sources(
        self,
        documents: list[Document],
        entities: list[Entity],
        relations: list[Relation],
    ) -> list[QuerySource]:
        """Format sources for the response"""
        sources: list[QuerySource] = []

        for document in documents:
            sources.append(
                QuerySource(
                    type="document",
                    id=document.id,
                    content=(
                        document.content[:200] + "..."
                        if len(document.content) > 200
                        else document.content
                    ),
                    document_id=document.id,
                )
            )

        for entity in entities:
            profile_text = entity.profile_value or entity.description
            sources.append(
                QuerySource(
                    type="entity",
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=(
                        profile_text[:200] + "..."
                        if len(profile_text) > 200
                        else profile_text
                    ),
                )
            )

        for relation in relations:
            profile_text = relation.profile_value or relation.description
            sources.append(
                QuerySource(
                    type="relation",
                    id=relation.id,
                    source=relation.source_entity.name,
                    relation_type=relation.relation_type,
                    target=relation.target_entity.name,
                    description=(
                        profile_text[:200] + "..."
                        if len(profile_text) > 200
                        else profile_text
                    ),
                )
            )

        return sources
