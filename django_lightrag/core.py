import hashlib
import time
from typing import Any

import requests
from django.db import transaction
from embed_gen.generator import generate_embeddings

from .config import LightRAGConfig, get_lightrag_settings
from .deduplication import DeduplicationResult, GraphDeduplicationService
from .entity_extraction import DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE
from .graph_builder import KnowledgeGraphBuilder, KnowledgeGraphBuilderConfig
from .llm import LLMService
from .models import Document, Entity, Relation
from .profiling import ProfilingConfig, ProfilingService
from .query_engine import QueryEngine
from .query_keywords import QueryKeywordConfig, QueryKeywordExtractor, QueryKeywords
from .storage import ChromaVectorStorage, LadybugGraphStorage
from .types import (
    DocumentVectorHit,
    DocumentVectorMatches,
    EntityVectorHit,
    EntityVectorMatches,
    GraphTraversalCaps,
    GraphTraversalResult,
    QueryParam,
    QueryResult,
    RelationVectorHit,
    RelationVectorMatches,
    VectorMatchingResult,
)
from .utils import Tokenizer


class LightRAGCore:
    """Core LightRAG functionality integrated with Django (Facade)"""

    def __init__(
        self,
        embedding_model: str,
        embedding_provider: str,
        embedding_base_url: str,
        llm_model: str,
        llm_temperature: float | None = None,
        *,
        llm_service: LLMService | None = None,
        graph_storage: LadybugGraphStorage | None = None,
        vector_storage: ChromaVectorStorage | None = None,
        tokenizer: Tokenizer | None = None,
        query_keyword_extractor: QueryKeywordExtractor | None = None,
    ):
        self.config: LightRAGConfig = get_lightrag_settings()

        self.tokenizer = tokenizer or Tokenizer()
        self.llm_service = llm_service or LLMService(
            model=llm_model,
            temperature=(
                llm_temperature
                if llm_temperature is not None
                else self.config.llm_temperature
            ),
        )

        self.graph_storage = graph_storage or LadybugGraphStorage()
        self.vector_storage = vector_storage or ChromaVectorStorage()
        self.profiling_service = ProfilingService(
            llm_service=self.llm_service,
            config=ProfilingConfig(profile_max_tokens=self.config.profile_max_tokens),
        )
        self.query_keyword_extractor = query_keyword_extractor or QueryKeywordExtractor(
            llm_service=self.llm_service,
            config=QueryKeywordConfig(
                query_keyword_max_tokens=self.config.query_keyword_max_tokens
            ),
        )
        self.deduplication_service = GraphDeduplicationService(
            graph_storage=self.graph_storage,
            vector_storage=self.vector_storage,
        )
        self.graph_builder = KnowledgeGraphBuilder(
            llm_service=self.llm_service,
            tokenizer=self.tokenizer,
            graph_storage=self.graph_storage,
            config=KnowledgeGraphBuilderConfig(
                entity_extract_max_gleaning=self.config.entity_extract_max_gleaning,
                extraction_language=(
                    self.config.extraction_language or DEFAULT_SUMMARY_LANGUAGE
                ),
                entity_types=self.config.entity_types or list(DEFAULT_ENTITY_TYPES),
                max_extract_input_tokens=self.config.max_extract_input_tokens,
            ),
        )
        self.query_engine = QueryEngine(
            llm_service=self.llm_service,
            vector_storage=self.vector_storage,
            tokenizer=self.tokenizer,
        )

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.top_k = self.config.top_k
        self.max_total_tokens = self.config.max_total_tokens

    def _generate_id(self, content: str) -> str:
        """Generate a consistent ID from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def ingest_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        track_id: str = "",
    ) -> str:
        """Orchestrate document ingestion"""
        document_id = self._generate_id(content)
        metadata = metadata or {}

        try:
            with transaction.atomic():
                # 1. Create document record
                document = Document.objects.create(
                    id=document_id,
                    content=content,
                    metadata=metadata,
                    track_id=track_id,
                )

                # 2. Extract KG and persist
                entities, relations = self.graph_builder.extract_and_persist(document)

                # 3. Deduplicate touched records before profiling and vector upserts
                dedup_result = self._deduplicate_graph_records(
                    include_entities=True,
                    include_relations=True,
                    entity_ids=[entity.id for entity in entities],
                    relation_ids=[relation.id for relation in relations],
                    profile_survivors=False,
                )

                # 4. Generate entity and relation profiles plus vector entries
                self._profile_knowledge_graph(
                    dedup_result.surviving_entities or entities,
                    dedup_result.surviving_relations or relations,
                )

                # 5. Generate and store document embeddings
                self._generate_document_embeddings(document)

            return document_id
        except Exception:
            Document.objects.filter(id=document_id).delete()
            raise

    def query(self, query_text: str, param: QueryParam = None) -> QueryResult:
        """Query the RAG system using the QueryEngine"""
        if param is None:
            param = QueryParam()
        else:
            param.low_level_keywords = self._normalize_keyword_values(
                param.low_level_keywords
            )
            param.high_level_keywords = self._normalize_keyword_values(
                param.high_level_keywords
            )

        start_time = time.time()
        extracted_keywords = self._resolve_query_keywords(query_text, param)
        param.low_level_keywords = extracted_keywords.low_level_keywords
        param.high_level_keywords = extracted_keywords.high_level_keywords

        # 1. Prepare and batch embed query texts
        document_branch = {
            "query_text": query_text,
            "query_source": "raw",
        }
        entity_branch = {
            "query_text": ", ".join(param.low_level_keywords)
            if param.low_level_keywords
            else query_text,
            "query_source": "keyword" if param.low_level_keywords else "fallback",
        }
        relation_branch = {
            "query_text": ", ".join(param.high_level_keywords)
            if param.high_level_keywords
            else query_text,
            "query_source": "keyword" if param.high_level_keywords else "fallback",
        }

        embeddings = self._get_embeddings(
            [
                document_branch["query_text"],
                entity_branch["query_text"],
                relation_branch["query_text"],
            ]
        )

        document_branch["embedding"] = embeddings[0]
        entity_branch["embedding"] = embeddings[1]
        relation_branch["embedding"] = embeddings[2]

        # 2. Vector Search (Raw Vector Matches)
        doc_vectors = self.query_engine.search_document_vectors(
            document_branch["embedding"], param.top_k
        )
        ent_vectors, rel_vectors = self._retrieve_knowledge_graph_vectors(
            entity_branch["embedding"], relation_branch["embedding"], param
        )

        # 3. Hydrate into ORM Models
        relevant_documents = self.query_engine.hydrate_documents(
            doc_vectors, param.top_k
        )
        relevant_entities = self.query_engine.hydrate_entities(ent_vectors)
        relevant_relations = self.query_engine.hydrate_relations(rel_vectors)

        # ONE-HOP GRAPH TRAVERSAL EXPANSION
        graph_traversal_debug = None
        if getattr(param, "one_hop_enabled", False):
            expanded_entities, expanded_relations = (
                self.query_engine.expand_one_hop_neighborhood(
                    relevant_entities,
                    relevant_relations,
                    getattr(param, "one_hop_max_entities", 10),
                    getattr(param, "one_hop_max_relations", 10),
                )
            )

            # Stable merge: vector-ranked seeds first, append graph-only neighbors after them, deduping by PK
            merged_entities = list(relevant_entities)
            seen_entity_ids = {e.id for e in relevant_entities}
            for e in expanded_entities:
                if e.id not in seen_entity_ids:
                    seen_entity_ids.add(e.id)
                    merged_entities.append(e)

            merged_relations = list(relevant_relations)
            seen_relation_ids = {r.id for r in relevant_relations}
            for r in expanded_relations:
                if r.id not in seen_relation_ids:
                    seen_relation_ids.add(r.id)
                    merged_relations.append(r)

            # Compute seed IDs correctly for debug output
            seed_entity_ids = {e.id for e in relevant_entities}
            for rel in relevant_relations:
                seed_entity_ids.add(rel.source_entity_id)
                seed_entity_ids.add(rel.target_entity_id)

            graph_traversal_debug = GraphTraversalResult(
                seed_entity_ids=list(seed_entity_ids),
                added_entity_ids=[e.id for e in expanded_entities],
                added_relation_ids=[r.id for r in expanded_relations],
                caps_applied=GraphTraversalCaps(
                    max_entities=getattr(param, "one_hop_max_entities", 10),
                    max_relations=getattr(param, "one_hop_max_relations", 10),
                ),
            )

            relevant_entities = merged_entities
            relevant_relations = merged_relations

        # 4. Build context and generate response
        context = self.query_engine.build_context(
            relevant_documents, relevant_entities, relevant_relations, param
        )

        # 5. Inject Vector Matching Debug Context
        context.vector_matching = VectorMatchingResult(
            documents=DocumentVectorMatches(
                query_text=document_branch["query_text"],
                query_source=document_branch["query_source"],
                hits=[
                    DocumentVectorHit(
                        id=hit["id"],
                        score=hit["score"],
                        rank=rank + 1,
                    )
                    for rank, hit in enumerate(doc_vectors)
                ],
                selected_ids=[d.document_id for d in context.documents],
            ),
            entities=EntityVectorMatches(
                query_text=entity_branch["query_text"],
                query_source=entity_branch["query_source"],
                hits=[
                    EntityVectorHit(
                        id=hit["metadata"].get("entity_id", hit["id"]),
                        name=hit["metadata"].get("name", "Unknown Entity"),
                        profile_key=hit["metadata"].get("profile_key", ""),
                        score=hit["score"],
                        rank=rank + 1,
                    )
                    for rank, hit in enumerate(ent_vectors)
                ],
                selected_ids=[e.id for e in relevant_entities[: len(context.entities)]],
            ),
            relations=RelationVectorMatches(
                query_text=relation_branch["query_text"],
                query_source=relation_branch["query_source"],
                hits=[
                    RelationVectorHit(
                        id=hit["metadata"].get("relation_id", hit["id"]),
                        source=hit["metadata"].get("source_entity_id", ""),
                        relation_type=hit["metadata"].get("relation_type", ""),
                        target=hit["metadata"].get("target_entity_id", ""),
                        profile_key=hit["metadata"].get("profile_key", ""),
                        score=hit["score"],
                        rank=rank + 1,
                    )
                    for rank, hit in enumerate(rel_vectors)
                ],
                selected_ids=[
                    r.id for r in relevant_relations[: len(context.relations)]
                ],
            ),
        )

        if graph_traversal_debug:
            context.graph_traversal = graph_traversal_debug

        response = context.aggregated_context

        query_time = time.time() - start_time

        return QueryResult(
            response=response,
            sources=self.query_engine.format_sources(
                relevant_documents, relevant_entities, relevant_relations
            ),
            context=context,
            query_time=query_time,
            tokens_used=context.total_tokens,
        )

    def _retrieve_knowledge_graph_vectors(
        self,
        entity_query_embedding: list[float],
        relation_query_embedding: list[float],
        param: QueryParam,
    ) -> tuple[list[dict], list[dict]]:
        mode = param.mode.lower()
        if mode == "local":
            return (
                self.query_engine.search_entity_vectors(
                    entity_query_embedding, param.top_k
                ),
                [],
            )
        if mode == "global":
            return (
                [],
                self.query_engine.search_relation_vectors(
                    relation_query_embedding, param.top_k
                ),
            )

        entities = self.query_engine.search_entity_vectors(
            entity_query_embedding, param.top_k
        )
        relations = self.query_engine.search_relation_vectors(
            relation_query_embedding, param.top_k
        )

        # Note: deduplication/merging is handled during ORM hydration
        return entities, relations

    def _resolve_query_keywords(
        self, query_text: str, param: QueryParam
    ) -> QueryKeywords:
        provided_low = self._normalize_keyword_values(param.low_level_keywords)
        provided_high = self._normalize_keyword_values(param.high_level_keywords)
        if provided_low and provided_high:
            return QueryKeywords(
                low_level_keywords=provided_low,
                high_level_keywords=provided_high,
            )

        extracted = QueryKeywords(low_level_keywords=[], high_level_keywords=[])
        try:
            extracted = self.query_keyword_extractor.extract(query_text)
        except Exception:
            extracted = QueryKeywords(low_level_keywords=[], high_level_keywords=[])

        low_level_keywords = provided_low or extracted.low_level_keywords
        high_level_keywords = provided_high or extracted.high_level_keywords

        return QueryKeywords(
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
        )

    def _normalize_keyword_values(self, values: list[str] | None) -> list[str]:
        if not values:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            keyword = " ".join(str(value).split()).strip()
            if not keyword:
                continue
            marker = keyword.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            normalized.append(keyword)
        return normalized

    def _keyword_text_or_query(self, keywords: list[str], query_text: str) -> str:
        return ", ".join(keywords) if keywords else query_text

    def _get_query_embedding(self, query_text: str) -> list[float]:
        return self._get_embeddings([query_text])[0]

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("No texts provided for embedding generation.")

        try:
            if generate_embeddings is not None:
                return generate_embeddings(
                    texts=texts,
                    model_name=self.embedding_model,
                    provider=self.embedding_provider,
                    base_url=self.embedding_base_url,
                )
            return self._get_embeddings_via_http(texts)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    def _get_embeddings_via_http(self, texts: list[str]) -> list[list[float]]:
        endpoint = f"{self.embedding_base_url.rstrip('/')}/embeddings"
        response = requests.post(
            endpoint,
            json={"model": self.embedding_model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response missing 'data' list.")

        try:
            ordered_rows = sorted(data, key=lambda item: item.get("index", 0))
            return [row["embedding"] for row in ordered_rows]
        except Exception as exc:
            raise RuntimeError(
                "Embedding response missing expected 'embedding' values."
            ) from exc

    def _generate_document_embeddings(self, document: Document):
        embedding = self._get_embeddings([document.content])[0]
        self.vector_storage.upsert_embedding(
            "document",
            document.id,
            embedding,
            metadata={
                "content": document.content[:500],
                "document_id": document.id,
            },
        )

    def _profile_knowledge_graph(
        self, entities: list[Entity], relations: list[Relation]
    ) -> None:
        unique_entities = list({entity.id: entity for entity in entities}.values())
        unique_relations = list(
            {relation.id: relation for relation in relations}.values()
        )

        for entity in unique_entities:
            self.profiling_service.profile_entity(entity)

        for relation in unique_relations:
            self.profiling_service.profile_relation(relation)

        self._upsert_entity_embeddings(unique_entities)
        self._upsert_relation_embeddings(unique_relations)

    def _upsert_entity_embeddings(self, entities: list[Entity]) -> None:
        profiled_entities = [
            entity
            for entity in entities
            if (entity.profile_key or "").strip()
            and (entity.profile_value or "").strip()
        ]
        if not profiled_entities:
            return

        embedding_inputs = [
            f"{entity.profile_key}\n{entity.name}\n{entity.profile_value}"
            for entity in profiled_entities
        ]
        embeddings = self._get_embeddings(embedding_inputs)

        for entity, embedding, embedding_input in zip(
            profiled_entities, embeddings, embedding_inputs, strict=True
        ):
            self.vector_storage.upsert_embedding(
                "entity",
                entity.id,
                embedding,
                metadata={
                    "entity_id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "profile_key": entity.profile_key,
                    "profile_value": entity.profile_value,
                    "content": embedding_input,
                },
            )

    def _upsert_relation_embeddings(self, relations: list[Relation]) -> None:
        profiled_relations = [
            relation
            for relation in relations
            if (relation.profile_key or "").strip()
            and (relation.profile_value or "").strip()
        ]
        if not profiled_relations:
            return

        embedding_inputs = [
            (
                f"{relation.profile_key}\n"
                f"{relation.source_entity.name} -> {relation.relation_type} -> "
                f"{relation.target_entity.name}\n"
                f"{relation.profile_value}"
            )
            for relation in profiled_relations
        ]
        embeddings = self._get_embeddings(embedding_inputs)

        for relation, embedding, embedding_input in zip(
            profiled_relations, embeddings, embedding_inputs, strict=True
        ):
            self.vector_storage.upsert_embedding(
                "relation",
                relation.id,
                embedding,
                metadata={
                    "relation_id": relation.id,
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "relation_type": relation.relation_type,
                    "profile_key": relation.profile_key,
                    "profile_value": relation.profile_value,
                    "content": embedding_input,
                },
            )

    def backfill_profiles(
        self, *, include_entities: bool = True, include_relations: bool = True
    ) -> dict[str, int]:
        entities = list(Entity.objects.all()) if include_entities else []
        relations = (
            list(Relation.objects.select_related("source_entity", "target_entity"))
            if include_relations
            else []
        )

        self._profile_knowledge_graph(entities, relations)
        return {
            "entities": len(entities),
            "relations": len(relations),
        }

    def deduplicate_graph(
        self,
        *,
        include_entities: bool = True,
        include_relations: bool = True,
        entity_ids: list[str] | None = None,
        relation_ids: list[str] | None = None,
    ) -> dict[str, int]:
        result = self._deduplicate_graph_records(
            include_entities=include_entities,
            include_relations=include_relations,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            profile_survivors=True,
        )
        return result.as_counts()

    def _deduplicate_graph_records(
        self,
        *,
        include_entities: bool,
        include_relations: bool,
        entity_ids: list[str] | None,
        relation_ids: list[str] | None,
        profile_survivors: bool,
    ) -> DeduplicationResult:
        result = self.deduplication_service.deduplicate(
            include_entities=include_entities,
            include_relations=include_relations,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
        )
        if profile_survivors and (
            result.surviving_entities or result.surviving_relations
        ):
            self._profile_knowledge_graph(
                result.surviving_entities,
                result.surviving_relations,
            )
        return result

    def delete_document(self, document_id: str) -> bool:
        try:
            document = Document.objects.get(id=document_id)
            self.vector_storage.delete_embedding("document", document.id)
            document.delete()
            return True
        except Document.DoesNotExist:
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to delete document: {e}") from e

    def list_documents(self) -> list[dict[str, Any]]:
        documents = list(Document.objects.all())
        result = []
        for doc in documents:
            result.append(
                {
                    "id": doc.id,
                    "track_id": doc.track_id,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )
        return result

    def close(self):
        self.graph_storage.close()
        self.vector_storage.close()
