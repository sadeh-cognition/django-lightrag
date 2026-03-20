import hashlib
import time
from typing import Any

from django.conf import settings

try:
    from embed_gen.generator import generate_embeddings
except ImportError:
    generate_embeddings = None

from .entity_extraction import DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE
from .graph_builder import KnowledgeGraphBuilder
from .llm import LLMService
from .models import Document, Entity, Relation
from .profiling import ProfilingService
from .query_engine import QueryEngine
from .storage import ChromaVectorStorage, LadybugGraphStorage
from .types import QueryParam, QueryResult
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
    ):
        # Load configuration from settings
        self.config = getattr(settings, "LIGHTRAG", {})

        self.tokenizer = tokenizer or Tokenizer()
        self.llm_service = llm_service or LLMService(
            model=llm_model,
            temperature=(
                llm_temperature
                if llm_temperature is not None
                else self.config.get("LLM_TEMPERATURE", 0.0)
            ),
        )

        self.graph_storage = graph_storage or LadybugGraphStorage()
        self.vector_storage = vector_storage or ChromaVectorStorage()
        self.profiling_service = ProfilingService(
            llm_service=self.llm_service,
            config={"PROFILE_MAX_TOKENS": self.config.get("PROFILE_MAX_TOKENS", 400)},
        )

        # Initialize specialized components
        self.graph_builder = KnowledgeGraphBuilder(
            llm_service=self.llm_service,
            tokenizer=self.tokenizer,
            graph_storage=self.graph_storage,
            config={
                "ENTITY_EXTRACT_MAX_GLEANING": self.config.get(
                    "ENTITY_EXTRACT_MAX_GLEANING", 1
                ),
                "EXTRACTION_LANGUAGE": self.config.get(
                    "EXTRACTION_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE
                ),
                "ENTITY_TYPES": self.config.get("ENTITY_TYPES", DEFAULT_ENTITY_TYPES),
                "MAX_EXTRACT_INPUT_TOKENS": self.config.get(
                    "MAX_EXTRACT_INPUT_TOKENS", 12000
                ),
            },
        )

        self.query_engine = QueryEngine(
            llm_service=self.llm_service,
            vector_storage=self.vector_storage,
            tokenizer=self.tokenizer,
        )

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url

        # Keep some config for backward compatibility or internal use
        self.top_k = self.config.get("TOP_K", 10)
        self.max_total_tokens = self.config.get("MAX_TOTAL_TOKENS", 12000)

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

        # 1. Create document record
        document = Document.objects.create(
            id=document_id,
            content=content,
            metadata=metadata,
            track_id=track_id,
        )

        try:
            # 2. Extract KG and persist
            entities, relations = self.graph_builder.extract_and_persist(document)

            # 3. Generate entity and relation profiles plus vector entries
            self._profile_knowledge_graph(entities, relations)

            # 4. Generate and store document embeddings
            self._generate_document_embeddings(document)

            return document_id
        except Exception:
            # Clean up if needed (original implementation didn't, but we could)
            raise

    def query(self, query_text: str, param: QueryParam = None) -> QueryResult:
        """Query the RAG system using the QueryEngine"""
        if param is None:
            param = QueryParam()

        start_time = time.time()

        # 1. Generate query embedding
        query_embedding = self._get_query_embedding(query_text)

        # 2. Retrieval using QueryEngine
        relevant_documents = self.query_engine.retrieve_documents(
            query_embedding, param.top_k
        )
        relevant_entities, relevant_relations = (
            self.query_engine.retrieve_knowledge_graph(query_embedding, param.top_k)
        )

        # 3. Build context and generate response
        context = self.query_engine.build_context(
            relevant_documents, relevant_entities, relevant_relations, param
        )
        response = self.query_engine.generate_response(query_text, context, param)

        query_time = time.time() - start_time

        return QueryResult(
            response=response,
            sources=self.query_engine.format_sources(
                relevant_documents, relevant_entities, relevant_relations
            ),
            context=context,
            query_time=query_time,
            tokens_used=self.tokenizer.count_tokens(response),
        )

    def _get_query_embedding(self, query_text: str) -> list[float]:
        return self._get_embeddings([query_text])[0]

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("No texts provided for embedding generation.")
        if generate_embeddings is None:
            raise RuntimeError(
                "embed_gen is not installed. Install embed_gen to generate embeddings."
            )

        try:
            return generate_embeddings(
                texts=texts,
                model_name=self.embedding_model,
                provider=self.embedding_provider,
                base_url=self.embedding_base_url,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

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
                    "title": doc.title,
                    "track_id": doc.track_id,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )
        return result

    def close(self):
        self.graph_storage.close()
        self.vector_storage.close()
