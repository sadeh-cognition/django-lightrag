import hashlib
import time
from typing import Any, Dict, List

from django.conf import settings
from embed_gen.generator import generate_embeddings

from .entity_extraction import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
)
from .models import (
    Document,
)
from .storage import ChromaVectorStorage, LadybugGraphStorage
from .types import QueryParam, QueryResult
from .utils import Tokenizer
from .llm import LLMService
from .graph_builder import KnowledgeGraphBuilder
from .query_engine import QueryEngine


class LightRAGCore:
    """Core LightRAG functionality integrated with Django (Facade)"""

    def __init__(
        self,
        embedding_model: str,
        embedding_provider: str,
        embedding_base_url: str,
        llm_model: str,
        llm_temperature: float | None = None,
    ):
        # Load configuration from settings
        self.config = getattr(settings, "LIGHTRAG", {})

        self.tokenizer = Tokenizer()
        self.llm_service = LLMService(
            model=llm_model,
            temperature=(
                llm_temperature
                if llm_temperature is not None
                else self.config.get("LLM_TEMPERATURE", 0.0)
            ),
        )

        self.graph_storage = LadybugGraphStorage()
        self.vector_storage = ChromaVectorStorage()

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
            self.graph_builder.extract_and_persist(document)

            # 3. Generate and store embeddings
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
            self.query_engine.retrieve_knowledge_graph(query_text, param.top_k)
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

    def _get_query_embedding(self, query_text: str) -> List[float]:
        return self._get_embeddings([query_text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
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

    def _generate_document_embeddings(self, document: Document):
        embedding = self._get_embeddings([document.content])[0]
        self.vector_storage.add_embedding(
            "document",
            document.id,
            embedding,
            metadata={
                "content": document.content[:500],
                "document_id": document.id,
            },
        )

    def delete_document(self, document_id: str) -> bool:
        try:
            document = Document.objects.get(id=document_id)
            self.vector_storage.delete_embedding("document", document.id)
            document.delete()
            return True
        except Document.DoesNotExist:
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to delete document: {e}")

    def list_documents(self) -> List[Dict[str, Any]]:
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
