import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import tiktoken
from django.conf import settings
from embed_gen.generator import generate_embeddings

from .entity_extraction import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    extract_entities,
)
from .models import (
    Document,
    Entity,
    Relation,
    VectorEmbedding,
)
from .storage import ChromaVectorStorage, LadybugGraphStorage


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
    sources: list[dict[str, Any]]
    context: dict[str, Any]
    query_time: float
    tokens_used: int


class Tokenizer:
    """Tokenizer using tiktoken"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
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

    def __init__(self, llm_model: str = "", llm_temperature: float | None = None):
        self.tokenizer = Tokenizer()
        self.graph_storage = LadybugGraphStorage()
        self.vector_storage = ChromaVectorStorage()

        # Load configuration from settings
        self.config = getattr(
            settings, "LIGHTRAG", {}
        )  # TODO: Review the settings, document them
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
        self.llm_model = llm_model or self.config.get("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else self.config.get("LLM_TEMPERATURE", 0.0)
        )
        self.entity_extract_max_gleaning = self.config.get(
            "ENTITY_EXTRACT_MAX_GLEANING", 1
        )
        self.max_extract_input_tokens = self.config.get(
            "MAX_EXTRACT_INPUT_TOKENS", 12000
        )
        self.extraction_language = self.config.get(
            "EXTRACTION_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE
        )
        self.entity_types = self.config.get("ENTITY_TYPES", DEFAULT_ENTITY_TYPES)

    def _generate_id(self, content: str) -> str:
        """Generate a consistent ID from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def ingest_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        track_id: str = "",  # TODO this should be a `dict` of `metadata`
    ) -> str:
        """Ingest a document into the RAG system"""
        document_id = self._generate_id(content)
        metadata = metadata or {}

        # Create document
        document = Document.objects.create(
            id=document_id,
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

        except Exception:
            raise

    def _extract_knowledge_graph_from_document(self, document: Document):
        """Extract entities and relations from a document"""
        extracted_entities, extracted_relations = self._llm_extract_entities_relations(
            document
        )

        entity_objects = self._persist_entities(document, extracted_entities)
        self._persist_relations(document, extracted_relations, entity_objects)

    def _call_llm(
        self,
        user_prompt: str,
        _llm_model_func: Any,
        system_prompt: str | None = None,
        history_messages: List[Dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Adapter callable for `extract_entities`.

        `extract_entities` passes a second positional arg containing the llm func;
        we accept and ignore it for compatibility.
        """
        try:
            from django_llm_chat.chat import Chat, DuplicateSystemMessageError
            from django_llm_chat.models import Message
        except ImportError as exc:
            raise RuntimeError(
                "django-llm-chat package not installed. Add it to dependencies to "
                "enable LLM extraction."
            ) from exc

        chat = Chat.create()

        if system_prompt:
            try:
                chat.create_system_message(system_prompt)
            except DuplicateSystemMessageError:
                pass

        if history_messages:
            for msg in history_messages:
                role = (msg.get("role") or msg.get("type") or "user").lower()
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "system":
                    try:
                        chat.create_system_message(content)
                    except DuplicateSystemMessageError:
                        continue
                elif role == "assistant":
                    Message.create_llm_message(
                        chat=chat.chat_db_model,
                        text=content,
                        user=chat.llm_user,
                    )
                else:
                    chat.create_user_message(content)

        llm_msg, _, _ = chat.send_user_msg_to_llm(
            self.llm_model,
            user_prompt,
            include_chat_history=True,
            temperature=self.llm_temperature,
            max_tokens=max_tokens,
        )
        return llm_msg.text or ""

    def _relation_type_from_keywords(self, keywords: str) -> str:
        if not keywords:
            return "related_to"
        primary = keywords.split(",")[0].strip()
        return primary[:100] if primary else "related_to"

    def _llm_extract_entities_relations(
        self, document: Document
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        document_payload = {
            document.id: {
                "tokens": self.tokenizer.count_tokens(document.content),
                "content": document.content,
                "full_doc_id": document.id,
                "chunk_order_index": 0,
            }
        }

        global_config = {
            "llm_model_func": self._call_llm,
            "entity_extract_max_gleaning": self.entity_extract_max_gleaning,
            "addon_params": {
                "language": self.extraction_language,
                "entity_types": self.entity_types,
            },
            "tokenizer": self.tokenizer,
            "max_extract_input_tokens": self.max_extract_input_tokens,
        }

        document_results = extract_entities(document_payload, global_config)

        entity_by_name: Dict[str, Dict[str, Any]] = {}
        relation_by_key: Dict[str, Dict[str, Any]] = {}

        for maybe_nodes, maybe_edges in document_results:
            for entity_name, entity_list in maybe_nodes.items():
                if not entity_list:
                    continue
                best = max(
                    entity_list, key=lambda item: len(item.get("description", "") or "")
                )
                existing = entity_by_name.get(entity_name)
                if existing is None or len(best.get("description", "")) > len(
                    existing.get("description", "")
                ):
                    entity_by_name[entity_name] = best

            for (src_name, tgt_name), relation_list in maybe_edges.items():
                if not relation_list:
                    continue
                best = max(
                    relation_list,
                    key=lambda item: len(item.get("description", "") or ""),
                )
                relation_type = self._relation_type_from_keywords(
                    best.get("keywords", "")
                )
                sorted_key = "::".join(sorted([src_name, tgt_name]) + [relation_type])
                existing = relation_by_key.get(sorted_key)
                if existing is None or len(best.get("description", "")) > len(
                    existing.get("description", "")
                ):
                    relation_by_key[sorted_key] = {
                        **best,
                        "relation_type": relation_type,
                    }

        return entity_by_name, relation_by_key

    def _persist_entities(
        self, document: Document, entity_by_name: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Entity]:
        entity_objects: Dict[str, Entity] = {}
        for entity_name, entity_data in entity_by_name.items():
            entity_type = entity_data.get("entity_type", "other") or "other"
            entity_id = self._generate_id(f"entity:{entity_name}:{entity_type}")
            defaults = {
                "name": entity_name,
                "entity_type": entity_type,
                "description": entity_data.get("description", ""),
                "source_ids": [document.id],
                "metadata": {
                    "source_id": entity_data.get("source_id"),
                    "timestamp": entity_data.get("timestamp"),
                },
            }

            entity, created = Entity.objects.get_or_create(
                id=entity_id, defaults=defaults
            )

            if not created:
                updated = False
                if entity_data.get("description") and len(
                    entity_data["description"]
                ) > len(entity.description or ""):
                    entity.description = entity_data["description"]
                    updated = True
                if document.id not in entity.source_ids:
                    entity.source_ids.append(document.id)
                    updated = True
                if updated:
                    entity.save()

            if created:
                self.graph_storage.add_entity(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "metadata": entity.metadata,
                    }
                )

            entity_objects[entity_name] = entity

        return entity_objects

    def _get_or_create_placeholder_entity(
        self, document: Document, entity_objects: Dict[str, Entity], entity_name: str
    ) -> Entity:
        if entity_name in entity_objects:
            return entity_objects[entity_name]

        entity_type = "other"
        entity_id = self._generate_id(f"entity:{entity_name}:{entity_type}")
        entity, created = Entity.objects.get_or_create(
            id=entity_id,
            defaults={
                "name": entity_name,
                "entity_type": entity_type,
                "description": "",
                "source_ids": [document.id],
                "metadata": {"auto_created": True},
            },
        )
        if created:
            self.graph_storage.add_entity(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "metadata": entity.metadata,
                }
            )
        entity_objects[entity_name] = entity
        return entity

    def _persist_relations(
        self,
        document: Document,
        relation_by_key: Dict[str, Dict[str, Any]],
        entity_objects: Dict[str, Entity],
    ) -> None:
        for relation_data in relation_by_key.values():
            src_name = relation_data.get("src_id")
            tgt_name = relation_data.get("tgt_id")
            if not src_name or not tgt_name:
                continue

            source_entity = self._get_or_create_placeholder_entity(
                document, entity_objects, src_name
            )
            target_entity = self._get_or_create_placeholder_entity(
                document, entity_objects, tgt_name
            )

            relation_type = relation_data.get(
                "relation_type"
            ) or self._relation_type_from_keywords(relation_data.get("keywords", ""))
            relation_type = relation_type[:100] if relation_type else "related_to"

            relation_id = self._generate_id(
                f"relation:{min(source_entity.id, target_entity.id)}:"
                f"{max(source_entity.id, target_entity.id)}:{relation_type}"
            )

            defaults = {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relation_type": relation_type,
                "description": relation_data.get("description", ""),
                "source_ids": [document.id],
                "weight": relation_data.get("weight", 1.0),
                "metadata": {
                    "keywords": relation_data.get("keywords", ""),
                    "source_id": relation_data.get("source_id"),
                    "timestamp": relation_data.get("timestamp"),
                },
            }

            relation, created = Relation.objects.get_or_create(
                id=relation_id, defaults=defaults
            )

            if not created:
                updated = False
                if relation_data.get("description") and len(
                    relation_data["description"]
                ) > len(relation.description or ""):
                    relation.description = relation_data["description"]
                    updated = True
                if document.id not in relation.source_ids:
                    relation.source_ids.append(document.id)
                    updated = True
                if updated:
                    relation.save()

            if created:
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
