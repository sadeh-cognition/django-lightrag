import json
import time
from typing import Any, Dict, List, Tuple
from .models import Document, Entity, Relation
from .storage import ChromaVectorStorage
from .llm import LLMService
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
        start_time = time.time()

        # Generate query embedding (this might need to be passed in or handled by a separate service if we want to be super clean)
        # For now, let's assume we can get it or the caller provides it.
        # However, to keep it simple and consistent with core.py, we'll keep the retrieval logic here.
        # Actually core.py does the embedding generation.

        # We'll need a way to get embeddings. Let's assume the caller provides the embedding for now,
        # or we pass the embedding function to the engine.

        # To match the original implementation's flow:
        # 1. Get embedding (handled by core for now, but engine needs it)
        # 2. Retrieve docs
        # 3. Retrieve graph
        # 4. Build context
        # 5. Generate response

        # I'll update the signature to accept the query_embedding if it's cleaner,
        # but let's stick to what core.py expects: core calls retrieve_documents etc.
        pass

    def retrieve_documents(
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

    def retrieve_knowledge_graph(
        self, query_text: str, top_k: int
    ) -> Tuple[List[Entity], List[Relation]]:
        """Retrieve relevant entities and relations"""
        # Placeholder implementation - in practice, use graph traversal or entity matching
        entities = list(Entity.objects.all()[:top_k])
        relations = list(Relation.objects.all()[:top_k])

        return entities, relations

    def build_context(
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

    def generate_response(
        self, query_text: str, context: Dict[str, Any], param: QueryParam
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
                    "content": (
                        document.content[:200] + "..."
                        if len(document.content) > 200
                        else document.content
                    ),
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
                    "description": (
                        entity.description[:200] + "..."
                        if len(entity.description) > 200
                        else entity.description
                    ),
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
                    "description": (
                        relation.description[:200] + "..."
                        if len(relation.description) > 200
                        else relation.description
                    ),
                }
            )

        return sources
