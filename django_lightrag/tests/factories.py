from django_lightrag.core import LightRAGCore
from django_lightrag.models import Document, Entity, Relation
from django_lightrag.utils import Tokenizer


class EndpointVectorStorage:
    def __init__(self):
        self.records = {
            "document": {},
            "entity": {},
            "relation": {},
        }

    def upsert_embedding(self, vector_type, content_id, embedding, metadata=None):
        self.records[vector_type][content_id] = {
            "embedding": embedding,
            "metadata": metadata or {},
        }
        return content_id

    def search_similar(self, vector_type, query_embedding, top_k=10, where=None):
        scored = []
        for content_id, record in self.records[vector_type].items():
            distance = sum(
                (query_value - value) ** 2
                for query_value, value in zip(
                    query_embedding, record["embedding"], strict=True
                )
            )
            scored.append(
                {
                    "id": content_id,
                    "score": distance,
                    "metadata": record["metadata"],
                }
            )
        scored.sort(key=lambda item: item["score"])
        return scored[:top_k]

    def close(self):
        return None


class EndpointGraphStorage:
    def close(self):
        return None


class EndpointCore(LightRAGCore):
    """Core used for endpoint tests that need to avoid real external calls."""

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            embeddings.append(
                [
                    1.0 if "policy engine" in lowered else 0.0,
                    1.0 if "governance" in lowered else 0.0,
                    1.0 if "document" in lowered else 0.0,
                ]
            )
        return embeddings


def make_test_core(
    *,
    embedding_model: str,
    embedding_provider: str,
    embedding_base_url: str,
    llm_model: str,
):
    vector_storage = EndpointVectorStorage()
    core = EndpointCore(
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        llm_model=llm_model,
        llm_service=EndpointLLMService(),
        graph_storage=EndpointGraphStorage(),
        vector_storage=vector_storage,
        tokenizer=Tokenizer(),
    )

    document = Document.objects.get(id="endpoint-doc")
    entity = Entity.objects.get(id="endpoint-entity")
    relation = Relation.objects.get(id="endpoint-relation")

    vector_storage.upsert_embedding(
        "document",
        document.id,
        core._get_embeddings([document.content])[0],
        metadata={"document_id": document.id},
    )
    vector_storage.upsert_embedding(
        "entity",
        entity.id,
        core._get_embeddings(["Policy Engine"])[0],
        metadata={"entity_id": entity.id, "profile_key": entity.profile_key},
    )
    vector_storage.upsert_embedding(
        "relation",
        relation.id,
        core._get_embeddings(["governance"])[0],
        metadata={"relation_id": relation.id, "profile_key": relation.profile_key},
    )
    return core
