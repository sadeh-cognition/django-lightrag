import json

import pytest
from django.test import override_settings
from ninja.testing import TestClient

from django_lightrag.core import LightRAGCore
from django_lightrag.models import Document, Entity, Relation
from django_lightrag.types import QueryParam
from django_lightrag.utils import Tokenizer
from django_lightrag.views import router


class QueryLLMService:
    def __init__(self, response: str):
        self.response = response

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        max_tokens: int | None = None,
    ) -> str:
        return self.response


class QueryGraphStorage:
    def close(self):
        return None


class QueryVectorStorage:
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


class DeterministicQueryCore(LightRAGCore):
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


def build_core(llm_response: str) -> DeterministicQueryCore:
    vector_storage = QueryVectorStorage()
    core = DeterministicQueryCore(
        embedding_model="test-embedding",
        embedding_provider="test",
        embedding_base_url="http://test.invalid",
        llm_model="test-llm",
        llm_service=QueryLLMService(llm_response),
        graph_storage=QueryGraphStorage(),
        vector_storage=vector_storage,
        tokenizer=Tokenizer(),
    )

    doc = Document.objects.create(
        id="query-doc",
        content="This document explains how Policy Engine decisions are recorded.",
    )
    entity = Entity.objects.create(
        id="query-entity",
        name="Policy Engine",
        entity_type="concept",
        description="Entity fallback description.",
        profile_key="Policy Engine",
        profile_value="Entity profile for the Policy Engine.",
        source_ids=[doc.id],
        metadata={},
    )
    target = Entity.objects.create(
        id="query-target",
        name="Control Plane",
        entity_type="concept",
        description="Target description.",
        profile_key="Control Plane",
        profile_value="Target profile.",
        source_ids=[doc.id],
        metadata={},
    )
    relation = Relation.objects.create(
        id="query-relation",
        source_entity=entity,
        target_entity=target,
        relation_type="governs",
        description="Relation fallback description.",
        profile_key="governance",
        profile_value="Relation profile for governance workflows.",
        source_ids=[doc.id],
        metadata={},
    )

    vector_storage.upsert_embedding(
        "document",
        doc.id,
        core._get_embeddings([doc.content])[0],
        metadata={"document_id": doc.id},
    )
    vector_storage.upsert_embedding(
        "entity",
        entity.id,
        core._get_embeddings(["Policy Engine"])[0],
        metadata={"entity_id": entity.id},
    )
    vector_storage.upsert_embedding(
        "relation",
        relation.id,
        core._get_embeddings(["governance"])[0],
        metadata={"relation_id": relation.id},
    )
    return core


@pytest.mark.django_db
def test_query_uses_split_keywords_for_retrieval_and_context():
    core = build_core(
        json.dumps(
            {
                "low_level_keywords": ["Policy Engine"],
                "high_level_keywords": ["governance"],
            }
        )
    )

    result = core.query("How are decisions enforced?", QueryParam(mode="hybrid"))

    assert [source["type"] for source in result.sources] == [
        "document",
        "entity",
        "relation",
    ]
    assert result.context["query_keywords"] == {
        "low_level_keywords": ["Policy Engine"],
        "high_level_keywords": ["governance"],
    }
    assert result.context["entities"][0]["name"] == "Policy Engine"
    assert result.context["relations"][0]["relation_type"] == "governs"


@pytest.mark.django_db
def test_query_mode_controls_knowledge_graph_retrieval():
    core = build_core(
        json.dumps(
            {
                "low_level_keywords": ["Policy Engine"],
                "high_level_keywords": ["governance"],
            }
        )
    )

    local_result = core.query("How are decisions enforced?", QueryParam(mode="local"))
    global_result = core.query("How are decisions enforced?", QueryParam(mode="global"))

    assert [item["name"] for item in local_result.context["entities"]] == [
        "Policy Engine"
    ]
    assert local_result.context["relations"] == []
    assert global_result.context["entities"] == []
    assert [item["relation_type"] for item in global_result.context["relations"]] == [
        "governs"
    ]


@pytest.mark.django_db
def test_query_falls_back_to_raw_query_when_keyword_extraction_fails():
    core = build_core("not valid json")

    result = core.query("Policy Engine governance document", QueryParam(mode="hybrid"))

    assert result.context["query_keywords"] == {
        "low_level_keywords": [],
        "high_level_keywords": [],
    }
    assert [item["name"] for item in result.context["entities"]] == ["Policy Engine"]
    assert [item["relation_type"] for item in result.context["relations"]] == [
        "governs"
    ]


@pytest.mark.django_db
@override_settings(
    LIGHTRAG={
        "EMBEDDING_PROVIDER": "test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_BASE_URL": "http://test.invalid",
        "LLM_MODEL": "test-llm",
        "LLM_TEMPERATURE": 0.0,
        "PROFILE_MAX_TOKENS": 200,
        "CORE_FACTORY": "django_lightrag.tests.factories.make_test_core",
    }
)
def test_query_endpoint_returns_extracted_keywords_in_context():
    doc = Document.objects.create(
        id="endpoint-doc",
        content="A document about Policy Engine governance.",
    )
    entity = Entity.objects.create(
        id="endpoint-entity",
        name="Policy Engine",
        entity_type="concept",
        description="Entity fallback description.",
        profile_key="Policy Engine",
        profile_value="Entity profile for the Policy Engine.",
        source_ids=[doc.id],
        metadata={},
    )
    target = Entity.objects.create(
        id="endpoint-target",
        name="Control Plane",
        entity_type="concept",
        description="Target description.",
        profile_key="Control Plane",
        profile_value="Target profile.",
        source_ids=[doc.id],
        metadata={},
    )
    Relation.objects.create(
        id="endpoint-relation",
        source_entity=entity,
        target_entity=target,
        relation_type="governs",
        description="Relation fallback description.",
        profile_key="governance",
        profile_value="Relation profile for governance workflows.",
        source_ids=[doc.id],
        metadata={},
    )

    client = TestClient(router)
    response = client.post(
        "/query",
        json={
            "query": "How does this work?",
            "param": {"mode": "hybrid", "top_k": 5},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["query_keywords"] == {
        "low_level_keywords": ["Policy Engine"],
        "high_level_keywords": ["governance"],
    }
