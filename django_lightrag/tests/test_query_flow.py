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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_calls = []

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.embedding_calls.append(texts)
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
        metadata={"entity_id": entity.id, "profile_key": entity.profile_key},
    )
    vector_storage.upsert_embedding(
        "relation",
        relation.id,
        core._get_embeddings(["governance"])[0],
        metadata={"relation_id": relation.id, "profile_key": relation.profile_key},
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

    # Clear previous calls from build_core
    core.embedding_calls.clear()

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

    # Assert exactly one batched embedding call was made for retrieval
    assert len(core.embedding_calls) == 1
    assert core.embedding_calls[0] == [
        "How are decisions enforced?",
        "Policy Engine",
        "governance",
    ]

    # Assert vector_matching metadata
    vector_match = result.context["vector_matching"]
    assert vector_match["entities"]["query_source"] == "keyword"
    assert vector_match["entities"]["hits"][0]["profile_key"] == "Policy Engine"
    assert "score" in vector_match["entities"]["hits"][0]

    assert vector_match["relations"]["query_source"] == "keyword"
    assert vector_match["relations"]["hits"][0]["profile_key"] == "governance"

    assert vector_match["documents"]["query_source"] == "raw"


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
    assert len(local_result.context["vector_matching"]["entities"]["hits"]) > 0
    assert len(local_result.context["vector_matching"]["relations"]["hits"]) == 0

    assert global_result.context["entities"] == []
    assert [item["relation_type"] for item in global_result.context["relations"]] == [
        "governs"
    ]
    assert len(global_result.context["vector_matching"]["entities"]["hits"]) == 0
    assert len(global_result.context["vector_matching"]["relations"]["hits"]) > 0


@pytest.mark.django_db
def test_query_falls_back_to_raw_query_when_keyword_extraction_fails():
    core = build_core("not valid json")

    core.embedding_calls.clear()

    result = core.query("Policy Engine governance document", QueryParam(mode="hybrid"))

    assert result.context["query_keywords"] == {
        "low_level_keywords": [],
        "high_level_keywords": [],
    }
    assert [item["name"] for item in result.context["entities"]] == ["Policy Engine"]
    assert [item["relation_type"] for item in result.context["relations"]] == [
        "governs"
    ]

    # Verify fallback query text
    assert len(core.embedding_calls) == 1
    assert core.embedding_calls[0] == [
        "Policy Engine governance document",  # doc
        "Policy Engine governance document",  # entity fallback
        "Policy Engine governance document",  # relation fallback
    ]

    vmatch = result.context["vector_matching"]
    assert vmatch["entities"]["query_source"] == "fallback"
    assert vmatch["relations"]["query_source"] == "fallback"


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

    # Endpoint coverage for vector matching structure
    vmatch = payload["context"]["vector_matching"]
    assert "documents" in vmatch
    assert "entities" in vmatch
    assert "relations" in vmatch
    assert "hits" in vmatch["entities"]
    assert "query_source" in vmatch["entities"]


@pytest.mark.django_db
def test_query_one_hop_traversal_core_logic():
    core = build_core("{}")

    doc = Document.objects.get(id="query-doc")
    target = Entity.objects.get(id="query-target")

    neighbor_ent = Entity.objects.create(
        id="neighbor-ent",
        name="Unrelated",
        entity_type="concept",
        description="Neighbor.",
        profile_key="Unrelated",
        profile_value="Unrelated",
        source_ids=[doc.id],
        metadata={},
    )
    Relation.objects.create(
        id="neighbor-rel",
        source_entity=target,
        target_entity=neighbor_ent,
        relation_type="depends",
        description="Neighbor relation.",
        profile_key="depends",
        profile_value="depends",
        source_ids=[doc.id],
        metadata={},
    )

    # Test one_hop_enabled=True
    result = core.query(
        "Policy Engine governance",
        QueryParam(
            mode="hybrid",
            one_hop_enabled=True,
            one_hop_max_entities=10,
            one_hop_max_relations=10,
        ),
    )

    source_entity_ids = [e["id"] for e in result.sources if e["type"] == "entity"]
    source_relation_ids = [r["id"] for r in result.sources if r["type"] == "relation"]

    assert "neighbor-ent" in source_entity_ids
    assert "neighbor-rel" in source_relation_ids

    gt = result.context["graph_traversal"]
    assert "neighbor-ent" in gt["added_entity_ids"]
    assert "neighbor-rel" in gt["added_relation_ids"]
    assert gt["caps_applied"]["max_entities"] == 10


@pytest.mark.django_db
@override_settings(
    LIGHTRAG={
        "EMBEDDING_PROVIDER": "test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_BASE_URL": "http://test.invalid",
        "LLM_MODEL": "test-llm",
        "CORE_FACTORY": "django_lightrag.tests.factories.make_test_core",
    }
)
def test_query_endpoint_one_hop_schema_parsing():
    doc = Document.objects.create(id="endpoint-doc", content="Test content")
    entity = Entity.objects.create(
        id="endpoint-entity",
        name="Policy Engine",
        entity_type="concept",
        profile_key="Policy Engine",
        metadata={},
        source_ids=[doc.id],
    )
    target = Entity.objects.create(
        id="endpoint-target",
        name="Control Plane",
        entity_type="concept",
        profile_key="Control Plane",
        metadata={},
        source_ids=[doc.id],
    )
    Relation.objects.create(
        id="endpoint-relation",
        source_entity=entity,
        target_entity=target,
        relation_type="governs",
        profile_key="governance",
        metadata={},
        source_ids=[doc.id],
    )
    client = TestClient(router)
    response = client.post(
        "/query",
        json={
            "query": "How does this work?",
            "param": {
                "mode": "hybrid",
                "top_k": 5,
                "one_hop_enabled": True,
                "one_hop_max_entities": 2,
                "one_hop_max_relations": 3,
            },
        },
    )
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["graph_traversal"]["caps_applied"]["max_entities"] == 2
