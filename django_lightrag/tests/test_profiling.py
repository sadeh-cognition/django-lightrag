import json

import pytest

from django_lightrag.core import LightRAGCore
from django_lightrag.graph_builder import KnowledgeGraphBuilder
from django_lightrag.models import Document, Entity, Relation
from django_lightrag.profiling import ProfilingService
from django_lightrag.types import QueryParam
from django_lightrag.utils import Tokenizer


class FakeGraphStorage:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.relations: list[dict] = []

    def add_entity_if_not_exists(self, entity_data):
        self.entities.setdefault(entity_data["id"], entity_data)

    def add_relation(self, relation_data):
        self.relations.append(relation_data)

    def close(self):
        return None


class RecordingLLMService:
    def __init__(self):
        self.calls: list[dict] = []

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        max_tokens: int | None = None,
    ) -> str:
        payload = json.loads(user_prompt)
        self.calls.append(payload)

        if payload["kind"] == "entity":
            if payload["name"] == "Policy Engine":
                return json.dumps(
                    {
                        "key": "governance policy",
                        "value": "Governance policy oversight for the Policy Engine.",
                    }
                )
            return json.dumps(
                {
                    "key": f"{payload['name']} summary",
                    "value": " ".join(payload["merged_descriptions"])
                    or " ".join(
                        document["content"] for document in payload["documents"]
                    ),
                }
            )

        return json.dumps(
            {
                "key": "governance link"
                if payload["source"] == "Policy Engine"
                else f"{payload['relation_type']} link",
                "value": (
                    "Governance oversight connects Policy Engine to Control Plane."
                    if payload["source"] == "Policy Engine"
                    else " ".join(payload["merged_descriptions"])
                ),
            }
        )


class InMemoryCollection:
    def __init__(self):
        self.records: dict[str, dict] = {}

    def get(self, ids):
        metadatas = [
            self.records[record_id]["metadata"]
            for record_id in ids
            if record_id in self.records
        ]
        found_ids = [record_id for record_id in ids if record_id in self.records]
        return {"ids": found_ids, "metadatas": metadatas}


class InMemoryVectorStorage:
    def __init__(self):
        self.records = {
            "document": {},
            "entity": {},
            "relation": {},
        }
        self.collections = {
            vector_type: InMemoryCollection() for vector_type in self.records
        }

    def upsert_embedding(self, vector_type, content_id, embedding, metadata=None):
        record = {"embedding": embedding, "metadata": metadata or {}}
        self.records[vector_type][content_id] = record
        self.collections[vector_type].records[content_id] = record
        return content_id

    def delete_embedding(self, vector_type, content_id):
        self.records[vector_type].pop(content_id, None)
        self.collections[vector_type].records.pop(content_id, None)
        return True

    def search_similar(self, vector_type, query_embedding, top_k=10, where=None):
        scored = []
        for content_id, record in self.records[vector_type].items():
            embedding = record["embedding"]
            distance = sum(
                (query_value - value) ** 2
                for query_value, value in zip(query_embedding, embedding, strict=True)
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


class DeterministicCore(LightRAGCore):
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        vocabulary = ["governance"]
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [1.0 if term in lowered else 0.0 for term in vocabulary]
            if not any(vector):
                vector[0] = 0.0
            embeddings.append(vector)
        return embeddings


@pytest.mark.django_db
def test_merge_provenance_across_multiple_documents():
    doc1 = Document.objects.create(id="doc-1", content="Acme builds the Policy Engine.")
    doc2 = Document.objects.create(
        id="doc-2", content="Acme aligns the Policy Engine with the Control Plane."
    )

    builder = KnowledgeGraphBuilder(
        llm_service=RecordingLLMService(),
        tokenizer=Tokenizer(),
        graph_storage=FakeGraphStorage(),
        config={},
    )

    entity_objects = builder._persist_entities(
        {
            "Acme": {
                "entity_type": "organization",
                "descriptions": ["Acme builds policy software."],
                "source_ids": [doc1.id],
                "metadata": {"source_id": doc1.id},
            },
            "Policy Engine": {
                "entity_type": "concept",
                "descriptions": ["Policy Engine evaluates service rules."],
                "source_ids": [doc1.id],
                "metadata": {"source_id": doc1.id},
            },
        }
    )
    builder._persist_relations(
        {
            "acme-policy": {
                "src_id": "Acme",
                "tgt_id": "Policy Engine",
                "relation_type": "builds",
                "descriptions": ["Acme creates the Policy Engine."],
                "keywords_list": ["builds"],
                "source_ids": [doc1.id],
                "weight": 1.0,
                "metadata": {"source_id": doc1.id},
            }
        },
        entity_objects,
    )

    entity_objects = builder._persist_entities(
        {
            "Acme": {
                "entity_type": "organization",
                "descriptions": ["Acme governs rollout policy for the platform."],
                "source_ids": [doc2.id],
                "metadata": {"source_id": doc2.id},
            }
        }
    )
    builder._persist_relations(
        {
            "acme-policy": {
                "src_id": "Acme",
                "tgt_id": "Policy Engine",
                "relation_type": "builds",
                "descriptions": [
                    "Acme aligns the Policy Engine with the Control Plane."
                ],
                "keywords_list": ["governance", "control-plane"],
                "source_ids": [doc2.id],
                "weight": 2.0,
                "metadata": {"source_id": doc2.id},
            }
        },
        entity_objects,
    )

    entity = Entity.objects.get(name="Acme")
    relation = Relation.objects.select_related("source_entity", "target_entity").get(
        source_entity__name="Acme",
        target_entity__name="Policy Engine",
        relation_type="builds",
    )

    assert entity.source_ids == [doc1.id, doc2.id]
    assert entity.metadata["description_fragments"] == [
        "Acme builds policy software.",
        "Acme governs rollout policy for the platform.",
    ]
    assert "Acme builds policy software." in entity.description
    assert "Acme governs rollout policy for the platform." in entity.description

    assert relation.source_ids == [doc1.id, doc2.id]
    assert relation.metadata["description_fragments"] == [
        "Acme creates the Policy Engine.",
        "Acme aligns the Policy Engine with the Control Plane.",
    ]
    assert relation.metadata["keywords_list"] == [
        "builds",
        "governance",
        "control-plane",
    ]
    assert relation.weight == 2.0


@pytest.mark.django_db
def test_profile_generation_and_incremental_refresh():
    doc = Document.objects.create(
        id="doc-profile", content="Policy Engine enforces governance rules."
    )
    entity = Entity.objects.create(
        id="entity-policy",
        name="Policy Engine",
        entity_type="concept",
        description="Policy Engine evaluates service rules.",
        source_ids=[doc.id],
        metadata={"description_fragments": ["Policy Engine evaluates service rules."]},
    )

    llm_service = RecordingLLMService()
    profiling_service = ProfilingService(llm_service)

    assert profiling_service.profile_entity(entity) is True
    entity.refresh_from_db()
    assert entity.profile_key == "governance policy"
    assert entity.profile_value == "Governance policy oversight for the Policy Engine."
    assert entity.profile_input_hash
    assert entity.profile_updated_at is not None
    assert len(llm_service.calls) == 1

    assert profiling_service.profile_entity(entity) is False
    assert len(llm_service.calls) == 1

    entity.description = (
        "Policy Engine evaluates service rules and coordinates governance."
    )
    entity.metadata["description_fragments"] = [
        "Policy Engine evaluates service rules.",
        "Policy Engine coordinates governance.",
    ]
    entity.save()

    assert profiling_service.profile_entity(entity) is True
    assert len(llm_service.calls) == 2


@pytest.mark.django_db
def test_entity_and_relation_vector_upserts_and_profile_retrieval():
    doc = Document.objects.create(
        id="doc-retrieval",
        content="The Policy Engine works with the Control Plane on policy decisions.",
    )
    other_doc = Document.objects.create(
        id="doc-other",
        content="Index Router optimizes latency for document retrieval.",
    )

    policy_engine = Entity.objects.create(
        id="entity-policy-engine",
        name="Policy Engine",
        entity_type="concept",
        description="A rules component for service evaluation.",
        source_ids=[doc.id],
        metadata={
            "description_fragments": ["A rules component for service evaluation."]
        },
    )
    control_plane = Entity.objects.create(
        id="entity-control-plane",
        name="Control Plane",
        entity_type="concept",
        description="A control component for orchestration.",
        source_ids=[doc.id],
        metadata={"description_fragments": ["A control component for orchestration."]},
    )
    index_router = Entity.objects.create(
        id="entity-index-router",
        name="Index Router",
        entity_type="concept",
        description="Routes retrieval traffic.",
        source_ids=[other_doc.id],
        metadata={"description_fragments": ["Routes retrieval traffic."]},
    )

    relation = Relation.objects.create(
        id="relation-policy-control",
        source_entity=policy_engine,
        target_entity=control_plane,
        relation_type="coordinates",
        description="The two components exchange decisions.",
        source_ids=[doc.id],
        metadata={
            "keywords": "coordinates",
            "keywords_list": ["coordinates"],
            "description_fragments": ["The two components exchange decisions."],
        },
    )
    Relation.objects.create(
        id="relation-index-router",
        source_entity=index_router,
        target_entity=policy_engine,
        relation_type="routes",
        description="The router forwards requests.",
        source_ids=[other_doc.id],
        metadata={
            "keywords": "routes",
            "keywords_list": ["routes"],
            "description_fragments": ["The router forwards requests."],
        },
    )

    vector_storage = InMemoryVectorStorage()
    core = DeterministicCore(
        embedding_model="test-embedding",
        embedding_provider="test",
        embedding_base_url="http://test.invalid",
        llm_model="test-llm",
        llm_service=RecordingLLMService(),
        graph_storage=FakeGraphStorage(),
        vector_storage=vector_storage,
        tokenizer=Tokenizer(),
    )

    result = core.backfill_profiles()
    assert result == {"entities": 3, "relations": 2}

    entity_record = vector_storage.collections["entity"].get(ids=[policy_engine.id])
    relation_record = vector_storage.collections["relation"].get(ids=[relation.id])
    assert entity_record["ids"] == [policy_engine.id]
    assert relation_record["ids"] == [relation.id]
    assert "Governance policy oversight" in entity_record["metadatas"][0]["content"]
    assert "Governance oversight connects" in relation_record["metadatas"][0]["content"]

    query_embedding = core._get_query_embedding("governance")
    ent_vectors = core.query_engine.search_entity_vectors(query_embedding, top_k=2)
    rel_vectors = core.query_engine.search_relation_vectors(query_embedding, top_k=2)
    entities = core.query_engine.hydrate_entities(ent_vectors)
    relations = core.query_engine.hydrate_relations(rel_vectors)

    assert [entity.id for entity in entities][0] == policy_engine.id
    assert [relation.id for relation in relations][0] == relation.id

    context = core.query_engine.build_context([], entities, relations, QueryParam())
    assert (
        context["entities"][0]["description"]
        == "Governance policy oversight for the Policy Engine."
    )
    assert (
        context["relations"][0]["description"]
        == "Governance oversight connects Policy Engine to Control Plane."
    )

    core.close()
