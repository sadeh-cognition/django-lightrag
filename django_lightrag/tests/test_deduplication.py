import pytest

from django_lightrag.core import LightRAGCore
from django_lightrag.deduplication import canonical_entity_id, canonical_relation_id
from django_lightrag.models import Document, Entity, Relation
from django_lightrag.storage import LadybugGraphStorage
from django_lightrag.utils import Tokenizer


class TrackingGraphStorage:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.relations: set[tuple[str, str]] = set()

    def add_entity_if_not_exists(self, entity_data):
        self.entities.setdefault(entity_data["id"], dict(entity_data))

    def add_relation(self, relation_data):
        source_id = relation_data["source_entity"]
        target_id = relation_data["target_entity"]
        self.relations.add(tuple(sorted([source_id, target_id])))

    def upsert_entity_node(self, entity_data):
        self.entities[entity_data["id"]] = dict(entity_data)
        return entity_data["id"]

    def remove_entity_node(self, entity_id):
        self.entities.pop(entity_id, None)
        self.relations = {pair for pair in self.relations if entity_id not in pair}
        return True

    def upsert_relation_edge(self, relation_data):
        source_id = relation_data["source_entity"]
        target_id = relation_data["target_entity"]
        self.relations.add(tuple(sorted([source_id, target_id])))
        return relation_data["id"]

    def remove_relation_edge(self, source_id, target_id):
        self.relations.discard(tuple(sorted([source_id, target_id])))
        return True

    def close(self):
        return None


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

    def close(self):
        return None


class RecordingProfilingService:
    def __init__(self):
        self.entity_ids: list[str] = []
        self.relation_ids: list[str] = []

    def profile_entity(self, entity):
        self.entity_ids.append(entity.id)
        entity.profile_key = entity.name
        entity.profile_value = entity.description or entity.name
        entity.profile_input_hash = f"entity:{entity.id}"
        entity.save(
            update_fields=[
                "profile_key",
                "profile_value",
                "profile_input_hash",
                "updated_at",
            ]
        )
        return True

    def profile_relation(self, relation):
        self.relation_ids.append(relation.id)
        relation.profile_key = relation.relation_type
        relation.profile_value = relation.description or relation.relation_type
        relation.profile_input_hash = f"relation:{relation.id}"
        relation.save(
            update_fields=[
                "profile_key",
                "profile_value",
                "profile_input_hash",
                "updated_at",
            ]
        )
        return True


class DeterministicCore(LightRAGCore):
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def make_core(*, graph_storage=None, vector_storage=None) -> DeterministicCore:
    core = DeterministicCore(
        embedding_model="test-embedding",
        embedding_provider="test",
        embedding_base_url="http://test.invalid",
        llm_model="test-llm",
        graph_storage=graph_storage or TrackingGraphStorage(),
        vector_storage=vector_storage or InMemoryVectorStorage(),
        tokenizer=Tokenizer(),
    )
    core.profiling_service = RecordingProfilingService()
    return core


@pytest.mark.django_db
def test_backfill_deduplicates_entities_and_repoints_relations():
    doc1 = Document.objects.create(id="doc-1", content="Acme launches Atlas.")
    doc2 = Document.objects.create(id="doc-2", content="Acme scales Atlas.")

    duplicate_entity = Entity.objects.create(
        id="legacy-entity-acme",
        name="Acme",
        entity_type="company",
        description="Acme builds systems.",
        source_ids=[doc1.id],
        metadata={"description_fragments": ["Acme builds systems."]},
    )
    canonical_entity = Entity.objects.create(
        id=canonical_entity_id("Acme", "company"),
        name="Acme",
        entity_type="company",
        description="Acme operates globally.",
        source_ids=[doc2.id],
        metadata={"description_fragments": ["Acme operates globally."]},
    )
    target = Entity.objects.create(
        id="entity-atlas",
        name="Atlas",
        entity_type="product",
        description="Atlas is a product.",
        source_ids=[doc1.id],
        metadata={"description_fragments": ["Atlas is a product."]},
    )

    relation = Relation.objects.create(
        id="legacy-relation",
        source_entity=duplicate_entity,
        target_entity=target,
        relation_type="builds",
        description="Acme builds Atlas.",
        source_ids=[doc1.id],
        metadata={
            "keywords": "builds",
            "keywords_list": ["builds"],
            "description_fragments": ["Acme builds Atlas."],
        },
    )

    vector_storage = InMemoryVectorStorage()
    vector_storage.upsert_embedding("entity", duplicate_entity.id, [1.0], {})
    vector_storage.upsert_embedding("entity", canonical_entity.id, [2.0], {})
    graph_storage = TrackingGraphStorage()
    graph_storage.add_entity_if_not_exists(
        {
            "id": duplicate_entity.id,
            "name": duplicate_entity.name,
            "entity_type": duplicate_entity.entity_type,
            "description": duplicate_entity.description,
            "metadata": duplicate_entity.metadata,
        }
    )
    graph_storage.add_entity_if_not_exists(
        {
            "id": canonical_entity.id,
            "name": canonical_entity.name,
            "entity_type": canonical_entity.entity_type,
            "description": canonical_entity.description,
            "metadata": canonical_entity.metadata,
        }
    )

    core = make_core(graph_storage=graph_storage, vector_storage=vector_storage)

    result = core.deduplicate_graph()

    assert result["merged_entities"] == 1
    assert result["deleted_entities"] == 1

    survivor = Entity.objects.get(id=canonical_entity.id)
    relation.refresh_from_db()

    assert survivor.source_ids == [doc1.id, doc2.id]
    assert survivor.metadata["description_fragments"] == [
        "Acme builds systems.",
        "Acme operates globally.",
    ]
    assert relation.source_entity_id == survivor.id
    assert Entity.objects.filter(id=duplicate_entity.id).count() == 0
    assert duplicate_entity.id not in vector_storage.records["entity"]
    assert canonical_entity.id in vector_storage.records["entity"]
    assert duplicate_entity.id not in graph_storage.entities

    core.close()


@pytest.mark.django_db
def test_backfill_deduplicates_relations_and_keeps_max_weight():
    doc1 = Document.objects.create(id="doc-r1", content="Acme operates Atlas.")
    doc2 = Document.objects.create(id="doc-r2", content="Atlas depends on Acme.")

    source = Entity.objects.create(
        id="entity-acme",
        name="Acme",
        entity_type="company",
        description="Company",
        metadata={"description_fragments": ["Company"]},
    )
    target = Entity.objects.create(
        id="entity-atlas",
        name="Atlas",
        entity_type="product",
        description="Product",
        metadata={"description_fragments": ["Product"]},
    )

    first_relation = Relation.objects.create(
        id="legacy-rel-1",
        source_entity=source,
        target_entity=target,
        relation_type="depends_on",
        description="Acme supports Atlas.",
        source_ids=[doc1.id],
        weight=1.5,
        metadata={
            "keywords": "supports",
            "keywords_list": ["supports"],
            "description_fragments": ["Acme supports Atlas."],
        },
    )
    second_relation = Relation.objects.create(
        id=canonical_relation_id(source.id, target.id, "depends_on"),
        source_entity=target,
        target_entity=source,
        relation_type="depends_on",
        description="Atlas depends on Acme.",
        source_ids=[doc2.id],
        weight=3.0,
        metadata={
            "keywords": "depends",
            "keywords_list": ["depends"],
            "description_fragments": ["Atlas depends on Acme."],
        },
    )

    vector_storage = InMemoryVectorStorage()
    vector_storage.upsert_embedding("relation", first_relation.id, [1.0], {})
    vector_storage.upsert_embedding("relation", second_relation.id, [2.0], {})

    core = make_core(vector_storage=vector_storage)
    result = core.deduplicate_graph()

    assert result["merged_relations"] == 1
    assert result["deleted_relations"] == 1

    survivor = Relation.objects.get(id=second_relation.id)
    assert survivor.source_ids == [doc1.id, doc2.id]
    assert survivor.metadata["description_fragments"] == [
        "Acme supports Atlas.",
        "Atlas depends on Acme.",
    ]
    assert survivor.metadata["keywords_list"] == ["supports", "depends"]
    assert survivor.metadata["keywords"] == "supports, depends"
    assert survivor.weight == 3.0
    assert Relation.objects.filter(id=first_relation.id).count() == 0
    assert first_relation.id not in vector_storage.records["relation"]
    assert second_relation.id in vector_storage.records["relation"]

    core.close()


@pytest.mark.django_db
def test_ingest_runs_targeted_dedup_before_profiling():
    legacy_entity = Entity.objects.create(
        id="legacy-acme",
        name="Acme",
        entity_type="company",
        description="Legacy Acme.",
        metadata={"description_fragments": ["Legacy Acme."]},
    )
    target = Entity.objects.create(
        id="entity-atlas",
        name="Atlas",
        entity_type="product",
        description="Atlas.",
        metadata={"description_fragments": ["Atlas."]},
    )
    legacy_relation = Relation.objects.create(
        id="legacy-builds",
        source_entity=legacy_entity,
        target_entity=target,
        relation_type="builds",
        description="Legacy relation.",
        metadata={
            "keywords": "builds",
            "keywords_list": ["builds"],
            "description_fragments": ["Legacy relation."],
        },
    )

    graph_storage = TrackingGraphStorage()
    vector_storage = InMemoryVectorStorage()
    core = make_core(graph_storage=graph_storage, vector_storage=vector_storage)

    canonical_acme_id = canonical_entity_id("Acme", "company")
    canonical_relation = canonical_relation_id(canonical_acme_id, target.id, "builds")

    class StubGraphBuilder:
        def extract_and_persist(self, document):
            fresh_entity = Entity.objects.create(
                id=canonical_acme_id,
                name="Acme",
                entity_type="company",
                description="Fresh Acme.",
                source_ids=[document.id],
                metadata={"description_fragments": ["Fresh Acme."]},
            )
            fresh_relation = Relation.objects.create(
                id=canonical_relation,
                source_entity=fresh_entity,
                target_entity=target,
                relation_type="builds",
                description="Fresh relation.",
                source_ids=[document.id],
                metadata={
                    "keywords": "builds",
                    "keywords_list": ["builds"],
                    "description_fragments": ["Fresh relation."],
                },
            )
            return [fresh_entity], [fresh_relation]

    core.graph_builder = StubGraphBuilder()
    document_id = core.ingest_document("Acme builds Atlas.")

    assert document_id
    assert core.profiling_service.entity_ids == [canonical_acme_id]
    assert core.profiling_service.relation_ids == [canonical_relation]
    assert Entity.objects.filter(name="Acme", entity_type="company").count() == 1
    assert Relation.objects.filter(relation_type="builds").count() == 1
    assert (
        Relation.objects.get(id=canonical_relation).source_entity_id
        == canonical_acme_id
    )
    assert legacy_relation.id not in vector_storage.records["relation"]

    core.close()


@pytest.mark.django_db
def test_same_name_entities_with_different_types_remain_separate():
    Entity.objects.create(
        id="entity-acme-company",
        name="Acme",
        entity_type="company",
        description="Company",
        metadata={"description_fragments": ["Company"]},
    )
    Entity.objects.create(
        id="entity-acme-project",
        name="Acme",
        entity_type="project",
        description="Project",
        metadata={"description_fragments": ["Project"]},
    )

    core = make_core()
    result = core.deduplicate_graph()

    assert result["merged_entities"] == 0
    assert Entity.objects.filter(name="Acme").count() == 2

    core.close()


@pytest.mark.django_db
def test_placeholder_other_entity_does_not_merge_into_typed_entity():
    Entity.objects.create(
        id="entity-acme-other",
        name="Acme",
        entity_type="other",
        description="Placeholder",
        metadata={"description_fragments": ["Placeholder"]},
    )
    Entity.objects.create(
        id="entity-acme-company",
        name="Acme",
        entity_type="company",
        description="Typed",
        metadata={"description_fragments": ["Typed"]},
    )

    core = make_core()
    result = core.deduplicate_graph()

    assert result["merged_entities"] == 0
    assert Entity.objects.filter(name="Acme").count() == 2

    core.close()


@pytest.mark.django_db
def test_ladybug_storage_reflects_survivor_nodes_and_edges():
    pytest.importorskip("real_ladybug")

    doc = Document.objects.create(id="doc-ladybug", content="Acme builds Atlas.")

    legacy_entity = Entity.objects.create(
        id="legacy-acme",
        name="Acme",
        entity_type="company",
        description="Legacy Acme.",
        source_ids=[doc.id],
        metadata={"description_fragments": ["Legacy Acme."]},
    )
    canonical_entity = Entity.objects.create(
        id=canonical_entity_id("Acme", "company"),
        name="Acme",
        entity_type="company",
        description="Canonical Acme.",
        source_ids=[doc.id],
        metadata={"description_fragments": ["Canonical Acme."]},
    )
    atlas = Entity.objects.create(
        id="entity-atlas",
        name="Atlas",
        entity_type="product",
        description="Atlas.",
        source_ids=[doc.id],
        metadata={"description_fragments": ["Atlas."]},
    )
    Relation.objects.create(
        id="legacy-rel",
        source_entity=legacy_entity,
        target_entity=atlas,
        relation_type="builds",
        description="Legacy builds.",
        source_ids=[doc.id],
        metadata={
            "keywords": "builds",
            "keywords_list": ["builds"],
            "description_fragments": ["Legacy builds."],
        },
    )
    Relation.objects.create(
        id=canonical_relation_id(canonical_entity.id, atlas.id, "builds"),
        source_entity=canonical_entity,
        target_entity=atlas,
        relation_type="builds",
        description="Canonical builds.",
        source_ids=[doc.id],
        metadata={
            "keywords": "builds",
            "keywords_list": ["builds"],
            "description_fragments": ["Canonical builds."],
        },
    )

    graph_storage = LadybugGraphStorage()
    for entity in [legacy_entity, canonical_entity, atlas]:
        graph_storage.add_entity_if_not_exists(
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "metadata": entity.metadata,
            }
        )
    graph_storage.add_relation(
        {
            "id": "legacy-rel",
            "source_entity": legacy_entity.id,
            "target_entity": atlas.id,
            "relation_type": "builds",
            "description": "Legacy builds.",
            "metadata": {},
        }
    )
    graph_storage.add_relation(
        {
            "id": "canonical-rel",
            "source_entity": canonical_entity.id,
            "target_entity": atlas.id,
            "relation_type": "builds",
            "description": "Canonical builds.",
            "metadata": {},
        }
    )

    core = make_core(graph_storage=graph_storage)
    core.deduplicate_graph()

    stored_entities = {entity["id"] for entity in graph_storage.get_all_entities()}
    stored_relations = {
        tuple(sorted([relation["source_entity"], relation["target_entity"]]))
        for relation in graph_storage.get_all_relations()
    }

    assert legacy_entity.id not in stored_entities
    assert canonical_entity.id in stored_entities
    assert tuple(sorted([canonical_entity.id, atlas.id])) in stored_relations
    assert tuple(sorted([legacy_entity.id, atlas.id])) not in stored_relations

    core.close()
