import os
import shutil
import tempfile

import pytest
import requests
from django.test import override_settings
from dotenv import load_dotenv
from ninja.testing import TestClient

from django_lightrag.models import Document, Entity, Relation
from django_lightrag.views import router

assert load_dotenv(".env")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def is_lmstudio_reachable():
    base_url = os.environ.get("LIGHTRAG_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    try:
        response = requests.get(f"{base_url}/models", timeout=2)
        if response.status_code != 200:
            print(f"DEBUG: LMStudio returned {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"DEBUG: LMStudio connection failed: {e}")
        return False


# Skip conditions
pytestmark = [
    pytest.mark.live_e2e,
    pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set"),
    pytest.mark.skipif(
        not is_lmstudio_reachable(),
        reason="LMStudio not reachable at LIGHTRAG_EMBEDDING_BASE_URL",
    ),
]


@pytest.fixture(scope="module")
def live_storage_dir():
    """Create a temporary directory for Chroma and Ladybug storage."""
    tmp_dir = tempfile.mkdtemp(prefix="lightrag_live_test_")
    chroma_dir = os.path.join(tmp_dir, "chroma")
    ladybug_path = os.path.join(tmp_dir, "ladybug.db")
    os.makedirs(chroma_dir, exist_ok=True)

    yield {"root": tmp_dir, "chroma": chroma_dir, "ladybug": ladybug_path}

    shutil.rmtree(tmp_dir)


@pytest.fixture(autouse=True)
def live_settings_override(live_storage_dir):
    """Override settings to use real dependencies with temporary storage."""
    with override_settings(
        LIGHTRAG={
            "EMBEDDING_PROVIDER": "LMStudio",
            "EMBEDDING_MODEL": os.environ.get(
                "LIGHTRAG_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF"
            ),
            "EMBEDDING_BASE_URL": os.environ.get(
                "LIGHTRAG_EMBEDDING_BASE_URL", "http://localhost:1234/v1"
            ),
            "LLM_MODEL": "groq/llama-3.1-8b-instant",
        },
        CHROMADB_DIR=live_storage_dir["chroma"],
        LADYBUGDB={
            "DATABASE_PATH": live_storage_dir["ladybug"],
        },
    ):
        yield


@pytest.mark.django_db(transaction=True)
def test_live_end_to_end_cycle():
    """
    Full cycle: Ingest -> State Reflection -> Query -> Delete
    """
    client = TestClient(router)

    # 1. Ingest Flow
    content = "The Antigravity agent is a powerful tool for coding. It uses Django and LightRAG."
    ingest_response = client.post(
        "/documents/ingest",
        json={
            "content": content,
            "metadata": {"source": "live_test"},
            "track_id": "test-track-1",
        },
    )
    assert ingest_response.status_code == 201
    doc_id = ingest_response.json()["document_id"]

    # Verify ORM
    doc = Document.objects.get(id=doc_id)
    assert doc.content == content

    # Verify Graph (LadybugDB through ORM reflections)
    entities = [
        entity for entity in Entity.objects.all() if doc_id in (entity.source_ids or [])
    ]
    relations = [
        relation
        for relation in Relation.objects.select_related(
            "source_entity", "target_entity"
        )
        if doc_id in (relation.source_ids or [])
    ]
    assert len(entities) > 0
    assert len(relations) > 0

    # Verify profiling (should not be empty if live LLM worked)
    for entity in entities:
        assert entity.profile_value != ""

    # 2. State Reflection
    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    assert any(
        e["id"] in [ent.id for ent in entities] for e in entities_response.json()
    )

    relations_response = client.get("/relations")
    assert relations_response.status_code == 200
    assert any(
        r["id"] in [rel.id for rel in relations] for r in relations_response.json()
    )

    # 3. Query Flow
    query_text = "What is the Antigravity agent used for?"
    query_response = client.post(
        "/query", json={"query": query_text, "param": {"mode": "hybrid"}}
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert "response" in payload
    assert len(payload["context"]["documents"]) > 0

    # 4. Delete Flow
    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code == 200
    assert not Document.objects.filter(id=doc_id).exists()


@pytest.mark.django_db(transaction=True)
def test_live_second_ingest_merging():
    """Verify merged state across two related documents."""
    client = TestClient(router)

    # First ingest
    client.post(
        "/documents/ingest",
        json={"content": "Alice works at Cyberdyne.", "track_id": "t1"},
    )
    # Second ingest (related)
    client.post(
        "/documents/ingest",
        json={"content": "Bob also works at Cyberdyne.", "track_id": "t2"},
    )

    # Assert entity 'Cyberdyne' has both source IDs
    cyberdyne = Entity.objects.get(name__icontains="Cyberdyne")
    assert len(cyberdyne.source_ids) >= 2
