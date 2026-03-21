import os
import shutil
import tempfile

import pytest
import requests
from django.test import override_settings
from dotenv import load_dotenv

from django_lightrag import run_update
from django_lightrag.models import Document, Entity, Relation

assert load_dotenv(".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def is_lmstudio_reachable():
    base_url = os.environ.get("LIGHTRAG_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    try:
        response = requests.get(f"{base_url}/models", timeout=2)
        return response.status_code == 200
    except Exception:
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
    tmp_dir = tempfile.mkdtemp(prefix="lightrag_run_update_test_")
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
def test_run_update_live_cycle():
    """
    Verify run_update function works end-to-end with real services.
    """
    content = (
        "The run_update function is a convenient way to ingest data into LightRAG."
    )
    metadata = {"source": "unit_test", "author": "Antigravity"}
    track_id = "test-run-update-track"

    result = run_update(content=content, metadata=metadata, track_id=track_id)

    assert "document_id" in result
    assert result["message"] == "Document ingested successfully"
    doc_id = result["document_id"]

    # Verify ORM
    doc = Document.objects.get(id=doc_id)
    assert doc.content == content
    assert doc.metadata == metadata
    assert doc.track_id == track_id

    # Verify Knowledge Graph (at least one entity and relation should be created)
    # Note: source_ids in Entity/Relation are lists of document IDs
    entities = [
        entity for entity in Entity.objects.all() if doc_id in (entity.source_ids or [])
    ]
    relations = [
        relation
        for relation in Relation.objects.all()
        if doc_id in (relation.source_ids or [])
    ]

    assert len(entities) > 0, "No entities extracted from content"
    assert len(relations) > 0, "No relations extracted from content"

    # Verify profiling
    for entity in entities:
        assert entity.profile_value != "", f"Entity {entity.name} was not profiled"
