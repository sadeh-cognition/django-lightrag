import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from ninja.testing import TestClient

from django_lightrag.config import get_ladybug_settings, get_lightrag_settings
from django_lightrag.views import router


@override_settings(LIGHTRAG=None)
def test_lightrag_setting_is_required():
    with pytest.raises(
        ImproperlyConfigured,
        match="Django setting LIGHTRAG must be defined as a dict.",
    ):
        get_lightrag_settings()


@override_settings(
    LIGHTRAG={
        "EMBEDDING_PROVIDER": "test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_BASE_URL": "http://test.invalid",
    }
)
def test_lightrag_required_keys_are_enforced():
    with pytest.raises(
        ImproperlyConfigured,
        match="Django setting LIGHTRAG is missing required keys: LLM_MODEL.",
    ):
        get_lightrag_settings()


@override_settings(LIGHTRAG=None)
def test_query_endpoint_returns_error_when_lightrag_setting_is_missing():
    client = TestClient(router)

    response = client.post(
        "/query",
        json={
            "query": "How does this work?",
            "param": {"mode": "hybrid", "top_k": 5},
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "query_failed",
        "message": "Django setting LIGHTRAG must be defined as a dict.",
        "details": None,
    }


@override_settings(LADYBUGDB=None)
def test_ladybug_setting_is_required():
    with pytest.raises(
        ImproperlyConfigured,
        match="Django setting LADYBUGDB must be defined as a dict.",
    ):
        get_ladybug_settings()


@override_settings(LADYBUGDB={})
def test_ladybug_database_path_is_required_when_not_in_memory():
    with pytest.raises(
        ImproperlyConfigured,
        match=(
            "Django setting LADYBUGDB must define DATABASE_PATH when IN_MEMORY is false."
        ),
    ):
        get_ladybug_settings()


@override_settings(LADYBUGDB=None)
@pytest.mark.django_db
def test_query_endpoint_returns_error_when_ladybug_setting_is_missing():
    client = TestClient(router)

    response = client.post(
        "/query",
        json={
            "query": "How does this work?",
            "param": {"mode": "hybrid", "top_k": 5},
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "query_failed",
        "message": "Django setting LADYBUGDB must be defined as a dict.",
        "details": None,
    }
