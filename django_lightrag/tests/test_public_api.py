from unittest.mock import patch

from django_lightrag import run_query
from django_lightrag.types import (
    QueryContext,
    QueryContextEntity,
    QueryParam,
    QueryResult,
    QuerySource,
)


class StubCore:
    last_init: dict[str, str] | None = None
    last_query_args: tuple[str, QueryParam] | None = None
    close_called = False

    def __init__(
        self,
        embedding_model: str,
        embedding_provider: str,
        embedding_base_url: str,
        llm_model: str,
    ):
        type(self).last_init = {
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "embedding_base_url": embedding_base_url,
            "llm_model": llm_model,
        }
        type(self).close_called = False

    def query(self, query_text: str, param: QueryParam) -> QueryResult:
        type(self).last_query_args = (query_text, param)
        return QueryResult(
            response="graph answer",
            sources=[QuerySource(type="entity", id="entity-1")],
            context=QueryContext(
                entities=[
                    QueryContextEntity(
                        name="Policy Engine",
                        entity_type="concept",
                        description="desc",
                        profile_key="policy",
                    )
                ]
            ),
            query_time=0.25,
            tokens_used=12,
        )

    def close(self) -> None:
        type(self).close_called = True


def test_run_query_returns_serialized_result_and_closes_core():
    with (
        patch(
            "django_lightrag.get_lightrag_core_settings",
            return_value={
                "EMBEDDING_MODEL": "test-embedding",
                "EMBEDDING_PROVIDER": "test-provider",
                "EMBEDDING_BASE_URL": "http://test.invalid",
                "LLM_MODEL": "test-llm",
            },
        ),
        patch("django_lightrag.core.LightRAGCore", StubCore),
    ):
        result = run_query(
            "How does the graph answer this?",
            {"mode": "local", "top_k": 3, "low_level_keywords": ["policy"]},
        )

    assert result == {
        "response": "graph answer",
        "sources": [{"type": "entity", "id": "entity-1"}],
        "context": {
            "documents": [],
            "entities": [
                {
                    "name": "Policy Engine",
                    "entity_type": "concept",
                    "description": "desc",
                    "profile_key": "policy",
                }
            ],
            "relations": [],
            "query_keywords": {
                "low_level_keywords": [],
                "high_level_keywords": [],
            },
            "total_tokens": 0,
            "aggregated_context": "",
        },
        "query_time": 0.25,
        "tokens_used": 12,
    }
    assert StubCore.last_init == {
        "embedding_model": "test-embedding",
        "embedding_provider": "test-provider",
        "embedding_base_url": "http://test.invalid",
        "llm_model": "test-llm",
    }
    assert StubCore.last_query_args is not None
    assert StubCore.last_query_args[0] == "How does the graph answer this?"
    assert StubCore.last_query_args[1] == QueryParam(
        mode="local",
        top_k=3,
        low_level_keywords=["policy"],
    )
    assert StubCore.close_called is True


def test_run_query_returns_error_payload():
    with patch(
        "django_lightrag.get_lightrag_core_settings",
        side_effect=RuntimeError("missing config"),
    ):
        result = run_query("How does the graph answer this?")

    assert result == {"error": "query_failed", "message": "missing config"}
