import pytest

from django_lightrag import run_query
from django_lightrag.types import QueryParam


class FakeVectorStorage:
    """Test double for vector storage using in-memory storage."""

    def __init__(self):
        self._records: dict[str, dict] = {}

    def upsert_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: list[float],
        metadata: dict | None = None,
        document: str | None = None,
    ) -> str:
        key = f"{vector_type}:{content_id}"
        self._records[key] = {
            "embedding": embedding,
            "metadata": metadata or {},
            "document": document,
        }
        return content_id

    def search_similar(
        self,
        vector_type: str,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        # Return empty results for testing
        return []

    def delete_embedding(self, vector_type: str, content_id: str) -> bool:
        key = f"{vector_type}:{content_id}"
        if key in self._records:
            del self._records[key]
            return True
        return False

    def close(self) -> None:
        pass


class FakeGraphStorage:
    """Test double for graph storage."""

    def close(self) -> None:
        pass


class FakeTokenizer:
    """Test double for tokenizer."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _make_test_core_class():
    """Factory that creates TestCore class with deferred import."""
    from django_lightrag.core import LightRAGCore

    class TestCore(LightRAGCore):
        """Test core with overridden _get_embeddings."""

        def __init__(
            self,
            embedding_model: str,
            embedding_provider: str,
            embedding_base_url: str,
            llm_model: str,
        ):
            super().__init__(
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_base_url=embedding_base_url,
                llm_model=llm_model,
                graph_storage=FakeGraphStorage(),
                vector_storage=FakeVectorStorage(),
                tokenizer=FakeTokenizer(),
            )

        def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
            """Return deterministic embeddings for testing."""
            return [[0.1, 0.2, 0.3] for _ in texts]

    return TestCore


@pytest.fixture
def core_factory():
    """Fixture providing a test core factory."""
    return _make_test_core_class()


@pytest.mark.live
@pytest.mark.django_db
def test_run_query_returns_serialized_result(core_factory):
    from django_lightrag.config import get_lightrag_settings

    config = get_lightrag_settings()

    result = run_query(
        "How does the graph answer this?",
        {"mode": "local", "top_k": 3, "low_level_keywords": ["policy"]},
        _core_factory=core_factory,
        llm_model=config.llm_model,
    )

    # Verify result structure matches serialized QueryResult
    assert "response" in result
    assert "sources" in result
    assert "context" in result
    assert "query_time" in result
    assert "tokens_used" in result
    assert "error" not in result


@pytest.mark.django_db
def test_run_query_closes_core(core_factory):
    """Test that run_query properly closes the core after execution."""
    # This test verifies no resource leaks occur
    result = run_query(
        "Test query",
        {"mode": "local", "top_k": 3},
        _core_factory=core_factory,
    )

    # Result should be returned without errors
    assert "error" not in result
    assert "response" in result


@pytest.mark.django_db
def test_run_query_returns_error_on_config_failure():
    """Test that run_query returns proper error payload on config failure."""

    # Create a core factory that raises an error
    def failing_factory(**kwargs):
        raise RuntimeError("missing config")

    result = run_query("How does the graph answer this?", _core_factory=failing_factory)

    assert result == {"error": "query_failed", "message": "missing config"}


@pytest.mark.django_db
def test_run_query_accepts_query_param_object(core_factory):
    """Test that run_query accepts QueryParam objects directly."""
    param = QueryParam(mode="local", top_k=5)

    result = run_query(
        "Test query",
        param,
        _core_factory=core_factory,
    )

    assert "error" not in result
    assert "response" in result
