import pytest
from dotenv import load_dotenv

load_dotenv(".env")
from django_lightrag.query_keywords import QueryKeywordExtractor


def test_parse_keyword_response_normalizes_and_dedupes():
    extractor = QueryKeywordExtractor(model="test-model")

    keywords = extractor.parse_response(
        """noise
{"low_level_keywords":[" Policy Engine ","","policy engine","Control Plane"],
"high_level_keywords":["Governance ","governance"," Policy automation "]}
tail"""
    )

    assert keywords.low_level_keywords == ["Policy Engine", "Control Plane"]
    assert keywords.high_level_keywords == ["Governance", "Policy automation"]


def test_parse_keyword_response_falls_back_to_empty_lists():
    extractor = QueryKeywordExtractor(model="test-model")

    keywords = extractor.parse_response("not json at all")

    assert keywords.low_level_keywords == []
    assert keywords.high_level_keywords == []


from django.test import override_settings


@pytest.mark.live
@pytest.mark.django_db
@override_settings(
    LIGHTRAG={
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_PROVIDER": "test",
        "EMBEDDING_BASE_URL": "http://test.invalid",
        "LLM_MODEL": "groq/llama-3.1-8b-instant",
    }
)
def test_extract_returns_keywords():
    # Use real model from settings/env for live test
    from django_lightrag.config import get_lightrag_settings

    config = get_lightrag_settings()

    extractor = QueryKeywordExtractor(model=config.llm_model)

    keywords = extractor.extract("How does the Policy Engine enforce governance?")

    # We expect some keywords to be returned by a real LLM
    assert len(keywords.low_level_keywords) > 0 or len(keywords.high_level_keywords) > 0
