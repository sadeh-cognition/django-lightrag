import json

from django_lightrag.query_keywords import QueryKeywordExtractor


class KeywordLLMService:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, str | int | None]] = []

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def test_parse_keyword_response_normalizes_and_dedupes():
    extractor = QueryKeywordExtractor(llm_service=KeywordLLMService("{}"))

    keywords = extractor.parse_response(
        """noise
{"low_level_keywords":[" Policy Engine ","","policy engine","Control Plane"],
"high_level_keywords":["Governance ","governance"," Policy automation "]}
tail"""
    )

    assert keywords.low_level_keywords == ["Policy Engine", "Control Plane"]
    assert keywords.high_level_keywords == ["Governance", "Policy automation"]


def test_parse_keyword_response_falls_back_to_empty_lists():
    extractor = QueryKeywordExtractor(llm_service=KeywordLLMService("{}"))

    keywords = extractor.parse_response("not json at all")

    assert keywords.low_level_keywords == []
    assert keywords.high_level_keywords == []


def test_extract_uses_strict_json_prompt():
    llm_service = KeywordLLMService(
        json.dumps(
            {
                "low_level_keywords": ["Policy Engine"],
                "high_level_keywords": ["Governance"],
            }
        )
    )
    extractor = QueryKeywordExtractor(llm_service=llm_service)

    keywords = extractor.extract("How does the Policy Engine enforce governance?")

    assert keywords.low_level_keywords == ["Policy Engine"]
    assert keywords.high_level_keywords == ["Governance"]
    assert llm_service.calls[0]["user_prompt"] == (
        "How does the Policy Engine enforce governance?"
    )
    assert "Return JSON only" in str(llm_service.calls[0]["system_prompt"])
