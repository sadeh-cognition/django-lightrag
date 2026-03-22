import json
import re
from dataclasses import dataclass
from typing import Any

KEYWORD_EXTRACTION_SYSTEM_PROMPT = """You extract retrieval keywords from a user query.

Return JSON only, with exactly this shape:
{"low_level_keywords": ["specific entity or detail"], "high_level_keywords": ["broad concept or theme"]}

Rules:
- Output valid JSON only. No markdown, no prose, no code fences.
- low_level_keywords must contain specific entities, proper nouns, product names, technical terms, or concrete details from the query.
- high_level_keywords must contain broad concepts, topics, themes, or the user intent from the query.
- Use concise words or meaningful phrases taken from the query.
- If the query is too vague or has no useful retrieval signal, return empty lists for both fields.
"""


@dataclass
class QueryKeywordConfig:
    query_keyword_max_tokens: int = 200


@dataclass
class QueryKeywords:
    low_level_keywords: list[str]
    high_level_keywords: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "low_level_keywords": self.low_level_keywords,
            "high_level_keywords": self.high_level_keywords,
        }


class QueryKeywordExtractor:
    def __init__(
        self, llm_service: Any, config: QueryKeywordConfig | None = None
    ) -> None:
        self.llm_service = llm_service
        self.config = config or QueryKeywordConfig()

    def extract(self, query_text: str) -> QueryKeywords:
        response = self.llm_service.call_llm(
            user_prompt=query_text,
            system_prompt=KEYWORD_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=self.config.query_keyword_max_tokens,
        )
        return self.parse_response(response)

    def parse_response(self, response: str) -> QueryKeywords:
        candidates = [response]

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return QueryKeywords(
                low_level_keywords=self._normalize_keywords(
                    parsed.get("low_level_keywords", [])
                ),
                high_level_keywords=self._normalize_keywords(
                    parsed.get("high_level_keywords", [])
                ),
            )

        return QueryKeywords(low_level_keywords=[], high_level_keywords=[])

    def _normalize_keywords(self, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, list):
            raw_values = value
        else:
            raw_values = []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            keyword = " ".join(str(item).split()).strip()
            if not keyword:
                continue
            marker = keyword.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            normalized.append(keyword)
        return normalized
