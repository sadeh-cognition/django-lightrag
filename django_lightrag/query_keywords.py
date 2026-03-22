import json
import re
from dataclasses import dataclass
from django.contrib.auth import get_user_model
from django_llm_chat.chat import Chat
from django_llm_chat.models import Project

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
    def __init__(self, model: str, config: QueryKeywordConfig | None = None) -> None:
        self.model = model
        self.config = config or QueryKeywordConfig()

    def extract(self, query_text: str) -> QueryKeywords:
        user, _ = get_user_model().objects.get_or_create(username="lightrag_django")
        project, _ = Project.objects.get_or_create(name="lightrag_django")
        chat = Chat.create(project=project)

        chat.create_system_message(KEYWORD_EXTRACTION_SYSTEM_PROMPT, user=user)

        chat.call_llm(
            model_name=self.model,
            message=query_text,
            user=user,
            include_chat_history=True,
            max_tokens=self.config.query_keyword_max_tokens,
            use_cache=True,
        )
        response = chat.last_llm_message.text if chat.last_llm_message else ""
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
