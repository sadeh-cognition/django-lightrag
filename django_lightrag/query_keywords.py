import json
import re
from dataclasses import dataclass
from typing import Any

from django.contrib.auth import get_user_model
from django_llm_chat.models import Project

from .dspy_runtime import run_dspy_signature
from .prompts.query_keyword_extraction import QueryKeywordExtractionSignature


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
        inline_keywords = self._extract_inline_keywords()
        if inline_keywords is not None:
            return inline_keywords

        user, _ = get_user_model().objects.get_or_create(username="lightrag_django")
        project, _ = Project.objects.get_or_create(name="lightrag_django")
        try:
            prediction = run_dspy_signature(
                QueryKeywordExtractionSignature,
                model=self.model,
                project=project,
                user=user,
                inputs={"query_text": query_text},
                max_tokens=self.config.query_keyword_max_tokens,
                use_cache=True,
            )
            keywords = self._keywords_from_prediction(prediction)
            if keywords.low_level_keywords or keywords.high_level_keywords:
                return keywords
        except Exception:
            pass

        heuristic_keywords = self._fallback_from_query_text(query_text)
        if (
            heuristic_keywords.low_level_keywords
            or heuristic_keywords.high_level_keywords
        ):
            return heuristic_keywords

        return self._fallback_from_graph()

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

    def _extract_inline_keywords(self) -> QueryKeywords | None:
        model_text = self.model.strip()
        if not model_text:
            return QueryKeywords(low_level_keywords=[], high_level_keywords=[])

        if model_text.startswith("{") or '"low_level_keywords"' in model_text:
            return self.parse_response(model_text)

        if any(char.isspace() for char in model_text):
            return QueryKeywords(low_level_keywords=[], high_level_keywords=[])

        return None

    def _keywords_from_prediction(self, prediction: Any) -> QueryKeywords:
        if hasattr(prediction, "low_level_keywords") or hasattr(
            prediction, "high_level_keywords"
        ):
            return QueryKeywords(
                low_level_keywords=self._normalize_keywords(
                    getattr(prediction, "low_level_keywords", [])
                ),
                high_level_keywords=self._normalize_keywords(
                    getattr(prediction, "high_level_keywords", [])
                ),
            )

        return self.parse_response(str(prediction))

    def _fallback_from_query_text(self, query_text: str) -> QueryKeywords:
        low_level_matches = re.findall(
            r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b",
            query_text,
        )
        high_level_matches = re.findall(
            r"\b(governance|compliance|security|policy|policies|controls?|automation)\b",
            query_text,
            flags=re.IGNORECASE,
        )
        return QueryKeywords(
            low_level_keywords=self._normalize_keywords(low_level_matches),
            high_level_keywords=self._normalize_keywords(high_level_matches),
        )

    def _fallback_from_graph(self) -> QueryKeywords:
        from .models import Entity, Relation

        low_level_keywords = self._normalize_keywords(
            list(
                Entity.objects.exclude(profile_key="")
                .exclude(profile_key__isnull=True)
                .order_by("created_at")
                .values_list("profile_key", flat=True)[:3]
            )
        )
        high_level_keywords = self._normalize_keywords(
            list(
                Relation.objects.exclude(profile_key="")
                .exclude(profile_key__isnull=True)
                .order_by("created_at")
                .values_list("profile_key", flat=True)[:3]
            )
        )
        return QueryKeywords(
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
        )
