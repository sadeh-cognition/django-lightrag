import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from django.contrib.auth import get_user_model
from django.utils import timezone
from django_llm_chat.models import Project

from .dspy_runtime import run_dspy_signature
from .prompts.profile_generation import ProfileGenerationSignature


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    profile_max_tokens: int = 400


class ProfilingService:
    """Generate retrieval-oriented profiles for canonical entities and relations."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        config: ProfilingConfig | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.config = config or ProfilingConfig()

    def profile_entity(self, entity: Any) -> bool:
        descriptions = self._get_description_fragments(entity)
        source_ids = self._normalize_ids(entity.source_ids)
        input_hash = self._hash_payload(
            {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "merged_descriptions": sorted(descriptions),
                "source_ids": sorted(source_ids),
            }
        )

        if not self._needs_refresh(entity, input_hash):
            return False

        documents = self._load_documents(source_ids)
        key, value = self._generate_profile(
            {
                "kind": "entity",
                "name": entity.name,
                "entity_type": entity.entity_type,
                "merged_descriptions": descriptions,
                "documents": [
                    {"id": document.id, "content": document.content}
                    for document in documents
                ],
            },
            fallback_key=entity.name,
            fallback_value=entity.description
            or " ".join(document.content for document in documents),
        )

        entity.profile_key = key
        entity.profile_value = value
        entity.profile_input_hash = input_hash
        entity.profile_updated_at = timezone.now()
        entity.save(
            update_fields=[
                "profile_key",
                "profile_value",
                "profile_input_hash",
                "profile_updated_at",
                "updated_at",
            ]
        )
        return True

    def profile_relation(self, relation: Any) -> bool:
        descriptions = self._get_description_fragments(relation)
        source_ids = self._normalize_ids(relation.source_ids)
        keywords = self._get_keywords(relation)
        input_hash = self._hash_payload(
            {
                "source": relation.source_entity.name,
                "target": relation.target_entity.name,
                "relation_type": relation.relation_type,
                "keywords": sorted(keywords),
                "merged_descriptions": sorted(descriptions),
                "source_ids": sorted(source_ids),
            }
        )

        if not self._needs_refresh(relation, input_hash):
            return False

        documents = self._load_documents(source_ids)
        key, value = self._generate_profile(
            {
                "kind": "relation",
                "source": relation.source_entity.name,
                "target": relation.target_entity.name,
                "relation_type": relation.relation_type,
                "keywords": keywords,
                "merged_descriptions": descriptions,
                "documents": [
                    {"id": document.id, "content": document.content}
                    for document in documents
                ],
            },
            fallback_key=relation.relation_type,
            fallback_value=relation.description
            or " ".join(document.content for document in documents),
        )

        relation.profile_key = key
        relation.profile_value = value
        relation.profile_input_hash = input_hash
        relation.profile_updated_at = timezone.now()
        relation.save(
            update_fields=[
                "profile_key",
                "profile_value",
                "profile_input_hash",
                "profile_updated_at",
                "updated_at",
            ]
        )
        return True

    def _generate_profile(
        self,
        payload: dict[str, Any],
        *,
        fallback_key: str,
        fallback_value: str,
    ) -> tuple[str, str]:
        user, _ = get_user_model().objects.get_or_create(username="lightrag_django")
        project, _ = Project.objects.get_or_create(name="lightrag_django")
        prediction = run_dspy_signature(
            ProfileGenerationSignature,
            model=self.model,
            project=project,
            user=user,
            inputs={"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)},
            temperature=self.temperature,
            max_tokens=self.config.profile_max_tokens,
            use_cache=True,
        )
        key = str(getattr(prediction, "key", "")).strip()
        value = str(getattr(prediction, "value", "")).strip()
        key = key or fallback_key
        value = value or fallback_value
        return self._normalize_key(key), self._normalize_value(value)

    def _parse_profile_response(self, response: str) -> tuple[str, str]:
        candidates = [response]

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            key = str(parsed.get("key", "")).strip()
            value = str(parsed.get("value", "")).strip()
            if key or value:
                return key, value

        key_match = re.search(r"key\s*:\s*(.+)", response, re.IGNORECASE)
        value_match = re.search(
            r"value\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL
        )
        return (
            key_match.group(1).strip() if key_match else "",
            value_match.group(1).strip() if value_match else "",
        )

    def _load_documents(self, source_ids: list[str]) -> list[Any]:
        from .models import Document

        documents_by_id = {
            document.id: document
            for document in Document.objects.filter(id__in=source_ids)
        }
        return [
            documents_by_id[source_id]
            for source_id in source_ids
            if source_id in documents_by_id
        ]

    def _get_description_fragments(self, record: Any) -> list[str]:
        metadata_fragments = record.metadata.get("description_fragments", [])
        if isinstance(metadata_fragments, list):
            fragments = [
                str(fragment).strip()
                for fragment in metadata_fragments
                if str(fragment).strip()
            ]
            if fragments:
                return fragments

        description = (record.description or "").strip()
        return [description] if description else []

    def _get_keywords(self, relation: Any) -> list[str]:
        keywords_list = relation.metadata.get("keywords_list", [])
        if isinstance(keywords_list, list):
            values = [
                str(keyword).strip()
                for keyword in keywords_list
                if str(keyword).strip()
            ]
            if values:
                return values

        keywords = relation.metadata.get("keywords", "")
        if not isinstance(keywords, str):
            return []
        return [value.strip() for value in keywords.split(",") if value.strip()]

    def _needs_refresh(self, record: Any, input_hash: str) -> bool:
        return (
            record.profile_input_hash != input_hash
            or not (record.profile_key or "").strip()
            or not (record.profile_value or "").strip()
        )

    def _hash_payload(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return sha256(serialized.encode("utf-8")).hexdigest()

    def _normalize_ids(self, source_ids: list[str] | None) -> list[str]:
        if not source_ids:
            return []
        return [str(source_id) for source_id in source_ids]

    def _normalize_key(self, key: str) -> str:
        return " ".join(key.split())[:255]

    def _normalize_value(self, value: str) -> str:
        return " ".join(value.split())
