import hashlib
from typing import Any

from .deduplication import canonical_entity_id, canonical_relation_id
from .entity_extraction import extract_entities
from .models import Document, Entity, Relation
from .storage import LadybugGraphStorage


class KnowledgeGraphBuilder:
    """Handles extraction and persistence of knowledge graph entities and relations."""

    def __init__(
        self,
        llm_service: Any,
        tokenizer: Any,
        graph_storage: LadybugGraphStorage,
        config: dict[str, Any],
    ):
        self.llm_service = llm_service
        self.tokenizer = tokenizer
        self.graph_storage = graph_storage
        self.config = config

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def _relation_type_from_keywords(self, keywords: str) -> str:
        if not keywords:
            return "related_to"
        primary = keywords.split(",")[0].strip()
        return primary[:100] if primary else "related_to"

    def extract_and_persist(
        self, document: Document
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract and persist canonical entities and relations from a document."""
        entity_by_name, relation_by_key = self._extract_knowledge_graph(document)
        entity_objects = self._persist_entities(entity_by_name)
        relation_objects = self._persist_relations(relation_by_key, entity_objects)
        return list(entity_objects.values()), relation_objects

    def _extract_knowledge_graph(
        self, document: Document
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        def llm_model_func(
            user_prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            max_tokens: int | None = None,
        ) -> str:
            return self.llm_service.call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                max_tokens=max_tokens,
            )

        document_payload = {
            document.id: {
                "tokens": self.tokenizer.count_tokens(document.content),
                "content": document.content,
                "full_doc_id": document.id,
                "chunk_order_index": 0,
            }
        }

        document_results = extract_entities(
            document_payload,
            llm_callable=llm_model_func,
            entity_extract_max_gleaning=self.config.get(
                "ENTITY_EXTRACT_MAX_GLEANING", 1
            ),
            language=self.config.get("EXTRACTION_LANGUAGE", "English"),
            entity_types=self.config.get("ENTITY_TYPES", []),
            tokenizer=self.tokenizer,
            max_extract_input_tokens=self.config.get("MAX_EXTRACT_INPUT_TOKENS", 12000),
        )
        entity_by_name: dict[str, dict[str, Any]] = {}
        relation_by_key: dict[str, dict[str, Any]] = {}

        for maybe_nodes, maybe_edges in document_results:
            for entity_name, entity_list in maybe_nodes.items():
                if not entity_list:
                    continue

                selected_entity = max(
                    entity_list, key=lambda item: len(item.get("description", "") or "")
                )
                entity_payload = entity_by_name.setdefault(
                    entity_name,
                    {
                        "entity_name": entity_name,
                        "entity_type": selected_entity.get("entity_type", "other")
                        or "other",
                        "descriptions": [],
                        "source_ids": [],
                        "metadata": {},
                    },
                )
                entity_payload["entity_type"] = (
                    selected_entity.get("entity_type", entity_payload["entity_type"])
                    or "other"
                )
                entity_payload["descriptions"] = self._merge_strings(
                    entity_payload["descriptions"],
                    [item.get("description", "") for item in entity_list],
                )
                entity_payload["source_ids"] = self._merge_strings(
                    entity_payload["source_ids"],
                    [item.get("source_id", "") for item in entity_list],
                )
                entity_payload["metadata"] = {
                    "source_id": selected_entity.get("source_id"),
                    "timestamp": selected_entity.get("timestamp"),
                }

            for (src_name, tgt_name), relation_list in maybe_edges.items():
                if not relation_list:
                    continue

                selected_relation = max(
                    relation_list,
                    key=lambda item: len(item.get("description", "") or ""),
                )
                relation_type = self._relation_type_from_keywords(
                    selected_relation.get("keywords", "")
                )
                sorted_key = "::".join(sorted([src_name, tgt_name]) + [relation_type])

                relation_payload = relation_by_key.setdefault(
                    sorted_key,
                    {
                        "src_id": src_name,
                        "tgt_id": tgt_name,
                        "relation_type": relation_type,
                        "descriptions": [],
                        "keywords_list": [],
                        "source_ids": [],
                        "weight": selected_relation.get("weight", 1.0),
                        "metadata": {},
                    },
                )
                relation_payload["descriptions"] = self._merge_strings(
                    relation_payload["descriptions"],
                    [item.get("description", "") for item in relation_list],
                )
                relation_payload["keywords_list"] = self._merge_strings(
                    relation_payload["keywords_list"],
                    [
                        keyword
                        for item in relation_list
                        for keyword in self._split_keywords(item.get("keywords", ""))
                    ],
                )
                relation_payload["source_ids"] = self._merge_strings(
                    relation_payload["source_ids"],
                    [item.get("source_id", "") for item in relation_list],
                )
                relation_payload["weight"] = max(
                    relation_payload["weight"],
                    *[float(item.get("weight", 1.0)) for item in relation_list],
                )
                relation_payload["metadata"] = {
                    "source_id": selected_relation.get("source_id"),
                    "timestamp": selected_relation.get("timestamp"),
                }

        return entity_by_name, relation_by_key

    def _persist_entities(
        self, entity_by_name: dict[str, dict[str, Any]]
    ) -> dict[str, Entity]:
        entity_objects: dict[str, Entity] = {}

        for entity_name, entity_data in entity_by_name.items():
            entity_type = entity_data.get("entity_type", "other") or "other"
            entity_id = canonical_entity_id(entity_name, entity_type)
            description_fragments = self._normalize_string_list(
                entity_data.get("descriptions", [])
            )
            source_ids = self._normalize_string_list(entity_data.get("source_ids", []))
            description = self._join_fragments(description_fragments)

            defaults = {
                "name": entity_name,
                "entity_type": entity_type,
                "description": description,
                "source_ids": source_ids,
                "metadata": {
                    **entity_data.get("metadata", {}),
                    "description_fragments": description_fragments,
                },
            }

            entity, created = Entity.objects.get_or_create(
                id=entity_id, defaults=defaults
            )

            if not created:
                metadata = dict(entity.metadata)
                existing_fragments = self._get_description_fragments(entity)
                merged_fragments = self._merge_strings(
                    existing_fragments, description_fragments
                )
                merged_source_ids = self._merge_strings(entity.source_ids, source_ids)

                entity.description = self._join_fragments(merged_fragments)
                entity.source_ids = merged_source_ids
                metadata.update(entity_data.get("metadata", {}))
                metadata["description_fragments"] = merged_fragments
                entity.metadata = metadata
                entity.save()

            self.graph_storage.add_entity_if_not_exists(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "metadata": entity.metadata,
                }
            )
            entity_objects[entity_name] = entity

        return entity_objects

    def _get_or_create_placeholder_entity(
        self, entity_objects: dict[str, Entity], entity_name: str, source_ids: list[str]
    ) -> Entity:
        if entity_name in entity_objects:
            return entity_objects[entity_name]

        existing_other_entity = (
            Entity.objects.filter(name=entity_name, entity_type="other")
            .order_by("-updated_at")
            .first()
        )
        if existing_other_entity is not None:
            merged_source_ids = self._merge_strings(
                existing_other_entity.source_ids, source_ids
            )
            if merged_source_ids != existing_other_entity.source_ids:
                existing_other_entity.source_ids = merged_source_ids
                existing_other_entity.save(update_fields=["source_ids", "updated_at"])
            entity_objects[entity_name] = existing_other_entity
            return existing_other_entity

        typed_entities = list(
            Entity.objects.filter(name=entity_name)
            .exclude(entity_type="other")
            .order_by("-updated_at")[:2]
        )
        if len(typed_entities) == 1:
            typed_entity = typed_entities[0]
            merged_source_ids = self._merge_strings(typed_entity.source_ids, source_ids)
            if merged_source_ids != typed_entity.source_ids:
                typed_entity.source_ids = merged_source_ids
                typed_entity.save(update_fields=["source_ids", "updated_at"])
            entity_objects[entity_name] = typed_entity
            return typed_entity

        entity_type = "other"
        entity_id = canonical_entity_id(entity_name, entity_type)
        entity, _ = Entity.objects.get_or_create(
            id=entity_id,
            defaults={
                "name": entity_name,
                "entity_type": entity_type,
                "description": "",
                "source_ids": self._normalize_string_list(source_ids),
                "metadata": {"auto_created": True, "description_fragments": []},
            },
        )
        self.graph_storage.add_entity_if_not_exists(
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "metadata": entity.metadata,
            }
        )
        entity_objects[entity_name] = entity
        return entity

    def _persist_relations(
        self,
        relation_by_key: dict[str, dict[str, Any]],
        entity_objects: dict[str, Entity],
    ) -> list[Relation]:
        relation_objects: list[Relation] = []

        for relation_data in relation_by_key.values():
            src_name = relation_data.get("src_id")
            tgt_name = relation_data.get("tgt_id")
            if not src_name or not tgt_name:
                continue

            source_ids = self._normalize_string_list(
                relation_data.get("source_ids", [])
            )
            source_entity = self._get_or_create_placeholder_entity(
                entity_objects, src_name, source_ids
            )
            target_entity = self._get_or_create_placeholder_entity(
                entity_objects, tgt_name, source_ids
            )

            relation_type = relation_data.get("relation_type") or "related_to"
            relation_type = relation_type[:100] if relation_type else "related_to"
            relation_id = canonical_relation_id(
                source_entity.id,
                target_entity.id,
                relation_type,
            )

            description_fragments = self._normalize_string_list(
                relation_data.get("descriptions", [])
            )
            keywords_list = self._normalize_string_list(
                relation_data.get("keywords_list", [])
            )
            description = self._join_fragments(description_fragments)

            defaults = {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relation_type": relation_type,
                "description": description,
                "source_ids": source_ids,
                "weight": relation_data.get("weight", 1.0),
                "metadata": {
                    **relation_data.get("metadata", {}),
                    "keywords": ", ".join(keywords_list),
                    "keywords_list": keywords_list,
                    "description_fragments": description_fragments,
                },
            }

            relation, created = Relation.objects.get_or_create(
                id=relation_id, defaults=defaults
            )

            if not created:
                metadata = dict(relation.metadata)
                existing_fragments = self._get_description_fragments(relation)
                merged_fragments = self._merge_strings(
                    existing_fragments, description_fragments
                )
                merged_source_ids = self._merge_strings(relation.source_ids, source_ids)
                merged_keywords = self._merge_strings(
                    self._normalize_string_list(metadata.get("keywords_list", [])),
                    keywords_list,
                )

                relation.description = self._join_fragments(merged_fragments)
                relation.source_ids = merged_source_ids
                relation.weight = max(relation.weight, relation_data.get("weight", 1.0))
                metadata.update(relation_data.get("metadata", {}))
                metadata["keywords_list"] = merged_keywords
                metadata["keywords"] = ", ".join(merged_keywords)
                metadata["description_fragments"] = merged_fragments
                relation.metadata = metadata
                relation.save()

            if created:
                self.graph_storage.add_relation(
                    {
                        "id": relation.id,
                        "source_entity": relation.source_entity.id,
                        "target_entity": relation.target_entity.id,
                        "relation_type": relation.relation_type,
                        "description": relation.description,
                        "metadata": relation.metadata,
                    }
                )

            relation_objects.append(relation)

        return relation_objects

    def _get_description_fragments(self, record: Entity | Relation) -> list[str]:
        metadata_fragments = record.metadata.get("description_fragments", [])
        if isinstance(metadata_fragments, list):
            fragments = self._normalize_string_list(metadata_fragments)
            if fragments:
                return fragments

        description = (record.description or "").strip()
        return [description] if description else []

    def _split_keywords(self, keywords: str) -> list[str]:
        if not keywords:
            return []
        return [value.strip() for value in keywords.split(",") if value.strip()]

    def _normalize_string_list(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized

    def _merge_strings(
        self, existing_values: list[Any], new_values: list[Any]
    ) -> list[str]:
        merged: list[str] = []
        for value in [*existing_values, *new_values]:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
        return merged

    def _join_fragments(self, fragments: list[str]) -> str:
        return "\n\n".join(fragment for fragment in fragments if fragment)
