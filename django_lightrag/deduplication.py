import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from django.db import transaction

from .models import Entity, Relation


def normalize_identity_value(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", str(value or "").strip())
    return collapsed.casefold()


def stable_unique_strings(values: list[Any] | tuple[Any, ...]) -> list[str]:
    merged: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def join_fragments(fragments: list[str]) -> str:
    return "\n\n".join(fragment for fragment in fragments if fragment)


def canonical_entity_id(name: str, entity_type: str) -> str:
    payload = f"entity:{name}:{entity_type}"
    return hashlib.md5(payload.encode()).hexdigest()


def canonical_relation_id(
    source_entity_id: str, target_entity_id: str, relation_type: str
) -> str:
    source_id, target_id = sorted([source_entity_id, target_entity_id])
    payload = f"relation:{source_id}:{target_id}:{relation_type}"
    return hashlib.md5(payload.encode()).hexdigest()


def get_description_fragments(record: Entity | Relation) -> list[str]:
    metadata_fragments = record.metadata.get("description_fragments", [])
    if isinstance(metadata_fragments, list):
        fragments = stable_unique_strings(metadata_fragments)
        if fragments:
            return fragments

    description = (record.description or "").strip()
    return [description] if description else []


def get_relation_keywords(record: Relation) -> list[str]:
    keywords_list = record.metadata.get("keywords_list", [])
    if isinstance(keywords_list, list):
        values = stable_unique_strings(keywords_list)
        if values:
            return values

    keywords = record.metadata.get("keywords", "")
    if not isinstance(keywords, str):
        return []
    return stable_unique_strings(keywords.split(","))


def build_entity_group_key(entity: Entity) -> tuple[str, str]:
    return (
        normalize_identity_value(entity.name),
        normalize_identity_value(entity.entity_type),
    )


def build_relation_group_key(
    relation: Relation,
) -> tuple[str, str, str]:
    source_id, target_id = sorted(
        [relation.source_entity_id, relation.target_entity_id]
    )
    return (
        source_id,
        target_id,
        normalize_identity_value(relation.relation_type),
    )


@dataclass
class DeduplicationResult:
    merged_entities: int = 0
    deleted_entities: int = 0
    merged_relations: int = 0
    deleted_relations: int = 0
    surviving_entities: list[Entity] = field(default_factory=list)
    surviving_relations: list[Relation] = field(default_factory=list)

    def as_counts(self) -> dict[str, int]:
        return {
            "merged_entities": self.merged_entities,
            "deleted_entities": self.deleted_entities,
            "merged_relations": self.merged_relations,
            "deleted_relations": self.deleted_relations,
        }


class GraphDeduplicationService:
    def __init__(self, *, graph_storage: Any, vector_storage: Any):
        self.graph_storage = graph_storage
        self.vector_storage = vector_storage

    def deduplicate(
        self,
        *,
        include_entities: bool = True,
        include_relations: bool = True,
        entity_ids: list[str] | None = None,
        relation_ids: list[str] | None = None,
    ) -> DeduplicationResult:
        result = DeduplicationResult()
        deleted_entity_ids: set[str] = set()
        deleted_relation_ids: set[str] = set()
        relation_pairs_to_resync: set[tuple[str, str]] = set()

        target_entity_ids = set(entity_ids or [])
        target_relation_ids = set(relation_ids or [])

        with transaction.atomic():
            if include_entities:
                entity_result = self._deduplicate_entities(target_entity_ids)
                result.merged_entities = entity_result["merged_groups"]
                result.deleted_entities = len(entity_result["deleted_ids"])
                result.surviving_entities = entity_result["survivors"]
                deleted_entity_ids.update(entity_result["deleted_ids"])
                relation_pairs_to_resync.update(entity_result["relation_pairs"])
                target_relation_ids.update(entity_result["relation_ids"])
                target_entity_ids.update(entity_result["survivor_ids"])

            if include_relations:
                relation_result = self._deduplicate_relations(
                    target_relation_ids=target_relation_ids,
                    target_entity_ids=target_entity_ids,
                )
                result.merged_relations = relation_result["merged_groups"]
                result.deleted_relations = len(relation_result["deleted_ids"])
                result.surviving_relations = relation_result["survivors"]
                deleted_relation_ids.update(relation_result["deleted_ids"])
                relation_pairs_to_resync.update(relation_result["relation_pairs"])

        self._delete_duplicate_embeddings(deleted_entity_ids, deleted_relation_ids)
        self._sync_graph_storage(
            surviving_entities=result.surviving_entities,
            deleted_entity_ids=deleted_entity_ids,
            surviving_relations=result.surviving_relations,
            relation_pairs_to_resync=relation_pairs_to_resync,
        )
        return result

    def _deduplicate_entities(self, target_entity_ids: set[str]) -> dict[str, Any]:
        queryset = Entity.objects.all().order_by("created_at", "id")
        entities = list(queryset)
        if target_entity_ids:
            target_keys = {
                build_entity_group_key(entity)
                for entity in entities
                if entity.id in target_entity_ids
            }
            entities = [
                entity
                for entity in entities
                if build_entity_group_key(entity) in target_keys
            ]

        groups: dict[tuple[str, str], list[Entity]] = {}
        for entity in entities:
            groups.setdefault(build_entity_group_key(entity), []).append(entity)

        merged_groups = 0
        deleted_ids: set[str] = set()
        relation_ids: set[str] = set()
        relation_pairs: set[tuple[str, str]] = set()
        survivor_ids: set[str] = set()
        survivors: list[Entity] = []

        for group in groups.values():
            if len(group) < 2:
                continue

            merged_groups += 1
            survivor, removed_entities = self._merge_entity_group(group)
            survivors.append(survivor)
            survivor_ids.add(survivor.id)
            deleted_ids.update(entity.id for entity in removed_entities)

            affected_relations = Relation.objects.filter(
                source_entity_id=survivor.id
            ) | Relation.objects.filter(target_entity_id=survivor.id)
            for relation in affected_relations.distinct():
                relation_ids.add(relation.id)
                relation_pairs.add(
                    tuple(
                        sorted([relation.source_entity_id, relation.target_entity_id])
                    )
                )

        return {
            "merged_groups": merged_groups,
            "deleted_ids": deleted_ids,
            "relation_ids": relation_ids,
            "relation_pairs": relation_pairs,
            "survivor_ids": survivor_ids,
            "survivors": survivors,
        }

    def _merge_entity_group(
        self, entities: list[Entity]
    ) -> tuple[Entity, list[Entity]]:
        ordered_entities = sorted(
            entities, key=lambda entity: (entity.created_at, entity.id)
        )
        template = next(
            (
                entity
                for entity in ordered_entities
                if entity.id == canonical_entity_id(entity.name, entity.entity_type)
            ),
            ordered_entities[0],
        )

        canonical_id = canonical_entity_id(template.name, template.entity_type)
        survivor = next(
            (entity for entity in ordered_entities if entity.id == canonical_id), None
        )
        if survivor is None:
            survivor = Entity(
                id=canonical_id,
                name=template.name,
                entity_type=template.entity_type,
            )

        merged_source_ids = stable_unique_strings(
            [
                source_id
                for entity in ordered_entities
                for source_id in entity.source_ids
            ]
        )
        merged_fragments = stable_unique_strings(
            [
                fragment
                for entity in ordered_entities
                for fragment in get_description_fragments(entity)
            ]
        )

        survivor_metadata = (
            dict(survivor.metadata)
            if survivor.pk and Entity.objects.filter(pk=survivor.pk).exists()
            else dict(template.metadata)
        )
        survivor_metadata["description_fragments"] = merged_fragments

        survivor.name = template.name
        survivor.entity_type = template.entity_type
        survivor.description = join_fragments(merged_fragments)
        survivor.source_ids = merged_source_ids
        survivor.metadata = survivor_metadata
        survivor.save()

        removed_entities = [
            entity for entity in ordered_entities if entity.id != survivor.id
        ]
        for entity in removed_entities:
            Relation.objects.filter(source_entity=entity).update(source_entity=survivor)
            Relation.objects.filter(target_entity=entity).update(target_entity=survivor)

        if removed_entities:
            Entity.objects.filter(
                id__in=[entity.id for entity in removed_entities]
            ).delete()

        survivor.refresh_from_db()
        return survivor, removed_entities

    def _deduplicate_relations(
        self,
        *,
        target_relation_ids: set[str],
        target_entity_ids: set[str],
    ) -> dict[str, Any]:
        queryset = Relation.objects.select_related(
            "source_entity", "target_entity"
        ).order_by("created_at", "id")
        relations = list(queryset)

        if target_relation_ids or target_entity_ids:
            target_keys = {
                build_relation_group_key(relation)
                for relation in relations
                if relation.id in target_relation_ids
                or relation.source_entity_id in target_entity_ids
                or relation.target_entity_id in target_entity_ids
            }
            relations = [
                relation
                for relation in relations
                if build_relation_group_key(relation) in target_keys
            ]

        groups: dict[tuple[str, str, str], list[Relation]] = {}
        for relation in relations:
            groups.setdefault(build_relation_group_key(relation), []).append(relation)

        merged_groups = 0
        deleted_ids: set[str] = set()
        relation_pairs: set[tuple[str, str]] = set()
        survivors: list[Relation] = []

        for group in groups.values():
            if len(group) < 2:
                if target_relation_ids or target_entity_ids:
                    survivor = group[0]
                    survivors.append(survivor)
                    relation_pairs.add(
                        tuple(
                            sorted(
                                [survivor.source_entity_id, survivor.target_entity_id]
                            )
                        )
                    )
                continue

            merged_groups += 1
            survivor, removed_relations = self._merge_relation_group(group)
            survivors.append(survivor)
            relation_pairs.add(
                tuple(sorted([survivor.source_entity_id, survivor.target_entity_id]))
            )
            for relation in removed_relations:
                deleted_ids.add(relation.id)
                relation_pairs.add(
                    tuple(
                        sorted([relation.source_entity_id, relation.target_entity_id])
                    )
                )

        return {
            "merged_groups": merged_groups,
            "deleted_ids": deleted_ids,
            "relation_pairs": relation_pairs,
            "survivors": survivors,
        }

    def _merge_relation_group(
        self, relations: list[Relation]
    ) -> tuple[Relation, list[Relation]]:
        ordered_relations = sorted(
            relations, key=lambda relation: (relation.created_at, relation.id)
        )
        template = ordered_relations[0]
        canonical_id = canonical_relation_id(
            template.source_entity_id,
            template.target_entity_id,
            template.relation_type,
        )
        survivor = next(
            (relation for relation in ordered_relations if relation.id == canonical_id),
            None,
        )
        if survivor is None:
            survivor = Relation(
                id=canonical_id,
                source_entity=template.source_entity,
                target_entity=template.target_entity,
                relation_type=template.relation_type,
            )

        merged_source_ids = stable_unique_strings(
            [
                source_id
                for relation in ordered_relations
                for source_id in relation.source_ids
            ]
        )
        merged_fragments = stable_unique_strings(
            [
                fragment
                for relation in ordered_relations
                for fragment in get_description_fragments(relation)
            ]
        )
        merged_keywords = stable_unique_strings(
            [
                keyword
                for relation in ordered_relations
                for keyword in get_relation_keywords(relation)
            ]
        )

        survivor_metadata = (
            dict(survivor.metadata)
            if survivor.pk and Relation.objects.filter(pk=survivor.pk).exists()
            else dict(template.metadata)
        )
        survivor_metadata["description_fragments"] = merged_fragments
        survivor_metadata["keywords_list"] = merged_keywords
        survivor_metadata["keywords"] = ", ".join(merged_keywords)

        survivor.source_entity = template.source_entity
        survivor.target_entity = template.target_entity
        survivor.relation_type = template.relation_type
        survivor.description = join_fragments(merged_fragments)
        survivor.source_ids = merged_source_ids
        survivor.weight = max(float(relation.weight) for relation in ordered_relations)
        survivor.metadata = survivor_metadata
        survivor.save()

        removed_relations = [
            relation for relation in ordered_relations if relation.id != survivor.id
        ]
        if removed_relations:
            Relation.objects.filter(
                id__in=[relation.id for relation in removed_relations]
            ).delete()

        survivor.refresh_from_db()
        return survivor, removed_relations

    def _delete_duplicate_embeddings(
        self, entity_ids: set[str], relation_ids: set[str]
    ) -> None:
        for entity_id in sorted(entity_ids):
            self.vector_storage.delete_embedding("entity", entity_id)
        for relation_id in sorted(relation_ids):
            self.vector_storage.delete_embedding("relation", relation_id)

    def _sync_graph_storage(
        self,
        *,
        surviving_entities: list[Entity],
        deleted_entity_ids: set[str],
        surviving_relations: list[Relation],
        relation_pairs_to_resync: set[tuple[str, str]],
    ) -> None:
        for entity_id in sorted(deleted_entity_ids):
            self.graph_storage.remove_entity_node(entity_id)

        for entity in surviving_entities:
            self.graph_storage.upsert_entity_node(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "metadata": entity.metadata,
                }
            )

        for source_id, target_id in sorted(relation_pairs_to_resync):
            self.graph_storage.remove_relation_edge(source_id, target_id)

        for relation in surviving_relations:
            self.graph_storage.upsert_relation_edge(
                {
                    "id": relation.id,
                    "source_entity": relation.source_entity_id,
                    "target_entity": relation.target_entity_id,
                    "relation_type": relation.relation_type,
                    "description": relation.description,
                    "metadata": relation.metadata,
                }
            )
