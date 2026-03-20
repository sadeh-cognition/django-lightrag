import hashlib
from typing import Any, Dict, List, Tuple
from .models import Document, Entity, Relation
from .storage import LadybugGraphStorage
from .entity_extraction import extract_entities


class KnowledgeGraphBuilder:
    """Handles extraction and persistence of Knowledge Graph entities and relations"""

    def __init__(
        self,
        llm_service: Any,
        tokenizer: Any,
        graph_storage: LadybugGraphStorage,
        config: Dict[str, Any],
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

    def extract_and_persist(self, document: Document):
        """Extract and persist entities and relations from a document"""
        entity_by_name, relation_by_key = self._extract_knowledge_graph(document)
        entity_objects = self._persist_entities(document, entity_by_name)
        self._persist_relations(document, relation_by_key, entity_objects)

    def _extract_knowledge_graph(
        self, document: Document
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        document_payload = {
            document.id: {
                "tokens": self.tokenizer.count_tokens(document.content),
                "content": document.content,
                "full_doc_id": document.id,
                "chunk_order_index": 0,
            }
        }

        global_config = {
            # Adapter for extract_entities which expects a callable
            "llm_model_func": lambda user_prompt, system_prompt=None, history_messages=None, max_tokens=None: (
                self.llm_service.call_llm(
                    user_prompt, system_prompt, history_messages, max_tokens
                )
            ),
            "entity_extract_max_gleaning": self.config.get(
                "ENTITY_EXTRACT_MAX_GLEANING", 1
            ),
            "addon_params": {
                "language": self.config.get("EXTRACTION_LANGUAGE", "English"),
                "entity_types": self.config.get("ENTITY_TYPES", []),
            },
            "tokenizer": self.tokenizer,
            "max_extract_input_tokens": self.config.get(
                "MAX_EXTRACT_INPUT_TOKENS", 12000
            ),
        }

        document_results = extract_entities(document_payload, global_config)

        entity_by_name: Dict[str, Dict[str, Any]] = {}
        relation_by_key: Dict[str, Dict[str, Any]] = {}

        for maybe_nodes, maybe_edges in document_results:
            for entity_name, entity_list in maybe_nodes.items():
                if not entity_list:
                    continue
                best = max(
                    entity_list, key=lambda item: len(item.get("description", "") or "")
                )
                existing = entity_by_name.get(entity_name)
                if existing is None or len(best.get("description", "")) > len(
                    existing.get("description", "")
                ):
                    entity_by_name[entity_name] = best

            for (src_name, tgt_name), relation_list in maybe_edges.items():
                if not relation_list:
                    continue
                best = max(
                    relation_list,
                    key=lambda item: len(item.get("description", "") or ""),
                )
                relation_type = self._relation_type_from_keywords(
                    best.get("keywords", "")
                )
                sorted_key = "::".join(sorted([src_name, tgt_name]) + [relation_type])
                existing = relation_by_key.get(sorted_key)
                if existing is None or len(best.get("description", "")) > len(
                    existing.get("description", "")
                ):
                    relation_by_key[sorted_key] = {
                        **best,
                        "relation_type": relation_type,
                    }

        return entity_by_name, relation_by_key

    def _persist_entities(
        self, document: Document, entity_by_name: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Entity]:
        entity_objects: Dict[str, Entity] = {}
        for entity_name, entity_data in entity_by_name.items():
            entity_type = entity_data.get("entity_type", "other") or "other"
            entity_id = self._generate_id(f"entity:{entity_name}:{entity_type}")
            defaults = {
                "name": entity_name,
                "entity_type": entity_type,
                "description": entity_data.get("description", ""),
                "source_ids": [document.id],
                "metadata": {
                    "source_id": entity_data.get("source_id"),
                    "timestamp": entity_data.get("timestamp"),
                },
            }

            entity, created = Entity.objects.get_or_create(
                id=entity_id, defaults=defaults
            )

            if not created:
                updated = False
                if entity_data.get("description") and len(
                    entity_data["description"]
                ) > len(entity.description or ""):
                    entity.description = entity_data["description"]
                    updated = True
                if document.id not in entity.source_ids:
                    entity.source_ids.append(document.id)
                    updated = True
                if updated:
                    entity.save()

            if created:
                self.graph_storage.add_entity(
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
        self, document: Document, entity_objects: Dict[str, Entity], entity_name: str
    ) -> Entity:
        if entity_name in entity_objects:
            return entity_objects[entity_name]

        entity_type = "other"
        entity_id = self._generate_id(f"entity:{entity_name}:{entity_type}")
        entity, created = Entity.objects.get_or_create(
            id=entity_id,
            defaults={
                "name": entity_name,
                "entity_type": entity_type,
                "description": "",
                "source_ids": [document.id],
                "metadata": {"auto_created": True},
            },
        )
        if created:
            self.graph_storage.add_entity(
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
        document: Document,
        relation_by_key: Dict[str, Dict[str, Any]],
        entity_objects: Dict[str, Entity],
    ) -> None:
        for relation_data in relation_by_key.values():
            src_name = relation_data.get("src_id")
            tgt_name = relation_data.get("tgt_id")
            if not src_name or not tgt_name:
                continue

            source_entity = self._get_or_create_placeholder_entity(
                document, entity_objects, src_name
            )
            target_entity = self._get_or_create_placeholder_entity(
                document, entity_objects, tgt_name
            )

            relation_type = relation_data.get(
                "relation_type"
            ) or self._relation_type_from_keywords(relation_data.get("keywords", ""))
            relation_type = relation_type[:100] if relation_type else "related_to"

            relation_id = self._generate_id(
                f"relation:{min(source_entity.id, target_entity.id)}:"
                f"{max(source_entity.id, target_entity.id)}:{relation_type}"
            )

            defaults = {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relation_type": relation_type,
                "description": relation_data.get("description", ""),
                "source_ids": [document.id],
                "weight": relation_data.get("weight", 1.0),
                "metadata": {
                    "keywords": relation_data.get("keywords", ""),
                    "source_id": relation_data.get("source_id"),
                    "timestamp": relation_data.get("timestamp"),
                },
            }

            relation, created = Relation.objects.get_or_create(
                id=relation_id, defaults=defaults
            )

            if not created:
                updated = False
                if relation_data.get("description") and len(
                    relation_data["description"]
                ) > len(relation.description or ""):
                    relation.description = relation_data["description"]
                    updated = True
                if document.id not in relation.source_ids:
                    relation.source_ids.append(document.id)
                    updated = True
                if updated:
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
