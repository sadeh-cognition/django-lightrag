"""
Storage implementations for LightRAG Django app using LadybugDB and ChromaDB.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import real_ladybug as lb
except ImportError:
    lb = None

try:
    import chromadb
except ImportError:
    chromadb = None

from django.conf import settings

from .config import get_ladybug_settings


class LadybugGraphStorage:
    """LadybugDB implementation for graph storage"""

    def __init__(self):
        self.db_path = self._get_db_path()
        self.conn = None
        self._initialize_connection()

    def _get_db_path(self) -> str:
        """Get the database path"""
        ladybug_settings = get_ladybug_settings()

        if ladybug_settings.get("IN_MEMORY", False):
            return ":memory:"

        base_path = Path(ladybug_settings["DATABASE_PATH"])

        # Use single database file
        return str(base_path)

    def _initialize_connection(self):
        """Initialize LadybugDB connection"""
        if lb is None:
            raise ImportError(
                "real_ladybug is not installed. Install with: pip install real-ladybug"
            )

        try:
            db = lb.Database(self.db_path)
            self.conn = lb.Connection(db)
            self._create_schema()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LadybugDB connection: {e}") from e

    def _create_schema(self):
        """Create the graph schema if it doesn't exist"""
        try:
            # Create node tables
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    entity_id STRING PRIMARY KEY,
                    name STRING,
                    entity_type STRING,
                    description STRING,
                    metadata STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Document (
                    document_id STRING PRIMARY KEY,
                    title STRING,
                    content STRING,
                    metadata STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            # Create relationship tables
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS (
                    FROM Document TO Entity,
                    created_at STRING,
                    updated_at STRING
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO (
                    FROM Entity TO Entity,
                    relation_id STRING,
                    relation_type STRING,
                    description STRING,
                    metadata STRING,
                    created_at STRING,
                    updated_at STRING
                )
            """)

        except Exception:
            # Schema might already exist, ignore
            pass

    def _quote(self, value: Any) -> str:
        return json.dumps("" if value is None else str(value))

    def _consume_rows(self, result: Any) -> list[list[Any]]:
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def _entity_pattern(self, alias: str, entity_id: str) -> str:
        return f"({alias}:Entity {{entity_id: {self._quote(entity_id)}}})"

    def add_entity(self, entity_data: dict[str, Any]) -> str:
        """Add an entity to the graph"""
        entity_id = entity_data.get("id", str(uuid.uuid4()))
        now = datetime.now().isoformat()

        query = (
            "CREATE (:Entity {"
            f"entity_id: {self._quote(entity_id)}, "
            f"name: {self._quote(entity_data['name'])}, "
            f"entity_type: {self._quote(entity_data['entity_type'])}, "
            f"description: {self._quote(entity_data.get('description', ''))}, "
            f"metadata: {self._quote(json.dumps(entity_data.get('metadata', {})))}, "
            f"created_at: {self._quote(now)}, "
            f"updated_at: {self._quote(now)}"
            "})"
        )

        try:
            self.conn.execute(query)
            return entity_id
        except Exception as e:
            raise RuntimeError(f"Failed to add entity: {e}") from e

    def upsert_entity_node(self, entity_data: dict[str, Any]) -> str:
        entity_id = entity_data["id"]
        self.remove_entity_node(entity_id)
        return self.add_entity(entity_data)

    def add_relation(self, relation_data: dict[str, Any]) -> str:
        """Add a relation to the graph"""
        relation_id = relation_data.get("id", str(uuid.uuid4()))

        # First ensure source and target entities exist
        source_ref = relation_data["source_entity"]
        target_ref = relation_data["target_entity"]

        source_id = source_ref["id"] if isinstance(source_ref, dict) else source_ref
        target_id = target_ref["id"] if isinstance(target_ref, dict) else target_ref

        self.add_entity_if_not_exists(source_ref)
        self.add_entity_if_not_exists(target_ref)
        now = datetime.now().isoformat()

        query = (
            f"MATCH (src:Entity {{entity_id: {self._quote(source_id)}}}), "
            f"(tgt:Entity {{entity_id: {self._quote(target_id)}}}) "
            "CREATE (src)-[:RELATED_TO {"
            f"relation_id: {self._quote(relation_id)}, "
            f"relation_type: {self._quote(relation_data.get('relation_type', 'related_to'))}, "
            f"description: {self._quote(relation_data.get('description', ''))}, "
            f"metadata: {self._quote(json.dumps(relation_data.get('metadata', {})))}, "
            f"created_at: {self._quote(now)}, "
            f"updated_at: {self._quote(now)}"
            "}]->(tgt)"
        )

        try:
            self.conn.execute(query)
            return relation_id
        except Exception as e:
            raise RuntimeError(f"Failed to add relation: {e}") from e

    def upsert_relation_edge(self, relation_data: dict[str, Any]) -> str:
        source_ref = relation_data["source_entity"]
        target_ref = relation_data["target_entity"]
        source_id = source_ref["id"] if isinstance(source_ref, dict) else source_ref
        target_id = target_ref["id"] if isinstance(target_ref, dict) else target_ref
        self.remove_relation_edge(source_id, target_id)
        return self.add_relation(relation_data)

    def add_entity_if_not_exists(self, entity_data: dict[str, Any] | str):
        """Add entity only if it doesn't exist"""
        if isinstance(entity_data, str):
            entity_payload = {
                "id": entity_data,
                "name": entity_data,
                "entity_type": "unknown",
                "description": "",
                "metadata": {},
            }
        else:
            entity_payload = entity_data

        entity_id = entity_payload["id"]

        result = self.conn.execute(
            f"MATCH {self._entity_pattern('e', entity_id)} RETURN e.entity_id LIMIT 1"
        )

        if not self._consume_rows(result):
            self.add_entity(entity_payload)

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get an entity by ID"""
        query = (
            f"MATCH {self._entity_pattern('e', entity_id)} "
            "RETURN e.entity_id, e.name, e.entity_type, e.description, "
            "e.metadata, e.created_at, e.updated_at LIMIT 1"
        )

        try:
            rows = self._consume_rows(self.conn.execute(query))
            if rows:
                entity_row = rows[0]
                return {
                    "id": entity_row[0],
                    "name": entity_row[1],
                    "entity_type": entity_row[2],
                    "description": entity_row[3],
                    "metadata": json.loads(entity_row[4]) if entity_row[4] else {},
                    "created_at": entity_row[5],
                    "updated_at": entity_row[6],
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get entity: {e}") from e

    def get_relation(self, source_id: str, target_id: str) -> dict[str, Any] | None:
        """Get a relation by source and target entity IDs"""
        query = (
            f"MATCH {self._entity_pattern('src', source_id)}"
            f"-[rel:RELATED_TO]->{self._entity_pattern('tgt', target_id)} "
            "RETURN src.entity_id, tgt.entity_id, src.name, tgt.name, "
            "rel.relation_id, rel.relation_type, rel.description, rel.metadata, "
            "rel.created_at, rel.updated_at LIMIT 1"
        )

        try:
            rows = self._consume_rows(self.conn.execute(query))
            if rows:
                rel_row = rows[0]
                return {
                    "source_entity": rel_row[0],
                    "target_entity": rel_row[1],
                    "source_name": rel_row[2],
                    "target_name": rel_row[3],
                    "id": rel_row[4],
                    "relation_type": rel_row[5],
                    "description": rel_row[6],
                    "metadata": json.loads(rel_row[7]) if rel_row[7] else {},
                    "created_at": rel_row[8],
                    "updated_at": rel_row[9],
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get relation: {e}") from e

    def get_all_entities(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get all entities"""
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = (
            "MATCH (e:Entity) "
            "RETURN e.entity_id, e.name, e.entity_type, e.description, "
            "e.metadata, e.created_at, e.updated_at"
            f"{limit_clause}"
        )

        try:
            result = self.conn.execute(query)
            entities = []
            for row in self._consume_rows(result):
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "created_at": row[5],
                        "updated_at": row[6],
                    }
                )
            return entities
        except Exception as e:
            raise RuntimeError(f"Failed to get all entities: {e}") from e

    def get_all_relations(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get all relations"""
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = (
            "MATCH (src:Entity)-[rel:RELATED_TO]->(tgt:Entity) "
            "RETURN src.entity_id, tgt.entity_id, src.name, tgt.name, "
            "src.entity_type, tgt.entity_type, rel.relation_id, rel.relation_type, "
            "rel.description, rel.metadata, rel.created_at, rel.updated_at"
            f"{limit_clause}"
        )

        try:
            result = self.conn.execute(query)
            relations = []
            for row in self._consume_rows(result):
                relations.append(
                    {
                        "source_entity": row[0],
                        "target_entity": row[1],
                        "source_name": row[2],
                        "target_name": row[3],
                        "source_type": row[4],
                        "target_type": row[5],
                        "id": row[6],
                        "relation_type": row[7],
                        "description": row[8],
                        "metadata": json.loads(row[9]) if row[9] else {},
                        "created_at": row[10],
                        "updated_at": row[11],
                    }
                )
            return relations
        except Exception as e:
            raise RuntimeError(f"Failed to get all relations: {e}") from e

    def get_entity_neighbors(
        self, entity_id: str, direction: str = "both"
    ) -> list[dict[str, Any]]:
        """Get neighboring entities"""
        if direction == "outgoing":
            query = (
                f"MATCH {self._entity_pattern('src', entity_id)}"
                "-[:RELATED_TO]->(tgt:Entity) "
                "RETURN tgt.entity_id, tgt.name, tgt.entity_type"
            )
        elif direction == "incoming":
            query = (
                f"MATCH (src:Entity)-[:RELATED_TO]->"
                f"{self._entity_pattern('tgt', entity_id)} "
                "RETURN src.entity_id, src.name, src.entity_type"
            )
        else:  # both
            query = (
                f"MATCH {self._entity_pattern('src', entity_id)}"
                "-[:RELATED_TO]->(tgt:Entity) "
                "RETURN tgt.entity_id, tgt.name, tgt.entity_type, 'outgoing' "
                "UNION "
                f"MATCH (src:Entity)-[:RELATED_TO]->"
                f"{self._entity_pattern('tgt', entity_id)} "
                "RETURN src.entity_id, src.name, src.entity_type, 'incoming'"
            )

        try:
            result = self.conn.execute(query)
            neighbors = []
            for row in self._consume_rows(result):
                if direction == "both":
                    neighbors.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "entity_type": row[2],
                            "direction": row[3],
                        }
                    )
                else:
                    neighbors.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "entity_type": row[2],
                        }
                    )
            return neighbors
        except Exception as e:
            raise RuntimeError(f"Failed to get entity neighbors: {e}") from e

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relations"""
        try:
            self.conn.execute(
                f"MATCH {self._entity_pattern('e', entity_id)} DETACH DELETE e"
            )

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete entity: {e}") from e

    def remove_entity_node(self, entity_id: str) -> bool:
        return self.delete_entity(entity_id)

    def delete_relation(self, source_id: str, target_id: str) -> bool:
        """Delete a relation between two entities"""
        try:
            self.conn.execute(
                f"MATCH {self._entity_pattern('src', source_id)}"
                "-[rel:RELATED_TO]->"
                f"{self._entity_pattern('tgt', target_id)} "
                "DELETE rel"
            )

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete relation: {e}") from e

    def remove_relation_edge(self, source_id: str, target_id: str) -> bool:
        return self.delete_relation(source_id, target_id)

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()


class ChromaVectorStorage:
    """ChromaDB implementation for vector storage"""

    def __init__(self):
        self.client = None
        self.collections = {}
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client"""
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            )

        try:
            if getattr(settings, "CHROMADB_IN_MEMORY", False):
                self.client = chromadb.Client()
            else:
                persist_directory = getattr(settings, "CHROMADB_DIR", None)
                if not persist_directory:
                    raise RuntimeError(
                        "CHROMADB_DIR must be set in settings.py for persistent storage."
                    )
                os.makedirs(persist_directory, exist_ok=True)

                self.client = chromadb.PersistentClient(path=str(persist_directory))

            self._initialize_collections()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}") from e

    def _initialize_collections(self):
        """Initialize collections for different vector types"""
        collection_map = {
            "entity": "entity_django_lightrag",
            "relation": "relation_django_lightrag",
            "document": "document_django_lightrag",
        }

        for vector_type, name in collection_map.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=name, metadata={"type": name}
                )
                self.collections[vector_type] = collection
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize collection {name}: {e}"
                ) from e

    def add_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: list[float],
        metadata: dict[str, Any] = None,
    ) -> str:
        """Add a vector embedding"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]
        metadata = metadata or {}
        metadata.update(
            {
                "vector_type": vector_type,
                "content_id": content_id,
                "created_at": datetime.now().isoformat(),
            }
        )

        try:
            collection.add(
                embeddings=[embedding], ids=[content_id], metadatas=[metadata]
            )
            return content_id
        except Exception as e:
            raise RuntimeError(f"Failed to add embedding: {e}") from e

    def upsert_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: list[float],
        metadata: dict[str, Any] = None,
    ) -> str:
        """Create or replace a vector embedding."""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]
        metadata = metadata or {}
        metadata.update(
            {
                "vector_type": vector_type,
                "content_id": content_id,
                "created_at": datetime.now().isoformat(),
            }
        )

        try:
            collection.upsert(
                embeddings=[embedding], ids=[content_id], metadatas=[metadata]
            )
            return content_id
        except Exception as e:
            raise RuntimeError(f"Failed to upsert embedding: {e}") from e

    def get_embedding(self, vector_type: str, content_id: str) -> list[float] | None:
        """Get a vector embedding by ID"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]

        try:
            result = collection.get(ids=[content_id])
            if result["embeddings"]:
                return result["embeddings"][0]
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}") from e

    def search_similar(
        self,
        vector_type: str,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if where:
                query_kwargs["where"] = dict(where)

            result = collection.query(**query_kwargs)

            similar_items = []
            if result["ids"] and result["ids"][0]:
                for i, item_id in enumerate(result["ids"][0]):
                    similar_items.append(
                        {
                            "id": item_id,
                            "score": result["distances"][0][i]
                            if result["distances"]
                            else 0.0,
                            "metadata": result["metadatas"][0][i]
                            if result["metadatas"]
                            else {},
                        }
                    )

            return similar_items
        except Exception as e:
            raise RuntimeError(f"Failed to search similar vectors: {e}") from e

    def delete_embedding(self, vector_type: str, content_id: str) -> bool:
        """Delete a vector embedding"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]

        try:
            collection.delete(ids=[content_id])
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete embedding: {e}") from e

    def update_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: list[float],
        metadata: dict[str, Any] = None,
    ) -> bool:
        """Update a vector embedding"""
        self.upsert_embedding(vector_type, content_id, embedding, metadata)
        return True

    def close(self):
        """No close for ChromaDB"""
