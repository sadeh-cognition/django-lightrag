"""
Storage implementations for LightRAG Django app using LadybugDB and ChromaDB.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import uuid
from datetime import datetime

try:
    import real_ladybug as lb
except ImportError:
    lb = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from django.conf import settings
from .models import Entity, Relation, VectorEmbedding, CacheEntry


class LadybugGraphStorage:
    """LadybugDB implementation for graph storage"""

    def __init__(self):
        self.db_path = self._get_db_path()
        self.conn = None
        self._initialize_connection()

    def _get_db_path(self) -> str:
        """Get the database path"""
        ladybug_settings = getattr(settings, "LADYBUGDB", {})
        base_path = Path(ladybug_settings.get("DATABASE_PATH", "ladybugdb.lbug"))

        if ladybug_settings.get("IN_MEMORY", False):
            return ":memory:"

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
            raise RuntimeError(f"Failed to initialize LadybugDB connection: {e}")

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
                    FROM Document TO Entity
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO (
                    FROM Entity TO Entity
                )
            """)

        except Exception as e:
            # Schema might already exist, ignore
            pass

    def add_entity(self, entity_data: Dict[str, Any]) -> str:
        """Add an entity to the graph"""
        entity_id = entity_data.get("id", str(uuid.uuid4()))

        query = f"""
            INSERT INTO Entity VALUES (
                '{entity_id}',
                '{entity_data["name"]}',
                '{entity_data["entity_type"]}',
                '{entity_data.get("description", "")}',
                '{json.dumps(entity_data.get("metadata", {}))}',
                '{datetime.now().isoformat()}',
                '{datetime.now().isoformat()}'
            )
        """

        try:
            self.conn.execute(query)
            return entity_id
        except Exception as e:
            raise RuntimeError(f"Failed to add entity: {e}")

    def add_relation(self, relation_data: Dict[str, Any]) -> str:
        """Add a relation to the graph"""
        relation_id = relation_data.get("id", str(uuid.uuid4()))

        # First ensure source and target entities exist
        self.add_entity_if_not_exists(relation_data["source_entity"])
        self.add_entity_if_not_exists(relation_data["target_entity"])

        query = f"""
            INSERT INTO RELATED_TO VALUES (
                (SELECT node_id FROM Entity WHERE entity_id = '{relation_data["source_entity"]}'),
                (SELECT node_id FROM Entity WHERE entity_id = '{relation_data["target_entity"]}')
            )
        """

        try:
            self.conn.execute(query)
            return relation_id
        except Exception as e:
            raise RuntimeError(f"Failed to add relation: {e}")

    def add_entity_if_not_exists(self, entity_data: Dict[str, Any]):
        """Add entity only if it doesn't exist"""
        entity_id = entity_data["id"]

        # Check if entity exists
        result = self.conn.execute(f"""
            SELECT entity_id FROM Entity WHERE entity_id = '{entity_id}'
        """)

        if not list(result):
            self.add_entity(entity_data)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID"""
        query = f"""
            SELECT * FROM Entity WHERE entity_id = '{entity_id}'
        """

        try:
            result = self.conn.execute(query)
            row = list(result)
            if row:
                entity_row = row[0]
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
            raise RuntimeError(f"Failed to get entity: {e}")

    def get_relation(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get a relation by source and target entity IDs"""
        query = f"""
            SELECT
                src.entity_id as source_id,
                tgt.entity_id as target_id,
                src.name as source_name,
                tgt.name as target_name
            FROM Entity src
            JOIN RELATED_TO rel ON src.node_id = rel._from
            JOIN Entity tgt ON tgt.node_id = rel._to
            WHERE src.entity_id = '{source_id}' AND tgt.entity_id = '{target_id}'
        """

        try:
            result = self.conn.execute(query)
            row = list(result)
            if row:
                rel_row = row[0]
                return {
                    "source_entity": rel_row[0],
                    "target_entity": rel_row[1],
                    "source_name": rel_row[2],
                    "target_name": rel_row[3],
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get relation: {e}")

    def get_all_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all entities"""
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
            SELECT * FROM Entity {limit_clause}
        """

        try:
            result = self.conn.execute(query)
            entities = []
            for row in result:
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
            raise RuntimeError(f"Failed to get all entities: {e}")

    def get_all_relations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all relations"""
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
            SELECT
                src.entity_id as source_id,
                tgt.entity_id as target_id,
                src.name as source_name,
                tgt.name as target_name,
                src.entity_type as source_type,
                tgt.entity_type as target_type
            FROM Entity src
            JOIN RELATED_TO rel ON src.node_id = rel._from
            JOIN Entity tgt ON tgt.node_id = rel._to
            {limit_clause}
        """

        try:
            result = self.conn.execute(query)
            relations = []
            for row in result:
                relations.append(
                    {
                        "source_entity": row[0],
                        "target_entity": row[1],
                        "source_name": row[2],
                        "target_name": row[3],
                        "source_type": row[4],
                        "target_type": row[5],
                    }
                )
            return relations
        except Exception as e:
            raise RuntimeError(f"Failed to get all relations: {e}")

    def get_entity_neighbors(
        self, entity_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities"""
        if direction == "outgoing":
            query = f"""
                SELECT tgt.entity_id, tgt.name, tgt.entity_type
                FROM Entity src
                JOIN RELATED_TO rel ON src.node_id = rel._from
                JOIN Entity tgt ON tgt.node_id = rel._to
                WHERE src.entity_id = '{entity_id}'
            """
        elif direction == "incoming":
            query = f"""
                SELECT src.entity_id, src.name, src.entity_type
                FROM Entity src
                JOIN RELATED_TO rel ON src.node_id = rel._from
                JOIN Entity tgt ON tgt.node_id = rel._to
                WHERE tgt.entity_id = '{entity_id}'
            """
        else:  # both
            query = f"""
                (SELECT tgt.entity_id, tgt.name, tgt.entity_type, 'outgoing' as direction
                 FROM Entity src
                 JOIN RELATED_TO rel ON src.node_id = rel._from
                 JOIN Entity tgt ON tgt.node_id = rel._to
                 WHERE src.entity_id = '{entity_id}')
                UNION
                (SELECT src.entity_id, src.name, src.entity_type, 'incoming' as direction
                 FROM Entity src
                 JOIN RELATED_TO rel ON src.node_id = rel._from
                 JOIN Entity tgt ON tgt.node_id = rel._to
                 WHERE tgt.entity_id = '{entity_id}')
            """

        try:
            result = self.conn.execute(query)
            neighbors = []
            for row in result:
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
            raise RuntimeError(f"Failed to get entity neighbors: {e}")

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relations"""
        try:
            # Delete relations first
            self.conn.execute(f"""
                DELETE FROM RELATED_TO
                WHERE _from IN (SELECT node_id FROM Entity WHERE entity_id = '{entity_id}')
                OR _to IN (SELECT node_id FROM Entity WHERE entity_id = '{entity_id}')
            """)

            # Delete entity
            self.conn.execute(f"""
                DELETE FROM Entity WHERE entity_id = '{entity_id}'
            """)

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete entity: {e}")

    def delete_relation(self, source_id: str, target_id: str) -> bool:
        """Delete a relation between two entities"""
        try:
            self.conn.execute(f"""
                DELETE FROM RELATED_TO
                WHERE _from IN (SELECT node_id FROM Entity WHERE entity_id = '{source_id}')
                AND _to IN (SELECT node_id FROM Entity WHERE entity_id = '{target_id}')
            """)

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete relation: {e}")

    def close(self):
        """Close the database connection"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass


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

        chroma_settings = getattr(settings, "CHROMADB", {})

        try:
            if chroma_settings.get("IN_MEMORY", False):
                self.client = chromadb.Client()
            else:
                persist_directory = chroma_settings.get(
                    "PERSIST_DIRECTORY", "./chromadb_storage"
                )
                os.makedirs(persist_directory, exist_ok=True)

                self.client = chromadb.PersistentClient(path=persist_directory)

            self._initialize_collections()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

    def _initialize_collections(self):
        """Initialize collections for different vector types"""
        collection_names = ["entities", "relations", "documents"]

        for name in collection_names:
            try:
                collection = self.client.get_or_create_collection(
                    name=name, metadata={"type": name}
                )
                self.collections[name] = collection
            except Exception as e:
                raise RuntimeError(f"Failed to initialize collection {name}: {e}")

    def add_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
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
            result = collection.add(
                embeddings=[embedding], ids=[content_id], metadatas=[metadata]
            )
            return content_id
        except Exception as e:
            raise RuntimeError(f"Failed to add embedding: {e}")

    def get_embedding(self, vector_type: str, content_id: str) -> Optional[List[float]]:
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
            raise RuntimeError(f"Failed to get embedding: {e}")

    def search_similar(
        self,
        vector_type: str,
        query_embedding: List[float],
        top_k: int = 10,
        where: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]

        try:
            where_clause = {}
            if where:
                where_clause.update(where)

            result = collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_clause
            )

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
            raise RuntimeError(f"Failed to search similar vectors: {e}")

    def delete_embedding(self, vector_type: str, content_id: str) -> bool:
        """Delete a vector embedding"""
        if vector_type not in self.collections:
            raise ValueError(f"Invalid vector type: {vector_type}")

        collection = self.collections[vector_type]

        try:
            collection.delete(ids=[content_id])
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete embedding: {e}")

    def update_embedding(
        self,
        vector_type: str,
        content_id: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Update a vector embedding"""
        self.delete_embedding(vector_type, content_id)
        self.add_embedding(vector_type, content_id, embedding, metadata)
        return True

    def close(self):
        """Close the ChromaDB client"""
        if self.client:
            try:
                # ChromaDB doesn't have explicit close method for persistent client
                pass
            except:
                pass
