"""
Tests for LightRAG API endpoints.
"""

import json
from unittest.mock import Mock, patch
from django.test import TestCase, Client
from django.contrib.auth.models import User

from lightrag_app.models import Workspace, Document


class LightRAGAPITest(TestCase):
    """Test cases for LightRAG API endpoints"""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.workspace = Workspace.objects.create(
            name="test-workspace", description="Test workspace", created_by=self.user
        )

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/api/lightrag/health/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "lightrag-django")

    def test_create_workspace(self):
        """Test workspace creation endpoint"""
        workspace_data = {
            "name": "new-workspace",
            "description": "A new test workspace",
            "is_active": True,
        }

        response = self.client.post(
            "/api/lightrag/workspaces/",
            data=json.dumps(workspace_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.content)
        self.assertEqual(data["name"], "new-workspace")
        self.assertEqual(data["description"], "A new test workspace")
        self.assertTrue(data["is_active"])

    def test_create_workspace_duplicate_name(self):
        """Test workspace creation with duplicate name"""
        workspace_data = {
            "name": "test-workspace",  # Already exists
            "description": "Duplicate workspace",
        }

        response = self.client.post(
            "/api/lightrag/workspaces/",
            data=json.dumps(workspace_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_list_workspaces(self):
        """Test listing workspaces endpoint"""
        response = self.client.get("/api/lightrag/workspaces/")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)  # At least our test workspace

        # Check structure of returned data
        workspace = data[0]
        self.assertIn("id", workspace)
        self.assertIn("name", workspace)
        self.assertIn("description", workspace)
        self.assertIn("created_by", workspace)
        self.assertIn("created_at", workspace)
        self.assertIn("updated_at", workspace)
        self.assertIn("is_active", workspace)

    def test_get_workspace(self):
        """Test getting a specific workspace"""
        response = self.client.get(f"/api/lightrag/workspaces/{self.workspace.id}/")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["id"], self.workspace.id)
        self.assertEqual(data["name"], self.workspace.name)
        self.assertEqual(data["description"], self.workspace.description)

    def test_get_nonexistent_workspace(self):
        """Test getting a non-existent workspace"""
        response = self.client.get("/api/lightrag/workspaces/99999/")
        self.assertEqual(response.status_code, 404)

    @patch("lightrag_app.views.LightRAGCore")
    def test_ingest_document(self, mock_core_class):
        """Test document ingestion endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.ingest_document = Mock(return_value="test-doc-id")

        document_data = {
            "content": "This is a test document content.",
            "title": "Test Document",
            "file_path": "/path/to/test.txt",
            "track_id": "test-track-123",
            "metadata": {"key": "value"},
        }

        response = self.client.post(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/ingest",
            data=json.dumps(document_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.content)
        self.assertEqual(data["document_id"], "test-doc-id")
        self.assertIn("message", data)

    @patch("lightrag_app.views.LightRAGCore")
    def test_ingest_document_nonexistent_workspace(self, mock_core_class):
        """Test document ingestion with non-existent workspace"""
        document_data = {"content": "Test content", "title": "Test Document"}

        response = self.client.post(
            "/api/lightrag/workspaces/nonexistent/documents/ingest",
            data=json.dumps(document_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)
        data = json.loads(response.content)
        self.assertEqual(data["error"], "workspace_not_found")

    @patch("lightrag_app.views.LightRAGCore")
    def test_ingest_document_failure(self, mock_core_class):
        """Test document ingestion failure"""
        # Mock the LightRAGCore to raise an exception
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.ingest_document = Mock(side_effect=Exception("Ingestion failed"))

        document_data = {"content": "Test content", "title": "Test Document"}

        response = self.client.post(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/ingest",
            data=json.dumps(document_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertEqual(data["error"], "ingestion_failed")

    @patch("lightrag_app.views.LightRAGCore")
    def test_list_documents(self, mock_core_class):
        """Test listing documents endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.list_documents = Mock(
            return_value=[
                {
                    "id": "doc1",
                    "title": "Document 1",
                    "status": "processed",
                    "chunks_count": 5,
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                },
                {
                    "id": "doc2",
                    "title": "Document 2",
                    "status": "pending",
                    "chunks_count": 0,
                    "created_at": "2023-01-02T00:00:00Z",
                    "updated_at": "2023-01-02T00:00:00Z",
                },
            ]
        )

        response = self.client.get(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

        # Check document structure
        doc = data[0]
        self.assertIn("id", doc)
        self.assertIn("title", doc)
        self.assertIn("status", doc)
        self.assertIn("chunks_count", doc)
        self.assertIn("created_at", doc)
        self.assertIn("updated_at", doc)

    @patch("lightrag_app.views.LightRAGCore")
    def test_get_document_status(self, mock_core_class):
        """Test getting document status endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.get_document_status = Mock(
            return_value={
                "document_id": "test-doc-123",
                "title": "Test Document",
                "status": "processed",
                "chunks_count": 5,
                "chunks_list": ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"],
                "error_message": "",
                "started_at": "2023-01-01T00:00:00Z",
                "completed_at": "2023-01-01T00:05:00Z",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:05:00Z",
            }
        )

        response = self.client.get(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/test-doc-123/status"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["document_id"], "test-doc-123")
        self.assertEqual(data["status"], "processed")
        self.assertEqual(data["chunks_count"], 5)

    @patch("lightrag_app.views.LightRAGCore")
    def test_query_rag(self, mock_core_class):
        """Test RAG query endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core

        # Mock query result
        mock_result = Mock()
        mock_result.response = "This is a test response."
        mock_result.sources = [
            {
                "type": "chunk",
                "id": "chunk1",
                "content": "Relevant chunk content...",
                "document_title": "Test Document",
            }
        ]
        mock_result.context = {
            "chunks": [],
            "entities": [],
            "relations": [],
            "total_tokens": 100,
        }
        mock_result.query_time = 0.5
        mock_result.tokens_used = 50

        mock_core.query = Mock(return_value=mock_result)

        query_data = {
            "query": "What is this about?",
            "param": {
                "mode": "hybrid",
                "top_k": 5,
                "max_tokens": 4000,
                "temperature": 0.1,
            },
        }

        response = self.client.post(
            f"/api/lightrag/workspaces/{self.workspace.name}/query",
            data=json.dumps(query_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["response"], "This is a test response.")
        self.assertIsInstance(data["sources"], list)
        self.assertIsInstance(data["context"], dict)
        self.assertIsInstance(data["query_time"], float)
        self.assertIsInstance(data["tokens_used"], int)

    @patch("lightrag_app.views.LightRAGCore")
    def test_delete_document(self, mock_core_class):
        """Test document deletion endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.delete_document = Mock(return_value=True)

        response = self.client.delete(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/test-doc-123"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data["success"])
        self.assertIn("message", data)

    @patch("lightrag_app.views.LightRAGCore")
    def test_delete_nonexistent_document(self, mock_core_class):
        """Test deletion of non-existent document"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.delete_document = Mock(return_value=False)

        response = self.client.delete(
            f"/api/lightrag/workspaces/{self.workspace.name}/documents/nonexistent-doc"
        )

        self.assertEqual(response.status_code, 404)
        data = json.loads(response.content)
        self.assertEqual(data["error"], "document_not_found")

    @patch("lightrag_app.views.LightRAGCore")
    def test_list_entities(self, mock_core_class):
        """Test listing entities endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.graph_storage.get_all_entities = Mock(
            return_value=[
                {
                    "id": "entity1",
                    "name": "Test Entity",
                    "entity_type": "PERSON",
                    "description": "A test person",
                    "source_ids": ["chunk1"],
                    "file_paths": [],
                    "metadata": {},
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                }
            ]
        )

        response = self.client.get(
            f"/api/lightrag/workspaces/{self.workspace.name}/entities/"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

        entity = data[0]
        self.assertEqual(entity["name"], "Test Entity")
        self.assertEqual(entity["entity_type"], "PERSON")

    @patch("lightrag_app.views.LightRAGCore")
    def test_list_relations(self, mock_core_class):
        """Test listing relations endpoint"""
        # Mock the LightRAGCore
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.graph_storage.get_all_relations = Mock(
            return_value=[
                {
                    "source_entity": "entity1",
                    "target_entity": "entity2",
                    "source_name": "Person A",
                    "target_name": "Person B",
                    "source_type": "PERSON",
                    "target_type": "PERSON",
                }
            ]
        )

        response = self.client.get(
            f"/api/lightrag/workspaces/{self.workspace.name}/relations/"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

        relation = data[0]
        self.assertEqual(relation["source_entity"], "entity1")
        self.assertEqual(relation["target_entity"], "entity2")

    def test_list_chunks(self):
        """Test listing chunks endpoint"""
        # Create a document and chunks
        document = Document.objects.create(
            id="test-doc-chunks",
            workspace=self.workspace,
            title="Test Document for Chunks",
            content="Test content for chunks",
        )

        from lightrag_app.models import TextChunk

        _chunk = TextChunk.objects.create(
            id="test-chunk-123",
            document=document,
            workspace=self.workspace,
            content="Test chunk content",
            tokens=5,
            chunk_order_index=0,
        )

        response = self.client.get(
            f"/api/lightrag/workspaces/{self.workspace.name}/chunks/"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

        chunk_data = data[0]
        self.assertEqual(chunk_data["id"], "test-chunk-123")
        self.assertEqual(chunk_data["content"], "Test chunk content")
        self.assertEqual(chunk_data["tokens"], 5)
        self.assertEqual(chunk_data["chunk_order_index"], 0)
