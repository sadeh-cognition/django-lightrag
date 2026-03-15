"""
Tests for LightRAG core functionality.
"""

from unittest.mock import Mock, patch
from django.test import TestCase
from django.contrib.auth.models import User
from lightrag_app.models import Document
from lightrag_app.core import LightRAGCore, QueryParam, Tokenizer


class TokenizerTest(TestCase):
    """Test cases for Tokenizer class"""

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        self.assertEqual(self.tokenizer.model_name, "gpt-4o-mini")
        self.assertIsNotNone(self.tokenizer.tokenizer)

    def test_count_tokens(self):
        """Test token counting"""
        text = "This is a test text for tokenization."
        token_count = self.tokenizer.count_tokens(text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)

    def test_encode_decode(self):
        """Test token encoding and decoding"""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)

    def test_truncate_by_tokens(self):
        """Test text truncation by tokens"""
        text = "This is a longer text that should be truncated."
        max_tokens = 5
        truncated = self.tokenizer.truncate_by_tokens(text, max_tokens)
        self.assertLessEqual(self.tokenizer.count_tokens(truncated), max_tokens)


class LightRAGCoreTest(TestCase):
    """Test cases for LightRAGCore class"""

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        # Mock storage classes to avoid external dependencies
        with patch("lightrag_app.core.LadybugGraphStorage"), patch(
            "lightrag_app.core.ChromaVectorStorage"
        ):
            self.core = LightRAGCore()

    def test_core_initialization(self):
        """Test LightRAGCore initialization"""
        self.assertIsNotNone(self.core.tokenizer)
        self.assertIsNotNone(self.core.graph_storage)
        self.assertIsNotNone(self.core.vector_storage)
        self.assertIsInstance(self.core.config, dict)

    def test_generate_id(self):
        """Test ID generation from content"""
        content = "Test content for ID generation"
        id1 = self.core._generate_id(content)
        id2 = self.core._generate_id(content)
        self.assertEqual(id1, id2)  # Should be consistent

        different_content = "Different content"
        id3 = self.core._generate_id(different_content)
        self.assertNotEqual(id1, id3)  # Should be different for different content

    @patch("lightrag_app.core.LightRAGCore._extract_entities_from_document")
    @patch("lightrag_app.core.LightRAGCore._extract_relations_from_document")
    @patch("lightrag_app.core.LightRAGCore._generate_document_embeddings")
    def test_ingest_document(self, mock_embeddings, mock_relations, mock_entities):
        """Test document ingestion"""
        # Mock the extraction methods to return empty lists
        mock_entities.return_value = []
        mock_relations.return_value = []
        mock_embeddings.return_value = None

        content = "This is a test document for ingestion."
        title = "Test Document"

        document_id = self.core.ingest_document(
            content=content,
            title=title,
            track_id="test-track-123",
        )

        # Verify document was created
        self.assertIsNotNone(document_id)
        document = Document.objects.get(id=document_id)
        self.assertEqual(document.title, title)
        self.assertEqual(document.content, content)

        # Verify status was updated
        self.assertEqual(document.status.status, "processed")
        self.assertEqual(document.status.documents_count, 1)
        self.assertIsNotNone(document.status.completed_at)

    def test_ingest_document_failure(self):
        """Test document ingestion failure handling"""
        content = "Test content"

        # Mock an exception during ingestion
        with patch.object(
            self.core,
            "_extract_knowledge_graph_from_document",
            side_effect=Exception("Test error"),
        ):
            with self.assertRaises(Exception):
                self.core.ingest_document(content=content, title="Test")

        # Verify document status is marked as failed
        documents = Document.objects.filter(title="Test")
        if documents.exists():
            doc = documents.first()
            self.assertEqual(doc.status.status, "failed")
            self.assertEqual(doc.status.error_message, "Test error")

    @patch("lightrag_app.core.LightRAGCore._get_query_embedding")
    @patch("lightrag_app.core.LightRAGCore._retrieve_documents")
    @patch("lightrag_app.core.LightRAGCore._retrieve_knowledge_graph")
    @patch("lightrag_app.core.LightRAGCore._generate_response")
    def test_query(self, mock_response, mock_kg, mock_documents, mock_embedding):
        """Test RAG query functionality"""
        # Mock the dependencies
        mock_embedding.return_value = [0.0] * 1536  # Mock embedding
        mock_documents.return_value = []  # Empty documents list
        mock_kg.return_value = ([], [])  # Empty entities and relations
        mock_response.return_value = "Test response"

        query_text = "What is this about?"
        param = QueryParam(mode="hybrid", top_k=5)

        result = self.core.query(query_text, param)

        # Verify result structure
        self.assertIsNotNone(result.response)
        self.assertIsInstance(result.sources, list)
        self.assertIsInstance(result.context, dict)
        self.assertIsInstance(result.query_time, float)
        self.assertIsInstance(result.tokens_used, int)

        # Verify methods were called
        mock_embedding.assert_called_once_with(query_text)
        mock_documents.assert_called_once()
        mock_kg.assert_called_once()
        mock_response.assert_called_once()

    def test_delete_document(self):
        """Test document deletion"""
        # Create a document first
        _document = Document.objects.create(
            id="test-doc-delete",
            title="Test Document for Deletion",
            content="Test content for deletion",
        )

        # Mock vector storage delete
        self.core.vector_storage.delete_embedding = Mock(return_value=True)

        # Delete the document
        success = self.core.delete_document("test-doc-delete")

        self.assertTrue(success)
        self.assertFalse(Document.objects.filter(id="test-doc-delete").exists())

    def test_delete_nonexistent_document(self):
        """Test deletion of non-existent document"""
        success = self.core.delete_document("non-existent-doc")
        self.assertFalse(success)

    def test_get_document_status(self):
        """Test getting document status"""
        # Create a document
        _document = Document.objects.create(
            id="test-doc-status",
            title="Test Document Status",
            content="Test content",
        )

        status = self.core.get_document_status("test-doc-status")

        self.assertIsNotNone(status)
        self.assertEqual(status["document_id"], "test-doc-status")
        self.assertEqual(status["title"], "Test Document Status")
        self.assertEqual(status["status"], "pending")

    def test_get_nonexistent_document_status(self):
        """Test getting status of non-existent document"""
        status = self.core.get_document_status("non-existent-doc")
        self.assertIsNone(status)

    def test_list_documents(self):
        """Test listing documents"""
        # Create multiple documents
        for i in range(3):
            _document = Document.objects.create(
                id=f"test-doc-{i}",
                title=f"Test Document {i}",
                content=f"Test content {i}",
            )

        documents = self.core.list_documents()

        self.assertEqual(len(documents), 3)
        for doc in documents:
            self.assertIn("id", doc)
            self.assertIn("title", doc)
            self.assertIn("status", doc)
            self.assertIn("documents_count", doc)
            self.assertIn("created_at", doc)
            self.assertIn("updated_at", doc)

    def test_close(self):
        """Test closing storage connections"""
        # Mock close methods
        self.core.graph_storage.close = Mock()
        self.core.vector_storage.close = Mock()

        self.core.close()

        # Verify close methods were called
        self.core.graph_storage.close.assert_called_once()
        self.core.vector_storage.close.assert_called_once()


class QueryParamTest(TestCase):
    """Test cases for QueryParam dataclass"""

    def test_default_values(self):
        """Test QueryParam default values"""
        param = QueryParam()
        self.assertEqual(param.mode, "hybrid")
        self.assertEqual(param.top_k, 10)
        self.assertEqual(param.max_tokens, 4000)
        self.assertEqual(param.temperature, 0.1)
        self.assertFalse(param.stream)

    def test_custom_values(self):
        """Test QueryParam with custom values"""
        param = QueryParam(
            mode="local", top_k=20, max_tokens=8000, temperature=0.5, stream=True
        )
        self.assertEqual(param.mode, "local")
        self.assertEqual(param.top_k, 20)
        self.assertEqual(param.max_tokens, 8000)
        self.assertEqual(param.temperature, 0.5)
        self.assertTrue(param.stream)
