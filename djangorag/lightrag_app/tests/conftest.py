"""
Pytest configuration for LightRAG Django app tests.
"""

import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from djangorag.lightrag_app.models import Document


@pytest.fixture
def test_user():
    """Create a test user for testing"""
    return User.objects.create_user(
        username="testuser", email="test@example.com", password="testpass123"
    )


@pytest.fixture
def test_document(test_user):
    """Create a test document for testing"""
    return Document.objects.create(
        id="test-doc-123",
        title="Test Document",
        content="This is a test document for testing purposes.",
        track_id="test-track-123",
    )


@pytest.fixture
def mock_ladybug_storage():
    """Mock LadybugDB storage for testing"""
    from unittest.mock import Mock

    mock_storage = Mock()
    mock_storage.add_entity = Mock(return_value="entity-id")
    mock_storage.add_relation = Mock(return_value="relation-id")
    mock_storage.get_entity = Mock(return_value=None)
    mock_storage.get_relation = Mock(return_value=None)
    mock_storage.get_all_entities = Mock(return_value=[])
    mock_storage.get_all_relations = Mock(return_value=[])
    mock_storage.get_entity_neighbors = Mock(return_value=[])
    mock_storage.delete_entity = Mock(return_value=True)
    mock_storage.delete_relation = Mock(return_value=True)
    mock_storage.close = Mock()
    return mock_storage


@pytest.fixture
def mock_chroma_storage():
    """Mock ChromaDB storage for testing"""
    from unittest.mock import Mock

    mock_storage = Mock()
    mock_storage.add_embedding = Mock(return_value="embedding-id")
    mock_storage.get_embedding = Mock(return_value=None)
    mock_storage.search_similar = Mock(return_value=[])
    mock_storage.delete_embedding = Mock(return_value=True)
    mock_storage.update_embedding = Mock(return_value=True)
    mock_storage.close = Mock()
    return mock_storage


@pytest.fixture
def sample_document_content():
    """Sample document content for testing"""
    return """
    This is a sample document for testing the LightRAG Django app.
    It contains multiple sentences and should be suitable for document ingestion.
    The content is designed to test various aspects of the RAG system,
    including text processing, entity extraction, and query functionality.

    Key concepts mentioned: Django, LightRAG, vector databases, graph databases.
    Entities: Django (framework), LightRAG (system), ChromaDB (vector DB), LadybugDB (graph DB).

    This document should provide enough content to test:
    1. Document ingestion
    2. Entity extraction (mock)
    3. Relation extraction (mock)
    4. Vector embedding generation (mock)
    5. Query processing and retrieval
    """


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What is LightRAG and how does it work with Django?"


# Django test case base class with common setup
class LightRAGTestCase(TestCase):
    """Base test case for LightRAG Django app"""

    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.document = Document.objects.create(
            id="test-doc-123",
            title="Test Document",
            content="This is a test document for testing purposes.",
            track_id="test-track-123",
        )
