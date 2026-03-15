"""
Tests for LightRAG Django models.
"""

from django.test import TestCase
from django.utils import timezone
from lightrag_app.models import Document, Entity, Relation


class DocumentModelTest(TestCase):
    """Test cases for Document model"""

    def setUp(self):
        self.document = Document.objects.create(
            id="test-doc-123",
            title="Test Document",
            content="This is a test document content.",
            track_id="track-123",
        )

    def test_document_creation(self):
        """Test document creation"""
        self.assertEqual(self.document.id, "test-doc-123")
        self.assertEqual(self.document.title, "Test Document")
        self.assertEqual(self.document.content, "This is a test document content.")
        self.assertEqual(self.document.track_id, "track-123")
        self.assertIsNotNone(self.document.created_at)
        self.assertIsNotNone(self.document.updated_at)

    def test_document_str_representation(self):
        """Test document string representation"""
        self.assertEqual(str(self.document), "Test Document")

    def test_document_status_auto_creation(self):
        """Test that DocumentStatus is automatically created"""
        self.assertTrue(hasattr(self.document, "status"))
        self.assertEqual(self.document.status.status, "pending")
        self.assertEqual(self.document.status.document, self.document)


class DocumentStatusModelTest(TestCase):
    """Test cases for DocumentStatus model"""

    def setUp(self):
        self.document = Document.objects.create(
            id="test-doc-123",
            title="Test Document",
            content="This is a test document content.",
        )

    def test_status_choices(self):
        """Test status field choices"""
        valid_statuses = ["pending", "processing", "processed", "failed"]
        for status in valid_statuses:
            self.document.status.status = status
            self.document.status.save()
            self.assertEqual(self.document.status.status, status)

    def test_status_progression(self):
        """Test status progression through processing states"""
        # Initial state
        self.assertEqual(self.document.status.status, "pending")

        # Start processing
        self.document.status.status = "processing"
        self.document.status.started_at = timezone.now()
        self.document.status.save()
        self.assertEqual(self.document.status.status, "processing")
        self.assertIsNotNone(self.document.status.started_at)

        # Complete processing
        self.document.status.status = "processed"
        self.document.status.documents_count = 5
        self.document.status.documents_list = [
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
        ]
        self.document.status.completed_at = timezone.now()
        self.document.status.save()
        self.assertEqual(self.document.status.status, "processed")
        self.assertEqual(self.document.status.documents_count, 5)
        self.assertEqual(len(self.document.status.documents_list), 5)
        self.assertIsNotNone(self.document.status.completed_at)


class EntityModelTest(TestCase):
    """Test cases for Entity model"""

    def setUp(self):
        self.entity = Entity.objects.create(
            id="test-entity-123",
            name="Test Entity",
            entity_type="PERSON",
            description="A test person entity",
            source_ids=["doc1", "doc2"],
        )

    def test_entity_creation(self):
        """Test entity creation"""
        self.assertEqual(self.entity.id, "test-entity-123")
        self.assertEqual(self.entity.name, "Test Entity")
        self.assertEqual(self.entity.entity_type, "PERSON")
        self.assertEqual(self.entity.description, "A test person entity")
        self.assertEqual(self.entity.source_ids, ["doc1", "doc2"])

    def test_entity_str_representation(self):
        """Test entity string representation"""
        expected = "Test Entity (PERSON)"
        self.assertEqual(str(self.entity), expected)


class RelationModelTest(TestCase):
    """Test cases for Relation model"""

    def setUp(self):
        # Create entities
        self.source_entity = Entity.objects.create(
            id="source-entity-123",
            name="Source Entity",
            entity_type="PERSON",
        )
        self.target_entity = Entity.objects.create(
            id="target-entity-456",
            name="Target Entity",
            entity_type="ORGANIZATION",
        )

        # Create relation
        self.relation = Relation.objects.create(
            id="test-relation-789",
            source_entity=self.source_entity,
            target_entity=self.target_entity,
            relation_type="WORKS_FOR",
            description="Source works for target",
            source_ids=["doc1"],
            weight=1.5,
        )

    def test_relation_creation(self):
        """Test relation creation"""
        self.assertEqual(self.relation.id, "test-relation-789")
        self.assertEqual(self.relation.source_entity, self.source_entity)
        self.assertEqual(self.relation.target_entity, self.target_entity)
        self.assertEqual(self.relation.relation_type, "WORKS_FOR")
        self.assertEqual(self.relation.description, "Source works for target")
        self.assertEqual(self.relation.source_ids, ["doc1"])
        self.assertEqual(self.relation.weight, 1.5)

    def test_relation_str_representation(self):
        """Test relation string representation"""
        expected = "Source Entity -> WORKS_FOR -> Target Entity"
        self.assertEqual(str(self.relation), expected)

    def test_relation_foreign_key_constraints(self):
        """Test relation foreign key constraints"""
        # Try to create relation with non-existent entities
        with self.assertRaises(Exception):
            Relation.objects.create(
                id="invalid-relation",
                source_entity_id="non-existent",
                target_entity=self.target_entity,
                relation_type="TEST_REL",
            )
