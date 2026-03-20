from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    """Document representation in the RAG system"""

    id = models.CharField(max_length=255, primary_key=True)  # MD5 hash or UUID
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    track_id = models.CharField(max_length=100, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "lightrag_documents"
        indexes = [
            models.Index(fields=["track_id"]),
        ]

    def __str__(self):
        return f"{self.id[:50]}..."


class Entity(models.Model):
    """Knowledge graph entities"""

    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=500, db_index=True)
    entity_type = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    source_ids = models.JSONField(default=list, blank=True)  # List of document IDs
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "lightrag_entities"
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["entity_type"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class Relation(models.Model):
    """Knowledge graph relationships"""

    id = models.CharField(max_length=255, primary_key=True)
    source_entity = models.ForeignKey(
        Entity, on_delete=models.CASCADE, related_name="outgoing_relations"
    )
    target_entity = models.ForeignKey(
        Entity, on_delete=models.CASCADE, related_name="incoming_relations"
    )
    relation_type = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    source_ids = models.JSONField(default=list, blank=True)  # List of document IDs
    weight = models.FloatField(default=1.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "lightrag_relations"
        indexes = [
            models.Index(fields=["source_entity", "target_entity"]),
            models.Index(fields=["relation_type"]),
        ]

    def __str__(self):
        return f"{self.source_entity.name} -> {self.relation_type} -> {self.target_entity.name}"
