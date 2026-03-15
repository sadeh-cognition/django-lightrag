from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.db import transaction
from .models import Document, DocumentStatus


@receiver(post_save, sender=Document)
def create_document_status(sender, instance, created, **kwargs):
    """Create DocumentStatus when a new Document is created"""
    if created:
        DocumentStatus.objects.get_or_create(
            document=instance, defaults={"status": "pending"}
        )


@receiver(pre_delete, sender=Document)
def cleanup_document_data(sender, instance, **kwargs):
    """Clean up related data when a Document is deleted"""
    from .models import VectorEmbedding

    VectorEmbedding.objects.filter(
        vector_type="document", content_id=instance.id
    ).delete()
