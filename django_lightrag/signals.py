from django.db.models.signals import pre_delete
from django.dispatch import receiver
from .models import Document


@receiver(pre_delete, sender=Document)
def cleanup_document_data(sender, instance, **kwargs):
    """Clean up related data when a Document is deleted"""
