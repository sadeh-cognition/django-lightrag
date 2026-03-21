"""
Django management command to ingest a document into LightRAG.
"""

from django.core.management.base import BaseCommand, CommandError

from django_lightrag.config import get_lightrag_core_settings
from django_lightrag.core import LightRAGCore


class Command(BaseCommand):
    help = "Ingest a document into the LightRAG system"

    def add_arguments(self, parser):
        parser.add_argument("--content", type=str, help="Direct text content to ingest")
        parser.add_argument(
            "--track-id", type=str, default="", help="Tracking ID for the document"
        )

    def handle(self, *args, **options):
        content = options.get("content")
        track_id = options.get("track_id", "")

        if not content:
            raise CommandError("--content must be provided")

        # Ingest document
        config = get_lightrag_core_settings()

        try:
            core = LightRAGCore(
                embedding_model=config["EMBEDDING_MODEL"],
                embedding_provider=config["EMBEDDING_PROVIDER"],
                embedding_base_url=config["EMBEDDING_BASE_URL"],
                llm_model=config["LLM_MODEL"],
            )
            try:
                document_id = core.ingest_document(
                    content=content,
                    track_id=track_id,
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully ingested document. ID: {document_id}"
                    )
                )
            finally:
                core.close()
        except Exception as e:
            raise CommandError(f"Failed to ingest document: {e}") from e
