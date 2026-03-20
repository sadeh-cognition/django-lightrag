"""
Django management command to ingest a document into LightRAG.
"""

from django.core.management.base import BaseCommand, CommandError
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
        from django.conf import settings

        config = getattr(settings, "LIGHTRAG", {})
        embedding_model = config.get(
            "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
        )
        embedding_provider = config.get("EMBEDDING_PROVIDER", "LMStudio")
        embedding_base_url = config.get("EMBEDDING_BASE_URL", "http://localhost:1234")
        llm_model = config.get("LLM_MODEL", "gpt-4o-mini")

        try:
            core = LightRAGCore(
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_base_url=embedding_base_url,
                llm_model=llm_model,
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
            raise CommandError(f"Failed to ingest document: {e}")
