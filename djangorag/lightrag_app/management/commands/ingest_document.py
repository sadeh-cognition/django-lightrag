"""
Django management command to ingest a document into LightRAG.
"""

import os
from django.core.management.base import BaseCommand, CommandError
from djangorag.lightrag_app.core import LightRAGCore


class Command(BaseCommand):
    help = "Ingest a document into the LightRAG system"

    def add_arguments(self, parser):
        parser.add_argument("--file", type=str, help="Path to the file to ingest")
        parser.add_argument("--content", type=str, help="Direct text content to ingest")
        parser.add_argument("--title", type=str, default="", help="Document title")
        parser.add_argument(
            "--track-id", type=str, default="", help="Tracking ID for the document"
        )

    def handle(self, *args, **options):
        file_path = options.get("file")
        content = options.get("content")
        title = options.get("title", "")
        track_id = options.get("track_id", "")

        if not file_path and not content:
            raise CommandError("Either --file or --content must be provided")

        if file_path and content:
            raise CommandError("Cannot provide both --file and --content")

        # Get content
        if file_path:
            if not os.path.exists(file_path):
                raise CommandError(f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not title:
                title = os.path.basename(file_path)

        # Ingest document
        try:
            core = LightRAGCore()
            try:
                document_id = core.ingest_document(
                    content=content,
                    title=title,
                    file_path=file_path or "",
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
