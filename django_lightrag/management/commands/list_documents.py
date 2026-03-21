"""
Django management command to list documents.
"""

from django.core.management.base import BaseCommand, CommandError

from django_lightrag.config import get_lightrag_core_settings
from django_lightrag.core import LightRAGCore


class Command(BaseCommand):
    help = "List documents in the LightRAG system"

    def add_arguments(self, parser):
        parser.add_argument(
            "--format",
            type=str,
            choices=["table", "json"],
            default="table",
            help="Output format (default: table)",
        )

    def handle(self, *args, **options):
        output_format = options["format"]

        # Get documents
        config = get_lightrag_core_settings()

        try:
            core = LightRAGCore(
                embedding_model=config["EMBEDDING_MODEL"],
                embedding_provider=config["EMBEDDING_PROVIDER"],
                embedding_base_url=config["EMBEDDING_BASE_URL"],
                llm_model=config["LLM_MODEL"],
            )
            try:
                documents = core.list_documents()

                if not documents:
                    self.stdout.write(self.style.WARNING("No documents found"))
                    return

                if output_format == "json":
                    import json

                    self.stdout.write(json.dumps(documents, indent=2))
                else:
                    # Table format
                    self.stdout.write(self.style.SUCCESS("Documents in the system:"))
                    self.stdout.write("-" * 88)
                    self.stdout.write(f"{'ID':<36} {'Track ID':<20} {'Created':<20}")
                    self.stdout.write("-" * 88)

                    for doc in documents:
                        created = doc["created_at"][:19].replace("T", " ")

                        self.stdout.write(
                            f"{doc['id']:<36} {doc['track_id']:<20} {created:<20}"
                        )

                    self.stdout.write("-" * 88)
                    self.stdout.write(f"Total: {len(documents)} documents")
            finally:
                core.close()
        except Exception as e:
            raise CommandError(f"Failed to list documents: {e}") from e
