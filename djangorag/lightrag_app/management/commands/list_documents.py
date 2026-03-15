"""
Django management command to list documents.
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "List documents in the LightRAG system"

    def add_arguments(self, parser):
        parser.add_argument(
            "--status",
            type=str,
            choices=["pending", "processing", "processed", "failed"],
            help="Filter by document status",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["table", "json"],
            default="table",
            help="Output format (default: table)",
        )

    def handle(self, *args, **options):
        status_filter = options.get("status")
        output_format = options["format"]

        # Get documents
        from djangorag.lightrag_app.core import LightRAGCore

        try:
            core = LightRAGCore()
            try:
                documents = core.list_documents(status_filter)

                if not documents:
                    self.stdout.write(self.style.WARNING("No documents found"))
                    return

                if output_format == "json":
                    import json

                    self.stdout.write(json.dumps(documents, indent=2))
                else:
                    # Table format
                    self.stdout.write(self.style.SUCCESS("Documents in the system:"))
                    self.stdout.write("-" * 100)
                    self.stdout.write(
                        f"{'ID':<36} {'Title':<30} {'Status':<12} {'Chunks':<8} {'Created':<20}"
                    )
                    self.stdout.write("-" * 100)

                    for doc in documents:
                        title = (
                            (doc["title"][:27] + "...")
                            if len(doc["title"]) > 30
                            else doc["title"]
                        )
                        created = doc["created_at"][:19].replace("T", " ")

                        self.stdout.write(
                            f"{doc['id']:<36} {title:<30} {doc['status']:<12} "
                            f"{doc['chunks_count']:<8} {created:<20}"
                        )

                    self.stdout.write("-" * 100)
                    self.stdout.write(f"Total: {len(documents)} documents")
            finally:
                core.close()
        except Exception as e:
            raise CommandError(f"Failed to list documents: {e}")
