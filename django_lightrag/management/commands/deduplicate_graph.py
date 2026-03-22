"""
Django management command to backfill strict entity and relation deduplication.
"""

from django.core.management.base import BaseCommand, CommandError

from django_lightrag.config import get_lightrag_core_settings
from django_lightrag.core import LightRAGCore


class Command(BaseCommand):
    help = (
        "Deduplicate existing entities and relations using strict graph identity keys"
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            choices=["entities", "relations", "all"],
            default="all",
            help="Limit deduplication scope (default: all)",
        )

    def handle(self, *args, **options):
        only = options["only"]

        config = get_lightrag_core_settings()

        include_entities = only in {"entities", "all"}
        include_relations = only in {"relations", "all"}

        try:
            core = LightRAGCore(
                embedding_model=config.embedding_model,
                embedding_provider=config.embedding_provider,
                embedding_base_url=config.embedding_base_url,
                llm_model=config.llm_model,
            )
            try:
                result = core.deduplicate_graph(
                    include_entities=include_entities,
                    include_relations=include_relations,
                )
            finally:
                core.close()
        except Exception as exc:
            raise CommandError(f"Failed to deduplicate graph: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Deduplicated graph "
                f"(merged_entities={result['merged_entities']}, "
                f"deleted_entities={result['deleted_entities']}, "
                f"merged_relations={result['merged_relations']}, "
                f"deleted_relations={result['deleted_relations']})"
            )
        )
