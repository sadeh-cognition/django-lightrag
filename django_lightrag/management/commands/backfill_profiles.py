"""
Django management command to backfill entity and relation profiles.
"""

from django.core.management.base import BaseCommand, CommandError

from django_lightrag.core import LightRAGCore


class Command(BaseCommand):
    help = "Backfill LLM-generated profiles and vector entries for existing entities and relations"

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            choices=["entities", "relations", "all"],
            default="all",
            help="Limit backfill scope (default: all)",
        )

    def handle(self, *args, **options):
        only = options["only"]

        from django.conf import settings

        config = getattr(settings, "LIGHTRAG", {})
        embedding_model = config.get(
            "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
        )
        embedding_provider = config.get("EMBEDDING_PROVIDER", "LMStudio")
        embedding_base_url = config.get("EMBEDDING_BASE_URL", "http://localhost:1234")
        llm_model = config.get("LLM_MODEL", "gpt-4o-mini")

        include_entities = only in {"entities", "all"}
        include_relations = only in {"relations", "all"}

        try:
            core = LightRAGCore(
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_base_url=embedding_base_url,
                llm_model=llm_model,
            )
            try:
                result = core.backfill_profiles(
                    include_entities=include_entities,
                    include_relations=include_relations,
                )
            finally:
                core.close()
        except Exception as exc:
            raise CommandError(f"Failed to backfill profiles: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Backfilled profiles "
                f"(entities={result['entities']}, relations={result['relations']})"
            )
        )
