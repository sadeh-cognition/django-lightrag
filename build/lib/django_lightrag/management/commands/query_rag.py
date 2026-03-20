"""
Django management command to query the LightRAG system.
"""

from django.core.management.base import BaseCommand, CommandError
from django_lightrag.core.core import LightRAGCore, QueryParam


class Command(BaseCommand):
    help = "Query the LightRAG system"

    def add_arguments(self, parser):
        parser.add_argument("query", type=str, help="Query text")
        parser.add_argument(
            "--mode",
            type=str,
            choices=["local", "global", "hybrid"],
            default="hybrid",
            help="Query mode (default: hybrid)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=10,
            help="Number of top results to retrieve (default: 10)",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=4000,
            help="Maximum tokens in context (default: 4000)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.1,
            help="Temperature for response generation (default: 0.1)",
        )
        parser.add_argument(
            "--include-sources",
            action="store_true",
            help="Include source information in output",
        )
        parser.add_argument(
            "--include-context",
            action="store_true",
            help="Include context information in output",
        )

    def handle(self, *args, **options):
        query_text = options["query"]
        mode = options["mode"]
        top_k = options["top_k"]
        max_tokens = options["max_tokens"]
        temperature = options["temperature"]
        include_sources = options["include_sources"]
        include_context = options["include_context"]

        # Create query parameters
        param = QueryParam(
            mode=mode, top_k=top_k, max_tokens=max_tokens, temperature=temperature
        )

        # Query the system
        try:
            core = LightRAGCore()
            try:
                result = core.query(query_text, param)
                # Output results
                self.stdout.write(self.style.SUCCESS("Query Results:"))
                self.stdout.write("=" * 50)
                self.stdout.write(f"Query: {query_text}")
                self.stdout.write(f"Mode: {mode}")
                self.stdout.write(f"Query Time: {result.query_time:.2f}s")
                self.stdout.write(f"Tokens Used: {result.tokens_used}")
                self.stdout.write("=" * 50)
                self.stdout.write("Response:")
                self.stdout.write(result.response)

                if include_sources and result.sources:
                    self.stdout.write("=" * 50)
                    self.stdout.write("Sources:")
                    for i, source in enumerate(result.sources, 1):
                        self.stdout.write(
                            f"{i}. [{source['type'].upper()}] {source.get('name', source.get('id', ''))}"
                        )
                        if "content" in source:
                            self.stdout.write(f"   {source['content'][:200]}...")

                if include_context:
                    self.stdout.write("=" * 50)
                    self.stdout.write("Context Summary:")
                    context = result.context
                    self.stdout.write(f"Documents: {len(context.get('documents', []))}")
                    self.stdout.write(f"Entities: {len(context.get('entities', []))}")
                    self.stdout.write(f"Relations: {len(context.get('relations', []))}")
                    self.stdout.write(
                        f"Total Context Tokens: {context.get('total_tokens', 0)}"
                    )
            finally:
                core.close()
        except Exception as e:
            raise CommandError(f"Failed to query system: {e}")
