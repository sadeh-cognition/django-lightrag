from dataclasses import dataclass
from typing import Any


@dataclass
class QueryParam:
    """Parameters for RAG queries"""

    mode: str = "hybrid"  # "local", "global", "hybrid" # TODO: make this an Enum
    top_k: int = 10
    max_tokens: int = 4000
    temperature: float = 0.1
    stream: bool = False  # TODO: remove this option


@dataclass
class QueryResult:
    """Result of a RAG query"""

    response: str
    sources: list[dict[str, Any]]
    context: dict[str, Any]
    query_time: float
    tokens_used: int
