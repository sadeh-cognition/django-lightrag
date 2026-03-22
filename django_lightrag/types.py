from dataclasses import dataclass, field


@dataclass
class QueryParam:
    """Parameters for RAG queries"""

    mode: str = "hybrid"  # "local", "global", "hybrid" # TODO: make this an Enum
    top_k: int = 10
    max_tokens: int = 4000
    temperature: float = 0.1
    stream: bool = False  # TODO: remove this option
    low_level_keywords: list[str] = field(default_factory=list)
    high_level_keywords: list[str] = field(default_factory=list)
    one_hop_enabled: bool = False
    one_hop_max_entities: int = 10
    one_hop_max_relations: int = 10


@dataclass
class QueryKeywordsResult:
    low_level_keywords: list[str] = field(default_factory=list)
    high_level_keywords: list[str] = field(default_factory=list)


@dataclass
class QueryContextDocument:
    content: str
    document_id: str


@dataclass
class QueryContextEntity:
    name: str
    entity_type: str
    description: str
    profile_key: str


@dataclass
class QueryContextRelation:
    source: str
    relation_type: str
    target: str
    description: str
    profile_key: str


@dataclass
class DocumentVectorHit:
    id: str
    score: float
    rank: int


@dataclass
class EntityVectorHit:
    id: str
    name: str
    profile_key: str
    score: float
    rank: int


@dataclass
class RelationVectorHit:
    id: str
    source: str
    relation_type: str
    target: str
    profile_key: str
    score: float
    rank: int


@dataclass
class DocumentVectorMatches:
    query_text: str
    query_source: str
    hits: list[DocumentVectorHit] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)


@dataclass
class EntityVectorMatches:
    query_text: str
    query_source: str
    hits: list[EntityVectorHit] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)


@dataclass
class RelationVectorMatches:
    query_text: str
    query_source: str
    hits: list[RelationVectorHit] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)


@dataclass
class VectorMatchingResult:
    documents: DocumentVectorMatches
    entities: EntityVectorMatches
    relations: RelationVectorMatches


@dataclass
class GraphTraversalCaps:
    max_entities: int
    max_relations: int


@dataclass
class GraphTraversalResult:
    seed_entity_ids: list[str]
    added_entity_ids: list[str]
    added_relation_ids: list[str]
    caps_applied: GraphTraversalCaps


@dataclass
class QueryContext:
    documents: list[QueryContextDocument] = field(default_factory=list)
    entities: list[QueryContextEntity] = field(default_factory=list)
    relations: list[QueryContextRelation] = field(default_factory=list)
    query_keywords: QueryKeywordsResult = field(default_factory=QueryKeywordsResult)
    total_tokens: int = 0
    aggregated_context: str = ""
    vector_matching: VectorMatchingResult | None = None
    graph_traversal: GraphTraversalResult | None = None


@dataclass
class QuerySource:
    type: str
    id: str
    name: str | None = None
    content: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    entity_type: str | None = None
    description: str | None = None
    source: str | None = None
    relation_type: str | None = None
    target: str | None = None


@dataclass
class QueryResult:
    """Result of a RAG query"""

    response: str
    sources: list[QuerySource]
    context: QueryContext
    query_time: float
    tokens_used: int
