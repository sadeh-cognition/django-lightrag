import dspy


class ProfileGenerationSignature(dspy.Signature):
    """You generate retrieval-oriented knowledge graph profiles.

    Return exactly two values:
    - key: a single word or short phrase optimized for retrieval
    - value: one grounded paragraph based only on the provided descriptions and snippets

    Rules:
    - Do not invent facts, citations, bullet points, or markdown.
    - Prefer concrete nouns and task-relevant phrasing in the key.
    """

    payload_json: str = dspy.InputField(
        desc="JSON payload containing the graph record, descriptions, and supporting documents."
    )
    key: str = dspy.OutputField(
        desc="A short retrieval phrase optimized for search and matching."
    )
    value: str = dspy.OutputField(
        desc="A single grounded paragraph based only on the provided payload."
    )
