import dspy


class QueryKeywordExtractionSignature(dspy.Signature):
    """You extract retrieval keywords from a user query.

    Rules:
    - low_level_keywords must contain specific entities, proper nouns, product names, technical terms, or concrete details from the query.
    - high_level_keywords must contain broad concepts, topics, themes, or the user intent from the query.
    - use concise words or meaningful phrases taken from the query.
    - if the query is too vague or has no useful retrieval signal, return empty lists for both fields.
    """

    query_text: str = dspy.InputField(desc="Original user query.")
    low_level_keywords: list[str] = dspy.OutputField(
        desc="Specific entities, proper nouns, product names, technical terms, or concrete details."
    )
    high_level_keywords: list[str] = dspy.OutputField(
        desc="Broad concepts, topics, themes, or user intent."
    )
