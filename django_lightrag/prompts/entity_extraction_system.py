import dspy
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(
        description="The entity name. If the name is case-insensitive, capitalize the first letter of each significant word. Keep naming consistent across the extraction."
    )
    type: str = Field(
        description="One of given entity types. If none apply, use `Other`."
    )
    description: str = Field(
        description="A concise but comprehensive description based only on the input text."
    )


class Relationship(BaseModel):
    source: Entity = Field(
        description="The source entity name, using the same naming convention as the entity output."
    )
    target: Entity = Field(
        description="The target entity name, using the same naming convention as the entity output."
    )
    keywords: list[str] = Field(
        description="One or more high-level keywords separated by commas. Do not use `{tuple_delimiter}` inside this field."
    )
    description: str = Field(description="A concise explanation of the relationship.")


class EntityExtractionSystemSignature(dspy.Signature):
    """You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.
    Instructions:
    1. Identify clearly defined and meaningful entities in the input text.
    2. Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    3. If a statement describes a relationship involving more than two entities, decompose it into multiple binary relationships.
    4. Treat relationships as undirected unless the text explicitly states otherwise, and avoid duplicates.
    5. Write descriptions in the third person and avoid vague pronouns such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.
    """

    entity_types: list[str] = dspy.InputField(
        desc="Allowed entity types for classification."
    )
    examples: str = dspy.InputField(
        desc="Few-shot examples that demonstrate the required output format."
    )
    extraction_output: list[Relationship] = dspy.OutputField()
