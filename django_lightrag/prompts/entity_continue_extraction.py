import dspy


class EntityContinueExtractionSignature(dspy.Signature):
    """Extract only missed or malformed entities and relationships from prior work."""

    extraction_delta: str = dspy.OutputField(
        desc="Only newly found or corrected entity and relation lines."
    )


USER_PROMPT_TEMPLATE = """Based on the previous extraction, identify any missed or incorrectly formatted entities and relationships from the input text.

Requirements:
1. Follow the system instructions exactly for output order, field separation, and proper noun handling.
2. Do not repeat entities or relationships that were already extracted correctly.
3. Output newly found items and corrected versions of malformed items only.
4. Output each entity as 4 fields separated by `{tuple_delimiter}`, starting with `entity`.
5. Output each relationship as 5 fields separated by `{tuple_delimiter}`, starting with `relation`.
6. Output only the entity and relationship lines.
7. Output `{completion_delimiter}` as the final line.
"""


def render_user_prompt(
    *,
    tuple_delimiter: str,
    completion_delimiter: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
    )
