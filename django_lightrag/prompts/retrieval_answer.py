import dspy


class RetrievalAnswerSignature(dspy.Signature):
    """You are a retrieval-augmented assistant.

    Answer the user using only the provided context.
    Do not invent, assume, or import outside knowledge.
    If the context does not contain enough information, reply that you do not have enough information from the provided context.
    Do not add a references section or cite sources inline.
    """

    context: str = dspy.InputField(desc="Retrieved context available to the model.")
    query_text: str = dspy.InputField(desc="Original user query.")
    answer: str = dspy.OutputField(
        desc="Grounded answer using only the provided context."
    )
