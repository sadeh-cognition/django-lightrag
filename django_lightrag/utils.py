try:
    import tiktoken
except ImportError:
    tiktoken = None


class Tokenizer:
    """Tokenizer using tiktoken"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = (
            tiktoken.encoding_for_model(model_name) if tiktoken is not None else None
        )

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return [index for index, _ in enumerate(text.split(), start=1)]
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(token) for token in tokens)
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])
