try:
    from django_llm_chat.chat import Chat, DuplicateSystemMessageError
    from django_llm_chat.models import Message
except ImportError:
    Chat = None
    Message = None

    class DuplicateSystemMessageError(Exception):
        pass


class LLMService:
    """Service to handle interactions with the LLM via django-llm-chat"""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Adapter for LLM completion using django-llm-chat.
        """
        if Chat is None or Message is None:
            raise RuntimeError(
                "django-llm-chat is not installed. Install django-llm-chat to use LLMService."
            )

        chat = Chat.create()

        if system_prompt:
            try:
                chat.create_system_message(system_prompt)
            except DuplicateSystemMessageError:
                pass

        if history_messages:
            for msg in history_messages:
                role = (msg.get("role") or msg.get("type") or "user").lower()
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "system":
                    try:
                        chat.create_system_message(content)
                    except DuplicateSystemMessageError:
                        continue
                elif role == "assistant":
                    Message.create_llm_message(
                        chat=chat.chat_db_model,
                        text=content,
                        user=chat.llm_user,
                    )
                else:
                    chat.create_user_message(content)

        llm_msg, _, _ = chat.send_user_msg_to_llm(
            self.model,
            user_prompt,
            include_chat_history=True,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return llm_msg.text or ""
