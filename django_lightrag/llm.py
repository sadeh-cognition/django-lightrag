from django.contrib.auth import get_user_model
from django_llm_chat.chat import Chat
from django_llm_chat.models import Project


class LLMService:
    """Service to handle interactions with the LLM via django-llm-chat"""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        user_model = get_user_model()
        self.user, _ = user_model.objects.get_or_create(username="lightrag_django")
        self.project, _ = Project.objects.get_or_create(name="lightrag_django")

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Adapter for LLM completion using django-llm-chat.
        """
        chat = Chat.create(project=self.project)

        if system_prompt:
            chat.create_system_message(system_prompt, user=self.user)

        chat.call_llm(
            model_name=self.model,
            message=user_prompt,
            user=self.user,
            include_chat_history=True,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        return chat.last_llm_message.text if chat.last_llm_message else ""
