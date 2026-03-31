from __future__ import annotations

from typing import Any

import dspy
from django_llm_chat.dspy_chat import DSPyChat
from django_llm_chat.models import Project


def build_dspy_chat_lm(
    *,
    model: str,
    project: Project,
    user: object | None = None,
    use_cache: bool = True,
    **lm_kwargs: Any,
) -> dspy.LM:
    dspy_chat = DSPyChat.create(project=project)
    return dspy_chat.as_lm(
        model=model,
        user=user,
        use_cache=use_cache,
        **lm_kwargs,
    )


def run_dspy_signature(
    signature: type[dspy.Signature],
    *,
    model: str,
    project: Project,
    user: object | None = None,
    use_cache: bool = True,
    inputs: dict[str, Any],
    **lm_kwargs: Any,
) -> Any:
    lm = build_dspy_chat_lm(
        model=model,
        project=project,
        user=user,
        use_cache=use_cache,
        **lm_kwargs,
    )
    predictor = dspy.Predict(signature)
    with dspy.context(lm=lm):
        return predictor(**inputs)


def extract_dspy_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content

    if isinstance(response, list) and response:
        first_item = response[0]
        if isinstance(first_item, str):
            return first_item

    return str(response)
