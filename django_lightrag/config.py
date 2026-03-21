from typing import Any, TypedDict, cast

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


class LightRAGSettings(TypedDict):
    EMBEDDING_MODEL: str
    EMBEDDING_PROVIDER: str
    EMBEDDING_BASE_URL: str
    LLM_MODEL: str


class LadybugSettings(TypedDict, total=False):
    DATABASE_PATH: str
    IN_MEMORY: bool


REQUIRED_LIGHTRAG_KEYS = (
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "LLM_MODEL",
)


def get_lightrag_settings() -> dict[str, Any]:
    config = getattr(settings, "LIGHTRAG", None)
    if not isinstance(config, dict):
        raise ImproperlyConfigured("Django setting LIGHTRAG must be defined as a dict.")

    missing_keys = [key for key in REQUIRED_LIGHTRAG_KEYS if not config.get(key)]
    if missing_keys:
        missing_keys_display = ", ".join(sorted(missing_keys))
        raise ImproperlyConfigured(
            f"Django setting LIGHTRAG is missing required keys: {missing_keys_display}."
        )

    return config


def get_lightrag_core_settings() -> LightRAGSettings:
    config = get_lightrag_settings()
    return cast(
        LightRAGSettings,
        {
            "EMBEDDING_MODEL": config["EMBEDDING_MODEL"],
            "EMBEDDING_PROVIDER": config["EMBEDDING_PROVIDER"],
            "EMBEDDING_BASE_URL": config["EMBEDDING_BASE_URL"],
            "LLM_MODEL": config["LLM_MODEL"],
        },
    )


def get_ladybug_settings() -> LadybugSettings:
    config = getattr(settings, "LADYBUGDB", None)
    if not isinstance(config, dict):
        raise ImproperlyConfigured(
            "Django setting LADYBUGDB must be defined as a dict."
        )

    if config.get("IN_MEMORY", False):
        return cast(LadybugSettings, config)

    if not config.get("DATABASE_PATH"):
        raise ImproperlyConfigured(
            "Django setting LADYBUGDB must define DATABASE_PATH when IN_MEMORY is false."
        )

    return cast(LadybugSettings, config)
