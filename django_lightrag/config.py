from dataclasses import dataclass, field
from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


@dataclass(frozen=True, slots=True)
class LightRAGCoreConfig:
    embedding_model: str
    embedding_provider: str
    embedding_base_url: str
    llm_model: str


@dataclass(frozen=True, slots=True)
class LightRAGConfig:
    embedding_model: str
    embedding_provider: str
    embedding_base_url: str
    llm_model: str
    llm_temperature: float = 0.0
    profile_max_tokens: int = 400
    query_keyword_max_tokens: int = 200
    entity_extract_max_gleaning: int = 1
    extraction_language: str = "English"
    entity_types: list[str] = field(default_factory=list)
    max_extract_input_tokens: int = 12000
    top_k: int = 10
    max_total_tokens: int = 12000
    core_factory: str | None = None

    def to_core_config(self) -> LightRAGCoreConfig:
        return LightRAGCoreConfig(
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_base_url=self.embedding_base_url,
            llm_model=self.llm_model,
        )


@dataclass(frozen=True, slots=True)
class LadybugConfig:
    database_path: str | None = None
    in_memory: bool = False


REQUIRED_LIGHTRAG_KEYS = (
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "LLM_MODEL",
)


def _get_required_dict_setting(name: str) -> dict[str, Any]:
    config = getattr(settings, name, None)
    if not isinstance(config, dict):
        raise ImproperlyConfigured(f"Django setting {name} must be defined as a dict.")
    return config


def get_lightrag_settings() -> LightRAGConfig:
    config = _get_required_dict_setting("LIGHTRAG")

    missing_keys = [key for key in REQUIRED_LIGHTRAG_KEYS if not config.get(key)]
    if missing_keys:
        missing_keys_display = ", ".join(sorted(missing_keys))
        raise ImproperlyConfigured(
            f"Django setting LIGHTRAG is missing required keys: {missing_keys_display}."
        )

    return LightRAGConfig(
        embedding_model=str(config["EMBEDDING_MODEL"]),
        embedding_provider=str(config["EMBEDDING_PROVIDER"]),
        embedding_base_url=str(config["EMBEDDING_BASE_URL"]),
        llm_model=str(config["LLM_MODEL"]),
        llm_temperature=float(config.get("LLM_TEMPERATURE", 0.0)),
        profile_max_tokens=int(config.get("PROFILE_MAX_TOKENS", 400)),
        query_keyword_max_tokens=int(config.get("QUERY_KEYWORD_MAX_TOKENS", 200)),
        entity_extract_max_gleaning=int(config.get("ENTITY_EXTRACT_MAX_GLEANING", 1)),
        extraction_language=str(config.get("EXTRACTION_LANGUAGE", "English")),
        entity_types=[str(item) for item in config.get("ENTITY_TYPES", [])],
        max_extract_input_tokens=int(config.get("MAX_EXTRACT_INPUT_TOKENS", 12000)),
        top_k=int(config.get("TOP_K", 10)),
        max_total_tokens=int(config.get("MAX_TOTAL_TOKENS", 12000)),
        core_factory=(
            str(config["CORE_FACTORY"]) if config.get("CORE_FACTORY") else None
        ),
    )


def get_lightrag_core_settings() -> LightRAGCoreConfig:
    return get_lightrag_settings().to_core_config()


def get_ladybug_settings() -> LadybugConfig:
    config = _get_required_dict_setting("LADYBUGDB")

    if config.get("IN_MEMORY", False):
        return LadybugConfig(in_memory=True)

    if not config.get("DATABASE_PATH"):
        raise ImproperlyConfigured(
            "Django setting LADYBUGDB must define DATABASE_PATH when IN_MEMORY is false."
        )

    return LadybugConfig(
        database_path=str(config["DATABASE_PATH"]),
        in_memory=bool(config.get("IN_MEMORY", False)),
    )
