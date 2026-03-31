from __future__ import annotations

import html
import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, TypedDict

from django.contrib.auth import get_user_model
from django_llm_chat.dspy_chat import DSPyChat
from django_llm_chat.models import Project

from .dspy_runtime import extract_dspy_response_text
from .prompts import (
    render_entity_continue_extraction_user_prompt,
    render_entity_extraction_examples,
    render_entity_extraction_system_prompt,
    render_entity_extraction_user_prompt,
)

logger = logging.getLogger(__name__)


class PipelineCancelledError(Exception):
    """Raised when a user cancels a pipeline operation."""


PipelineCancelledException = PipelineCancelledError


class DocumentSchema(TypedDict, total=False):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


DEFAULT_SUMMARY_LANGUAGE = "English"
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256
DEFAULT_TUPLE_DELIMITER = "<|#|>"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Date",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def remove_think_tags(text: str) -> str:
    return re.sub(
        r"^(<think>.*?</think>|.*</think>)", "", text, flags=re.DOTALL
    ).strip()


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    if not text:
        return text

    try:
        text = text.strip()
        if not text:
            return text

        text.encode("utf-8")

        sanitized = ""
        for char in text:
            code_point = ord(char)
            if 0xD800 <= code_point <= 0xDFFF:
                sanitized += replacement_char
                continue
            if code_point == 0xFFFE or code_point == 0xFFFF:
                sanitized += replacement_char
                continue
            sanitized += char

        sanitized = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", replacement_char, sanitized
        )

        sanitized.encode("utf-8")
        sanitized = html.unescape(sanitized)
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", sanitized)
        return sanitized.strip()

    except UnicodeEncodeError as e:
        error_msg = f"Text contains uncleanable UTF-8 encoding issues: {str(e)[:100]}"
        logger.error(f"Text sanitization failed: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        logger.error(f"Text sanitization: Unexpected error: {str(e)}")
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            raise ValueError(
                f"Text sanitization failed with unexpected error: {str(e)}"
            ) from e


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes: bool = False
) -> str:
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""


def normalize_extracted_info(name: str, remove_inner_quotes: bool = False) -> str:
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    name = name.replace("－", "-")
    name = name.replace("＋", "+")
    name = name.replace("／", "/")
    name = name.replace("＊", "*")

    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("—", "-").replace("－", "-")
    name = name.replace("　", " ")

    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    if len(name) >= 2:
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:
                name = inner_content

        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:
                name = inner_content

        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        name = name.replace("\u00a0", " ")
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    name = name.strip()

    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text: str) -> bool:
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        return ""

    return name


def fix_tuple_delimiter_corruption(
    record: str, delimiter_core: str, tuple_delimiter: str
) -> str:
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    escaped_delimiter_core = re.escape(delimiter_core)

    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )
    record = re.sub(rf"<\|\\{escaped_delimiter_core}\|>", tuple_delimiter, record)
    record = re.sub(r"<\|+>", tuple_delimiter, record)
    record = re.sub(rf"<.?\|{escaped_delimiter_core}\|.?>", tuple_delimiter, record)
    record = re.sub(rf"<\|?{escaped_delimiter_core}\|?>", tuple_delimiter, record)
    record = re.sub(
        rf"<[^|]{escaped_delimiter_core}\|>|<\|{escaped_delimiter_core}[^|]>",
        tuple_delimiter,
        record,
    )
    record = re.sub(rf"<\|{escaped_delimiter_core}\|+(?!>)", tuple_delimiter, record)
    record = re.sub(rf"<\|{escaped_delimiter_core}:(?!>)", tuple_delimiter, record)
    record = re.sub(rf"<\|+{escaped_delimiter_core}>", tuple_delimiter, record)
    record = re.sub(r"<\|\|(?!>)", tuple_delimiter, record)
    record = re.sub(rf"(?<!<)\|{escaped_delimiter_core}\|>", tuple_delimiter, record)
    record = re.sub(rf"<\|{escaped_delimiter_core}\|>\|", tuple_delimiter, record)
    record = re.sub(rf"\|\|{escaped_delimiter_core}\|\|", tuple_delimiter, record)

    return record


def create_prefixed_exception(original_exception: Exception, prefix: str) -> Exception:
    try:
        if hasattr(original_exception, "args") and original_exception.args:
            args = list(original_exception.args)
            found_str = False
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    args[i] = f"{prefix}: {arg}"
                    found_str = True
                    break

            if not found_str:
                args[0] = f"{prefix}: {args[0]}"

            return type(original_exception)(*args)
        return type(original_exception)(f"{prefix}: {str(original_exception)}")
    except (TypeError, ValueError, AttributeError) as construct_error:
        return RuntimeError(
            f"{prefix}: {type(original_exception).__name__}: {str(original_exception)} "
            f"(Original exception could not be reconstructed: {construct_error})"
        )


def use_llm_func(
    user_prompt: str,
    model_name: str,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
) -> str:
    safe_user_prompt = sanitize_text_for_encoding(user_prompt)
    safe_system_prompt = (
        sanitize_text_for_encoding(system_prompt) if system_prompt else None
    )

    safe_history_messages = None
    if history_messages:
        safe_history_messages = []
        for msg in history_messages:
            safe_msg = msg.copy()
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text_for_encoding(safe_msg["content"])
            safe_history_messages.append(safe_msg)

    try:
        user, _ = get_user_model().objects.get_or_create(username="lightrag_django")
        project, _ = Project.objects.get_or_create(name="lightrag_django")
        dspy_chat = DSPyChat.create(project=project)
        lm = dspy_chat.as_lm(
            model=model_name,
            user=user,
            use_cache=True,
            max_tokens=max_tokens,
        )
        messages: list[dict[str, str]] = []
        if safe_system_prompt:
            messages.append({"role": "system", "content": safe_system_prompt})
        if safe_history_messages:
            messages.extend(safe_history_messages)
        messages.append({"role": "user", "content": safe_user_prompt})
        response = lm.forward(messages=messages)
    except Exception as e:
        error_msg = f"[LLM func] {str(e)}"
        raise type(e)(error_msg) from e

    return remove_think_tags(extract_dspy_response_text(response))


def _truncate_entity_identifier(
    identifier: str, limit: int, document_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        document_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def _handle_single_entity_extraction(
    record_attributes: list[str],
    document_key: str,
    timestamp: int,
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{document_key}: LLM output format error; found {len(record_attributes)}/4 feilds on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Handle comma-separated entity types by finding the first non-empty token
        if "," in entity_type:
            original = entity_type
            tokens = [t.strip() for t in entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            if not non_empty:
                logger.warning(
                    f"Entity extraction error: all tokens empty after comma-split: '{original}'"
                )
                return None
            entity_type = non_empty[0]
            logger.warning(
                f"Entity type contains comma, taking first non-empty token: '{original}' -> '{entity_type}'"
            )

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": entity_description,
            "source_id": document_key,
            "timestamp": timestamp,
        }

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in document {document_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in document {document_key}: {e}"
        )
        return None


def _handle_single_relationship_extraction(
    record_attributes: list[str],
    document_key: str,
    timestamp: int,
):
    if len(record_attributes) != 5 or "relation" not in record_attributes[0]:
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{document_key}: LLM output format error; found {len(record_attributes)}/5 fields on REALTION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])
        if not edge_description.strip():
            logger.warning(
                f"Relationship extraction error: empty description for relation '{source}'~'{target}' in document '{document_key}'"
            )
            return None

        edge_source_id = document_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return {
            "src_id": source,
            "tgt_id": target,
            "weight": weight,
            "description": edge_description,
            "keywords": edge_keywords,
            "source_id": edge_source_id,
            "timestamp": timestamp,
        }

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in document {document_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in document {document_key}: {e}"
        )
        return None


def _process_extraction_result(
    result: str,
    document_key: str,
    timestamp: int,
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)."""
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning(
            f"{document_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to separate record instead of "\n"
    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{document_key}: LLM output format error; find LLM use {tuple_delimiter} as record separators instead new-line"
        )

    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            # change delimiter_core to lower case, and fix again
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = _handle_single_entity_extraction(
            record_attributes, document_key, timestamp
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                document_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = _handle_single_relationship_extraction(
            record_attributes, document_key, timestamp
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                document_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                document_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


def extract_entities(
    documents: dict[str, DocumentSchema],
    model_name: str,
    entity_extract_max_gleaning: int,
    language: str = DEFAULT_SUMMARY_LANGUAGE,
    entity_types: list[str] | None = None,
    tokenizer: Any | None = None,
    max_extract_input_tokens: int = 12000,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> list:
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledError("User cancelled during entity extraction")

    ordered_documents = list(documents.items())
    resolved_entity_types = entity_types or DEFAULT_ENTITY_TYPES

    example_context_base = {
        "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
        "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
        "entity_types": ", ".join(resolved_entity_types),
    }
    examples = render_entity_extraction_examples(**example_context_base)

    context_base = {
        "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
        "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
        "entity_types": ",".join(resolved_entity_types),
        "examples": examples,
    }

    processed_documents = 0
    total_documents = len(ordered_documents)

    def _process_single_content(document_key_dp: tuple[str, DocumentSchema]):
        """Process a single document
        Args:
            document_key_dp (tuple[str, DocumentSchema]):
                ("document-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_documents
        document_key = document_key_dp[0]
        document_dp = document_key_dp[1]
        content = document_dp["content"]

        # Get initial extraction
        # Format system prompt without input_text for each document (enables prompt caching across documents)
        entity_extraction_system_prompt = render_entity_extraction_system_prompt(
            entity_types=context_base["entity_types"],
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
            examples=context_base["examples"],
        )
        entity_extraction_user_prompt = render_entity_extraction_user_prompt(
            entity_types=context_base["entity_types"],
            completion_delimiter=context_base["completion_delimiter"],
            input_text=content,
        )
        entity_continue_extraction_user_prompt = (
            render_entity_continue_extraction_user_prompt(
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )
        )

        final_result = use_llm_func(
            entity_extraction_user_prompt,
            model_name=model_name,
            system_prompt=entity_extraction_system_prompt,
        )
        timestamp = int(time.time())

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction
        maybe_nodes, maybe_edges = _process_extraction_result(
            final_result,
            document_key,
            timestamp,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            # Calculate total tokens for the gleaning request to prevent context window overflow
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required when entity_extract_max_gleaning is greater than 0"
                )

            # Approximate total tokens: system prompt + history + user prompt.
            # This slightly underestimates actual API usage (missing role/framing tokens)
            # but is sufficient as a safety guard against context window overflow.
            history_str = json.dumps(history, ensure_ascii=False)
            full_context_str = (
                entity_extraction_system_prompt
                + history_str
                + entity_continue_extraction_user_prompt
            )
            token_count = len(tokenizer.encode(full_context_str))

            if token_count > max_extract_input_tokens:
                logger.warning(
                    f"Gleaning stopped for document {document_key}: Input tokens ({token_count}) exceeded limit ({max_extract_input_tokens})."
                )
            else:
                glean_result = use_llm_func(
                    entity_continue_extraction_user_prompt,
                    model_name=model_name,
                    system_prompt=entity_extraction_system_prompt,
                    history_messages=history,
                )
                timestamp = int(time.time())

                # Process gleaning result separately with file path
                glean_nodes, glean_edges = _process_extraction_result(
                    glean_result,
                    document_key,
                    timestamp,
                    tuple_delimiter=context_base["tuple_delimiter"],
                    completion_delimiter=context_base["completion_delimiter"],
                )

                # Merge results - compare description lengths to choose better version
                for entity_name, glean_entities in glean_nodes.items():
                    if entity_name in maybe_nodes:
                        # Compare description lengths and keep the better one
                        original_desc_len = len(
                            maybe_nodes[entity_name][0].get("description", "") or ""
                        )
                        glean_desc_len = len(
                            glean_entities[0].get("description", "") or ""
                        )

                        if glean_desc_len > original_desc_len:
                            maybe_nodes[entity_name] = list(glean_entities)
                        # Otherwise keep original version
                    else:
                        # New entity from gleaning stage
                        maybe_nodes[entity_name] = list(glean_entities)

                for edge_key, glean_edge_list in glean_edges.items():
                    if edge_key in maybe_edges:
                        # Compare description lengths and keep the better one
                        original_desc_len = len(
                            maybe_edges[edge_key][0].get("description", "") or ""
                        )
                        glean_desc_len = len(
                            glean_edge_list[0].get("description", "") or ""
                        )

                        if glean_desc_len > original_desc_len:
                            maybe_edges[edge_key] = list(glean_edge_list)
                        # Otherwise keep original version
                    else:
                        # New edge from gleaning stage
                        maybe_edges[edge_key] = list(glean_edge_list)

        processed_documents += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = (
            f"Document {processed_documents} of {total_documents} extracted "
            f"{entities_count} Ent + {relations_count} Rel {document_key}"
        )
        logger.info(log_message)
        if pipeline_status is not None:
            with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Process documents sequentially (synchronous processing)
    document_results = []
    first_exception = None

    for document in ordered_documents:
        # Check for cancellation before processing document
        if pipeline_status is not None and pipeline_status_lock is not None:
            with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledError(
                        "User cancelled during document processing"
                    )

        try:
            result = _process_single_content(document)
            document_results.append(result)
        except Exception as e:
            document_id = document[0]  # Extract document_id from document[0]
            prefixed_exception = create_prefixed_exception(e, document_id)
            if first_exception is None:
                first_exception = prefixed_exception
            # Stop processing on first exception
            break

    # If any task failed, raise the first exception with progress prefix
    if first_exception is not None:
        progress_prefix = f"D[{processed_documents + 1}/{total_documents}]"
        final_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise final_exception from first_exception

    # If all tasks completed successfully, document_results already contains the results
    # Return the document_results for later processing in merge_nodes_and_edges
    return document_results
