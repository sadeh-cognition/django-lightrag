from dataclasses import asdict
from typing import Any


def to_serializable(value: Any) -> Any:
    """Convert dataclass trees to plain Python values while pruning None fields."""
    return _prune_none(asdict(value))


def _prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _prune_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_prune_none(item) for item in value]
    return value
