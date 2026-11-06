"""Shared checks for the two validated Three Branches data documents.

Both ``rules.json`` and ``catalog.json`` are hand-authored, so they are validated on load rather
than trusted. Every check names the field it rejected so a bad edit points at itself.
"""

from __future__ import annotations

import re
from typing import Any

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def mapping(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    """Accept an object carrying exactly the named keys."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(value) - keys)
    missing = sorted(keys - set(value))
    if unknown or missing:
        raise ValueError(f"{name} has unknown keys {unknown} and missing keys {missing}")
    return value


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def nonnegative_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def text(value: Any, name: str, *, length: int | None = None) -> str:
    """Accept a non-empty string, optionally of one exact length."""
    if not isinstance(value, str) or not value or (length is not None and len(value) != length):
        raise ValueError(f"{name} must be a non-empty string of the expected length")
    return value


def token(value: Any, name: str, *, max_length: int = 16) -> str:
    """Accept a lowercase snake-case identifier that fits its observation text bound."""
    if not isinstance(value, str) or len(value) > max_length or not _SNAKE_CASE.fullmatch(value):
        raise ValueError(f"{name} must be a snake-case token of at most {max_length} characters")
    return value
