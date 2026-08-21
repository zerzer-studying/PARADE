"""Helpers for keeping saved artifacts free of machine-specific paths."""

import ntpath
import os
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_absolute_path(value: str) -> bool:
    """Recognize POSIX, Windows-drive, and UNC absolute paths."""
    return os.path.isabs(value) or ntpath.isabs(value)


def public_path(value: Any) -> Any:
    """Return a portable path without exposing its host filesystem prefix.

    Paths inside this repository become repository-relative. External absolute
    paths are replaced completely because even their final component may carry
    identifying information. Non-path values and already-relative paths are
    returned unchanged.
    """
    if not isinstance(value, (str, os.PathLike)):
        return value

    text = os.fspath(value)
    expanded = os.path.expanduser(text)
    if not _is_absolute_path(expanded):
        return text

    if ntpath.isabs(expanded) and not os.path.isabs(expanded):
        return "<external-path>"

    normalized = Path(expanded).resolve(strict=False)
    try:
        relative = normalized.relative_to(_PROJECT_ROOT)
    except ValueError:
        return "<external-path>"
    return relative.as_posix() or "."


def sanitize_for_publication(value: Any) -> Any:
    """Recursively redact absolute paths before serialization or logging."""
    if isinstance(value, dict):
        return {
            key: sanitize_for_publication(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_publication(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_publication(item) for item in value)
    return public_path(value)
