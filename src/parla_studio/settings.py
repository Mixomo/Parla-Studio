"""User preferences in data/settings.json (shared with the interface language)."""
from __future__ import annotations

import json

from . import paths


def _load() -> dict:
    try:
        return json.loads(paths.SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get(key: str, default=None):
    return _load().get(key, default)


def set(key: str, value) -> None:  # noqa: A001 - module-level accessor: settings.set(...)
    data = _load()
    data[key] = value
    try:
        paths.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        paths.SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
