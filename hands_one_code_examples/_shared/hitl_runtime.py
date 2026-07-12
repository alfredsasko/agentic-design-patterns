from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import is_dataclass
from typing import Any


def get_mutable_state(context: Any) -> dict[str, Any]:
    state = getattr(context, "state", None)
    if isinstance(state, dict):
        return state

    session = getattr(context, "session", None)
    session_state = getattr(session, "state", None)
    if isinstance(session_state, dict):
        return session_state

    raise AttributeError("The provided context does not expose a mutable state dict.")


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Human review input must be valid JSON matching the requested schema."
            ) from exc
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Human review input must be a JSON object.")
