from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from google.genai import types
except ImportError:
    types = None  # type: ignore[assignment]


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")


def validate_runtime_environment(model: str = "gemini-2.5-flash") -> None:
    if model.startswith("gemini") and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it before running this ADK example."
        )


def derive_session_id(base_session_id: str, index: int) -> str:
    if index < 0:
        raise ValueError("index must not be negative")
    return f"{base_session_id}-{index + 1}"


def require_google_adk(*required: Any) -> None:
    if any(item is None for item in required) or types is None:
        raise ImportError(
            "google-adk is not installed. Install it with `uv add google-adk` "
            "before running this ADK example."
        )


def build_user_message(request: str) -> Any:
    if types is None:
        raise ImportError("google-adk is not installed.")
    if not request.strip():
        raise ValueError("request must not be empty")
    user_content_type = getattr(types, "UserContent", types.Content)
    return user_content_type(parts=[types.Part(text=request)])


def content_to_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def event_output_to_text(event: Any) -> str:
    output = getattr(event, "output", None)
    if isinstance(output, str) and output.strip():
        return output.strip()
    return ""


def format_structured_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
