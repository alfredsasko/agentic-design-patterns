from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_project_environment(anchor_file: str) -> None:
    project_root = Path(anchor_file).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_google_adk_dependencies(
    dependencies: dict[str, Any],
    *,
    example_name: str,
) -> None:
    if any(value is None for value in dependencies.values()):
        raise ImportError(
            "google-adk is not installed. Install it with `uv add google-adk` "
            f"before running this {example_name}."
        )


def validate_runtime_environment(
    model: str,
    *,
    example_name: str,
) -> None:
    if model.startswith("gemini") and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            f"GOOGLE_API_KEY not found. Set it before running this {example_name}."
        )


def build_user_message(types_module: Any, request: str) -> Any:
    if not request.strip():
        raise ValueError("request must not be empty")

    user_content_type = getattr(types_module, "UserContent", types_module.Content)
    return user_content_type(parts=[types_module.Part(text=request)])


def derive_session_id(
    base_session_id: str,
    index: int,
    *,
    index_name: str,
) -> str:
    if index < 0:
        raise ValueError(f"{index_name} must not be negative")
    return f"{base_session_id}-{index + 1}"


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


def event_transfer_target(event: Any) -> str:
    actions = getattr(event, "actions", None)
    transfer_target = getattr(actions, "transfer_to_agent", None)
    if isinstance(transfer_target, str) and transfer_target.strip():
        return transfer_target.strip()
    return ""


def event_node_name(event: Any) -> str:
    node_info = getattr(event, "node_info", None)
    path = getattr(node_info, "path", "") or ""
    if not path:
        return ""
    return path.rsplit("/", 1)[-1].split("@", 1)[0]


async def ensure_session(
    runner: Any,
    *,
    user_id: str,
    session_id: str,
    initial_state: dict[str, Any] | None = None,
) -> Any:
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is not None:
        return session

    create_session_kwargs = {
        "app_name": runner.app_name,
        "user_id": user_id,
        "session_id": session_id,
    }
    if initial_state is not None:
        create_session_kwargs["state"] = initial_state
    return await runner.session_service.create_session(**create_session_kwargs)


async def close_agent_tree_async_clients(agent: Any) -> None:
    model = getattr(agent, "canonical_model", None)
    if model is not None:
        api_client = getattr(model, "api_client", None)
        if api_client is not None:
            base_api_client = getattr(api_client, "_api_client", None)
            if base_api_client is not None:
                aclose = getattr(base_api_client, "aclose", None)
                if callable(aclose):
                    await aclose()
            close = getattr(api_client, "close", None)
            if callable(close):
                close()

    for sub_agent in getattr(agent, "sub_agents", []) or []:
        await close_agent_tree_async_clients(sub_agent)


async def close_runner(runner: Any) -> None:
    close = getattr(runner, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def suppress_known_asyncio_shutdown_noise(
    loop: asyncio.AbstractEventLoop,
    context: dict[str, Any],
) -> None:
    message = str(context.get("message", ""))
    exception = context.get("exception")
    if "Fatal error on SSL transport" in message:
        return
    if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
        return
    loop.default_exception_handler(context)


def finalize_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    loop.run_until_complete(asyncio.sleep(0))
    loop.run_until_complete(asyncio.sleep(0))
    loop.run_until_complete(loop.shutdown_asyncgens())
    shutdown_default_executor = getattr(loop, "shutdown_default_executor", None)
    if callable(shutdown_default_executor):
        loop.run_until_complete(shutdown_default_executor())
    loop.close()
