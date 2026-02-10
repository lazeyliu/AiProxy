"""Helpers to build OpenAI /v1/responses payloads."""
from __future__ import annotations

from typing import Any

from .token_count import count_messages_tokens, count_text_tokens


def _model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return obj


def _normalize_usage(usage_obj: Any, messages: list, model_name: str, content: str) -> dict:
    usage_payload = None
    if usage_obj is not None:
        if hasattr(usage_obj, "model_dump"):
            usage_payload = usage_obj.model_dump()
        elif hasattr(usage_obj, "dict"):
            usage_payload = usage_obj.dict()
        elif isinstance(usage_obj, dict):
            usage_payload = usage_obj
    if usage_payload is None:
        input_tokens = count_messages_tokens(messages, model=model_name)
        output_tokens = count_text_tokens(content, model=model_name)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    input_tokens = usage_payload.get("input_tokens")
    output_tokens = usage_payload.get("output_tokens")
    total_tokens = usage_payload.get("total_tokens")
    if input_tokens is None:
        input_tokens = usage_payload.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage_payload.get("completion_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def build_response_payload_from_chat(
    response_obj: Any,
    *,
    response_id: str,
    created: int,
    model_id: str,
    model_name: str,
    messages: list,
) -> dict:
    choices = getattr(response_obj, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message else ""
    if isinstance(content, list):
        content = "".join(str(item) for item in content)
    usage_payload = _normalize_usage(getattr(response_obj, "usage", None), messages, model_name, content)
    return build_response_payload(
        response_id=response_id,
        created=created,
        model_id=model_id,
        content=content,
        usage_payload=usage_payload,
    )


def build_response_payload(
    *,
    response_id: str,
    created: int,
    model_id: str,
    content: str,
    usage_payload: dict,
) -> dict:
    try:
        from openai.types.responses import (
            Response,
            ResponseOutputMessage,
            ResponseOutputText,
            ResponseUsage,
        )
        response_usage = ResponseUsage(
            input_tokens=usage_payload.get("input_tokens"),
            output_tokens=usage_payload.get("output_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
        )
        message = ResponseOutputMessage(
            id=f"msg_{response_id}",
            type="message",
            status="completed",
            role="assistant",
            content=[ResponseOutputText(type="output_text", text=content, annotations=[])],
        )
        response_payload = Response(
            id=response_id,
            object="response",
            created_at=created,
            model=model_id,
            status="completed",
            output=[message],
            usage=response_usage,
        )
        return _model_dump(response_payload)
    except Exception:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model_id,
            "status": "completed",
            "output": [
                {
                    "id": f"msg_{response_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            "usage": usage_payload,
        }


def build_response_completed_payload(
    *,
    response_id: str,
    created: int,
    model_id: str,
    output_items: list,
    usage_payload: dict,
) -> dict:
    try:
        from openai.types.responses import Response, ResponseUsage
        response_payload = Response(
            id=response_id,
            object="response",
            created_at=created,
            model=model_id,
            status="completed",
            output=output_items,
            usage=ResponseUsage(
                input_tokens=usage_payload.get("input_tokens"),
                output_tokens=usage_payload.get("output_tokens"),
                total_tokens=usage_payload.get("total_tokens"),
            ),
        )
        return _model_dump(response_payload)
    except Exception:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model_id,
            "status": "completed",
            "output": output_items,
            "usage": usage_payload,
        }
