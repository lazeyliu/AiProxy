"""Helpers to build OpenAI /v1/responses payloads."""
from __future__ import annotations

import json
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


def dump_output_item(item: Any) -> Any:
    return _model_dump(item)


def build_output_message_item(
    *,
    item_id: str,
    text: str,
    status: str,
    role: str = "assistant",
) -> Any:
    try:
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        return ResponseOutputMessage(
            id=item_id,
            type="message",
            status=status,
            role=role,
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )
    except Exception:
        return {
            "id": item_id,
            "type": "message",
            "status": status,
            "role": role,
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }


def _normalize_usage(usage_obj: Any) -> dict:
    usage_payload = None
    if usage_obj is not None:
        if hasattr(usage_obj, "model_dump"):
            usage_payload = usage_obj.model_dump()
        elif hasattr(usage_obj, "dict"):
            usage_payload = usage_obj.dict()
        elif isinstance(usage_obj, dict):
            usage_payload = usage_obj
    if usage_payload is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
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


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text is None and item.get("type") in ("text", "input_text", "output_text"):
                    text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
                continue
            parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_tool_calls(message: Any) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    calls = []
    if isinstance(tool_calls, list):
        for idx, call in enumerate(tool_calls):
            payload = _model_dump(call)
            if not isinstance(payload, dict):
                continue
            function = payload.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            calls.append(
                {
                    "id": payload.get("id") or f"call_{idx}",
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                }
            )
    function_call = getattr(message, "function_call", None)
    if function_call is None and isinstance(message, dict):
        function_call = message.get("function_call")
    if isinstance(function_call, dict):
        calls.append(
            {
                "id": function_call.get("id") or "call_0",
                "name": function_call.get("name") or "",
                "arguments": function_call.get("arguments") or "",
            }
        )
    return calls


def _build_output_items_from_chat(response_obj: Any, response_id: str) -> tuple[list, list[str], list[str]]:
    choices = getattr(response_obj, "choices", None) or []
    indexed_choices = []
    for ordinal, choice in enumerate(choices):
        index = getattr(choice, "index", None)
        if index is None and isinstance(choice, dict):
            index = choice.get("index")
        indexed_choices.append((index if isinstance(index, int) else ordinal, ordinal, choice))
    indexed_choices.sort(key=lambda item: (item[0], item[1]))

    output_items: list[dict] = []
    output_texts: list[str] = []
    tool_args: list[str] = []

    for index, ordinal, choice in indexed_choices:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = _normalize_message_content(
            getattr(message, "content", None) if message is not None else ""
        )
        output_texts.append(content)
        message_id = f"msg_{response_id}_{index}"
        output_items.append(
            build_output_message_item(item_id=message_id, text=content, status="completed")
        )
        for call_index, call in enumerate(_extract_tool_calls(message)):
            call_id = call.get("id") or f"call_{index}_{call_index}"
            arguments = call.get("arguments") or ""
            tool_args.append(arguments)
            output_items.append(
                {
                    "id": f"tool_{call_id}",
                    "type": "tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": call.get("name") or "",
                    "arguments": arguments,
                }
            )

    if not output_items and not choices:
        output_items.append(
            build_output_message_item(item_id=f"msg_{response_id}", text="", status="completed")
        )
    return output_items, output_texts, tool_args


def _compute_usage_from_outputs(messages: list, model_name: str, output_texts: list[str], tool_args: list[str]) -> dict:
    input_tokens = count_messages_tokens(messages, model=model_name)
    output_tokens = 0
    for text in output_texts:
        if text:
            output_tokens += count_text_tokens(text, model=model_name)
    for args in tool_args:
        if args:
            output_tokens += count_text_tokens(args, model=model_name)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
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
    output_items, output_texts, tool_args = _build_output_items_from_chat(response_obj, response_id)
    computed_usage = _compute_usage_from_outputs(messages, model_name, output_texts, tool_args)
    usage_payload = _normalize_usage(getattr(response_obj, "usage", None))
    if usage_payload.get("input_tokens") is None:
        usage_payload["input_tokens"] = computed_usage["input_tokens"]
    if usage_payload.get("output_tokens") is None:
        usage_payload["output_tokens"] = computed_usage["output_tokens"]
    if usage_payload.get("total_tokens") is None:
        usage_payload["total_tokens"] = computed_usage["total_tokens"]
    return build_response_completed_payload(
        response_id=response_id,
        created=created,
        model_id=model_id,
        output_items=output_items,
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
        dumped_items = [dump_output_item(item) for item in output_items]
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model_id,
            "status": "completed",
            "output": dumped_items,
            "usage": usage_payload,
        }
