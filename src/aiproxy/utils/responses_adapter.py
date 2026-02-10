"""Helpers to build OpenAI /v1/responses payloads."""
from __future__ import annotations

import json
import time
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
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens_details": {"reasoning_tokens": None},
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
    input_details = dict(usage_payload.get("input_tokens_details") or {})
    if input_details.get("cached_tokens") is None:
        input_details["cached_tokens"] = 0 if input_tokens is not None else None
    output_details = dict(usage_payload.get("output_tokens_details") or {})
    if output_details.get("reasoning_tokens") is None:
        output_details["reasoning_tokens"] = 0 if output_tokens is not None else None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": input_details,
        "output_tokens_details": output_details,
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


def _normalize_reasoning(reasoning: Any) -> dict:
    if isinstance(reasoning, dict):
        return {
            "effort": reasoning.get("effort"),
            "summary": reasoning.get("summary"),
        }
    return {"effort": None, "summary": None}


def _normalize_text_format(text_payload: Any) -> dict:
    if isinstance(text_payload, dict):
        if "format" in text_payload and isinstance(text_payload["format"], dict):
            return {"format": text_payload["format"]}
        if "type" in text_payload:
            return {"format": text_payload}
    return {"format": {"type": "text"}}


def _normalize_metadata(metadata: Any) -> dict:
    return metadata if isinstance(metadata, dict) else {}


def _payload_has_input_file(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("type") == "input_file":
            return True
        for value in payload.values():
            if _payload_has_input_file(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_payload_has_input_file(item) for item in payload)
    return False


def _request_has_input_file(request_payload: dict | None) -> bool:
    if not isinstance(request_payload, dict):
        return False
    return _payload_has_input_file(request_payload.get("input")) or _payload_has_input_file(
        request_payload.get("messages")
    )


def _should_include_logprobs(request_payload: dict | None) -> bool:
    if not isinstance(request_payload, dict):
        return False
    include = request_payload.get("include")
    if isinstance(include, str):
        include = [include]
    if isinstance(include, list):
        for item in include:
            if item == "message.output_text.logprobs":
                return True
    if request_payload.get("logprobs"):
        return True
    if request_payload.get("top_logprobs") is not None:
        return True
    if _request_has_input_file(request_payload):
        return True
    return False


def build_response_envelope(
    *,
    response_id: str,
    created: int,
    model_id: str,
    status: str,
    output_items: list,
    usage_payload: dict | None,
    request_payload: dict | None = None,
    completed_at: int | None = None,
    error: dict | None = None,
    incomplete_details: dict | None = None,
) -> dict:
    wants_logprobs = _should_include_logprobs(request_payload)
    has_input_file = _request_has_input_file(request_payload)
    response_payload = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "completed_at": completed_at,
        "background": False if has_input_file else None,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model_id,
        "output": output_items,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "service_tier": "default" if has_input_file else None,
        "store": True,
        "temperature": 1,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0 if wants_logprobs else None,
        "top_p": 1,
        "truncation": "disabled",
        "usage": usage_payload,
        "user": None,
        "metadata": {},
    }
    if request_payload:
        if "instructions" in request_payload:
            response_payload["instructions"] = request_payload.get("instructions")
        elif isinstance(request_payload.get("system"), str):
            response_payload["instructions"] = request_payload.get("system")
        if "max_output_tokens" in request_payload:
            response_payload["max_output_tokens"] = request_payload.get("max_output_tokens")
        elif "max_tokens" in request_payload:
            response_payload["max_output_tokens"] = request_payload.get("max_tokens")
        if "parallel_tool_calls" in request_payload:
            response_payload["parallel_tool_calls"] = request_payload.get("parallel_tool_calls")
        if "previous_response_id" in request_payload:
            response_payload["previous_response_id"] = request_payload.get("previous_response_id")
        if "reasoning" in request_payload:
            response_payload["reasoning"] = _normalize_reasoning(request_payload.get("reasoning"))
        if "store" in request_payload:
            response_payload["store"] = request_payload.get("store")
        if "temperature" in request_payload:
            response_payload["temperature"] = request_payload.get("temperature")
        if "text" in request_payload:
            response_payload["text"] = _normalize_text_format(request_payload.get("text"))
        if "tool_choice" in request_payload:
            response_payload["tool_choice"] = request_payload.get("tool_choice")
        if "tools" in request_payload:
            response_payload["tools"] = request_payload.get("tools") or []
        if "top_p" in request_payload:
            response_payload["top_p"] = request_payload.get("top_p")
        if "truncation" in request_payload:
            response_payload["truncation"] = request_payload.get("truncation")
        if "user" in request_payload:
            response_payload["user"] = request_payload.get("user")
        if "metadata" in request_payload:
            response_payload["metadata"] = _normalize_metadata(request_payload.get("metadata"))
        if "background" in request_payload:
            response_payload["background"] = request_payload.get("background")
        if "max_tool_calls" in request_payload:
            response_payload["max_tool_calls"] = request_payload.get("max_tool_calls")
        if "service_tier" in request_payload:
            response_payload["service_tier"] = request_payload.get("service_tier")
        if "top_logprobs" in request_payload:
            response_payload["top_logprobs"] = request_payload.get("top_logprobs")
        for key in (
            "conversation",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
        ):
            if key in request_payload:
                response_payload[key] = request_payload.get(key)
    keep_null_background = isinstance(request_payload, dict) and "background" in request_payload
    keep_null_max_tool_calls = has_input_file or (
        isinstance(request_payload, dict) and "max_tool_calls" in request_payload
    )
    keep_null_service_tier = isinstance(request_payload, dict) and "service_tier" in request_payload
    keep_null_top_logprobs = isinstance(request_payload, dict) and "top_logprobs" in request_payload
    if response_payload.get("background") is None and not keep_null_background:
        response_payload.pop("background", None)
    if response_payload.get("max_tool_calls") is None and not keep_null_max_tool_calls:
        response_payload.pop("max_tool_calls", None)
    if response_payload.get("service_tier") is None and not keep_null_service_tier:
        response_payload.pop("service_tier", None)
    if response_payload.get("top_logprobs") is None and not keep_null_top_logprobs:
        response_payload.pop("top_logprobs", None)
    return response_payload


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


def _finish_reason_to_incomplete(reason: Any) -> dict | None:
    if reason == "length":
        return {"reason": "max_tokens"}
    if reason == "content_filter":
        return {"reason": "content_filter"}
    return None


def build_response_payload_from_chat(
    response_obj: Any,
    *,
    response_id: str,
    created: int,
    model_id: str,
    model_name: str,
    messages: list,
    request_payload: dict | None = None,
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
    choices = getattr(response_obj, "choices", None) or []
    finish_reason = None
    if choices:
        first_choice = choices[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        if finish_reason is None and isinstance(first_choice, dict):
            finish_reason = first_choice.get("finish_reason")
    incomplete_details = _finish_reason_to_incomplete(finish_reason)
    status = "incomplete" if incomplete_details else "completed"
    completed_at = None if status != "completed" else int(time.time())
    return build_response_completed_payload(
        response_id=response_id,
        created=created,
        model_id=model_id,
        output_items=output_items,
        usage_payload=usage_payload,
        request_payload=request_payload,
        status=status,
        completed_at=completed_at,
        incomplete_details=incomplete_details,
    )


def build_response_payload(
    *,
    response_id: str,
    created: int,
    model_id: str,
    content: str,
    usage_payload: dict,
    request_payload: dict | None = None,
) -> dict:
    output_items = [
        build_output_message_item(
            item_id=f"msg_{response_id}",
            text=content,
            status="completed",
        )
    ]
    return build_response_completed_payload(
        response_id=response_id,
        created=created,
        model_id=model_id,
        output_items=output_items,
        usage_payload=usage_payload,
        request_payload=request_payload,
        status="completed",
        completed_at=int(time.time()),
    )


def build_response_completed_payload(
    *,
    response_id: str,
    created: int,
    model_id: str,
    output_items: list,
    usage_payload: dict,
    request_payload: dict | None = None,
    status: str = "completed",
    completed_at: int | None = None,
    incomplete_details: dict | None = None,
    error: dict | None = None,
) -> dict:
    dumped_items = [dump_output_item(item) for item in output_items]
    if _should_include_logprobs(request_payload):
        for item in dumped_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and "logprobs" not in part:
                    part["logprobs"] = []
    return build_response_envelope(
        response_id=response_id,
        created=created,
        model_id=model_id,
        status=status,
        output_items=dumped_items,
        usage_payload=usage_payload,
        request_payload=request_payload,
        completed_at=completed_at,
        incomplete_details=incomplete_details,
        error=error,
    )
