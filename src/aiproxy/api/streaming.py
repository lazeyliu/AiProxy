"""SSE streaming helpers for chat and responses endpoints."""
import json
import time
import openai

from ..services.openai_service import stream_chat_completion
from ..utils.logging import log_event
from ..utils.token_count import count_messages_tokens, count_text_tokens


def _get_first_choice(chunk):
    """Return the first choice from a streaming chunk, if present."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return choices[0]


def _dump_delta(delta):
    """Serialize delta objects to dict, stripping None fields."""
    if delta is None:
        return {}
    if hasattr(delta, "model_dump"):
        return delta.model_dump(exclude_none=True)
    if isinstance(delta, dict):
        return {key: value for key, value in delta.items() if value is not None}
    return {}


def _stream_error_payload(message, response_id, created, response_model, error_type):
    """Build a stream-safe error chunk with choices for client compatibility."""
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": response_model,
        "choices": [
            {
                "delta": {"content": ""},
                "index": 0,
                "finish_reason": "error",
            }
        ],
        "error": {"message": message, "type": error_type},
    }


def _extract_tool_delta_text(delta_payload: dict) -> str:
    text_parts = []
    tool_calls = delta_payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    text_parts.append(name)
                args = function.get("arguments")
                if isinstance(args, str):
                    text_parts.append(args)
    function_call = delta_payload.get("function_call")
    if isinstance(function_call, dict):
        name = function_call.get("name")
        if isinstance(name, str):
            text_parts.append(name)
        args = function_call.get("arguments")
        if isinstance(args, str):
            text_parts.append(args)
    return "".join(text_parts)


def stream_chat_sse(client, model, messages, response_model, response_id, created, **kwargs):
    """Stream chat.completions and emit SSE in OpenAI format."""
    try:
        stream = stream_chat_completion(client, model, messages, **kwargs)
        output_text = ""
        output_tool_text = ""
        sent_usage = False
        for chunk in stream:
            choice = _get_first_choice(chunk)
            usage = getattr(chunk, "usage", None)
            usage_payload = None
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    usage_payload = usage.model_dump()
                elif hasattr(usage, "dict"):
                    usage_payload = usage.dict()
                elif isinstance(usage, dict):
                    usage_payload = usage
            if not choice:
                if usage_payload is not None:
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "choices": [],
                        "usage": usage_payload,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    sent_usage = True
                continue
            delta_payload = _dump_delta(getattr(choice, "delta", None))
            finish_reason = getattr(choice, "finish_reason", None)
            if not delta_payload and finish_reason is None:
                continue
            delta_content = delta_payload.get("content")
            if isinstance(delta_content, str):
                output_text += delta_content
            output_tool_text += _extract_tool_delta_text(delta_payload)
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response_model,
                "choices": [
                    {
                        "delta": delta_payload,
                        "index": getattr(choice, "index", 0),
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage_payload is not None:
                data["usage"] = usage_payload
                sent_usage = True
            yield f"data: {json.dumps(data)}\n\n"

        if not sent_usage:
            prompt_tokens = count_messages_tokens(messages, model=response_model)
            completion_tokens = count_text_tokens(output_text + output_tool_text, model=response_model)
            usage_payload = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response_model,
                "choices": [],
                "usage": usage_payload,
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except openai.APIConnectionError as e:
        error_payload = _stream_error_payload(
            f"Failed to connect to API: {e}",
            response_id,
            created,
            response_model,
            "api_connection_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.RateLimitError as e:
        error_payload = _stream_error_payload(
            f"API request exceeded rate limit: {e}",
            response_id,
            created,
            response_model,
            "rate_limit_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.APIStatusError as e:
        error_payload = _stream_error_payload(
            f"API returned an API Status Error: {e}",
            response_id,
            created,
            response_model,
            "api_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_payload = _stream_error_payload(
            f"An unexpected error occurred: {e}",
            response_id,
            created,
            response_model,
            "internal_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"


def _log_stream_error(
    err,
    request_id: str | None,
    upstream_url: str | None,
    request_payload: dict | None = None,
):
    response = getattr(err, "response", None)
    status = getattr(err, "status_code", None) or getattr(response, "status_code", None)
    body = None
    if response is not None:
        try:
            body = response.text
        except Exception:
            body = None
    if isinstance(body, str) and len(body) > 2000:
        body = body[:2000] + "...(truncated)"
    log_event(
        40,
        "upstream_error",
        request_id=request_id or "",
        upstream_url=upstream_url,
        status=status,
        body=body,
        payload=request_payload,
    )


def stream_responses_sse(
    stream,
    request_id: str | None = None,
    upstream_url: str | None = None,
    request_payload: dict | None = None,
):
    """Stream /v1/responses SSE events from the upstream responses API."""
    try:
        for event in stream:
            if hasattr(event, "model_dump"):
                payload = event.model_dump()
            elif hasattr(event, "dict"):
                payload = event.dict()
            elif isinstance(event, dict):
                payload = event
            else:
                payload = {"type": "response.output_text.delta", "delta": str(event)}
            event_type = payload.get("type", "response.output_text.delta")
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except openai.RateLimitError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "rate_limit_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIConnectionError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_connection_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIStatusError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "internal_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"


def stream_responses_sse_from_chat(
    client,
    model,
    messages,
    response_model,
    response_id,
    created,
    request_id: str | None = None,
    upstream_url: str | None = None,
    request_payload: dict | None = None,
    **kwargs,
):
    """Fallback: stream /v1/responses format using chat.completions."""
    try:
        stream = stream_chat_completion(client, model, messages, **kwargs)
        output_text = ""
        output_item_id = f"msg_{int(time.time() * 1000)}"
        sequence_number = 0
        created_event = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "model": response_model,
                "status": "in_progress",
                "output": [],
            },
        }
        yield "event: response.created\n"
        yield f"data: {json.dumps(created_event)}\n\n"
        added_event = {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": output_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            },
        }
        yield "event: response.output_item.added\n"
        yield f"data: {json.dumps(added_event)}\n\n"

        for chunk in stream:
            choice = _get_first_choice(chunk)
            if not choice:
                continue
            delta = getattr(getattr(choice, "delta", None), "content", None)
            if not delta:
                continue
            output_text += delta
            delta_event = {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": output_item_id,
                "delta": delta,
                "output_index": 0,
                "content_index": 0,
                "sequence_number": sequence_number,
            }
            sequence_number += 1
            yield "event: response.output_text.delta\n"
            yield f"data: {json.dumps(delta_event)}\n\n"

        done_event = {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": output_item_id,
            "text": output_text,
            "output_index": 0,
            "content_index": 0,
            "sequence_number": sequence_number,
        }
        sequence_number += 1
        yield "event: response.output_text.done\n"
        yield f"data: {json.dumps(done_event)}\n\n"

        input_tokens = count_messages_tokens(messages, model=response_model)
        output_tokens = count_text_tokens(output_text, model=response_model)
        completed_event = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "model": response_model,
                "status": "completed",
                "output": [
                    {
                        "id": output_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        }
        yield "event: response.completed\n"
        yield f"data: {json.dumps(completed_event)}\n\n"
    except openai.RateLimitError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "rate_limit_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIConnectionError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_connection_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIStatusError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "internal_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
