"""SSE streaming helpers for OpenAI passthrough."""
import json
import time
import openai

from ..services.openai_service import stream_chat_completion
from ..utils.logging import log_event
from ..utils.token_count import count_messages_tokens, count_text_tokens
from ..utils.responses_adapter import build_response_completed_payload


def _dump_event(event):
    if hasattr(event, "model_dump"):
        return event.model_dump()
    if hasattr(event, "dict"):
        return event.dict()
    if isinstance(event, dict):
        return event
    if isinstance(event, str):
        return event
    return {"data": str(event)}


def _get_first_choice(chunk):
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return choices[0]


def _dump_delta(delta):
    if delta is None:
        return {}
    if hasattr(delta, "model_dump"):
        return delta.model_dump(exclude_none=True)
    if isinstance(delta, dict):
        return {key: value for key, value in delta.items() if value is not None}
    return {}


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


def stream_openai_sse(
    stream,
    *,
    request_id: str | None = None,
    upstream_url: str | None = None,
    raw_response: bool = False,
):
    """Stream OpenAI SDK events as SSE without modification."""
    saw_done = False
    saw_event = False
    try:
        if raw_response:
            with stream as raw:
                for line in raw.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        try:
                            line = line.decode("utf-8")
                        except Exception:
                            line = line.decode("utf-8", errors="ignore")
                    if line.startswith("event:"):
                        saw_event = True
                    if line.strip() == "data: [DONE]":
                        saw_done = True
                    yield f"{line}\n\n"
        else:
            for event in stream:
                payload = _dump_event(event)
                if payload == "[DONE]":
                    saw_done = True
                    yield "data: [DONE]\n\n"
                    continue
                if isinstance(payload, dict) and payload.get("data") == "[DONE]":
                    saw_done = True
                    yield "data: [DONE]\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
    except openai.APIStatusError as e:
        log_event(
            40,
            "upstream_error",
            request_id=request_id or "",
            upstream_url=upstream_url,
            status=getattr(e, "status_code", None),
            body=str(e),
            payload=None,
        )
        payload = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(payload)}\n\n"
    except Exception as e:
        log_event(
            40,
            "upstream_error",
            request_id=request_id or "",
            upstream_url=upstream_url,
            status=None,
            body=str(e),
            payload=None,
        )
        payload = {"error": {"message": str(e), "type": "internal_error"}}
        yield f"data: {json.dumps(payload)}\n\n"
    if not saw_done and not (raw_response and saw_event):
        yield "data: [DONE]\n\n"


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
        message_output_index = None
        tool_items: dict[int, dict] = {}
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

        def ensure_message_item():
            nonlocal message_output_index
            if message_output_index is not None:
                return
            message_output_index = 0
            added_event = {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": message_output_index,
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
            text_added_event = {
                "type": "response.output_text.added",
                "response_id": response_id,
                "item_id": output_item_id,
                "output_index": message_output_index,
                "content_index": 0,
                "text": "",
            }
            yield "event: response.output_text.added\n"
            yield f"data: {json.dumps(text_added_event)}\n\n"

        for chunk in stream:
            choice = _get_first_choice(chunk)
            if not choice:
                continue
            delta_obj = getattr(choice, "delta", None)
            delta = getattr(delta_obj, "content", None)
            if isinstance(delta, str) and delta:
                output_text += delta
            tool_calls = getattr(delta_obj, "tool_calls", None)
            if not isinstance(tool_calls, list) or not tool_calls:
                function_call = getattr(delta_obj, "function_call", None)
                if isinstance(function_call, dict):
                    tool_calls = [{"index": 0, "function": function_call}]
            if (isinstance(delta, str) and delta) or (isinstance(tool_calls, list) and tool_calls):
                yield from ensure_message_item()
            if isinstance(tool_calls, list):
                for i, call in enumerate(tool_calls):
                    call_payload = _dump_delta(call)
                    index = call_payload.get("index", i)
                    if not isinstance(index, int):
                        try:
                            index = int(index)
                        except (TypeError, ValueError):
                            index = i
                    call_id = call_payload.get("id") or f"call_{index}"
                    function = call_payload.get("function") or {}
                    name = function.get("name")
                    args_delta = function.get("arguments") or ""
                    state = tool_items.get(index)
                    if state is None:
                        output_index = index + 1 if isinstance(index, int) and index >= 0 else len(tool_items) + 1
                        state = {
                            "output_index": output_index,
                            "item_id": f"tool_{call_id}",
                            "call_id": call_id,
                            "name": name or "",
                            "arguments": "",
                        }
                        tool_items[index] = state
                        tool_added_event = {
                            "type": "response.output_item.added",
                            "response_id": response_id,
                            "output_index": output_index,
                            "item": {
                                "id": state["item_id"],
                                "type": "tool_call",
                                "status": "in_progress",
                                "call_id": call_id,
                                "name": state["name"],
                                "arguments": "",
                            },
                        }
                        yield "event: response.output_item.added\n"
                        yield f"data: {json.dumps(tool_added_event)}\n\n"
                    if name and not state["name"]:
                        state["name"] = name
                    if isinstance(args_delta, str) and args_delta:
                        state["arguments"] += args_delta
                        args_event = {
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "item_id": state["item_id"],
                            "output_index": state["output_index"],
                            "delta": args_delta,
                        }
                        yield "event: response.function_call_arguments.delta\n"
                        yield f"data: {json.dumps(args_event)}\n\n"
            if isinstance(delta, str) and delta:
                delta_event = {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "item_id": output_item_id,
                    "delta": delta,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "sequence_number": sequence_number,
                }
                sequence_number += 1
                yield "event: response.output_text.delta\n"
                yield f"data: {json.dumps(delta_event)}\n\n"

        if message_output_index is not None:
            done_event = {
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": output_item_id,
                "text": output_text,
                "output_index": message_output_index,
                "content_index": 0,
                "sequence_number": sequence_number,
            }
            sequence_number += 1
            yield "event: response.output_text.done\n"
            yield f"data: {json.dumps(done_event)}\n\n"
            item_done_event = {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": message_output_index,
                "item": {
                    "id": output_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                },
            }
            yield "event: response.output_item.done\n"
            yield f"data: {json.dumps(item_done_event)}\n\n"
        for index in sorted(tool_items.keys()):
            state = tool_items.get(index)
            if not state:
                continue
            args_done_event = {
                "type": "response.function_call_arguments.done",
                "response_id": response_id,
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "arguments": state["arguments"],
            }
            yield "event: response.function_call_arguments.done\n"
            yield f"data: {json.dumps(args_done_event)}\n\n"
            tool_done_event = {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": state["output_index"],
                "item": {
                    "id": state["item_id"],
                    "type": "tool_call",
                    "status": "completed",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                },
            }
            yield "event: response.output_item.done\n"
            yield f"data: {json.dumps(tool_done_event)}\n\n"

        input_tokens = count_messages_tokens(messages, model=response_model)
        tool_args_text = "".join(state["arguments"] for state in tool_items.values() if state.get("arguments"))
        output_tokens = count_text_tokens(output_text + tool_args_text, model=response_model)
        output_items = []
        if message_output_index is not None:
            output_items.append(
                {
                    "id": output_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                }
            )
        for index in sorted(tool_items.keys()):
            state = tool_items.get(index)
            if not state:
                continue
            output_items.append(
                {
                    "id": state["item_id"],
                    "type": "tool_call",
                    "status": "completed",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                }
            )
        response_dump = build_response_completed_payload(
            response_id=response_id,
            created=created,
            model_id=response_model,
            output_items=output_items,
            usage_payload={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        completed_event = {
            "type": "response.completed",
            "response": response_dump,
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
