"""Route handlers for AIProxy endpoints."""
import time
import json
import os
import uuid
import base64
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime

import openai
from flask import Response, jsonify, render_template, request, stream_with_context, g
from pydantic import ValidationError

from ..core.config import (
    get_config_errors,
    get_models_response,
    load_config,
    resolve_model_config,
    get_logging_config,
    get_responses_config,
)
from ..utils.http import error_response
from ..utils.logging import get_file_logger, log_event, redact_payload
from ..utils.token_count import count_messages_tokens, count_text_tokens
from ..utils.local_store import LocalStore
from ..utils.params import (
    coerce_messages_for_chat,
    extract_chat_params,
    extract_chat_params_from_responses,
    normalize_messages_from_input,
)
from .schemas import ChatCompletionsRequest, CompletionsRequest, EmbeddingsRequest, ResponsesRequest
from ..services.openai_service import (
    create_client,
    create_chat_completion,
    create_response,
    stream_response,
)
from .streaming import stream_chat_sse, stream_responses_sse, stream_responses_sse_from_chat

_local_store = None


def _get_local_store(settings) -> LocalStore:
    global _local_store
    if _local_store is None:
        base_dir = os.getenv("LOCAL_STORE_DIR", settings.log_dir)
        _local_store = LocalStore(base_dir)
    return _local_store


def _get_default_model_id() -> str:
    config = load_config()
    return config.get("defaults", {}).get("model") or "unknown-model"


def _build_client(settings, base_url, api_key):
    return create_client(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.upstream_timeout,
        max_retries=settings.upstream_max_retries,
    )


def _accepts_sse(req):
    accept = (req.headers.get("Accept") or "").lower()
    return "text/event-stream" in accept


def _ordered_payload(payload: dict, key_order: list[str]) -> dict:
    ordered = {key: payload[key] for key in key_order if key in payload}
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _resolve_request_model(payload_dict):
    requested_id = payload_dict.get("model")
    resolved = resolve_model_config(requested_id)
    if resolved is None:
        return None, "Model not found"
    if not resolved.get("api_key"):
        return None, "Provider API key not configured"
    allowed_models = getattr(g, "allowed_models", None)
    if allowed_models is not None and resolved.get("id") not in allowed_models:
        return None, "Model not allowed for this key"
    g.resolved_model = resolved.get("id")
    g.resolved_provider = resolved.get("provider_name") or resolved.get("base_url")
    g.resolved_provider_url = resolved.get("base_url")
    return resolved, None


def _make_response_id(prefix):
    return f"{prefix}-{int(time.time() * 1000)}"


def _extract_usage_dict(usage_obj):
    if usage_obj is None:
        return None
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    if hasattr(usage_obj, "dict"):
        return usage_obj.dict()
    if isinstance(usage_obj, dict):
        return usage_obj
    return None


def _log_upstream_error(
    err: Exception,
    request_id: str,
    upstream_url: str,
    payload: dict | None = None,
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
        request_id=request_id,
        upstream_url=upstream_url,
        status=status,
        body=body,
        payload=payload,
    )


def _build_upstream_url(base_url: str, endpoint: str) -> str:
    if not base_url:
        return ""
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _build_upstream_url_with_query(base_url: str, endpoint: str) -> str:
    url = _build_upstream_url(base_url, endpoint)
    if not url:
        return url
    if request.query_string:
        qs = request.query_string.decode()
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{qs}"
    return url


def _build_multipart_body():
    boundary = uuid.uuid4().hex
    body = bytearray()

    def add_bytes(value: bytes):
        body.extend(value)

    def add_line(value: str = ""):
        add_bytes(value.encode("utf-8"))
        add_bytes(b"\r\n")

    for key, value in request.form.items(multi=True):
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{key}"')
        add_line()
        add_line(str(value))

    for key, storage in request.files.items(multi=True):
        filename = storage.filename or "file"
        content_type = storage.mimetype or "application/octet-stream"
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"')
        add_line(f"Content-Type: {content_type}")
        add_line()
        file_bytes = storage.stream.read()
        add_bytes(file_bytes)
        add_bytes(b"\r\n")
        try:
            storage.stream.seek(0)
        except Exception:
            pass

    add_line(f"--{boundary}--")
    return bytes(body), boundary


def _forward_request(
    settings,
    upstream_url: str,
    api_key: str,
    *,
    body_override: bytes | None = None,
    headers_override: dict | None = None,
    expect_json: bool = False,
):
    if not upstream_url:
        return error_response("Upstream URL not configured", 502, "api_error")
    body = request.get_data() if body_override is None else body_override
    headers = {}
    for key, value in request.headers:
        if key.lower() in ("host", "content-length", "authorization"):
            continue
        headers[key] = value
    if headers_override:
        headers.update(headers_override)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(upstream_url, data=body, headers=headers, method=request.method)
    try:
        with urlopen(req, timeout=settings.upstream_timeout) as resp:
            status = resp.status
            resp_body = resp.read()
            resp_headers = resp.headers
    except HTTPError as e:
        status = e.code
        resp_body = e.read()
        resp_headers = e.headers
    except URLError as e:
        return error_response(str(e), 502, "api_connection_error")
    content_type = resp_headers.get("Content-Type", "")
    if expect_json and "application/json" not in content_type.lower():
        return error_response(
            f"Upstream returned non-JSON response (status {status})",
            status or 502,
            "api_error",
        )
    response = Response(resp_body, status=status)
    if content_type:
        response.headers["Content-Type"] = content_type
    return response


def _build_responses_payload(data: dict, resolved_model: str) -> dict:
    allowed = {
        "background",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "modalities",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
        "user",
        "stop",
    }
    payload = {key: data[key] for key in data if key in allowed}
    if "input" not in payload and isinstance(data.get("messages"), list):
        payload["input"] = data["messages"]
    if "text" not in payload and "response_format" in payload:
        payload["text"] = {"format": payload["response_format"]}
        payload.pop("response_format", None)
    payload["model"] = resolved_model
    return payload


def _apply_instructions(messages, instructions):
    if not instructions or not isinstance(instructions, str):
        return messages
    if not isinstance(messages, list):
        return messages
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return messages
    return [{"role": "system", "content": instructions}] + messages


def register_routes(app, settings):
    """Register Flask routes on the app."""
    store = _get_local_store(settings)

    def _list_response(items):
        items_sorted = sorted(items, key=lambda item: item.get("created_at", 0), reverse=True)
        first_id = items_sorted[0]["id"] if items_sorted else None
        last_id = items_sorted[-1]["id"] if items_sorted else None
        return jsonify(
            {
                "object": "list",
                "data": items_sorted,
                "first_id": first_id,
                "last_id": last_id,
                "has_more": False,
            }
        )

    def _not_found(name: str, item_id: str):
        return error_response(f"{name} '{item_id}' not found", 404, "invalid_request_error")

    def _parse_completion_window_seconds(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip().lower()
        match = re.match(r"^(\d+)\s*([smhd])$", text)
        if not match:
            return None
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            return amount
        if unit == "m":
            return amount * 60
        if unit == "h":
            return amount * 3600
        if unit == "d":
            return amount * 86400
        return None

    def _calculate_expires_at(created_at: int | None, completion_window) -> int | None:
        seconds = _parse_completion_window_seconds(completion_window)
        if not seconds:
            return None
        base = created_at or int(time.time())
        return base + seconds

    def _normalize_assistant_data(payload, existing=None):
        base = existing or {}
        return {
            "name": payload.get("name", base.get("name")),
            "description": payload.get("description", base.get("description")),
            "model": payload.get("model") or base.get("model") or _get_default_model_id(),
            "instructions": payload.get("instructions", base.get("instructions")),
            "tools": payload.get("tools", base.get("tools", [])) or [],
            "tool_resources": payload.get("tool_resources", base.get("tool_resources", {})) or {},
            "metadata": payload.get("metadata", base.get("metadata", {})) or {},
            "top_p": payload.get("top_p", base.get("top_p", 1.0)),
            "temperature": payload.get("temperature", base.get("temperature", 1.0)),
            "response_format": payload.get("response_format", base.get("response_format", "auto")),
        }

    def _normalize_thread_data(payload, existing=None):
        base = existing or {}
        return {
            "metadata": payload.get("metadata", base.get("metadata", {})) or {},
            "tool_resources": payload.get("tool_resources", base.get("tool_resources", {})) or {},
        }

    def _normalize_message_content(content):
        if content is None:
            return []
        if isinstance(content, list):
            normalized = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text") or {}
                    if isinstance(text, str):
                        text = {"value": text, "annotations": []}
                    if "annotations" not in text:
                        text["annotations"] = []
                    normalized.append({"type": "text", "text": text})
                elif isinstance(part, dict) and "text" in part and isinstance(part.get("text"), str):
                    normalized.append({"type": "text", "text": {"value": part["text"], "annotations": []}})
            return normalized
        if isinstance(content, str):
            return [{"type": "text", "text": {"value": content, "annotations": []}}]
        return [{"type": "text", "text": {"value": json.dumps(content), "annotations": []}}]

    def _normalize_message_data(payload, thread_id, run_id=None, assistant_id=None):
        run_id = run_id or payload.get("run_id")
        assistant_id = assistant_id or payload.get("assistant_id")
        return {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "run_id": run_id,
            "role": payload.get("role", "user"),
            "content": _normalize_message_content(payload.get("content")),
            "attachments": payload.get("attachments", []) or [],
            "metadata": payload.get("metadata", {}) or {},
            "status": "completed",
            "incomplete_details": None,
        }

    def _normalize_run_data(payload, thread_id, assistant_id=None, existing=None):
        base = existing or {}
        created_at = base.get("created_at", int(time.time()))
        status = payload.get("status", base.get("status", "queued"))
        completed_at = payload["completed_at"] if "completed_at" in payload else base.get("completed_at")
        cancelled_at = payload["cancelled_at"] if "cancelled_at" in payload else base.get("cancelled_at")
        started_at = payload["started_at"] if "started_at" in payload else base.get("started_at", created_at)
        if status == "completed" and completed_at is None:
            completed_at = created_at
        return {
            "thread_id": thread_id,
            "assistant_id": payload.get("assistant_id", assistant_id or base.get("assistant_id")),
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "cancelled_at": cancelled_at,
            "failed_at": payload.get("failed_at", base.get("failed_at")),
            "expires_at": payload.get("expires_at", base.get("expires_at")),
            "model": payload.get("model") or base.get("model") or _get_default_model_id(),
            "instructions": payload.get("instructions", base.get("instructions")),
            "tools": payload.get("tools", base.get("tools", [])),
            "metadata": payload.get("metadata", base.get("metadata", {})) or {},
            "response_format": payload.get("response_format", base.get("response_format", "auto")),
            "tool_choice": payload.get("tool_choice", base.get("tool_choice", "auto")),
            "parallel_tool_calls": payload.get(
                "parallel_tool_calls",
                base.get("parallel_tool_calls", True),
            ),
            "truncation_strategy": payload.get(
                "truncation_strategy",
                base.get("truncation_strategy", {"type": "auto", "last_messages": None}),
            ),
            "max_prompt_tokens": payload.get("max_prompt_tokens", base.get("max_prompt_tokens")),
            "max_completion_tokens": payload.get("max_completion_tokens", base.get("max_completion_tokens")),
            "temperature": payload.get("temperature", base.get("temperature")),
            "top_p": payload.get("top_p", base.get("top_p")),
            "incomplete_details": payload.get("incomplete_details", base.get("incomplete_details")),
            "last_error": payload.get("last_error", base.get("last_error")),
            "required_action": payload.get("required_action", base.get("required_action")),
            "usage": payload.get("usage", base.get("usage")),
        }

    def _normalize_run_step_data(
        thread_id,
        run_id,
        message_id,
        assistant_id=None,
        status: str = "completed",
        usage: dict | None = None,
    ):
        now = int(time.time())
        started_at = now if status in ("in_progress", "completed") else None
        completed_at = now if status == "completed" else None
        return {
            "thread_id": thread_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
            "type": "message_creation",
            "status": status,
            "step_details": {"type": "message_creation", "message_creation": {"message_id": message_id}},
            "last_error": None,
            "usage": usage,
            "started_at": started_at,
            "completed_at": completed_at,
        }

    def _compute_run_usage(thread_id: str, run: dict) -> dict:
        thread_messages = store.list_items("thread.message", parent_id=thread_id)
        run_created_at = run.get("created_at", 0)
        prompt_messages = [msg for msg in thread_messages if msg.get("created_at", 0) <= run_created_at]
        completion_messages = [
            msg for msg in thread_messages if msg.get("run_id") == run.get("id") and msg.get("role") == "assistant"
        ]
        model = run.get("model")
        prompt_tokens = count_messages_tokens(prompt_messages, model=model)
        completion_tokens = count_messages_tokens(completion_messages, model=model)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _refresh_run_record(thread_id: str, run: dict) -> dict:
        existing_status = run.get("status")
        if existing_status in ("cancelled", "failed", "expired", "requires_action"):
            return run
        usage = _compute_run_usage(thread_id, run)
        thread_messages = store.list_items("thread.message", parent_id=thread_id)
        completion_messages = [
            msg for msg in thread_messages if msg.get("run_id") == run.get("id") and msg.get("role") == "assistant"
        ]
        has_activity = bool(completion_messages)
        has_output = count_messages_tokens(completion_messages, model=run.get("model")) > 0
        completed_at = run.get("completed_at")
        started_at = run.get("started_at")
        status = existing_status or "queued"
        if has_output:
            status = "completed"
            if not completed_at:
                latest_completion = max((msg.get("created_at", 0) for msg in completion_messages), default=0)
                completed_at = latest_completion or int(time.time())
            if not started_at:
                started_at = run.get("created_at")
        else:
            if has_activity:
                status = "in_progress"
                if not started_at:
                    started_at = run.get("created_at")
            else:
                if status not in ("queued", "in_progress"):
                    status = "queued"
                if status == "in_progress" and not started_at:
                    started_at = run.get("created_at")
                if status == "queued":
                    started_at = None

        updates = _normalize_run_data(
            {
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at,
                "usage": usage,
            },
            thread_id,
            run.get("assistant_id"),
            run,
        )
        updated = store.update_item("thread.run", run["id"], updates)
        return updated or run

    def _refresh_run_step_record(thread_id: str, step: dict) -> dict:
        run_id = step.get("run_id")
        if not run_id:
            return step
        run = store.get_item("thread.run", run_id)
        if not run:
            return step
        if step.get("status") in ("cancelled", "failed", "expired"):
            return step
        message_id = (
            (step.get("step_details") or {}).get("message_creation", {}).get("message_id")
        )
        message = store.get_item("thread.message", message_id) if message_id else None
        prompt_messages = [
            msg
            for msg in store.list_items("thread.message", parent_id=thread_id)
            if msg.get("created_at", 0) <= run.get("created_at", 0)
        ]
        prompt_tokens = count_messages_tokens(prompt_messages, model=run.get("model"))
        completion_tokens = count_messages_tokens([message], model=run.get("model")) if message else 0
        status = "completed" if completion_tokens > 0 else "in_progress"
        updates = {
            "status": status,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "started_at": step.get("started_at") or run.get("started_at") or run.get("created_at") or int(time.time()),
            "completed_at": step.get("completed_at"),
        }
        if status == "completed" and updates["completed_at"] is None:
            updates["completed_at"] = message.get("created_at") if message else int(time.time())
        updated = store.update_item("thread.run.step", step["id"], updates)
        return updated or step

    def _extract_response_output_text(output) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    if isinstance(item.get("output_text"), str):
                        parts.append(item.get("output_text"))
                    content = item.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    if isinstance(content, list):
                        for piece in content:
                            if not isinstance(piece, dict):
                                continue
                            if piece.get("type") in ("output_text", "text"):
                                text = piece.get("text")
                                if isinstance(text, dict):
                                    parts.append(str(text.get("value") or text.get("text") or ""))
                                else:
                                    parts.append(str(text or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    def _strip_upload_part_payload(item: dict) -> dict:
        if not isinstance(item, dict):
            return item
        sanitized = dict(item)
        sanitized.pop("data_b64", None)
        sanitized.pop("path", None)
        sanitized.pop("sha256", None)
        return sanitized

    def _normalize_upload_part_object(item: dict) -> dict:
        if not isinstance(item, dict):
            return item
        if item.get("object") == "upload.part":
            return item
        normalized = dict(item)
        normalized["object"] = "upload.part"
        return normalized

    def _normalize_vector_store_data(payload, existing=None):
        base = existing or {}
        return {
            "name": payload.get("name", base.get("name")),
            "status": "completed",
            "usage_bytes": payload.get("usage_bytes", base.get("usage_bytes", 0)) or 0,
            "file_counts": payload.get(
                "file_counts",
                base.get(
                    "file_counts",
                    {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                ),
            ),
            "last_active_at": payload.get("last_active_at", base.get("last_active_at")),
            "last_used_at": payload.get("last_used_at", base.get("last_used_at")),
            "expires_after": payload.get("expires_after", base.get("expires_after")),
            "expires_at": payload.get("expires_at", base.get("expires_at")),
            "metadata": payload.get("metadata", base.get("metadata", {})) or {},
        }

    def _normalize_vector_store_file_data(vector_store_id, file_id):
        return {
            "vector_store_id": vector_store_id,
            "file_id": file_id,
            "status": "completed",
            "last_error": None,
        }

    def _normalize_file_data(payload, existing=None):
        base = existing or {}
        return {
            "bytes": payload.get("bytes", base.get("bytes", 0)) or 0,
            "filename": payload.get("filename", base.get("filename")),
            "purpose": payload.get("purpose", base.get("purpose")),
            "status": payload.get("status", base.get("status", "processed")),
            "status_details": payload.get("status_details", base.get("status_details")),
            "expires_at": payload.get("expires_at", base.get("expires_at")),
        }

    def _normalize_upload_data(payload, existing=None):
        base = existing or {}
        created_at = base.get("created_at")
        expires_at = payload.get("expires_at", base.get("expires_at"))
        if expires_at is None:
            expires_at = _calculate_expires_at(created_at, payload.get("completion_window") or "1h")
        return {
            "bytes": payload.get("bytes", base.get("bytes", 0)) or 0,
            "filename": payload.get("filename", base.get("filename")),
            "purpose": payload.get("purpose", base.get("purpose")),
            "status": payload.get("status", base.get("status", "pending")),
            "expires_at": expires_at,
            "mime_type": payload.get("mime_type", base.get("mime_type")),
            "file_id": payload.get("file_id", base.get("file_id")),
        }

    def _normalize_batch_data(payload, existing=None):
        base = existing or {}
        created_at = base.get("created_at")
        status = payload.get("status", base.get("status", "validating"))
        expires_at = payload.get("expires_at", base.get("expires_at"))
        if expires_at is None:
            expires_at = _calculate_expires_at(created_at, payload.get("completion_window") or base.get("completion_window"))
        now = int(time.time())
        in_progress_at = payload.get("in_progress_at", base.get("in_progress_at"))
        finalizing_at = payload.get("finalizing_at", base.get("finalizing_at"))
        completed_at = payload.get("completed_at", base.get("completed_at"))
        failed_at = payload.get("failed_at", base.get("failed_at"))
        cancelled_at = payload.get("cancelled_at", base.get("cancelled_at"))
        if status == "in_progress" and in_progress_at is None:
            in_progress_at = now
        if status == "finalizing" and finalizing_at is None:
            finalizing_at = now
        if status == "completed" and completed_at is None:
            completed_at = now
        if status == "failed" and failed_at is None:
            failed_at = now
        if status == "cancelled" and cancelled_at is None:
            cancelled_at = now
        base_counts = base.get("request_counts") or {"total": 0, "completed": 0, "failed": 0}
        payload_counts = payload.get("request_counts") or {}
        merged_counts = {**base_counts, **payload_counts}
        return {
            "endpoint": payload.get("endpoint", base.get("endpoint")),
            "input_file_id": payload.get("input_file_id", base.get("input_file_id")),
            "completion_window": payload.get("completion_window", base.get("completion_window", "24h")),
            "status": status,
            "output_file_id": payload.get("output_file_id", base.get("output_file_id")),
            "error_file_id": payload.get("error_file_id", base.get("error_file_id")),
            "request_counts": merged_counts,
            "metadata": payload.get("metadata", base.get("metadata", {})) or {},
            "in_progress_at": in_progress_at,
            "expires_at": expires_at,
            "finalizing_at": finalizing_at,
            "completed_at": completed_at,
            "failed_at": failed_at,
            "cancelled_at": cancelled_at,
        }

    @app.route('/', methods=['GET'])
    def index():
        return render_template("index.html")

    @app.route('/completions', methods=['POST'])
    @app.route('/v1/completions', methods=['POST'])
    def completions():
        try:
            data = request.get_json(silent=True) or {}
            if "model" not in data and "modelId" in data:
                data["model"] = data["modelId"]
            try:
                payload = CompletionsRequest.model_validate(data)
            except ValidationError as e:
                return error_response(str(e), 400, "invalid_request_error")
            payload_dict = payload.model_dump(exclude_none=True)
            prompt = payload.prompt
            if prompt is None:
                return error_response("No prompt provided", 400, "invalid_request_error")
            if isinstance(prompt, list):
                prompt_text = "\n".join(str(item) for item in prompt if item is not None)
            elif isinstance(prompt, str):
                prompt_text = prompt
            else:
                return error_response("Invalid prompt", 400, "invalid_request_error")
            resolved, error_message = _resolve_request_model(payload_dict)
            if error_message:
                return error_response(error_message, 400)
            params = extract_chat_params(payload_dict)
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")

            if payload.stream:
                return error_response(
                    "Streaming not supported for /v1/completions; use /v1/chat/completions",
                    400,
                    "invalid_request_error",
                )

            log_cfg = get_logging_config()
            upstream_payload = redact_payload(
                {
                    "model": resolved["model"],
                    "messages": [{"role": "user", "content": prompt_text}],
                    **params,
                },
                log_cfg.get("redact_keys", []),
            )
            if log_cfg.get("include_body"):
                try:
                    log_event(
                        20,
                        "upstream_request",
                        request_id=g.request_id,
                        provider=resolved.get("provider_name") or resolved.get("base_url"),
                        upstream_url=g.upstream_url,
                        payload=upstream_payload,
                    )
                except Exception as e:
                    log_event(40, "upstream_log_failed", error=str(e))

            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            try:
                response_obj = create_chat_completion(
                    client,
                    resolved["model"],
                    [{"role": "user", "content": prompt_text}],
                    **params,
                )
            except openai.RateLimitError as e:
                return error_response(str(e), 429, "rate_limit_error")
            except openai.APIConnectionError as e:
                return error_response(str(e), 502, "api_connection_error")
            except openai.APIStatusError as e:
                _log_upstream_error(e, g.request_id, g.upstream_url, payload=upstream_payload)
                status = getattr(e, "status_code", 502) or 502
                return error_response(str(e), status, "api_error")
            except Exception as e:
                return error_response(str(e), 500, "internal_error")

            choices = getattr(response_obj, "choices", None) or []
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None) if first_choice else None
            content = getattr(message, "content", None) if message else ""
            finish_reason = getattr(first_choice, "finish_reason", None) if first_choice else None
            usage = _extract_usage_dict(getattr(response_obj, "usage", None))
            if usage is None:
                prompt_tokens = count_text_tokens(prompt_text, model=resolved["model"])
                completion_tokens = count_text_tokens(content, model=resolved["model"])
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            return jsonify(
                {
                    "id": _make_response_id("cmpl"),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": resolved["id"],
                    "choices": [
                        {
                            "index": 0,
                            "text": content,
                            "finish_reason": finish_reason or "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": usage,
                }
            )

        except Exception as e:
            log_event(40, "completions_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/chat/completions', methods=['POST'])
    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        try:
            data = request.get_json(silent=True) or {}
            if "model" not in data and "modelId" in data:
                data["model"] = data["modelId"]
            try:
                payload = ChatCompletionsRequest.model_validate(data)
            except ValidationError as e:
                return error_response(str(e), 400, "invalid_request_error")
            payload_dict = payload.model_dump(exclude_none=True)
            messages = [msg.model_dump(exclude_none=True) for msg in payload.messages]
            messages = coerce_messages_for_chat(messages)
            stream = payload.stream
            if stream and not _accepts_sse(request):
                stream = False
            params = extract_chat_params(payload_dict)
            resolved, error_message = _resolve_request_model(payload_dict)
            if error_message:
                return error_response(error_message, 400)
            if not messages:
                return error_response("No messages provided", 400)
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")

            if settings.create_log:
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_logger = get_file_logger(settings.log_dir)
                    payload = "\n".join(
                        [
                            f"=== {timestamp} ===",
                            f"Model: {resolved['id']}",
                            f"ProviderModel: {resolved['model']}",
                            f"BaseUrl: {resolved['base_url']}",
                            f"IP: {request.remote_addr}",
                            json.dumps(data, indent=2, ensure_ascii=False),
                            "",
                        ]
                    )
                    file_logger.info(payload)
                except Exception as e:
                    log_event(40, "file_logging_failed", error=str(e))

            log_cfg = get_logging_config()
            upstream_payload = redact_payload(
                {"model": resolved["model"], "messages": messages, **params},
                log_cfg.get("redact_keys", []),
            )
            if log_cfg.get("include_body"):
                try:
                    log_event(
                        20,
                        "upstream_request",
                        request_id=g.request_id,
                        provider=resolved.get("provider_name") or resolved.get("base_url"),
                        upstream_url=g.upstream_url,
                        payload=upstream_payload,
                    )
                except Exception as e:
                    log_event(40, "upstream_log_failed", error=str(e))

            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if stream:
                response_id = _make_response_id("chatcmpl")
                created = int(time.time())
                return Response(
                    stream_with_context(
                        stream_chat_sse(
                            client,
                            resolved["model"],
                            messages,
                            resolved["id"],
                            response_id,
                            created,
                            **params,
                        )
                    ),
                    mimetype='text/event-stream',
                )

            try:
                response_obj = create_chat_completion(client, resolved["model"], messages, **params)
            except openai.RateLimitError as e:
                return error_response(str(e), 429, "rate_limit_error")
            except openai.APIConnectionError as e:
                return error_response(str(e), 502, "api_connection_error")
            except openai.APIStatusError as e:
                _log_upstream_error(e, g.request_id, g.upstream_url, payload=upstream_payload)
                status = getattr(e, "status_code", 502) or 502
                return error_response(str(e), status, "api_error")
            except Exception as e:
                return error_response(str(e), 500, "internal_error")

            response_payload = None
            if hasattr(response_obj, "model_dump"):
                response_payload = response_obj.model_dump()
            elif hasattr(response_obj, "dict"):
                response_payload = response_obj.dict()
            elif isinstance(response_obj, dict):
                response_payload = response_obj
            if response_payload:
                response_payload["model"] = resolved["id"]
                if "id" not in response_payload:
                    response_payload["id"] = _make_response_id("chatcmpl")
                if "object" not in response_payload:
                    response_payload["object"] = "chat.completion"
                if "created" not in response_payload:
                    response_payload["created"] = int(time.time())
            choices = (response_payload or {}).get("choices") if response_payload else (getattr(response_obj, "choices", None) or [])
            first_choice = choices[0] if choices else None
            message = (first_choice or {}).get("message") if isinstance(first_choice, dict) else getattr(first_choice, "message", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None) if message else ""
            finish_reason = (first_choice or {}).get("finish_reason") if isinstance(first_choice, dict) else getattr(first_choice, "finish_reason", None)
            usage = _extract_usage_dict((response_payload or {}).get("usage") if response_payload else getattr(response_obj, "usage", None))
            if usage is None:
                prompt_tokens = count_messages_tokens(messages, model=resolved["model"])
                completion_tokens = count_text_tokens(content, model=resolved["model"])
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                }
            legacy = request.path in ("/chat/completions",)
            if legacy:
                return jsonify(
                    {
                        "id": _make_response_id("cmpl"),
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": resolved["id"],
                        "choices": [
                            {
                                "index": 0,
                                "text": content,
                                "finish_reason": finish_reason or "stop",
                                "logprobs": None,
                            }
                        ],
                        "usage": usage,
                    }
                )
            if response_payload:
                response_payload["usage"] = response_payload.get("usage") or usage
                response_payload = _ordered_payload(
                    response_payload,
                    ["id", "object", "created", "model", "choices", "usage"],
                )
                return jsonify(response_payload)
            return jsonify(
                {
                    "id": _make_response_id("chatcmpl"),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved["id"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        except Exception as e:
            log_event(40, "chat_completions_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/responses', methods=['POST'])
    @app.route('/v1/responses', methods=['POST'])
    def responses():
        try:
            data = request.get_json(silent=True) or {}
            if "model" not in data and "modelId" in data:
                data["model"] = data["modelId"]
            try:
                payload = ResponsesRequest.model_validate(data)
            except ValidationError as e:
                return error_response(str(e), 400, "invalid_request_error")
            payload_dict = payload.model_dump(exclude_none=True)
            stream = payload.stream
            if stream and not _accepts_sse(request):
                stream = False
            resolved, error_message = _resolve_request_model(payload_dict)
            if error_message:
                return error_response(error_message, 400)
            responses_cfg = get_responses_config()
            model_responses = resolved.get("responses", {}) if isinstance(resolved, dict) else {}
            provider_responses = resolved.get("provider_responses", {}) if isinstance(resolved, dict) else {}
            mode = (
                (model_responses or {}).get("mode")
                or (provider_responses or {}).get("mode")
                or responses_cfg.get("mode", "auto")
            )

            if mode == "chat":
                fallback_messages = normalize_messages_from_input(payload_dict)
                fallback_messages = _apply_instructions(
                    fallback_messages,
                    payload_dict.get("instructions") or payload_dict.get("system"),
                )
                fallback_messages = coerce_messages_for_chat(fallback_messages)
                if not fallback_messages:
                    return error_response("No input provided", 400, "invalid_request_error")
                fallback_params = extract_chat_params_from_responses(payload_dict)
                g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")

                log_cfg = get_logging_config()
                fallback_payload = redact_payload(
                    {"model": resolved["model"], "messages": fallback_messages, **fallback_params},
                    log_cfg.get("redact_keys", []),
                )
                if log_cfg.get("include_body"):
                    try:
                        log_event(
                            20,
                            "upstream_request",
                            request_id=g.request_id,
                            provider=resolved.get("provider_name") or resolved.get("base_url"),
                            upstream_url=g.upstream_url,
                            payload=fallback_payload,
                        )
                    except Exception as e:
                        log_event(40, "upstream_log_failed", error=str(e))

                client = _build_client(settings, resolved["base_url"], resolved["api_key"])
                if stream:
                    response_id = _make_response_id("resp")
                    created = int(time.time())
                    return Response(
                        stream_with_context(
                            stream_responses_sse_from_chat(
                                client,
                                resolved["model"],
                                fallback_messages,
                                resolved["id"],
                                response_id,
                                created,
                                request_id=g.request_id,
                                upstream_url=g.upstream_url,
                                **fallback_params,
                            )
                        ),
                        mimetype='text/event-stream',
                    )
                try:
                    response_obj = create_chat_completion(client, resolved["model"], fallback_messages, **fallback_params)
                except openai.RateLimitError as e:
                    return error_response(str(e), 429, "rate_limit_error")
                except openai.APIConnectionError as e:
                    return error_response(str(e), 502, "api_connection_error")
                except openai.APIStatusError as e:
                    _log_upstream_error(e, g.request_id, g.upstream_url, payload=fallback_payload)
                    status = getattr(e, "status_code", 502) or 502
                    return error_response(str(e), status, "api_error")
                except Exception as e:
                    return error_response(str(e), 500, "internal_error")
                choices = getattr(response_obj, "choices", None) or []
                message = getattr(choices[0], "message", None) if choices else None
                content = getattr(message, "content", None) if message else ""
                usage = _extract_usage_dict(getattr(response_obj, "usage", None))
                if usage is None:
                    input_tokens = count_messages_tokens(fallback_messages, model=resolved["model"])
                    output_tokens = count_text_tokens(content, model=resolved["model"])
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                response_payload = {
                    "id": _make_response_id("resp"),
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": resolved["id"],
                    "status": "completed",
                    "output": [
                        {
                            "id": _make_response_id("msg"),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content, "annotations": []}],
                        }
                    ],
                    "usage": usage,
                }
                response_payload = _ordered_payload(
                    response_payload,
                    ["id", "object", "created_at", "model", "status", "output", "usage"],
                )
                return jsonify(response_payload)
            responses_payload = _build_responses_payload(data, resolved["model"])
            if "input" not in responses_payload:
                return error_response("No input provided", 400)
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "responses")

            log_cfg = get_logging_config()
            upstream_payload = redact_payload(responses_payload, log_cfg.get("redact_keys", []))
            if log_cfg.get("include_body"):
                try:
                    log_event(
                        20,
                        "upstream_request",
                        request_id=g.request_id,
                        provider=resolved.get("provider_name") or resolved.get("base_url"),
                        upstream_url=g.upstream_url,
                        payload=upstream_payload,
                    )
                except Exception as e:
                    log_event(40, "upstream_log_failed", error=str(e))

            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if stream:
                responses_payload.pop("stream", None)
                def response_stream():
                    try:
                        stream_iter = stream_response(client, **responses_payload)
                        for chunk in stream_responses_sse(
                            stream_iter,
                            request_id=g.request_id,
                            upstream_url=g.upstream_url,
                            request_payload=upstream_payload,
                        ):
                            yield chunk
                    except openai.NotFoundError:
                        # Provider doesn't support /responses; fall back to chat.completions.
                        if mode != "auto":
                            yield 'event: response.failed\n'
                            yield 'data: {"type":"response.failed","error":{"message":"Responses API not supported by provider","type":"api_error"}}\n\n'
                            return
                        fallback_messages = normalize_messages_from_input(payload_dict)
                        fallback_messages = _apply_instructions(
                            fallback_messages,
                            payload_dict.get("instructions") or payload_dict.get("system"),
                        )
                        fallback_messages = coerce_messages_for_chat(fallback_messages)
                        if not fallback_messages:
                            yield 'event: response.failed\n'
                            yield 'data: {"type":"response.failed","error":{"message":"No input provided","type":"invalid_request_error"}}\n\n'
                            return
                        fallback_params = extract_chat_params_from_responses(payload_dict)
                        g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")
                        fallback_payload = redact_payload(
                            {"model": resolved["model"], "messages": fallback_messages, **fallback_params},
                            log_cfg.get("redact_keys", []),
                        )
                        log_event(
                            20,
                            "responses_fallback",
                            request_id=g.request_id,
                            provider=resolved.get("provider_name") or resolved.get("base_url"),
                            upstream_url=g.upstream_url,
                            reason="responses_not_supported",
                        )
                        response_id = _make_response_id("resp")
                        created = int(time.time())
                        for chunk in stream_responses_sse_from_chat(
                            client,
                            resolved["model"],
                            fallback_messages,
                            resolved["id"],
                            response_id,
                            created,
                            request_id=g.request_id,
                            upstream_url=g.upstream_url,
                            request_payload=fallback_payload,
                            **fallback_params,
                        ):
                            yield chunk

                return Response(stream_with_context(response_stream()), mimetype='text/event-stream')

            try:
                response_obj = create_response(client, **responses_payload)
            except openai.NotFoundError:
                # Provider doesn't support /responses; fall back to chat.completions.
                if mode != "auto":
                    return error_response("Responses API not supported by provider", 502, "api_error")
                fallback_messages = normalize_messages_from_input(payload_dict)
                fallback_messages = _apply_instructions(
                    fallback_messages,
                    payload_dict.get("instructions") or payload_dict.get("system"),
                )
                fallback_messages = coerce_messages_for_chat(fallback_messages)
                if not fallback_messages:
                    return error_response("No input provided", 400, "invalid_request_error")
                fallback_params = extract_chat_params_from_responses(payload_dict)
                g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")
                fallback_payload = redact_payload(
                    {"model": resolved["model"], "messages": fallback_messages, **fallback_params},
                    log_cfg.get("redact_keys", []),
                )
                log_event(
                    20,
                    "responses_fallback",
                    request_id=g.request_id,
                    provider=resolved.get("provider_name") or resolved.get("base_url"),
                    upstream_url=g.upstream_url,
                    reason="responses_not_supported",
                )
                try:
                    response_obj = create_chat_completion(client, resolved["model"], fallback_messages, **fallback_params)
                except openai.RateLimitError as e:
                    return error_response(str(e), 429, "rate_limit_error")
                except openai.APIConnectionError as e:
                    return error_response(str(e), 502, "api_connection_error")
                except openai.APIStatusError as e:
                    _log_upstream_error(e, g.request_id, g.upstream_url, payload=fallback_payload)
                    status = getattr(e, "status_code", 502) or 502
                    return error_response(str(e), status, "api_error")
                except Exception as e:
                    return error_response(str(e), 500, "internal_error")
                choices = getattr(response_obj, "choices", None) or []
                message = getattr(choices[0], "message", None) if choices else None
                content = getattr(message, "content", None) if message else ""
                usage = _extract_usage_dict(getattr(response_obj, "usage", None))
                if usage is None:
                    input_tokens = count_messages_tokens(fallback_messages, model=resolved["model"])
                    output_tokens = count_text_tokens(content, model=resolved["model"])
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                response_payload = {
                    "id": _make_response_id("resp"),
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": resolved["id"],
                    "status": "completed",
                    "output": [
                        {
                            "id": _make_response_id("msg"),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content, "annotations": []}],
                        }
                    ],
                    "usage": usage,
                }
                return jsonify(response_payload)
            except openai.RateLimitError as e:
                return error_response(str(e), 429, "rate_limit_error")
            except openai.APIConnectionError as e:
                return error_response(str(e), 502, "api_connection_error")
            except openai.APIStatusError as e:
                _log_upstream_error(e, g.request_id, g.upstream_url, payload=upstream_payload)
                status = getattr(e, "status_code", 502) or 502
                return error_response(str(e), status, "api_error")
            except Exception as e:
                return error_response(str(e), 500, "internal_error")

            if hasattr(response_obj, "model_dump"):
                response_payload = response_obj.model_dump()
            elif hasattr(response_obj, "dict"):
                response_payload = response_obj.dict()
            elif isinstance(response_obj, dict):
                response_payload = response_obj
            else:
                response_payload = {"output": response_obj}

            if not response_payload.get("usage"):
                input_messages = normalize_messages_from_input(responses_payload)
                input_messages = coerce_messages_for_chat(input_messages)
                input_tokens = count_messages_tokens(input_messages, model=resolved["model"])
                output_text = _extract_response_output_text(response_payload.get("output"))
                output_tokens = count_text_tokens(output_text, model=resolved["model"])
                response_payload["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            response_payload = _ordered_payload(
                response_payload,
                ["id", "object", "created_at", "model", "status", "output", "usage"],
            )
            return jsonify(response_payload)

        except Exception as e:
            log_event(40, "responses_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/embeddings', methods=['POST'])
    @app.route('/v1/embeddings', methods=['POST'])
    def embeddings():
        try:
            data = request.get_json(silent=True) or {}
            if "model" not in data and "modelId" in data:
                data["model"] = data["modelId"]
            try:
                payload = EmbeddingsRequest.model_validate(data)
            except ValidationError as e:
                return error_response(str(e), 400, "invalid_request_error")
            payload_dict = payload.model_dump(exclude_none=True)
            if payload.input is None:
                return error_response("No input provided", 400, "invalid_request_error")

            resolved, error_message = _resolve_request_model(payload_dict)
            if error_message:
                return error_response(error_message, 400)

            params = {}
            if "input" in payload_dict:
                params["input"] = payload_dict["input"]
            if "dimensions" in payload_dict:
                params["dimensions"] = payload_dict["dimensions"]
            if "encoding_format" in payload_dict:
                params["encoding_format"] = payload_dict["encoding_format"]
            if "user" in payload_dict:
                params["user"] = payload_dict["user"]

            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "embeddings")

            log_cfg = get_logging_config()
            upstream_payload = redact_payload(
                {"model": resolved["model"], **params},
                log_cfg.get("redact_keys", []),
            )
            if log_cfg.get("include_body"):
                try:
                    log_event(
                        20,
                        "upstream_request",
                        request_id=g.request_id,
                        provider=resolved.get("provider_name") or resolved.get("base_url"),
                        upstream_url=g.upstream_url,
                        payload=upstream_payload,
                    )
                except Exception as e:
                    log_event(40, "upstream_log_failed", error=str(e))

            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            try:
                response_obj = client.embeddings.create(model=resolved["model"], **params)
            except openai.RateLimitError as e:
                return error_response(str(e), 429, "rate_limit_error")
            except openai.APIConnectionError as e:
                return error_response(str(e), 502, "api_connection_error")
            except openai.APIStatusError as e:
                _log_upstream_error(e, g.request_id, g.upstream_url, payload=upstream_payload)
                status = getattr(e, "status_code", 502) or 502
                return error_response(str(e), status, "api_error")
            except Exception as e:
                return error_response(str(e), 500, "internal_error")

            response_payload = None
            if hasattr(response_obj, "model_dump"):
                response_payload = response_obj.model_dump()
            elif hasattr(response_obj, "dict"):
                response_payload = response_obj.dict()
            elif isinstance(response_obj, dict):
                response_payload = response_obj
            if response_payload is None:
                return jsonify(response_obj)
            if response_payload.get("object") is None:
                response_payload["object"] = "list"
            if response_payload.get("model") is None:
                response_payload["model"] = resolved["id"]
            data_items = response_payload.get("data")
            if isinstance(data_items, list):
                for idx, item in enumerate(data_items):
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("object", "embedding")
                    item.setdefault("index", idx)
            if not response_payload.get("usage"):
                raw_input = params.get("input")
                texts = []
                if isinstance(raw_input, list):
                    texts = [str(v) for v in raw_input]
                elif raw_input is not None:
                    texts = [str(raw_input)]
                prompt_tokens = sum(count_text_tokens(text, model=resolved["model"]) for text in texts)
                response_payload["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                }
            response_payload = _ordered_payload(
                response_payload,
                ["object", "data", "model", "usage"],
            )
            return jsonify(response_payload)

        except Exception as e:
            log_event(40, "embeddings_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/images/generations', methods=['POST'])
    @app.route('/v1/images/generations', methods=['POST'])
    def images_generations():
        try:
            data = request.get_json(silent=True) or {}
            if "model" not in data and "modelId" in data:
                data["model"] = data["modelId"]
            resolved, error_message = _resolve_request_model(data)
            if error_message:
                return error_response(error_message, 400)
            payload = dict(data)
            payload["model"] = resolved["model"]
            body = json.dumps(payload).encode("utf-8")
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "images/generations")
            return _forward_request(
                settings,
                g.upstream_url,
                resolved.get("api_key", ""),
                body_override=body,
                headers_override={"Content-Type": "application/json"},
                expect_json=True,
            )
        except Exception as e:
            log_event(40, "images_generations_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/images/edits', methods=['POST'])
    @app.route('/v1/images/edits', methods=['POST'])
    def images_edits():
        try:
            model = request.form.get("model") or request.args.get("model")
            if not model and request.is_json:
                data = request.get_json(silent=True) or {}
                model = data.get("model") or data.get("modelId")
            resolved, error_message = _resolve_request_model({"model": model} if model else {})
            if error_message:
                return error_response(error_message, 400)
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "images/edits")
            body, boundary = _build_multipart_body()
            return _forward_request(
                settings,
                g.upstream_url,
                resolved.get("api_key", ""),
                body_override=body,
                headers_override={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                expect_json=True,
            )
        except Exception as e:
            log_event(40, "images_edits_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    def _resolve_json_model_payload():
        payload = request.get_json(silent=True) or {}
        if "model" not in payload and "modelId" in payload:
            payload["model"] = payload["modelId"]
        resolved, error_message = _resolve_request_model(payload)
        if error_message:
            return None, None, None, error_response(error_message, 400)
        if "model" in payload:
            payload["model"] = resolved["model"]
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return resolved, body, headers, None

    def _resolve_model_from_form_or_query():
        model = request.form.get("model") or request.args.get("model")
        if not model and request.is_json:
            data = request.get_json(silent=True) or {}
            model = data.get("model") or data.get("modelId")
        resolved, error_message = _resolve_request_model({"model": model} if model else {})
        if error_message:
            return None, error_response(error_message, 400)
        return resolved, None

    def _forward_openai_endpoint(endpoint: str, *, expect_json: bool):
        if request.is_json:
            resolved, body, headers, error = _resolve_json_model_payload()
            if error:
                return error
            g.upstream_url = _build_upstream_url_with_query(resolved.get("base_url", ""), endpoint)
            return _forward_request(
                settings,
                g.upstream_url,
                resolved.get("api_key", ""),
                body_override=body,
                headers_override=headers,
                expect_json=expect_json,
            )
        if request.mimetype == "multipart/form-data":
            resolved, error = _resolve_model_from_form_or_query()
            if error:
                return error
            body, boundary = _build_multipart_body()
            g.upstream_url = _build_upstream_url_with_query(resolved.get("base_url", ""), endpoint)
            return _forward_request(
                settings,
                g.upstream_url,
                resolved.get("api_key", ""),
                body_override=body,
                headers_override={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                expect_json=expect_json,
            )
        resolved, error_message = _resolve_request_model({})
        if error_message:
            return error_response("Model not specified and no default configured", 400, "invalid_request_error")
        g.upstream_url = _build_upstream_url_with_query(resolved.get("base_url", ""), endpoint)
        return _forward_request(
            settings,
            g.upstream_url,
            resolved.get("api_key", ""),
            expect_json=expect_json,
        )

    @app.route('/moderations', methods=['POST'])
    @app.route('/v1/moderations', methods=['POST'])
    def moderations():
        try:
            return _forward_openai_endpoint("moderations", expect_json=True)
        except Exception as e:
            log_event(40, "moderations_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/audio/transcriptions', methods=['POST'])
    @app.route('/v1/audio/transcriptions', methods=['POST'])
    def audio_transcriptions():
        try:
            return _forward_openai_endpoint("audio/transcriptions", expect_json=True)
        except Exception as e:
            log_event(40, "audio_transcriptions_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/audio/translations', methods=['POST'])
    @app.route('/v1/audio/translations', methods=['POST'])
    def audio_translations():
        try:
            return _forward_openai_endpoint("audio/translations", expect_json=True)
        except Exception as e:
            log_event(40, "audio_translations_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/audio/speech', methods=['POST'])
    @app.route('/v1/audio/speech', methods=['POST'])
    def audio_speech():
        try:
            return _forward_openai_endpoint("audio/speech", expect_json=False)
        except Exception as e:
            log_event(40, "audio_speech_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/assistants', methods=['GET', 'POST', 'PATCH'])
    @app.route('/assistants/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/assistants', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/assistants/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def assistants(subpath=None):
        payload = request.get_json(silent=True) or {}
        if "model" not in payload and "modelId" in payload:
            payload["model"] = payload["modelId"]
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("assistant"))
            if request.method in ("POST", "PATCH"):
                item = store.create_item("assistant", _normalize_assistant_data(payload), id_prefix="asst_")
                return jsonify(item)
        parts = subpath.split("/")
        assistant_id = parts[0]
        if len(parts) > 1 and parts[1] == "files":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("assistant.file", parent_id=assistant_id))
                if request.method == "POST":
                    file_id = payload.get("file_id") or payload.get("file")
                    item = store.create_item(
                        "assistant.file",
                        {"assistant_id": assistant_id, "file_id": file_id},
                        parent_id=assistant_id,
                        id_prefix="asst_file_",
                    )
                    return jsonify(item)
            if len(parts) == 3 and request.method == "DELETE":
                ok = store.delete_item("assistant.file", parts[2])
                return jsonify({"id": parts[2], "object": "assistant.file.deleted", "deleted": ok})
        if request.method == "GET":
            item = store.get_item("assistant", assistant_id)
            return jsonify(item) if item else _not_found("assistant", assistant_id)
        if request.method in ("POST", "PATCH"):
            existing = store.get_item("assistant", assistant_id)
            item = store.update_item("assistant", assistant_id, _normalize_assistant_data(payload, existing))
            return jsonify(item) if item else _not_found("assistant", assistant_id)
        if request.method == "DELETE":
            ok = store.delete_item("assistant", assistant_id)
            return jsonify({"id": assistant_id, "object": "assistant.deleted", "deleted": ok})
        return error_response("Method not allowed", 405, "invalid_request_error")

    @app.route('/threads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/threads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/threads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/threads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def threads(subpath=None):
        payload = request.get_json(silent=True) or {}
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("thread"))
            if request.method in ("POST", "PATCH"):
                messages = payload.pop("messages", None)
                thread = store.create_item("thread", _normalize_thread_data(payload), id_prefix="thread_")
                if isinstance(messages, list):
                    for msg in messages:
                        store.create_item(
                            "thread.message",
                            _normalize_message_data(msg or {}, thread["id"]),
                            parent_id=thread["id"],
                            id_prefix="msg_",
                        )
                return jsonify(thread)
        parts = subpath.split("/")
        thread_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("thread", thread_id)
                return jsonify(item) if item else _not_found("thread", thread_id)
            if request.method in ("POST", "PATCH"):
                existing = store.get_item("thread", thread_id)
                item = store.update_item("thread", thread_id, _normalize_thread_data(payload, existing))
                return jsonify(item) if item else _not_found("thread", thread_id)
            if request.method == "DELETE":
                ok = store.delete_item("thread", thread_id)
                return jsonify({"id": thread_id, "object": "thread.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "messages":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("thread.message", parent_id=thread_id))
                if request.method == "POST":
                    run_id = payload.get("run_id")
                    assistant_id = payload.get("assistant_id")
                    run = None
                    if run_id:
                        run = store.get_item("thread.run", run_id)
                        if not run:
                            return _not_found("thread.run", run_id)
                        assistant_id = assistant_id or run.get("assistant_id")
                    msg = store.create_item(
                        "thread.message",
                        _normalize_message_data(payload, thread_id, run_id, assistant_id),
                        parent_id=thread_id,
                        id_prefix="msg_",
                    )
                    if run_id and msg.get("role") == "assistant":
                        usage = _compute_run_usage(thread_id, run or {"id": run_id, "model": None, "created_at": 0})
                        completion_tokens = count_messages_tokens([msg], model=(run or {}).get("model"))
                        step_status = "completed" if completion_tokens > 0 else "in_progress"
                        store.create_item(
                            "thread.run.step",
                            _normalize_run_step_data(
                                thread_id,
                                run_id,
                                msg["id"],
                                assistant_id=assistant_id,
                                status=step_status,
                                usage={
                                    "prompt_tokens": usage["prompt_tokens"],
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": usage["prompt_tokens"] + completion_tokens,
                                },
                            ),
                            parent_id=run_id,
                            id_prefix="step_",
                        )
                        if run:
                            _refresh_run_record(thread_id, run)
                    return jsonify(msg)
            if len(parts) == 3 and request.method == "GET":
                item = store.get_item("thread.message", parts[2])
                return jsonify(item) if item else _not_found("thread.message", parts[2])
        if len(parts) >= 2 and parts[1] == "runs":
            if len(parts) == 2:
                if request.method == "GET":
                    runs = [
                        _refresh_run_record(thread_id, run)
                        for run in store.list_items("thread.run", parent_id=thread_id)
                    ]
                    return _list_response(runs)
                if request.method == "POST":
                    run = store.create_item(
                        "thread.run",
                        _normalize_run_data(payload, thread_id),
                        parent_id=thread_id,
                        id_prefix="run_",
                    )
                    run = _refresh_run_record(thread_id, run)
                    return jsonify(run)
            if len(parts) >= 3:
                run_id = parts[2]
                if len(parts) == 3 and request.method == "GET":
                    item = store.get_item("thread.run", run_id)
                    if not item:
                        return _not_found("thread.run", run_id)
                    item = _refresh_run_record(thread_id, item)
                    return jsonify(item)
                if len(parts) == 4 and parts[3] == "cancel" and request.method in ("POST", "PATCH"):
                    existing = store.get_item("thread.run", run_id)
                    if not existing:
                        return _not_found("thread.run", run_id)
                    updates = {"status": "cancelled", "cancelled_at": int(time.time())}
                    item = store.update_item(
                        "thread.run",
                        run_id,
                        _normalize_run_data(updates, thread_id, existing.get("assistant_id"), existing),
                    )
                    return jsonify(item)
                if len(parts) >= 4 and parts[3] == "steps":
                    if len(parts) == 4 and request.method == "GET":
                        steps = [
                            _refresh_run_step_record(thread_id, step)
                            for step in store.list_items("thread.run.step", parent_id=run_id)
                        ]
                        return _list_response(steps)
                    if len(parts) == 5 and request.method == "GET":
                        item = store.get_item("thread.run.step", parts[4])
                        if not item:
                            return _not_found("thread.run.step", parts[4])
                        item = _refresh_run_step_record(thread_id, item)
                        return jsonify(item)
        return error_response("Not found", 404, "invalid_request_error")

    @app.route('/vector_stores', methods=['GET', 'POST', 'PATCH'])
    @app.route('/vector_stores/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/vector_stores', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/vector_stores/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def vector_stores(subpath=None):
        payload = request.get_json(silent=True) or {}
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("vector_store"))
            if request.method in ("POST", "PATCH"):
                item = store.create_item("vector_store", _normalize_vector_store_data(payload), id_prefix="vs_")
                return jsonify(item)
        parts = subpath.split("/")
        store_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("vector_store", store_id)
                return jsonify(item) if item else _not_found("vector_store", store_id)
            if request.method in ("POST", "PATCH"):
                existing = store.get_item("vector_store", store_id)
                item = store.update_item("vector_store", store_id, _normalize_vector_store_data(payload, existing))
                return jsonify(item) if item else _not_found("vector_store", store_id)
            if request.method == "DELETE":
                ok = store.delete_item("vector_store", store_id)
                return jsonify({"id": store_id, "object": "vector_store.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "files":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("vector_store.file", parent_id=store_id))
                if request.method == "POST":
                    file_id = payload.get("file_id") or payload.get("file")
                    item = store.create_item(
                        "vector_store.file",
                        _normalize_vector_store_file_data(store_id, file_id),
                        parent_id=store_id,
                        id_prefix="vsf_",
                    )
                    existing_store = store.get_item("vector_store", store_id)
                    if existing_store:
                        counts = existing_store.get("file_counts") or {}
                        updated_counts = {
                            "in_progress": counts.get("in_progress", 0),
                            "completed": counts.get("completed", 0) + 1,
                            "failed": counts.get("failed", 0),
                            "cancelled": counts.get("cancelled", 0),
                            "total": counts.get("total", 0) + 1,
                        }
                        store.update_item(
                            "vector_store",
                            store_id,
                            _normalize_vector_store_data({"file_counts": updated_counts}, existing_store),
                        )
                    return jsonify(item)
            if len(parts) == 3 and request.method == "DELETE":
                ok = store.delete_item("vector_store.file", parts[2])
                return jsonify({"id": parts[2], "object": "vector_store.file.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "file_batches":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("vector_store.file_batch", parent_id=store_id))
                if request.method == "POST":
                    file_ids = payload.get("file_ids") or []
                    batch = store.create_item(
                        "vector_store.file_batch",
                        {
                            "vector_store_id": store_id,
                            "status": "completed",
                            "file_counts": {
                                "in_progress": 0,
                                "completed": len(file_ids),
                                "failed": 0,
                                "cancelled": 0,
                                "total": len(file_ids),
                            },
                            "completed_at": int(time.time()),
                        },
                        parent_id=store_id,
                        id_prefix="vsfb_",
                    )
                    for file_id in file_ids:
                        store.create_item(
                            "vector_store.file",
                            _normalize_vector_store_file_data(store_id, file_id),
                            parent_id=store_id,
                            id_prefix="vsf_",
                        )
                    existing_store = store.get_item("vector_store", store_id)
                    if existing_store:
                        counts = existing_store.get("file_counts") or {}
                        updated_counts = {
                            "in_progress": counts.get("in_progress", 0),
                            "completed": counts.get("completed", 0) + len(file_ids),
                            "failed": counts.get("failed", 0),
                            "cancelled": counts.get("cancelled", 0),
                            "total": counts.get("total", 0) + len(file_ids),
                        }
                        store.update_item(
                            "vector_store",
                            store_id,
                            _normalize_vector_store_data({"file_counts": updated_counts}, existing_store),
                        )
                    return jsonify(batch)
            if len(parts) >= 3:
                batch_id = parts[2]
                if len(parts) == 3 and request.method == "GET":
                    item = store.get_item("vector_store.file_batch", batch_id)
                    return jsonify(item) if item else _not_found("vector_store.file_batch", batch_id)
                if len(parts) == 4 and parts[3] == "files" and request.method == "GET":
                    return _list_response(store.list_items("vector_store.file", parent_id=store_id))
        return error_response("Not found", 404, "invalid_request_error")

    @app.route('/files', methods=['GET', 'POST', 'PATCH'])
    @app.route('/files/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/files', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/files/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def files(subpath=None):
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("file"))
            if request.method in ("POST", "PATCH"):
                if request.mimetype == "multipart/form-data":
                    storage = request.files.get("file")
                    if not storage:
                        return error_response("No file provided", 400, "invalid_request_error")
                    content = storage.read()
                    info = store.create_file(
                        storage.filename or "file",
                        storage.mimetype or "application/octet-stream",
                        content,
                    )
                    purpose = request.form.get("purpose")
                    item = store.create_item(
                        "file",
                        _normalize_file_data(
                            {
                                "bytes": info["bytes"],
                                "filename": info["filename"],
                                "purpose": purpose,
                                "status": info.get("status", "processed"),
                                "expires_at": None,
                            }
                        ),
                        item_id=info["id"],
                        id_prefix="file-",
                    )
                    purpose = request.form.get("purpose")
                    if purpose:
                        item = store.update_item("file", info["id"], {"purpose": purpose})
                    return jsonify(item)
                payload = request.get_json(silent=True) or {}
                item = store.create_item("file", _normalize_file_data(payload), id_prefix="file-")
                return jsonify(item)
        parts = subpath.split("/")
        file_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("file", file_id)
                return jsonify(item) if item else _not_found("file", file_id)
            if request.method == "DELETE":
                ok = store.delete_item("file", file_id)
                store.delete_file(file_id)
                return jsonify({"id": file_id, "object": "file.deleted", "deleted": ok})
        if len(parts) == 2 and parts[1] == "content" and request.method == "GET":
            file_info = store.get_file(file_id)
            if not file_info:
                return _not_found("file", file_id)
            with open(file_info["path"], "rb") as f:
                data = f.read()
            response = Response(data, status=200)
            response.headers["Content-Type"] = file_info.get("content_type") or "application/octet-stream"
            return response
        return error_response("Not found", 404, "invalid_request_error")

    @app.route('/uploads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/uploads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/uploads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/uploads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def uploads(subpath=None):
        payload = request.get_json(silent=True) or {}
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("upload"))
            if request.method in ("POST", "PATCH"):
                item = store.create_item("upload", _normalize_upload_data(payload), id_prefix="upload_")
                return jsonify(item)
        parts = subpath.split("/")
        upload_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("upload", upload_id)
                return jsonify(item) if item else _not_found("upload", upload_id)
            if request.method == "DELETE":
                ok = store.delete_item("upload", upload_id)
                store.delete_upload_parts(upload_id)
                for legacy_type in ("upload.part", "upload_part"):
                    for legacy in store.list_items(legacy_type, parent_id=upload_id):
                        store.delete_item(legacy_type, legacy["id"])
                return jsonify({"id": upload_id, "object": "upload.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "parts":
            existing_upload = store.get_item("upload", upload_id)
            if not existing_upload:
                return _not_found("upload", upload_id)
            if request.method == "GET":
                legacy_parts = store.list_items("upload_part", parent_id=upload_id)
                new_parts = store.list_items("upload.part", parent_id=upload_id)
                file_parts = store.list_upload_parts(upload_id)
                parts_items = [
                    _normalize_upload_part_object(_strip_upload_part_payload(item))
                    for item in (file_parts + legacy_parts + new_parts)
                ]
                parts_items = sorted(parts_items, key=lambda item: item.get("created_at", 0))
                return jsonify(
                    {
                        "object": "list",
                        "data": parts_items,
                        "first_id": parts_items[0]["id"] if parts_items else None,
                        "last_id": parts_items[-1]["id"] if parts_items else None,
                        "has_more": False,
                    }
                )
            if request.method == "POST":
                data = request.get_data() or b""
                part = store.create_upload_part(upload_id, data)
                updated_bytes = (existing_upload.get("bytes") or 0) + len(data)
                updates = _normalize_upload_data({"bytes": updated_bytes, "status": "in_progress"}, existing_upload)
                store.update_item("upload", upload_id, updates)
                return jsonify(_normalize_upload_part_object(_strip_upload_part_payload(part)))
        if len(parts) == 2 and parts[1] == "complete" and request.method in ("POST", "PATCH"):
            existing = store.get_item("upload", upload_id)
            if not existing:
                return _not_found("upload", upload_id)
            file_parts = store.list_upload_parts(upload_id)
            if file_parts:
                parts_list = file_parts
            else:
                new_parts = store.list_items("upload.part", parent_id=upload_id)
                parts_list = new_parts if new_parts else store.list_items("upload_part", parent_id=upload_id)
            parts_list = sorted(parts_list, key=lambda item: item.get("created_at", 0))
            filename = existing.get("filename") or "upload"
            mime_type = existing.get("mime_type") or "application/octet-stream"

            def iter_parts():
                for part in parts_list:
                    path = part.get("path")
                    if path and os.path.exists(path):
                        with open(path, "rb") as f:
                            while True:
                                chunk = f.read(1024 * 1024)
                                if not chunk:
                                    break
                                yield chunk
                    else:
                        data_b64 = part.get("data_b64")
                        if data_b64:
                            try:
                                yield base64.b64decode(data_b64)
                            except Exception:
                                continue

            file_info = store.create_file_from_iterator(filename, mime_type, iter_parts())
            file_item = store.create_item(
                "file",
                _normalize_file_data(
                    {
                        "bytes": file_info["bytes"],
                        "filename": file_info["filename"],
                        "purpose": existing.get("purpose"),
                        "status": file_info.get("status", "processed"),
                        "expires_at": None,
                    }
                ),
                item_id=file_info["id"],
                id_prefix="file-",
            )
            store.update_item(
                "upload",
                upload_id,
                _normalize_upload_data({"status": "completed", "file_id": file_item["id"]}, existing),
            )
            store.delete_upload_parts(upload_id)
            for legacy_type in ("upload.part", "upload_part"):
                for legacy in store.list_items(legacy_type, parent_id=upload_id):
                    store.delete_item(legacy_type, legacy["id"])
            return jsonify(file_item)
        if len(parts) == 2 and parts[1] == "cancel" and request.method in ("POST", "PATCH"):
            existing = store.get_item("upload", upload_id)
            if not existing:
                return _not_found("upload", upload_id)
            item = store.update_item(
                "upload",
                upload_id,
                _normalize_upload_data({"status": "cancelled"}, existing),
            )
            store.delete_upload_parts(upload_id)
            for legacy_type in ("upload.part", "upload_part"):
                for legacy in store.list_items(legacy_type, parent_id=upload_id):
                    store.delete_item(legacy_type, legacy["id"])
            return jsonify(item)
        return error_response("Not found", 404, "invalid_request_error")

    @app.route('/batches', methods=['GET', 'POST', 'PATCH'])
    @app.route('/batches/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/batches', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/batches/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def batches(subpath=None):
        payload = request.get_json(silent=True) or {}
        if not subpath:
            if request.method == "GET":
                return _list_response(store.list_items("batch"))
            if request.method in ("POST", "PATCH"):
                item = store.create_item("batch", _normalize_batch_data(payload), id_prefix="batch_")
                return jsonify(item)
        parts = subpath.split("/") if subpath else []
        batch_id = parts[0] if parts else ""
        if len(parts) == 2 and parts[1] == "cancel" and request.method in ("POST", "PATCH"):
            existing = store.get_item("batch", batch_id)
            if not existing:
                return _not_found("batch", batch_id)
            item = store.update_item("batch", batch_id, _normalize_batch_data({"status": "cancelled"}, existing))
            return jsonify(item)
        if request.method == "GET":
            item = store.get_item("batch", batch_id)
            return jsonify(item) if item else _not_found("batch", batch_id)
        if request.method in ("POST", "PATCH"):
            existing = store.get_item("batch", batch_id)
            item = store.update_item("batch", batch_id, _normalize_batch_data(payload, existing))
            return jsonify(item) if item else _not_found("batch", batch_id)
        if request.method == "DELETE":
            ok = store.delete_item("batch", batch_id)
            return jsonify({"id": batch_id, "object": "batch.deleted", "deleted": ok})
        return error_response("Not found", 404, "invalid_request_error")

    @app.route('/models', methods=['GET'])
    @app.route('/v1/models', methods=['GET'])
    def list_models():
        models = get_models_response()
        allowed_models = getattr(g, "allowed_models", None)
        if allowed_models is not None:
            models = [model for model in models if model.get("id") in allowed_models]
        return jsonify({"object": "list", "data": models})

    @app.route('/healthz', methods=['GET'])
    def health():
        errors = get_config_errors()
        status = "ok" if not errors else "warn"
        verbose = request.args.get("verbose") == "1"
        if not verbose:
            return jsonify({"status": status})
        return jsonify(
            {
                "status": status,
                "uptime_seconds": int(time.time() - app.config.get("APP_STARTED_AT", time.time())),
                "version": settings.app_version,
                "config_errors": errors,
            }
        )

    @app.route('/version', methods=['GET'])
    def version():
        return jsonify({"version": settings.app_version})

    @app.errorhandler(404)
    def handle_not_found(error):
        return error_response("Not found", 404, "invalid_request_error")

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        return error_response("Method not allowed", 405, "invalid_request_error")

    @app.errorhandler(413)
    def handle_payload_too_large(error):
        return error_response("Request body too large", 413, "invalid_request_error")
