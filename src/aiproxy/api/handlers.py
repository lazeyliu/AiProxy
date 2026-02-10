"""Route handlers for AIProxy endpoints."""
import time
import json
import os
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime

import openai
from flask import Response, jsonify, render_template, request, stream_with_context, g
from pydantic import ValidationError

from ..core.config import (
    get_config_errors,
    get_models_response,
    resolve_model_config,
    get_logging_config,
    get_responses_config,
)
from ..utils.http import error_response
from ..utils.logging import get_file_logger, log_event, redact_payload
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


def _build_client(settings, base_url, api_key):
    return create_client(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.upstream_timeout,
        max_retries=settings.upstream_max_retries,
    )


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


def _build_multipart_body(model_override: str | None):
    boundary = uuid.uuid4().hex
    body = bytearray()

    def add_bytes(value: bytes):
        body.extend(value)

    def add_line(value: str = ""):
        add_bytes(value.encode("utf-8"))
        add_bytes(b"\r\n")

    for key, value in request.form.items(multi=True):
        if key == "model" and model_override:
            value = model_override
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
        return jsonify({"object": "list", "data": items, "has_more": False})

    def _not_found(name: str, item_id: str):
        return error_response(f"{name} '{item_id}' not found", 404, "invalid_request_error")

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
                content = create_chat_completion(
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
                            "finish_reason": "stop",
                        }
                    ],
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
                content = create_chat_completion(client, resolved["model"], messages, **params)
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

            return jsonify(
                {
                    "id": _make_response_id("chatcmpl"),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved["id"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
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
                    content = create_chat_completion(client, resolved["model"], fallback_messages, **fallback_params)
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
                response_payload = {
                    "id": _make_response_id("resp"),
                    "object": "response",
                    "created": int(time.time()),
                    "model": resolved["id"],
                    "output": [
                        {
                            "id": _make_response_id("msg"),
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    ],
                }
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
                    content = create_chat_completion(client, resolved["model"], fallback_messages, **fallback_params)
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
                response_payload = {
                    "id": _make_response_id("resp"),
                    "object": "response",
                    "created": int(time.time()),
                    "model": resolved["id"],
                    "output": [
                        {
                            "id": _make_response_id("msg"),
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    ],
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
                return jsonify(response_obj.model_dump())
            if hasattr(response_obj, "dict"):
                return jsonify(response_obj.dict())
            return jsonify(response_obj)

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

            if hasattr(response_obj, "model_dump"):
                return jsonify(response_obj.model_dump())
            if hasattr(response_obj, "dict"):
                return jsonify(response_obj.dict())
            return jsonify(response_obj)

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
            body, boundary = _build_multipart_body(resolved["model"])
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

    def _forward_openai_endpoint(endpoint: str, *, expect_json: bool, rewrite_model_in_multipart: bool = False):
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
            model_override = resolved["model"] if rewrite_model_in_multipart else None
            body, boundary = _build_multipart_body(model_override)
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
            return _forward_openai_endpoint("audio/transcriptions", expect_json=True, rewrite_model_in_multipart=True)
        except Exception as e:
            log_event(40, "audio_transcriptions_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/audio/translations', methods=['POST'])
    @app.route('/v1/audio/translations', methods=['POST'])
    def audio_translations():
        try:
            return _forward_openai_endpoint("audio/translations", expect_json=True, rewrite_model_in_multipart=True)
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
                item = store.create_item("assistant", payload)
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
                    )
                    return jsonify(item)
            if len(parts) == 3 and request.method == "DELETE":
                ok = store.delete_item("assistant.file", parts[2])
                return jsonify({"id": parts[2], "object": "assistant.file.deleted", "deleted": ok})
        if request.method == "GET":
            item = store.get_item("assistant", assistant_id)
            return jsonify(item) if item else _not_found("assistant", assistant_id)
        if request.method in ("POST", "PATCH"):
            item = store.update_item("assistant", assistant_id, payload)
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
                thread = store.create_item("thread", payload)
                if isinstance(messages, list):
                    for msg in messages:
                        store.create_item(
                            "thread.message",
                            {"thread_id": thread["id"], **(msg or {})},
                            parent_id=thread["id"],
                        )
                return jsonify(thread)
        parts = subpath.split("/")
        thread_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("thread", thread_id)
                return jsonify(item) if item else _not_found("thread", thread_id)
            if request.method in ("POST", "PATCH"):
                item = store.update_item("thread", thread_id, payload)
                return jsonify(item) if item else _not_found("thread", thread_id)
            if request.method == "DELETE":
                ok = store.delete_item("thread", thread_id)
                return jsonify({"id": thread_id, "object": "thread.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "messages":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("thread.message", parent_id=thread_id))
                if request.method == "POST":
                    msg = store.create_item(
                        "thread.message",
                        {"thread_id": thread_id, **payload},
                        parent_id=thread_id,
                    )
                    return jsonify(msg)
            if len(parts) == 3 and request.method == "GET":
                item = store.get_item("thread.message", parts[2])
                return jsonify(item) if item else _not_found("thread.message", parts[2])
        if len(parts) >= 2 and parts[1] == "runs":
            if len(parts) == 2:
                if request.method == "GET":
                    return _list_response(store.list_items("thread.run", parent_id=thread_id))
                if request.method == "POST":
                    run_data = {"thread_id": thread_id, "status": "completed", **payload}
                    run = store.create_item("thread.run", run_data, parent_id=thread_id)
                    # Create a placeholder assistant message to keep clients unblocked.
                    store.create_item(
                        "thread.message",
                        {
                            "thread_id": thread_id,
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": ""}],
                        },
                        parent_id=thread_id,
                    )
                    return jsonify(run)
            if len(parts) >= 3:
                run_id = parts[2]
                if len(parts) == 3 and request.method == "GET":
                    item = store.get_item("thread.run", run_id)
                    return jsonify(item) if item else _not_found("thread.run", run_id)
                if len(parts) == 4 and parts[3] == "steps" and request.method == "GET":
                    return _list_response([])
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
                item = store.create_item("vector_store", payload)
                return jsonify(item)
        parts = subpath.split("/")
        store_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("vector_store", store_id)
                return jsonify(item) if item else _not_found("vector_store", store_id)
            if request.method in ("POST", "PATCH"):
                item = store.update_item("vector_store", store_id, payload)
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
                        {"vector_store_id": store_id, "file_id": file_id},
                        parent_id=store_id,
                    )
                    return jsonify(item)
            if len(parts) == 3 and request.method == "DELETE":
                ok = store.delete_item("vector_store.file", parts[2])
                return jsonify({"id": parts[2], "object": "vector_store.file.deleted", "deleted": ok})
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
                    info = store.create_file(storage.filename or "file", storage.mimetype or "application/octet-stream", content)
                    item = store.create_item("file", {k: v for k, v in info.items() if k != "id"}, item_id=info["id"])
                    purpose = request.form.get("purpose")
                    if purpose:
                        item = store.update_item("file", info["id"], {"purpose": purpose})
                    return jsonify(item)
                payload = request.get_json(silent=True) or {}
                item = store.create_item("file", payload)
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
                item = store.create_item("upload", payload)
                return jsonify(item)
        parts = subpath.split("/")
        upload_id = parts[0]
        if len(parts) == 1:
            if request.method == "GET":
                item = store.get_item("upload", upload_id)
                return jsonify(item) if item else _not_found("upload", upload_id)
            if request.method == "DELETE":
                ok = store.delete_item("upload", upload_id)
                return jsonify({"id": upload_id, "object": "upload.deleted", "deleted": ok})
        if len(parts) >= 2 and parts[1] == "parts" and request.method == "POST":
            data = request.get_data() or b""
            item = store.create_item(
                "upload.part",
                {"upload_id": upload_id, "bytes": len(data)},
                parent_id=upload_id,
            )
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
                item = store.create_item("batch", payload)
                return jsonify(item)
        batch_id = subpath.split("/")[0] if subpath else ""
        if request.method == "GET":
            item = store.get_item("batch", batch_id)
            return jsonify(item) if item else _not_found("batch", batch_id)
        if request.method in ("POST", "PATCH"):
            item = store.update_item("batch", batch_id, payload)
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
