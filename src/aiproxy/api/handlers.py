"""Route handlers for AIProxy endpoints."""
import time
import json
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
from ..utils.params import (
    coerce_messages_for_chat,
    extract_chat_params,
    extract_chat_params_from_responses,
    normalize_messages_from_input,
)
from .schemas import ChatCompletionsRequest, CompletionsRequest, ResponsesRequest
from ..services.openai_service import (
    create_client,
    create_chat_completion,
    create_response,
    stream_response,
)
from .streaming import stream_chat_sse, stream_responses_sse, stream_responses_sse_from_chat


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
