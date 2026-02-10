"""Route handlers for AIProxy endpoints."""
import time
import json
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from flask import Response, jsonify, render_template, request, stream_with_context, g

from ..core.config import (
    get_config_errors,
    get_models_response,
    resolve_model_config,
)
from ..utils.http import error_response
from ..utils.logging import log_event
from ..services.openai_service import create_client
from .streaming import stream_openai_sse


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


def _build_multipart_body(overrides: dict | None = None):
    boundary = uuid.uuid4().hex
    body = bytearray()
    overrides = overrides or {}

    def add_bytes(value: bytes):
        body.extend(value)

    def add_line(value: str = ""):
        add_bytes(value.encode("utf-8"))
        add_bytes(b"\r\n")

    for key, value in request.form.items(multi=True):
        if key in overrides:
            value = overrides[key]
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


def _model_dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return obj


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
            resolved, error_message = _resolve_request_model(data)
            if error_message:
                return error_response(error_message, 400)
            payload = dict(data)
            payload["model"] = resolved["model"]
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "completions")
            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if payload.get("stream"):
                if not hasattr(client.completions, "with_streaming_response") or not hasattr(client.completions, "with_raw_response"):
                    stream = client.completions.create(**payload)
                    return Response(
                        stream_with_context(
                            stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url)
                        ),
                        mimetype='text/event-stream',
                    )
                stream = client.completions.with_streaming_response.create(**payload)
                return Response(
                    stream_with_context(
                        stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url, raw_response=True)
                    ),
                    mimetype='text/event-stream',
                )
            if not hasattr(client.completions, "with_raw_response"):
                response_obj = client.completions.create(**payload)
                return jsonify(_model_dump(response_obj))
            response_obj = client.completions.with_raw_response.create(**payload)
            if hasattr(response_obj, "parse"):
                return jsonify(_model_dump(response_obj.parse()))
            return jsonify(_model_dump(response_obj))
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
            resolved, error_message = _resolve_request_model(data)
            if error_message:
                return error_response(error_message, 400)
            payload = dict(data)
            payload["model"] = resolved["model"]
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "chat/completions")
            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if payload.get("stream"):
                if not hasattr(client.chat.completions, "with_streaming_response") or not hasattr(client.chat.completions, "with_raw_response"):
                    stream = client.chat.completions.create(**payload)
                    return Response(
                        stream_with_context(
                            stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url)
                        ),
                        mimetype='text/event-stream',
                    )
                stream = client.chat.completions.with_streaming_response.create(**payload)
                return Response(
                    stream_with_context(
                        stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url, raw_response=True)
                    ),
                    mimetype='text/event-stream',
                )
            if not hasattr(client.chat.completions, "with_raw_response"):
                response_obj = client.chat.completions.create(**payload)
                return jsonify(_model_dump(response_obj))
            response_obj = client.chat.completions.with_raw_response.create(**payload)
            if hasattr(response_obj, "parse"):
                return jsonify(_model_dump(response_obj.parse()))
            return jsonify(_model_dump(response_obj))
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
            resolved, error_message = _resolve_request_model(data)
            if error_message:
                return error_response(error_message, 400)
            payload = dict(data)
            payload["model"] = resolved["model"]
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "responses")
            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if payload.get("stream"):
                if not hasattr(client.responses, "with_streaming_response") or not hasattr(client.responses, "with_raw_response"):
                    stream = client.responses.create(**payload)
                    return Response(
                        stream_with_context(
                            stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url)
                        ),
                        mimetype='text/event-stream',
                    )
                stream = client.responses.with_streaming_response.create(**payload)
                return Response(
                    stream_with_context(
                        stream_openai_sse(stream, request_id=g.request_id, upstream_url=g.upstream_url, raw_response=True)
                    ),
                    mimetype='text/event-stream',
                )
            if not hasattr(client.responses, "with_raw_response"):
                response_obj = client.responses.create(**payload)
                return jsonify(_model_dump(response_obj))
            response_obj = client.responses.with_raw_response.create(**payload)
            if hasattr(response_obj, "parse"):
                return jsonify(_model_dump(response_obj.parse()))
            return jsonify(_model_dump(response_obj))
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
            resolved, error_message = _resolve_request_model(data)
            if error_message:
                return error_response(error_message, 400)
            payload = dict(data)
            payload["model"] = resolved["model"]
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "embeddings")
            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if not hasattr(client.embeddings, "with_raw_response"):
                response_obj = client.embeddings.create(**payload)
                return jsonify(_model_dump(response_obj))
            response_obj = client.embeddings.with_raw_response.create(**payload)
            if hasattr(response_obj, "parse"):
                return jsonify(_model_dump(response_obj.parse()))
            return jsonify(_model_dump(response_obj))

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
            g.upstream_url = _build_upstream_url(resolved.get("base_url", ""), "images/generations")
            payload = dict(data)
            payload["model"] = resolved["model"]
            client = _build_client(settings, resolved["base_url"], resolved["api_key"])
            if not hasattr(client.images, "with_raw_response"):
                response_obj = client.images.generate(**payload)
                return jsonify(_model_dump(response_obj))
            response_obj = client.images.with_raw_response.generate(**payload)
            if hasattr(response_obj, "parse"):
                return jsonify(_model_dump(response_obj.parse()))
            return jsonify(_model_dump(response_obj))
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
            body, boundary = _build_multipart_body({"model": resolved["model"]})
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
            body, boundary = _build_multipart_body({"model": resolved["model"]})
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
        try:
            endpoint = "assistants" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "assistants_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/threads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/threads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/threads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/threads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def threads(subpath=None):
        try:
            endpoint = "threads" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "threads_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/vector_stores', methods=['GET', 'POST', 'PATCH'])
    @app.route('/vector_stores/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/vector_stores', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/vector_stores/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def vector_stores(subpath=None):
        try:
            endpoint = "vector_stores" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "vector_stores_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/files', methods=['GET', 'POST', 'PATCH'])
    @app.route('/files/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/files', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/files/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def files(subpath=None):
        try:
            endpoint = "files" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "files_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/uploads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/uploads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/uploads', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/uploads/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def uploads(subpath=None):
        try:
            endpoint = "uploads" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "uploads_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

    @app.route('/batches', methods=['GET', 'POST', 'PATCH'])
    @app.route('/batches/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    @app.route('/v1/batches', methods=['GET', 'POST', 'PATCH'])
    @app.route('/v1/batches/<path:subpath>', methods=['GET', 'POST', 'DELETE', 'PATCH'])
    def batches(subpath=None):
        try:
            endpoint = "batches" + (f"/{subpath}" if subpath else "")
            return _forward_openai_endpoint(endpoint, expect_json=True)
        except Exception as e:
            log_event(40, "batches_error", error=str(e), request_id=g.request_id)
            return error_response(str(e), 500, "internal_error")

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
