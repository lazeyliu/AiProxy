"""SSE streaming helpers for OpenAI passthrough."""
import json
import openai

from ..utils.logging import log_event


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


def stream_openai_sse(
    stream,
    *,
    request_id: str | None = None,
    upstream_url: str | None = None,
    raw_response: bool = False,
):
    """Stream OpenAI SDK events as SSE without modification."""
    saw_done = False
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
    if not saw_done:
        yield "data: [DONE]\n\n"
