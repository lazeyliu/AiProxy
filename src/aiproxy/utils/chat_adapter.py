"""Helpers to normalize chat completion responses to OpenAI model classes."""
from __future__ import annotations

from typing import Any


def _model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return obj


def ensure_chat_completion_model(obj: Any) -> Any:
    try:
        from openai.types.chat import ChatCompletion
    except Exception:
        return obj
    if isinstance(obj, ChatCompletion):
        return obj
    payload = _model_dump(obj)
    if not isinstance(payload, dict):
        return obj
    try:
        return ChatCompletion.model_validate(payload)
    except Exception:
        return obj
