"""Pydantic request schemas for API endpoints."""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    input_text: Optional[str] = None
    input_audio: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    store: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    stream_options: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None

    class Config:
        # Allow forward-compat fields from clients; we ignore unsupported ones.
        extra = "allow"


class CompletionsRequest(BaseModel):
    model: Optional[str] = None
    prompt: Optional[Union[str, List[str]]] = None
    stream: bool = False

    class Config:
        # Allow forward-compat fields from clients; we ignore unsupported ones.
        extra = "allow"


class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: Optional[Union[str, List[Any]]] = None
    messages: Optional[List[ChatMessage]] = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    store: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    modalities: Optional[List[str]] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    include: Optional[List[str]] = None

    class Config:
        # Allow forward-compat fields from clients; we ignore unsupported ones.
        extra = "allow"
