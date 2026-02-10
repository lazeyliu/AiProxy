"""Request parameter normalization and mapping."""
import base64
import json


def _bytes_from_array(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return bytes(value)
    if isinstance(value, dict) and value.get("type") == "Buffer" and isinstance(value.get("data"), list):
        return bytes(value["data"])
    return None


def _payload_has_input_file(payload) -> bool:
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


def request_has_input_file(payload_dict) -> bool:
    """Detect whether a responses request contains an input_file part."""
    if not isinstance(payload_dict, dict):
        return False
    return _payload_has_input_file(payload_dict.get("input")) or _payload_has_input_file(
        payload_dict.get("messages")
    )


def _extract_image_url(part):
    if not isinstance(part, dict):
        return None
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    image = part.get("image")
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        url = image.get("url") or image.get("href")
        if isinstance(url, str):
            return url
        data = image.get("data")
        media_type = image.get("mediaType") or image.get("mimeType")
        raw = _bytes_from_array(data)
        if raw and isinstance(media_type, str) and media_type:
            encoded = base64.b64encode(raw).decode("ascii")
            return f"data:{media_type};base64,{encoded}"
    data = part.get("data")
    media_type = part.get("mediaType") or part.get("mimeType")
    raw = _bytes_from_array(data)
    if raw and isinstance(media_type, str) and media_type:
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
    return None


def normalize_messages_from_input(data):
    """Normalize /v1/responses input into chat-style messages."""
    if "messages" in data and isinstance(data["messages"], list):
        return data["messages"]

    input_data = data.get("input")
    if input_data is None:
        return []

    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if isinstance(item, dict):
                if "role" in item and "content" in item:
                    messages.append({"role": item["role"], "content": item["content"]})
                    continue
                if item.get("type") == "message":
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                                text_parts.append(part.get("text", ""))
                        content = "".join(text_parts)
                    messages.append({"role": role, "content": content})
                    continue
            messages.append({"role": "user", "content": json.dumps(item)})
        return messages

    return [{"role": "user", "content": str(input_data)}]


def extract_chat_params(payload_dict):
    """Return chat.completions params excluding model/messages/stream."""
    excluded = {"model", "messages", "stream"}
    allowed = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "n",
        "stop",
        "seed",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "metadata",
        "store",
        "parallel_tool_calls",
        "stream_options",
        "modalities",
    }
    params = {key: value for key, value in payload_dict.items() if key in allowed and key not in excluded}
    if "max_tokens" not in params and "max_output_tokens" in params:
        params["max_tokens"] = params.pop("max_output_tokens")
    else:
        params.pop("max_output_tokens", None)
    return params


def extract_chat_params_from_responses(payload_dict):
    """Map /v1/responses params to chat.completions params."""
    params = {}
    if "temperature" in payload_dict:
        params["temperature"] = payload_dict["temperature"]
    if "top_p" in payload_dict:
        params["top_p"] = payload_dict["top_p"]
    if "seed" in payload_dict:
        params["seed"] = payload_dict["seed"]
    if "stop" in payload_dict:
        params["stop"] = payload_dict["stop"]
    if "tools" in payload_dict:
        params["tools"] = _normalize_tools_for_chat(payload_dict["tools"])
    if "tool_choice" in payload_dict:
        params["tool_choice"] = _normalize_tool_choice_for_chat(payload_dict["tool_choice"])
    if "response_format" in payload_dict:
        params["response_format"] = payload_dict["response_format"]
    if "metadata" in payload_dict:
        params["metadata"] = payload_dict["metadata"]
    if "store" in payload_dict:
        params["store"] = payload_dict["store"]
    if "parallel_tool_calls" in payload_dict:
        params["parallel_tool_calls"] = payload_dict["parallel_tool_calls"]
    if "modalities" in payload_dict:
        params["modalities"] = payload_dict["modalities"]
    if "n" in payload_dict:
        params["n"] = payload_dict["n"]
    if "logprobs" in payload_dict:
        params["logprobs"] = payload_dict["logprobs"]
    if "top_logprobs" in payload_dict:
        params["top_logprobs"] = payload_dict["top_logprobs"]
    include = payload_dict.get("include")
    if isinstance(include, str):
        include = [include]
    if isinstance(include, list):
        for item in include:
            if item == "message.output_text.logprobs":
                params["logprobs"] = True
    if "max_output_tokens" in payload_dict:
        params["max_tokens"] = payload_dict["max_output_tokens"]
    return params


def _normalize_tools_for_chat(tools):
    """Convert responses-style tools into chat.completions format."""
    if not isinstance(tools, list):
        return tools
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            if "function" in tool:
                normalized.append(tool)
                continue
            func = {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            }
            func = {k: v for k, v in func.items() if v is not None}
            normalized.append({"type": "function", "function": func})
        # Drop non-function tools when falling back to chat.completions.
    return normalized


def _normalize_tool_choice_for_chat(tool_choice):
    """Normalize responses-style tool_choice into chat.completions format."""
    if not isinstance(tool_choice, dict):
        return tool_choice
    if "function" in tool_choice:
        return tool_choice
    if tool_choice.get("type") == "function" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    return tool_choice


def coerce_messages_for_chat(messages):
    """Normalize message content parts for chat.completions providers."""
    if not isinstance(messages, list):
        return messages
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue
        content = msg.get("content")
        # Convert content dicts with explicit type into supported forms.
        if isinstance(content, dict):
            part_type = content.get("type")
            if part_type in ("text", "input_text", "output_text") or "text" in content:
                new_msg = dict(msg)
                new_msg["content"] = (
                    content.get("text")
                    or content.get("input_text")
                    or content.get("output_text")
                    or ""
                )
                normalized.append(new_msg)
                continue
            if part_type in ("image_url", "image", "input_image"):
                new_msg = dict(msg)
                url = _extract_image_url(content)
                if url:
                    new_msg["content"] = [{"type": "image_url", "image_url": {"url": url}}]
                else:
                    new_msg["content"] = ""
                normalized.append(new_msg)
                continue
        if isinstance(content, list):
            ordered_parts = []
            has_image = False
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("text", "input_text", "output_text") or "text" in part:
                    text = part.get("text") or part.get("input_text") or part.get("output_text") or ""
                    ordered_parts.append(("text", text))
                elif part_type in ("image_url", "image", "input_image"):
                    url = _extract_image_url(part)
                    if url:
                        has_image = True
                        ordered_parts.append(("image", url))
            new_msg = dict(msg)
            if not ordered_parts:
                new_msg["content"] = ""
            elif has_image:
                parts = []
                for kind, value in ordered_parts:
                    if kind == "image":
                        parts.append({"type": "image_url", "image_url": {"url": value}})
                    elif kind == "text" and value:
                        parts.append({"type": "text", "text": value})
                new_msg["content"] = parts if parts else ""
            else:
                new_msg["content"] = "".join(value for _, value in ordered_parts)
            normalized.append(new_msg)
            continue
        normalized.append(msg)
    return normalized
