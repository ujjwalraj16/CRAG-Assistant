"""Lightweight HTTP client and JSON extraction helpers for local Ollama inference."""

import json
import re
from typing import Any, Callable

import httpx

from app.config import settings


LLMCallable = Callable[[str, str | None, bool], str]

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached or returns an HTTP error."""


def call_ollama(
    prompt: str,
    system_prompt: str | None = None,
    expect_json: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout_seconds: float | None = None,
) -> str:
    """Send a chat completion request to the local Ollama server and return text output."""
    target_url = (base_url or settings.ollama_base_url).rstrip("/")
    target_model = model or settings.ollama_model
    timeout = timeout_seconds or settings.ollama_timeout_seconds

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if expect_json:
        payload["format"] = "json"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{target_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        raise OllamaUnavailableError(
            f"Unable to reach Ollama at {target_url} with model '{target_model}': {exc}"
        ) from exc

    message = data.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise OllamaUnavailableError("Ollama returned an empty response payload.")
    return content.strip()


def check_ollama_health(
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Check whether Ollama is running and whether the configured model is available locally."""
    target_url = (base_url or settings.ollama_base_url).rstrip("/")
    target_model = model or settings.ollama_model
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(f"{target_url}/api/tags")
            response.raise_for_status()
            data = response.json()
        models = [item.get("name", "") for item in data.get("models", [])]
        model_available = any(
            name == target_model or name.startswith(f"{target_model}:")
            for name in models
        )
        return {
            "reachable": True,
            "base_url": target_url,
            "configured_model": target_model,
            "model_available": model_available,
            "available_models": models,
        }
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        return {
            "reachable": False,
            "base_url": target_url,
            "configured_model": target_model,
            "model_available": False,
            "available_models": [],
            "error": str(exc),
        }


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM text, tolerating markdown fences and extra text."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response cannot be parsed as JSON.")

    stripped = raw_text.strip()

    # 1. Try direct parse first.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Try fenced ```json ... ``` block.
    fence_match = JSON_FENCE_RE.search(stripped)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Scan for the outermost JSON object inside surrounding prose.
    start_idx = stripped.find("{")
    end_idx = stripped.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        candidate = stripped[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in LLM output: {stripped[:160]}")

