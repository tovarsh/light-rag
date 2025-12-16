"""Vertex AI LLM and embedding bindings for LightRAG."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

import pipmaster as pm

if not pm.is_installed("numpy"):
    pm.install("numpy")

import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from lightrag.utils import logger, wrap_embedding_func_with_attrs

if not pm.is_installed("google-cloud-aiplatform"):
    pm.install("google-cloud-aiplatform")

from google.api_core import exceptions as google_api_exceptions  # type: ignore
from google.oauth2 import service_account  # type: ignore
import vertexai  # type: ignore
from vertexai.generative_models import (  # type: ignore
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
    SafetySetting,
)
from vertexai.language_models import TextEmbeddingModel  # type: ignore

LOG = logging.getLogger(__name__)

DEFAULT_VERTEX_API_ENDPOINT = "https://us-central1-aiplatform.googleapis.com"


def _resolve_project_id(project_id: str | None) -> str | None:
    return project_id or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


def _resolve_location(location: str | None) -> str | None:
    return location or os.getenv("VERTEX_LOCATION") or os.getenv("VERTEX_REGION")


def _resolve_endpoint(api_endpoint: str | None) -> str:
    return api_endpoint or os.getenv("VERTEX_API_ENDPOINT") or DEFAULT_VERTEX_API_ENDPOINT


def _resolve_credentials_path(credentials_path: str | None) -> str | None:
    return credentials_path or os.getenv("VERTEX_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def _load_credentials(credentials_path: str | None):
    if not credentials_path:
        return None

    return service_account.Credentials.from_service_account_file(credentials_path)


@lru_cache(maxsize=8)
def _init_vertex_ai(
    project_id: str | None,
    location: str | None,
    api_endpoint: str | None,
    credentials_path: str | None,
):
    init_kwargs: dict[str, Any] = {}
    resolved_project = _resolve_project_id(project_id)
    resolved_location = _resolve_location(location)
    resolved_endpoint = _resolve_endpoint(api_endpoint)
    resolved_credentials = _load_credentials(_resolve_credentials_path(credentials_path))

    if resolved_project:
        init_kwargs["project"] = resolved_project
    if resolved_location:
        init_kwargs["location"] = resolved_location
    if resolved_endpoint:
        init_kwargs["api_endpoint"] = resolved_endpoint
    if resolved_credentials:
        init_kwargs["credentials"] = resolved_credentials

    vertexai.init(**init_kwargs)
    return True


def _build_generation_config(**config_values: Any) -> GenerationConfig | None:
    sanitized = {
        key: value
        for key, value in config_values.items()
        if value not in (None, "")
    }
    if not sanitized:
        return None
    stop_sequences = sanitized.pop("stop_sequences", None)
    if stop_sequences:
        sanitized["stop_sequences"] = stop_sequences
    return GenerationConfig(**sanitized)


def _build_safety_settings(settings: dict | None) -> list[SafetySetting] | None:
    if not settings:
        return None

    result: list[SafetySetting] = []
    for category, threshold in settings.items():
        try:
            result.append(
                SafetySetting(
                    category=category,
                    threshold=threshold,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOG.warning("Skipping invalid safety setting %s=%s: %s", category, threshold, exc)
    return result or None


def _format_history(history_messages: list[dict[str, Any]] | None) -> list[Content]:
    if not history_messages:
        return []

    formatted: list[Content] = []
    for message in history_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted.append(Content(role=role, parts=[Part.from_text(str(content))]))
    return formatted


def _extract_response_text(response: Any) -> str:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=(
        retry_if_exception_type(google_api_exceptions.InternalServerError)
        | retry_if_exception_type(google_api_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_api_exceptions.ResourceExhausted)
        | retry_if_exception_type(google_api_exceptions.GatewayTimeout)
        | retry_if_exception_type(google_api_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_api_exceptions.Aborted)
        | retry_if_exception_type(google_api_exceptions.Unknown)
    ),
)
async def vertex_ai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    project_id: str | None = None,
    location: str | None = None,
    api_endpoint: str | None = None,
    credentials_path: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    generation_config: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_output_tokens: int | None = None,
    candidate_count: int | None = None,
    response_mime_type: str | None = None,
    stop_sequences: list[str] | None = None,
    safety_settings: dict | None = None,
    timeout: int | None = None,
    **_: Any,
) -> str | AsyncIterator[str]:
    _init_vertex_ai(project_id, location, api_endpoint, credentials_path)

    model_kwargs: dict[str, Any] = {}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt

    chat_model = GenerativeModel(model, **model_kwargs)
    history = _format_history(history_messages)
    chat = chat_model.start_chat(history=history) if history else chat_model.start_chat()

    generation = _build_generation_config(
        **(generation_config or {}),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
        response_mime_type=("application/json" if keyword_extraction else response_mime_type),
        stop_sequences=stop_sequences,
    )
    safety_overrides = _build_safety_settings(safety_settings)

    request_kwargs: dict[str, Any] = {}
    if generation:
        request_kwargs["generation_config"] = generation
    if safety_overrides:
        request_kwargs["safety_settings"] = safety_overrides
    if timeout is not None:
        request_kwargs["timeout"] = timeout

    use_stream = bool(stream)
    if use_stream:
        response = chat.send_message(prompt, stream=True, **request_kwargs)

        async def _stream_iterator():
            for chunk in response:
                text = _extract_response_text(chunk)
                if text:
                    yield text

        return _stream_iterator()

    response = chat.send_message(prompt, **request_kwargs)
    text = _extract_response_text(response)

    if token_tracker and hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        token_tracker.add_usage(
            {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        )

    logger.debug("Vertex AI response length: %s", len(text))
    return text


async def vertex_ai_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    hashing_kv = kwargs.get("hashing_kv")
    model_name = None
    if hashing_kv is not None:
        model_name = hashing_kv.global_config.get("llm_model_name")
    if model_name is None:
        model_name = kwargs.pop("model_name", None)
    if model_name is None:
        raise ValueError("Vertex AI model name not provided in configuration.")

    return await vertex_ai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=2048)
async def vertex_ai_embed(
    texts: list[str],
    model: str = "text-embedding-004",
    project_id: str | None = None,
    location: str | None = None,
    api_endpoint: str | None = None,
    credentials_path: str | None = None,
    task_type: str | None = None,
    title: str | None = None,
    output_dimensionality: int | None = None,
    token_tracker: Any | None = None,
    timeout: int | None = None,
) -> np.ndarray:
    _init_vertex_ai(project_id, location, api_endpoint, credentials_path)

    loop = asyncio.get_running_loop()
    embedding_model = TextEmbeddingModel.from_pretrained(model)

    def _call_embed():
        return embedding_model.get_embeddings(
            texts,
            task_type=task_type,
            title=title,
            output_dimensionality=output_dimensionality,
            timeout=timeout,
        )

    responses = await loop.run_in_executor(None, _call_embed)

    embeddings = np.array([np.array(resp.values, dtype=np.float32) for resp in responses])

    if token_tracker:
        total_tokens = 0
        for resp in responses:
            total_tokens += getattr(resp, "token_count", 0)
        if total_tokens:
            token_tracker.add_usage({"prompt_tokens": total_tokens, "total_tokens": total_tokens})

    return embeddings


__all__ = [
    "vertex_ai_complete_if_cache",
    "vertex_ai_model_complete",
    "vertex_ai_embed",
]
