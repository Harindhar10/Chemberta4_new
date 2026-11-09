"""Talk to a vLLM OpenAI-compatible server.

Start the server yourself, then point this at it::

    vllm serve Qwen/Qwen3-8B --max-model-len 17408 --enable-prefix-caching
    export HOSTED_VLLM_API_BASE=http://localhost:8000/v1

The model id carries litellm's ``hosted_vllm/`` prefix, e.g.
``hosted_vllm/Qwen/Qwen3-8B``. Requests go out concurrently and vLLM batches
them server-side, so throughput comes from raising ``--workers`` rather than
from batching here.

Serving with ``--reasoning-parser`` (``qwen3`` for Qwen) returns the reasoning
trace in its own field, which is cleaner than splitting an inline ``<think>``
block. Without one, :func:`chemberta4.evaluation.thinking.split_thinking`
separates it, which matters because a greedy JSON regex would otherwise match
braces inside the trace.

The litellm parameters, including vLLM's ``chat_template_kwargs`` and the
Qwen3.8 ``reasoning_effort`` mapping, come from the copied
:func:`build_completion_params` so a vLLM run is set up exactly as DeepRetro
sets it up.
"""

from __future__ import annotations

import os
import time
from typing import Any

from chemberta4.evaluation.backends.base import ModelBackend, Response
from chemberta4.evaluation.thinking import split_thinking
from chemberta4.deepretro.llm_helpers import build_completion_params
from chemberta4.deepretro.llm_interface import reasoning_token_count
from chemberta4.deepretro.llm_trace import reasoning_text

DEFAULT_API_BASE = "http://localhost:8000/v1"


def api_base() -> str:
    """The server URL, from ``HOSTED_VLLM_API_BASE`` or the local default."""
    return os.getenv("HOSTED_VLLM_API_BASE") or DEFAULT_API_BASE


class VLLMServerBackend(ModelBackend):
    """A model served by vLLM over its OpenAI-compatible endpoint.

    Parameters
    ----------
    model:
        litellm model id, e.g. ``hosted_vllm/Qwen/Qwen3-8B``.
    temperature:
        Default sampling temperature. 0.0 for scored evaluations.
    max_output_tokens:
        Default output cap. Thinking runs need far more; the task sets it.
    enable_thinking, thinking_effort:
        Reasoning controls, sent as the server's ``chat_template_kwargs``.
    max_retries:
        Attempts per call. A failure returns a non-200 ``Response`` rather than
        raising, so the caller can leave the record out of the checkpoint and
        pick it up on the next run.
    metadata:
        litellm metadata for callbacks, used for Langfuse grouping.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        enable_thinking: bool = False,
        thinking_effort: str = "medium",
        max_retries: int = 3,
        retry_wait_s: float = 2.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.thinking_effort = thinking_effort
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self.metadata = metadata or {}

    def completion_params(
        self, messages: list[dict[str, Any]], **overrides: Any
    ) -> dict[str, Any]:
        """litellm keyword arguments for one call.

        Everything vLLM-specific is decided by the copied builder; this only
        resolves the defaults against per-call overrides.
        """
        params = build_completion_params(
            model=self.model,
            messages=messages,
            max_completion_tokens=overrides.pop("max_output_tokens", self.max_output_tokens),
            temperature=overrides.pop("temperature", self.temperature),
            enable_thinking=overrides.pop("enable_thinking", self.enable_thinking),
            thinking_effort=overrides.pop("thinking_effort", self.thinking_effort),
            metadata={**self.metadata, **overrides.pop("metadata", {})},
        )
        params["api_base"] = overrides.pop("api_base", api_base())
        params.update(overrides)
        return params

    def generate(self, messages: list[dict[str, Any]], **overrides: Any) -> Response:
        """Complete ``messages``, retrying a failed call before giving up.

        A reasoning trace is taken from the server's own field when present,
        otherwise split off the text.
        """
        from litellm import completion

        params = self.completion_params(messages, **overrides)
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = completion(**params)
            except Exception as exc:  # litellm raises a provider-specific tree
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_s * attempt)
                continue

            message = response.choices[0].message
            text = message.content or ""
            reasoning = reasoning_text(message)
            if not reasoning:
                reasoning, text = split_thinking(text)
            usage = getattr(response, "usage", None)
            return Response(
                text=text.strip(),
                reasoning=reasoning,
                status=200,
                usage={
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "reasoning_tokens": reasoning_token_count(response),
                    "reasoning_chars": len(reasoning),
                },
            )
        return Response(text=last_error, reasoning="", status=502, usage={})
