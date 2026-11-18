"""Claude and GPT through their own SDKs, in real time or through the Batch API.

The model id names the provider: ``anthropic/claude-sonnet-5-5`` or
``openai/gpt-5.1``. Keys come from ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``;
``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` point the SDKs elsewhere (a
LiteLLM proxy, say) without a code change.

One request builder per provider feeds both modes, so a batch run and a
real-time run send the same request body. Responses are parsed from plain
dicts (the SDK object's ``to_dict()``, or a batch result line), which is what
the tests feed in.

Settings are sent as given and never quietly substituted: thinking off is sent
as "off", a temperature is sent only when one was set. A model that cannot
honor a setting rejects the request, and the run reports that error instead of
producing results under different settings. Rejected batch requests are not
billed.

Cost savings:

- Batch API (default): half price on every token.
- Prompt caching. Claude: the system prompt (instructions and few-shot
  examples, identical across a group) carries a ``cache_control`` breakpoint,
  with a 1-hour lifetime in batch mode, where requests run out of order over a
  long window, and 5 minutes in real time. OpenAI caches automatically; the
  ``prompt_cache_key`` is a hash of the system prompt, so requests sharing a
  prefix are routed to the same cache.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from chemberta4.evaluation import pricing
from chemberta4.evaluation.backends.base import ModelBackend, Response

PROVIDERS = ("anthropic", "openai")


def provider_of(model: str) -> str | None:
    """``"anthropic"`` or ``"openai"`` for a closed model id, else ``None``.

    >>> provider_of("anthropic/claude-sonnet-5-5"), provider_of("openai/gpt-5.1")
    ('anthropic', 'openai')
    >>> provider_of("hosted_vllm/Qwen/Qwen3-8B") is None
    True
    """
    prefix = model.split("/", 1)[0]
    return prefix if prefix in PROVIDERS and "/" in model else None


def split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The system text (string content) and the remaining chat messages."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class ClosedBackend(ModelBackend):
    """Shared plumbing; subclasses build and parse one provider's requests.

    Parameters
    ----------
    model:
        ``anthropic/<id>`` or ``openai/<id>``.
    temperature:
        Sent only when not ``None``. Current Claude models reject any
        temperature, and OpenAI reasoning models reject anything but the default.
    batch:
        Whether requests go through the Batch API. Sets the cache lifetime and
        the 50% discount in the recorded cost.
    client:
        An SDK client; built from the environment when omitted.
    """

    provider = ""

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        enable_thinking: bool = False,
        thinking_effort: str = "medium",
        batch: bool = False,
        client: Any = None,
    ) -> None:
        self.model = model
        self.name = model.split("/", 1)[1]
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.thinking_effort = thinking_effort
        self.batch = batch
        self.cache_ttl = "1h" if batch else "5m"
        self.prices = pricing.rates(model)
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self.make_client()
        return self._client

    def generate(self, messages: list[dict[str, Any]], **overrides: Any) -> Response:
        """One real-time call. The SDK retries 429, 5xx and connection errors."""
        overrides.pop("metadata", None)  # Langfuse metadata: vLLM path only
        body = self.request(messages, **overrides)
        try:
            raw = self.create(body)
        except Exception as exc:  # the SDKs raise a tree of API errors
            return Response(text=f"{type(exc).__name__}: {exc}", status=502)
        return self.parse(raw)

    def priced(self, usage: dict[str, Any]) -> dict[str, Any]:
        usage["cost_usd"] = pricing.cost(
            self.prices, usage, batch=self.batch, cache_ttl=self.cache_ttl
        )
        return usage

    # Provider-specific -----------------------------------------------------
    def make_client(self) -> Any:
        raise NotImplementedError

    def request(self, messages: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
        raise NotImplementedError

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def parse(self, raw: dict[str, Any]) -> Response:
        raise NotImplementedError

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
        """Send ``(custom_id, body)`` pairs as one batch; return its id."""
        raise NotImplementedError

    def finished(self, batch_id: str) -> bool:
        raise NotImplementedError

    def results(self, batch_id: str) -> dict[str, Response]:
        """Responses of a finished batch by custom_id; failures are non-200."""
        raise NotImplementedError


class AnthropicBackend(ClosedBackend):
    """Claude through the Messages API and Message Batches.

    >>> b = AnthropicBackend("anthropic/claude-sonnet-5-5", batch=True, client=object())
    >>> body = b.request([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    >>> body["system"][0]["cache_control"], body["thinking"], "temperature" in body
    ({'type': 'ephemeral', 'ttl': '1h'}, {'type': 'disabled'}, False)
    """

    provider = "anthropic"

    def make_client(self) -> Any:
        import anthropic

        return anthropic.Anthropic(max_retries=5)

    def request(self, messages: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
        system, chat = split_system(messages)
        body: dict[str, Any] = {
            "model": self.name,
            "max_tokens": overrides.pop("max_output_tokens", self.max_output_tokens),
            "messages": chat,
        }
        if system:
            body["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
            }]
        if self.enable_thinking:
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            body["output_config"] = {"effort": self.thinking_effort}
        else:
            body["thinking"] = {"type": "disabled"}
        temperature = overrides.pop("temperature", self.temperature)
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        # Streaming keeps a large max_tokens clear of the SDK's HTTP timeout.
        with self.client.messages.stream(**body) as stream:
            return stream.get_final_message().to_dict()

    def parse(self, raw: dict[str, Any]) -> Response:
        """
        >>> b = AnthropicBackend("anthropic/claude-x", client=object())
        >>> r = b.parse({"content": [{"type": "thinking", "thinking": "hm"},
        ...                          {"type": "text", "text": " 42 "}],
        ...              "stop_reason": "end_turn",
        ...              "usage": {"input_tokens": 5, "cache_read_input_tokens": 900,
        ...                        "cache_creation_input_tokens": 0, "output_tokens": 3}})
        >>> r.text, r.reasoning, r.usage["cache_read_tokens"], r.truncated
        ('42', 'hm', 900, False)
        """
        blocks = raw.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        reasoning = "\n\n".join(
            b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking")
        )
        u = raw.get("usage") or {}
        usage = self.priced({
            "input_tokens": u.get("input_tokens") or 0,
            "cache_read_tokens": u.get("cache_read_input_tokens") or 0,
            "cache_write_tokens": u.get("cache_creation_input_tokens") or 0,
            "output_tokens": u.get("output_tokens") or 0,
            "reasoning_tokens": None,  # counted inside output_tokens
            "reasoning_chars": len(reasoning),
        })
        return Response(
            text=text.strip(),
            reasoning=reasoning,
            usage=usage,
            stop_reason=raw.get("stop_reason") or "",
        )

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": body} for cid, body in requests]
        )
        return batch.id

    def finished(self, batch_id: str) -> bool:
        return self.client.messages.batches.retrieve(batch_id).processing_status == "ended"

    def results(self, batch_id: str) -> dict[str, Response]:
        out: dict[str, Response] = {}
        for item in self.client.messages.batches.results(batch_id):
            item = item.to_dict()
            result = item["result"]
            if result["type"] == "succeeded":
                out[item["custom_id"]] = self.parse(result["message"])
            else:
                detail = result.get("error") or ""
                out[item["custom_id"]] = Response(
                    text=f"batch {result['type']}: {detail}", status=502
                )
        return out


class OpenAIBackend(ClosedBackend):
    """GPT through Chat Completions and the Batch API.

    >>> b = OpenAIBackend("openai/gpt-5.1", client=object())
    >>> body = b.request([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    >>> body["reasoning_effort"], body["max_completion_tokens"], len(body["prompt_cache_key"])
    ('none', 1024, 16)
    >>> "reasoning_effort" in OpenAIBackend("openai/gpt-4.1", client=object()).request(
    ...     [{"role": "user", "content": "U"}])
    False
    """

    provider = "openai"
    endpoint = "/v1/chat/completions"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        if self.enable_thinking and self.thinking_effort == "max":
            raise ValueError(
                "OpenAI has no 'max' reasoning effort; use --thinking-effort low, medium or high"
            )

    def make_client(self) -> Any:
        import openai

        return openai.OpenAI(max_retries=5)

    def request(self, messages: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
        from chemberta4.deepretro.llm_helpers import looks_like_openai_reasoning_model

        system, _ = split_system(messages)
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "max_completion_tokens": overrides.pop("max_output_tokens", self.max_output_tokens),
        }
        if system:
            body["prompt_cache_key"] = hashlib.sha256(system.encode()).hexdigest()[:16]
        if self.enable_thinking:
            body["reasoning_effort"] = self.thinking_effort
        elif looks_like_openai_reasoning_model(self.name):
            body["reasoning_effort"] = "none"  # rejected by models that always reason
        temperature = overrides.pop("temperature", self.temperature)
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.client.chat.completions.create(**body).to_dict()

    def parse(self, raw: dict[str, Any]) -> Response:
        """
        >>> b = OpenAIBackend("openai/gpt-x", client=object())
        >>> r = b.parse({"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        ...              "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
        ...                        "prompt_tokens_details": {"cached_tokens": 896},
        ...                        "completion_tokens_details": {"reasoning_tokens": 50}}})
        >>> r.usage["input_tokens"], r.usage["cache_read_tokens"], r.truncated
        (104, 896, True)
        """
        choice = (raw.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = raw.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = self.priced({
            "input_tokens": (u.get("prompt_tokens") or 0) - cached,
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
            "output_tokens": u.get("completion_tokens") or 0,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "reasoning_chars": 0,
        })
        return Response(
            text=text.strip(), usage=usage, stop_reason=choice.get("finish_reason") or ""
        )

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
        lines = "\n".join(
            json.dumps({"custom_id": cid, "method": "POST", "url": self.endpoint, "body": body})
            for cid, body in requests
        )
        upload = self.client.files.create(file=("batch.jsonl", lines.encode()), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=upload.id, endpoint=self.endpoint, completion_window="24h"
        )
        return batch.id

    def finished(self, batch_id: str) -> bool:
        status = self.client.batches.retrieve(batch_id).status
        return status in ("completed", "failed", "expired", "cancelled")

    def results(self, batch_id: str) -> dict[str, Response]:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "failed":
            print(f"[batch] {batch_id} failed validation: {batch.errors}")
        out: dict[str, Response] = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item.get("response") or {}
                if response.get("status_code") == 200:
                    out[item["custom_id"]] = self.parse(response["body"])
                else:
                    detail = item.get("error") or response.get("body")
                    out[item["custom_id"]] = Response(text=f"batch error: {detail}", status=502)
        return out


def make_closed_backend(model: str, **kwargs: Any) -> ClosedBackend:
    """The backend for ``model``'s provider."""
    cls = {"anthropic": AnthropicBackend, "openai": OpenAIBackend}[provider_of(model)]
    return cls(model, **kwargs)
