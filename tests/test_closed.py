"""Tests for the Claude/GPT backends and the Batch API loop.

No network and no SDK: the backends get fake clients, and the batch loop gets
a fake provider. What matters here:

- a batch run and a real-time run send the same request body, with the cache
  breakpoint on the shared system prompt (1 h in batch, 5 min in real time);
- settings are sent as given, never substituted (no temperature unless set,
  thinking off sent as off);
- the batch loop never submits a request twice, stores results in the same
  checkpoint, and leaves failures out so a re-run sends them again;
- ``--no-wait`` submits and returns, and the next run collects.
"""

from __future__ import annotations

import json

import pytest

from chemberta4.evaluation.backends.base import Response
from chemberta4.evaluation.backends.closed import AnthropicBackend, OpenAIBackend, provider_of
from chemberta4.evaluation.batch import load_manifest, run_batch
from chemberta4.evaluation.runner import load_jsonl

MESSAGES = [
    {"role": "system", "content": "instructions and examples"},
    {"role": "user", "content": "SMILES: CCO"},
]


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------
def test_provider_of() -> None:
    assert provider_of("anthropic/claude-opus-5-5") == "anthropic"
    assert provider_of("openai/gpt-5.1") == "openai"
    assert provider_of("hosted_vllm/Qwen/Qwen3-8B") is None
    assert provider_of("anthropic") is None


@pytest.mark.parametrize("batch, ttl", [(True, "1h"), (False, "5m")])
def test_anthropic_caches_the_system_prompt(batch: bool, ttl: str) -> None:
    body = AnthropicBackend("anthropic/claude-x", batch=batch, client=object()).request(MESSAGES)
    assert body["model"] == "claude-x"
    assert body["system"] == [{
        "type": "text",
        "text": "instructions and examples",
        "cache_control": {"type": "ephemeral", "ttl": ttl},
    }]
    assert body["messages"] == [MESSAGES[1]], "the system prompt moves to the system field"


def test_anthropic_sends_settings_as_given() -> None:
    off = AnthropicBackend("anthropic/claude-x", client=object()).request(MESSAGES)
    assert off["thinking"] == {"type": "disabled"}
    assert "temperature" not in off and "output_config" not in off

    on = AnthropicBackend(
        "anthropic/claude-x", enable_thinking=True, thinking_effort="high",
        temperature=1.0, client=object(),
    ).request(MESSAGES)
    assert on["thinking"]["type"] == "adaptive"
    assert on["output_config"] == {"effort": "high"}
    assert on["temperature"] == 1.0


def test_openai_reasoning_controls() -> None:
    reasoning = OpenAIBackend("openai/gpt-5.1", client=object()).request(MESSAGES)
    assert reasoning["reasoning_effort"] == "none"
    assert "temperature" not in reasoning

    plain = OpenAIBackend("openai/gpt-4.1", client=object()).request(MESSAGES)
    assert "reasoning_effort" not in plain

    thinking = OpenAIBackend(
        "openai/gpt-5.1", enable_thinking=True, thinking_effort="low", client=object()
    ).request(MESSAGES)
    assert thinking["reasoning_effort"] == "low"

    with pytest.raises(ValueError, match="max"):
        OpenAIBackend("openai/gpt-5.1", enable_thinking=True, thinking_effort="max")


def test_openai_cache_key_follows_the_system_prompt() -> None:
    backend = OpenAIBackend("openai/gpt-4.1", client=object())
    other_user = [MESSAGES[0], {"role": "user", "content": "SMILES: CCN"}]
    other_system = [{"role": "system", "content": "different"}, MESSAGES[1]]
    key = backend.request(MESSAGES)["prompt_cache_key"]
    assert backend.request(other_user)["prompt_cache_key"] == key
    assert backend.request(other_system)["prompt_cache_key"] != key


def test_batch_and_realtime_send_the_same_body_apart_from_cache_ttl() -> None:
    batch = AnthropicBackend("anthropic/claude-x", batch=True, client=object()).request(MESSAGES)
    live = AnthropicBackend("anthropic/claude-x", batch=False, client=object()).request(MESSAGES)
    batch["system"][0]["cache_control"].pop("ttl")
    live["system"][0]["cache_control"].pop("ttl")
    assert batch == live


# --------------------------------------------------------------------------
# Real-time calls
# --------------------------------------------------------------------------
class _Final:
    def __init__(self, data: dict) -> None:
        self.data = data

    def to_dict(self) -> dict:
        return self.data


class _Stream:
    def __init__(self, data: dict) -> None:
        self.data = data

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def get_final_message(self) -> _Final:
        return _Final(self.data)


class FakeAnthropic:
    def __init__(self, reply: dict | Exception) -> None:
        self.reply, self.sent = reply, []
        self.messages = self

    def stream(self, **body):
        self.sent.append(body)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _Stream(self.reply)


def test_anthropic_generate_parses_text_and_usage() -> None:
    client = FakeAnthropic({
        "content": [{"type": "text", "text": '{"t1": 0.7}'}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "cache_read_input_tokens": 1500,
                  "cache_creation_input_tokens": 0, "output_tokens": 8},
    })
    backend = AnthropicBackend("anthropic/claude-x", client=client)
    response = backend.generate(MESSAGES, metadata={"ignored": True})
    assert response.ok and response.text == '{"t1": 0.7}'
    assert response.usage["cache_read_tokens"] == 1500
    assert "metadata" not in client.sent[0], "Langfuse metadata is not sent to the API"


def test_anthropic_generate_reports_an_api_error_without_raising() -> None:
    backend = AnthropicBackend("anthropic/claude-x", client=FakeAnthropic(RuntimeError("400")))
    response = backend.generate(MESSAGES)
    assert not response.ok and "400" in response.text


def test_truncated_means_cap_hit_and_no_answer() -> None:
    assert Response(text="", stop_reason="max_tokens").truncated
    assert not Response(text="42", stop_reason="max_tokens").truncated
    assert Response(text="", reasoning="thinking...").truncated
    assert not Response(text="", stop_reason="end_turn").truncated


# --------------------------------------------------------------------------
# Batch loop
# --------------------------------------------------------------------------
class FakeProvider:
    """Batches finish after ``polls`` checks; ``fail`` custom ids come back errored."""

    batch = True

    def __init__(self, fail: set[str] = frozenset(), polls: int = 1) -> None:
        self.fail, self.polls = set(fail), polls
        self.batches: dict[str, list[str]] = {}
        self.checks: dict[str, int] = {}

    def request(self, messages):
        return {"messages": messages}

    def submit(self, requests):
        batch_id = f"b{len(self.batches)}"
        self.batches[batch_id] = [cid for cid, _ in requests]
        self.checks[batch_id] = 0
        return batch_id

    def finished(self, batch_id):
        self.checks[batch_id] += 1
        return self.checks[batch_id] >= self.polls

    def results(self, batch_id):
        return {
            cid: Response(text="boom", status=502) if cid in self.fail else Response(text=cid)
            for cid in self.batches[batch_id]
        }


def _run(provider, items, tmp_path, **kwargs):
    return run_batch(
        items=items,
        backend=provider,
        custom_id=lambda n: f"item-{n}",
        messages_of=lambda n: [{"role": "user", "content": str(n)}],
        finish=lambda n, messages, r: {"id": n, "text": r.text,
                                       "status": "ok" if r.ok else "api_error"},
        out_path=tmp_path / "results.jsonl",
        manifest_path=tmp_path / "batches.json",
        keep=lambda rec: rec["status"] != "api_error",
        poll_s=0,
        **kwargs,
    )


def test_batch_writes_results_and_drops_failures(tmp_path) -> None:
    provider = FakeProvider(fail={"item-2"})
    counts = _run(provider, [1, 2, 3], tmp_path)
    assert counts["ok"] == 2 and counts["dropped"] == 1
    assert sorted(r["id"] for r in load_jsonl(tmp_path / "results.jsonl")) == [1, 3]
    assert all(b["collected"] for b in load_manifest(tmp_path / "batches.json"))

    # The re-run sends only the failed item.
    provider.fail.clear()
    _run(provider, [2], tmp_path)
    assert provider.batches["b1"] == ["item-2"]
    assert sorted(r["id"] for r in load_jsonl(tmp_path / "results.jsonl")) == [1, 2, 3]


def test_no_wait_submits_once_and_the_next_run_collects(tmp_path) -> None:
    provider = FakeProvider()
    counts = _run(provider, [1, 2], tmp_path, wait=False)
    assert counts["pending"] == 2
    assert not (tmp_path / "results.jsonl").exists() or not load_jsonl(tmp_path / "results.jsonl")

    # Same items again: already in flight, so nothing new is submitted.
    _run(provider, [1, 2], tmp_path)
    assert list(provider.batches) == ["b0"], "an in-flight request must not be sent twice"
    assert sorted(r["id"] for r in load_jsonl(tmp_path / "results.jsonl")) == [1, 2]


def test_batch_reports_finished_groups(tmp_path) -> None:
    done: list[str] = []
    _run(
        FakeProvider(), [1, 2, 3], tmp_path,
        group_of=lambda n: "odd" if n % 2 else "even", on_group_done=done.append,
    )
    assert sorted(done) == ["even", "odd"]


def test_batch_polls_until_finished(tmp_path) -> None:
    provider = FakeProvider(polls=3)
    _run(provider, [1], tmp_path)
    assert provider.checks["b0"] == 3
    assert json.loads((tmp_path / "batches.json").read_text())[0]["collected"]
