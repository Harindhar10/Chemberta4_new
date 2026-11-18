"""Dollar cost of Claude and GPT calls, from litellm's built-in price list.

litellm ships a price map that is updated with each release, so no table is
kept here. A model the map does not know yet has no price; the cost is then
reported as ``None`` rather than guessed.

Two discounts are applied on top of the per-token rates, both documented by
the providers: the Batch API halves every token price, and an Anthropic
1-hour cache write costs twice the input rate (a 5-minute write is priced by
the map itself).
"""

from __future__ import annotations

from typing import Any

BATCH_DISCOUNT = 0.5


def rates(model: str) -> dict[str, float] | None:
    """Per-token prices for ``model`` (``anthropic/...`` or ``openai/...``), or ``None``.

    Keys: ``input``, ``output``, ``cache_read``, ``cache_write_5m``.
    """
    try:
        from litellm import model_cost
    except ImportError:
        return None
    bare = model.split("/", 1)[-1]
    entry = model_cost.get(bare) or model_cost.get(model)
    if not entry or not entry.get("input_cost_per_token"):
        return None
    inp = float(entry["input_cost_per_token"])
    return {
        "input": inp,
        "output": float(entry.get("output_cost_per_token") or 0.0),
        "cache_read": float(entry.get("cache_read_input_token_cost") or inp),
        "cache_write_5m": float(entry.get("cache_creation_input_token_cost") or inp),
    }


def cost(
    prices: dict[str, float] | None,
    usage: dict[str, Any],
    *,
    batch: bool,
    cache_ttl: str = "5m",
) -> float | None:
    """Dollars for one call's normalized ``usage``.

    ``usage`` holds ``input_tokens`` (uncached), ``cache_read_tokens``,
    ``cache_write_tokens`` and ``output_tokens``.

    >>> p = {"input": 1e-6, "output": 5e-6, "cache_read": 1e-7, "cache_write_5m": 1.25e-6}
    >>> u = {"input_tokens": 100, "cache_read_tokens": 1000, "cache_write_tokens": 0,
    ...      "output_tokens": 10}
    >>> round(cost(p, u, batch=False), 8)
    0.00025
    >>> round(cost(p, u, batch=True), 8)
    0.000125
    >>> cost(None, u, batch=False) is None
    True
    """
    if prices is None:
        return None
    write_rate = 2 * prices["input"] if cache_ttl == "1h" else prices["cache_write_5m"]
    total = (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("cache_read_tokens", 0) * prices["cache_read"]
        + usage.get("cache_write_tokens", 0) * write_rate
        + usage.get("output_tokens", 0) * prices["output"]
    )
    return total * (BATCH_DISCOUNT if batch else 1.0)


def estimate(
    prices: dict[str, float] | None,
    prompt_chars: int,
    n_requests: int,
    max_output_tokens: int,
    *,
    batch: bool,
) -> tuple[float, float] | None:
    """A rough upper bound before sending: ``(input $, output $)``.

    Input assumes ~4 characters per token and no cache hits; output assumes
    every answer runs to ``max_output_tokens``. Both overestimate, which is the
    safe direction for a spending check.

    >>> p = {"input": 1e-6, "output": 5e-6, "cache_read": 1e-7, "cache_write_5m": 1.25e-6}
    >>> [round(x, 6) for x in estimate(p, 4000, 10, 100, batch=False)]
    [0.001, 0.005]
    """
    if prices is None:
        return None
    discount = BATCH_DISCOUNT if batch else 1.0
    return (
        prompt_chars / 4 * prices["input"] * discount,
        n_requests * max_output_tokens * prices["output"] * discount,
    )
