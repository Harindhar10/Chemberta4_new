"""Record what the vLLM server was doing while a run generated.

The server publishes Prometheus metrics at ``/metrics``; the copied
:class:`PerfRun` polls them for the length of a generation window and reports
throughput, time-to-first-token, time-per-output-token, queue depth and KV
cache use. That is the number you want when comparing serving setups
(tensor-parallel size, ``--max-num-seqs``, prefix caching) rather than models.

Only meaningful with ``--perf``, and only for a server this run has to itself:
the metrics are server-wide, so another client's traffic lands in the window too.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from chemberta4.evaluation.backends.vllm_server import api_base
from chemberta4.deepretro.vllm_metrics import PerfRun


def perf_window(enabled: bool) -> PerfRun | nullcontext:
    """A :class:`PerfRun` over the configured server, or a no-op context."""
    return PerfRun(api_base()) if enabled else nullcontext()


def write_perf(perf: PerfRun, run_dir: str | Path, n_requests: int) -> dict[str, Any]:
    """Write ``perf.json``, print the headline numbers and return the summary."""
    summary = perf.summary(n_requests)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "perf.json", "w") as f:
        json.dump({"summary": summary, "gauge_samples": perf.samples}, f, indent=2)

    def fmt(key: str, scale: float = 1.0, unit: str = "") -> str:
        value = summary.get(key)
        return "n/a" if value is None else f"{value * scale:.1f}{unit}"

    print(
        f"perf: {fmt('output_tok_per_s')} out tok/s, {fmt('mol_per_s')} req/s | "
        f"TTFT p50/p90 {fmt('ttft_p50_s', 1000)}/{fmt('ttft_p90_s', 1000, 'ms')} | "
        f"TPOT p50/p90 {fmt('tpot_p50_s', 1000)}/{fmt('tpot_p90_s', 1000, 'ms')} | "
        f"queue p90 {fmt('queue_p90_s', 1000, 'ms')}"
    )
    print(
        f"perf: running mean/max {fmt('running_mean')}/{fmt('running_max')}, "
        f"waiting mean/max {fmt('waiting_mean')}/{fmt('waiting_max')}, "
        f"KV max {fmt('kv_pct_max', unit='%')}, preemptions {fmt('preemptions')}"
    )
    return summary
