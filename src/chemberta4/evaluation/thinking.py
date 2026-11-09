"""Separate a reasoning trace from the answer that follows it.

Only needed when the vLLM server runs without ``--reasoning-parser``; with one,
the trace arrives in its own ``reasoning_content`` field and none of this runs.

The tag patterns come from the ChemLLMBench runner
(``Chemberata4_ChemLLMBench/utils.py``), which covers more dialects than
DeepRetro's splitter: ``<think>`` and its synonyms, ``[thinking]``,
Kimi's ``<|begin_of_thought|>`` and the ``◁think▷`` form.

Three shapes have to be handled. A closed block is the easy case. An orphan
closing tag appears when the chat template put the opening tag in the prompt,
so the model's output starts mid-thought. An unclosed opening tag means the
token budget ran out before the model finished thinking, which leaves no answer
at all; callers treat that as a truncated record rather than a parse failure.
"""

from __future__ import annotations

import re

_THINK_NAMES = r"(?:think|thinking|thought|reasoning|reflection|scratchpad|cot)"

# (opening-tag pattern, closing-tag pattern)
_THINK_PAIRS = [
    (rf"<\s*{_THINK_NAMES}\b[^>]*>", rf"<\s*/\s*{_THINK_NAMES}\s*>"),
    (r"\[\s*think(?:ing)?\s*\]", r"\[\s*/\s*think(?:ing)?\s*\]"),
    (r"<\|\s*begin_of_thought\s*\|>", r"<\|\s*end_of_thought\s*\|>"),
    (r"◁\s*think\s*▷", r"◁\s*/\s*think\s*▷"),
]

# Markers deleted on sight; what they wrap is the answer, so it is kept.
_MARKER_ONLY = re.compile(r"<\|\s*(?:begin|end)_of_solution\s*\|>", re.IGNORECASE)

_FLAGS = re.DOTALL | re.IGNORECASE


def split_thinking(text: str) -> tuple[str, str]:
    """Return ``(reasoning, answer)`` for one completion.

    ``reasoning`` is every thinking block found, joined by blank lines, and
    ``answer`` is what is left. Text with no thinking tags comes back unchanged
    as the answer.

    Examples
    --------
    >>> split_thinking("<think>first I check the ring</think>{\\"t1\\": 0.8}")
    ('first I check the ring', '{"t1": 0.8}')

    An orphan closing tag: the prompt opened the block, so everything before the
    tag was thinking.

    >>> split_thinking("still weighing it</think>answer: 3")
    ('still weighing it', 'answer: 3')

    An unclosed block means the trace was cut off and no answer arrived.

    >>> split_thinking("<think>I should start by")
    ('I should start by', '')

    >>> split_thinking("plain answer")
    ('', 'plain answer')
    >>> split_thinking("")
    ('', '')
    """
    if not text:
        return "", ""

    traces: list[str] = []
    for open_re, close_re in _THINK_PAIRS:
        # 1) closed blocks
        def take(match: re.Match[str]) -> str:
            traces.append(match.group(1))
            return ""

        text = re.sub(rf"{open_re}(.*?){close_re}", take, text, flags=_FLAGS)

        # 2) orphan closing tag: the opening tag was in the prompt
        closers = list(re.finditer(close_re, text, flags=_FLAGS))
        if closers:
            traces.append(text[: closers[-1].start()])
            text = text[closers[-1].end():]

        # 3) unclosed opening tag: the rest is a truncated trace
        opener = re.search(open_re, text, flags=_FLAGS)
        if opener:
            traces.append(text[opener.end():])
            text = text[: opener.start()]

    text = _MARKER_ONLY.sub("", text)
    reasoning = "\n\n".join(t.strip() for t in traces if t.strip())
    return reasoning, text.strip()
