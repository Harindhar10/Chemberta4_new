"""The one interface every model goes through.

A task builds chat messages and asks a backend to complete them; it never knows
whether the answer came from a vLLM server, Claude or GPT. Only the vLLM server
backend exists today (see ``vllm_server.py``); closed-source backends slot in
behind the same ``generate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Response:
    """One completion.

    Attributes
    ----------
    text:
        The answer with any reasoning trace removed, ready to parse.
    reasoning:
        The reasoning trace, from the server's own field or split off ``text``.
        Empty when the model did not think.
    status:
        HTTP-style status: 200 on success, anything else means ``text`` holds an
        error message and the caller should not checkpoint the record.
    usage:
        Token counts as reported, with whatever keys the provider sent.
    """

    text: str
    reasoning: str = ""
    status: int = 200
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == 200


class ModelBackend(ABC):
    """A model that completes chat messages."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], **overrides: Any) -> Response:
        """Complete ``messages`` and return the answer.

        Parameters
        ----------
        messages:
            OpenAI-format chat messages. String content only: a vLLM server
            will not take the ``[{"type": "text", ...}]`` parts form.
        **overrides:
            Per-call sampling overrides, e.g. ``temperature`` or
            ``max_output_tokens``, taking precedence over the backend's defaults.

        Returns
        -------
        Response
            Never raises for a server-side failure; those come back with a
            non-200 ``status`` so a run can record the failure and retry later.
        """
