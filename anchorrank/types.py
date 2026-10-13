from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class TextGenerator(Protocol):
    """Minimal interface implemented by an LLM inference backend."""

    def generate(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]: ...


@dataclass(frozen=True)
class Document:
    doc_id: str
    content: str
    retrieval_score: float = 0.0


@dataclass(frozen=True)
class QueryExample:
    qid: str
    query: str
    documents: list[Document]
