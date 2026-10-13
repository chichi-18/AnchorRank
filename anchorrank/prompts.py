from __future__ import annotations

import json
from typing import Iterable

from .types import Document


def _payload(query: str, documents: Iterable[Document]) -> str:
    return json.dumps(
        {"query": query, "documents": [{"doc_id": d.doc_id, "content": d.content} for d in documents]},
        ensure_ascii=False,
    )


def evidence_extraction(query: str, documents: list[Document]) -> str:
    return """You are an evidence extractor for reasoning-intensive retrieval.
Given a query and candidate documents, extract only concise, concrete evidence useful for judging relevance. Omit documents with no useful evidence. Do not answer beyond what documents support.
Return JSON only in this exact shape: {"evidence":[{"doc_id":"...","evidence":"..."}]}.
Input:\n""" + _payload(query, documents)


def evidence_aggregation(query: str, evidence: list[dict[str, str]]) -> str:
    return """You are the evidence aggregator for AnchorRank. Given a query and extracted evidence, synthesize one concise shared relevance anchor. Retain query intent, key concepts, mechanisms, and constraints; remove duplicates and unsupported claims. Do not rank documents.
Return JSON only: {"anchor":"..."}.
Input:\n""" + json.dumps({"query": query, "evidence": evidence}, ensure_ascii=False)


def generated_answer(query: str) -> str:
    return """Write a concise factual reference answer for the query below. It will be used only as a stable relevance anchor. Include the key concepts, mechanisms, and constraints needed to judge supporting documents. Do not mention this instruction.
Query:\n""" + query


def pointwise(query: str, anchor: str, document: Document) -> str:
    return """You are a pointwise reranker for AnchorRank. Score how well the candidate document addresses the query using the shared relevance anchor as a fixed reference. Use a consistent absolute 0-100 scale: 81-100 highly relevant, 61-80 relevant, 41-60 moderate, 21-40 slight, 0-20 irrelevant. Do not provide reasoning.
Return JSON only: {"score": <integer 0-100>}.
Input:\n""" + json.dumps({"query": query, "anchor": anchor, "document": {"content": document.content}}, ensure_ascii=False)


def listwise(query: str, anchor: str, documents: list[Document]) -> str:
    return """You are a listwise reranker for AnchorRank. Score every candidate document against the query, using the shared relevance anchor as a fixed reference across windows. Use an absolute integer 0-100 scale; evaluate each document individually while considering local comparisons. Do not provide reasoning.
Return JSON only, mapping every input document id to its score, for example {"D1": 80, "D2": 20}.
Input:\n""" + _payload_with_anchor(query, anchor, documents)


def _payload_with_anchor(query: str, anchor: str, documents: list[Document]) -> str:
    return json.dumps(
        {"query": query, "anchor": anchor, "documents": [{"doc_id": d.doc_id, "content": d.content} for d in documents]},
        ensure_ascii=False,
    )
