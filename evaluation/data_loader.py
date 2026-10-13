from __future__ import annotations

import json
from pathlib import Path


class AnchorRankEvalDataLoader:
    """Loads the same local corpus/queries layout used by CuRLRank evaluation."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)

    def load_corpus(self, dataset_name: str) -> dict[str, dict[str, str]]:
        return {str(row["id"]): {"text": row["text"]} for row in self._jsonl(dataset_name, "corpus.jsonl")}

    def load_queries(self, dataset_name: str) -> dict[str, str]:
        return {str(row["id"]): row["text"] for row in self._jsonl(dataset_name, "queries.jsonl")}

    def load_qrels(self, dataset_name: str) -> dict[str, dict[str, int]]:
        path = self.dataset_dir / dataset_name / "qrels.jsonl"
        if not path.is_file():
            return {}
        qrels: dict[str, dict[str, int]] = {}
        for row in self._jsonl(dataset_name, "qrels.jsonl"):
            qrels.setdefault(str(row["qid"]), {})[str(row["docid"])] = int(row["relevance"])
        return qrels

    def _jsonl(self, dataset_name: str, filename: str):
        with (self.dataset_dir / dataset_name / filename).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
