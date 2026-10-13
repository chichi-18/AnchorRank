from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchorrank import AnchorBuilder, AnchorReranker, Document, QueryExample
from anchorrank.sglang_backend import SGLangGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="AnchorRank JSONL reranking")
    parser.add_argument("--input", required=True, help="JSONL: qid, query, documents[{doc_id, content, retrieval_score}]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["pointwise", "listwise"], default="listwise")
    parser.add_argument("--anchor-strategy", choices=["ae", "trd", "ga"], default="ae")
    parser.add_argument("--anchor-cache", default="anchors.json")
    parser.add_argument("--anchor-top-k", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=12288)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    args = parser.parse_args()
    generator = SGLangGenerator(args.model, args.context_length, args.tp_size, args.dp_size)
    builder = AnchorBuilder(generator, args.anchor_strategy, args.anchor_top_k, cache_path=args.anchor_cache)
    reranker = AnchorReranker(generator, builder, args.mode, args.window_size)
    with Path(args.input).open(encoding="utf-8") as source, Path(args.output).open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            docs = [Document(str(doc["doc_id"]), doc["content"], float(doc.get("retrieval_score", 0))) for doc in row["documents"]]
            anchor, ranked = reranker.rerank(QueryExample(str(row["qid"]), row["query"], docs))
            target.write(json.dumps({"qid": row["qid"], "anchor": anchor, "results": [{"doc_id": doc.doc_id, "score": score} for doc, score in ranked]}, ensure_ascii=False) + "\n")
    generator.shutdown()


if __name__ == "__main__":
    main()
