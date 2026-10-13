from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AnchorRank pointwise labels to LLaMA-Factory SFT JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    with Path(args.input).open(encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            document = sample.get("document", sample.get("content"))
            score = max(0, min(100, int(round(float(sample["score"])))) )
            rows.append({"instruction": "Assign an integer relevance score from 0 to 100. Return JSON only as {\\\"score\\\": <integer>}.", "input": json.dumps({"query": sample["query"], "document": {"content": document}}, ensure_ascii=False), "output": json.dumps({"score": score})})
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
