"""Install AnchorRank's reward module into an existing verl checkout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-dir", required=True)
    args = parser.parse_args()
    target = Path(args.verl_dir) / "verl" / "utils" / "reward_score" / "anchorrank.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).parent / "reward_score" / "anchorrank.py", target)
    dispatcher = target.parent / "__init__.py"
    text = dispatcher.read_text(encoding="utf-8")
    branch = '''    elif data_source in ["anchorrank_pointwise", "anchorrank_listwise"]:
        from . import anchorrank
        res = anchorrank.compute_score(data_source, solution_str, ground_truth, extra_info=extra_info)
'''
    marker = '    elif data_source in ["ranking"]:'
    if "anchorrank_pointwise" not in text:
        if marker not in text:
            raise RuntimeError(f"Could not find the ranking reward branch in {dispatcher}")
        dispatcher.write_text(text.replace(marker, branch + marker), encoding="utf-8")
    print(target)
    print(f"Registered AnchorRank reward sources in {dispatcher}")


if __name__ == "__main__":
    main()
