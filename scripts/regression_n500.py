"""Cổng hồi quy C6: code mới phải cho ĐÚNG từng bit các số n=500 đã công bố.

Chạy lại bốn checkpoint v1 với ``--limit 500`` bằng code hiện tại, ghi ra một thư
mục tạm, rồi so với các ``eval_*_validation.json`` n=500 gốc. Mọi metric tổng và
mọi dự đoán mẫu phải bằng nhau CHÍNH XÁC (so float bằng ``==``, không dung sai):
code ghi thêm bằng chứng (preds JSONL, cửa sổ, selection.json) không được phép
đổi một con số nào đã có.

    python scripts/run_eval.py --models baseline xlmr mbert visobert --limit 500 \\
        --out-dir /tmp/reg500
    python scripts/regression_n500.py /tmp/reg500

Exit 1 nếu có bất kỳ khác biệt nào, kèm danh sách.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNS = ("baseline", "xlmr", "mbert", "visobert")

#: (khối, khoá) phải trùng từng bit.
FIELDS = [("overall", "EM"), ("overall", "F1"), ("overall", "count"),
          ("answerable_only", "EM"), ("answerable_only", "F1"), ("answerable_only", "count"),
          ("impossible_only", "EM"), ("impossible_only", "count")]


def reference_dir(results_dir: Path) -> Path:
    """``history/n500`` khi đã dời, ngược lại chính ``results/`` (trước khi dời)."""
    history = results_dir / "history" / "n500"
    return history if history.is_dir() else results_dir


def compare(ref_dir: Path, new_dir: Path, runs=RUNS) -> list[str]:
    diffs = []
    for run in runs:
        ref_path = ref_dir / f"eval_{run}_validation.json"
        new_path = new_dir / f"eval_{run}_validation.json"
        if not ref_path.exists() or not new_path.exists():
            diffs.append(f"{run}: thiếu tệp ({ref_path.exists()=}, {new_path.exists()=})")
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        if ref["n"] != 500 or new["n"] != 500:
            diffs.append(f"{run}: n phải là 500 (ref={ref['n']}, new={new['n']})")
        for block, key in FIELDS:
            a, b = ref[block][key], new[block][key]
            if a != b:
                diffs.append(f"{run}: {block}.{key} {a!r} != {b!r}")
        for name in ("by_context_length", "by_question_type"):
            if ref.get(name) != new.get(name):
                diffs.append(f"{run}: {name} khác")
        ref_s = [(s["qid"], s["prediction"]) for s in ref.get("sample_predictions", [])]
        new_s = [(s["qid"], s["prediction"]) for s in new.get("sample_predictions", [])]
        if ref_s != new_s:
            diffs.append(f"{run}: sample_predictions khác")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_dir", help="thư mục chứa eval JSON vừa chạy lại (n=500)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    ref = reference_dir(Path(args.results_dir))
    diffs = compare(ref, Path(args.new_dir))
    if diffs:
        print(f"HỒI QUY: {len(diffs)} khác biệt so với {ref}:")
        for d in diffs:
            print(f"  {d}")
        return 1
    print(f"OK: {len(RUNS)} run trùng từng bit với {ref} "
          f"({len(FIELDS)} metric + breakdown + dự đoán mẫu).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
