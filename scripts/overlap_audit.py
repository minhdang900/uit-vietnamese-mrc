"""Chồng lấp giữa các mẫu ``reproducible_subset`` — W2, Phụ lục A.

W2 (điểm yếu đã thừa nhận): validation "vừa chọn epoch vừa báo cáo" — mẫu 300
câu dùng để chọn epoch lúc huấn luyện và mẫu 200 câu dùng để quét ``null_threshold``
đều lấy từ CÙNG một hàm ``reproducible_subset(seed=42)`` trên CÙNG validation split
mà mẫu báo cáo cuối (500 câu) cũng lấy từ đó — nên chúng chồng lấp nhau, không độc
lập. Script này đo đúng phần chồng lấp đó bằng ``qid``, cộng với việc xác nhận
train/validation không chia sẻ ``title`` nào (D3: dev split theo bài viết là
tách biệt NGHIÊM NGẶT hơn tách theo context).

    python scripts/overlap_audit.py
"""

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
from datetime import datetime, timezone

from mrc.data import load_squad_file, reproducible_subset
from mrc.evaluate import _git_commit


def compute_overlap(
    validation: list, train: list,
    sizes: tuple[int, ...] = (300, 200), report_n: int = 500, seed: int = 42,
) -> dict:
    """Hàm THUẦN: mọi số trong W2/N6 tính từ đây, không đường dẫn nào cứng."""
    report_subset = reproducible_subset(validation, report_n, seed=seed)
    report_qids = {e.qid for e in report_subset}

    overlaps = {}
    for n in sizes:
        subset = reproducible_subset(validation, n, seed=seed)
        subset_qids = {e.qid for e in subset}
        overlaps[str(n)] = {
            "n": n,
            "overlap_with_report_n": len(subset_qids & report_qids),
            "report_n": report_n,
        }

    shared_titles_train_val = {e.title for e in train} & {e.title for e in validation}

    return {
        "seed": seed,
        "report_n": report_n,
        "report_subset_contexts": len({e.context for e in report_subset}),
        "report_subset_articles": len({e.title for e in report_subset}),
        "overlaps": overlaps,
        "shared_titles_train_validation": len(shared_titles_train_val),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--sizes", nargs="+", type=int, default=[300, 200])
    p.add_argument("--report-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = _Path(args.data_dir)
    validation = load_squad_file(data_dir / "viquad2_validation.json")
    train = load_squad_file(data_dir / "viquad2_train.json")

    payload = compute_overlap(
        validation, train, sizes=tuple(args.sizes), report_n=args.report_n, seed=args.seed,
    )
    payload["commit"] = _git_commit()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results_dir = _Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "sample_overlap.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
