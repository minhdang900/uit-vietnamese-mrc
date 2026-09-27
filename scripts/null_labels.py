"""Đo tỉ lệ nhãn ``[CLS]`` (null) trong dữ liệu huấn luyện — Phụ lục A, N1.

Kế thừa ``00_STUDY_PACK_CS116/09_bang_chung/dem_nhan_null_theo_cua_so.py`` (đo lại
27/09/2026), nhưng phần đếm giờ nằm ở ``mrc.audit.null_label_rates`` — một hàm
THUẦN có test riêng — thay vì lặp trong script.

    python scripts/null_labels.py --models mbert visobert --split train
"""

# Bootstrap sys.path TRƯỚC mọi import của dự án — xem scripts/run_eval.py.
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
from datetime import datetime, timezone

from mrc.audit import null_label_rates
from mrc.data import load_squad_file
from mrc.evaluate import _git_commit


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                    help="tên thư mục dưới models/ (vd: mbert visobert)")
    p.add_argument("--split", default="train", choices=("train", "validation", "test"))
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models",
                    help="thư mục chứa checkpoint (mặc định models/)")
    p.add_argument("--data-dir", default="data/raw")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from transformers import AutoTokenizer

    data_path = _Path(args.data_dir) / f"viquad2_{args.split}.json"
    examples = load_squad_file(data_path)

    results_dir = _Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = _git_commit()

    records = []
    for model in args.models:
        tokenizer_path = str(_Path(args.models_dir) / model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        rates = null_label_rates(
            examples, tokenizer,
            max_length=args.max_length, doc_stride=args.doc_stride,
        )
        record = {
            "model": model,
            "split": args.split,
            "max_length": args.max_length,
            "doc_stride": args.doc_stride,
            "commit": commit,
            "timestamp": timestamp,
            "tokenizer_path": tokenizer_path,
            **rates,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    out_path = results_dir / f"null_labels_{args.split}.json"
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
