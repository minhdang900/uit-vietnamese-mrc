"""Chạy đánh giá một hoặc nhiều model trên một split và ghi kết quả ra results/.

Ví dụ:
    python scripts/run_eval.py --models baseline --limit 500
    python scripts/run_eval.py --models baseline xlmr --full
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from mrc.data import assert_gradeable, compute_stats, load_squad_file
from mrc.evaluate import run_evaluation


def subset(examples, n, seed=42):
    """Lấy mẫu con TÁI LẬP ĐƯỢC. Không lấy n câu đầu — chúng thiên lệch theo
    article đầu tiên của file."""
    if n is None or n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(list(examples), n)


def build(kind: str):
    if kind == "baseline":
        from mrc.baseline_tfidf import TfidfRetriever

        return TfidfRetriever()
    if kind == "xlmr":
        from mrc.transformer_qa import TransformerQA

        return TransformerQA("deepset/xlm-roberta-base-squad2", name="XLM-R (squad2, zero-shot)")
    # model đã fine-tune, đọc từ đĩa
    from mrc.transformer_qa import TransformerQA

    path = Path("models") / kind
    if not path.exists():
        raise SystemExit(f"Chưa có model fine-tuned tại {path}. Chạy scripts/finetune.py trước.")
    return TransformerQA(str(path), name=f"{kind} (fine-tuned)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="dùng toàn bộ split")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    examples = load_squad_file(Path(args.data_dir) / f"viquad2_{args.split}.json")
    # Cổng: từ chối split không chấm được, TRƯỚC khi tốn thời gian chạy model.
    assert_gradeable(examples)

    n = None if args.full else args.limit
    examples = subset(examples, n)
    stats = compute_stats(examples)
    print(f"split={args.split}  n={stats['num_questions']}  contexts={stats['num_contexts']}  "
          f"impossible={stats['num_impossible']} ({stats['impossible_pct']}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in args.models:
        print(f"\n=== {kind} ===", flush=True)
        result = run_evaluation(build(kind), examples, split=args.split)
        o, a, i = result["overall"], result["answerable_only"], result["impossible_only"]
        print(f"  overall     EM {o['EM']:6.2f}  F1 {o['F1']:6.2f}  (n={o['count']})")
        print(f"  answerable  EM {a['EM']:6.2f}  F1 {a['F1']:6.2f}  (n={a['count']})")
        print(f"  impossible  EM {i['EM']:6.2f}                    (n={i['count']})")
        print(f"  latency     {result['avg_latency_ms']} ms/câu   device={result['device']}")

        path = out_dir / f"eval_{kind}_{args.split}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
