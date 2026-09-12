"""Chạy đánh giá một hoặc nhiều model và ghi kết quả có provenance.

Đặt trong ``src/`` chứ không phải ``scripts/`` vì nó chứa quyết định cần test
(lấy mẫu, chọn ``max_answer_len``), không chỉ là nối dây. ``scripts/run_eval.py``
là vỏ mỏng gọi vào đây.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrc.data import assert_gradeable, compute_stats, load_squad_file, reproducible_subset
from mrc.evaluate import run_evaluation

__all__ = [
    "subset", "max_answer_len_for", "build_predictor", "parse_args",
    "resolve_limit", "main", "MAX_ANSWER_LEN", "DEFAULT_MAX_ANSWER_LEN",
]

#: Độ dài span tối đa theo TOKEN, đo riêng cho từng tokenizer.
#: Cùng một đáp án sinh ra số token khác nhau tuỳ vocab, nên một hằng số chung là sai.
#: Giá trị = p95 độ dài gold answer trên 4.000 mẫu train:
#:   mBERT / XLM-R (vocab 119k / 250k) -> p95 = 40 token  (dùng 30, vượt 10,8%)
#:   ViSoBERT      (vocab  15k)        -> p95 = 64 token  (dùng 30, vượt 25,2%)
MAX_ANSWER_LEN = {"visobert": 64}
DEFAULT_MAX_ANSWER_LEN = 30


def subset(examples, n, seed: int = 42):
    """Mẫu con ngẫu nhiên, tái lập được — KHÔNG phải n câu đầu file.

    Uỷ cho :func:`mrc.data.reproducible_subset` để đường cong huấn luyện và bảng
    kết quả cuối dùng chung đúng một hàm lấy mẫu.
    """
    return reproducible_subset(examples, n, seed=seed)


def max_answer_len_for(model_kind: str, override: int | None = None) -> int:
    """``max_answer_len`` phù hợp với tokenizer của model."""
    if override:
        return override
    return MAX_ANSWER_LEN.get(model_kind, DEFAULT_MAX_ANSWER_LEN)


def build_predictor(kind: str, max_answer_len: int | None = None):
    """Dựng predictor theo tên. Model fine-tuned chưa có trên đĩa thì FAIL TO ỒN."""
    if kind == "baseline":
        from mrc.baseline_tfidf import TfidfRetriever

        return TfidfRetriever()

    from mrc.transformer_qa import TransformerQA

    span_limit = max_answer_len_for(kind, max_answer_len)
    if kind == "xlmr":
        return TransformerQA("deepset/xlm-roberta-base-squad2",
                             name="XLM-R (squad2, zero-shot)",
                             max_answer_len=span_limit)

    path = Path("models") / kind
    if not path.exists():
        raise SystemExit(
            f"Chưa có model fine-tuned tại {path}. "
            f"Chạy: python scripts/finetune.py --model <hf-name> --out models/{kind}"
        )
    return TransformerQA(str(path), name=f"{kind} (fine-tuned)", max_answer_len=span_limit)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Đánh giá model MRC trên UIT-ViQuAD 2.0")
    ap.add_argument("--models", nargs="+", default=["baseline"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="dùng toàn bộ split")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-answer-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def resolve_limit(args: argparse.Namespace) -> int | None:
    """``--full`` thắng ``--limit``."""
    return None if args.full else args.limit


def main(argv=None) -> None:
    args = parse_args(argv)

    examples = load_squad_file(Path(args.data_dir) / f"viquad2_{args.split}.json")
    # Cổng: từ chối split không chấm được TRƯỚC khi tốn thời gian chạy model.
    assert_gradeable(examples)

    examples = subset(examples, resolve_limit(args), seed=args.seed)
    stats = compute_stats(examples)
    print(f"split={args.split}  n={stats['num_questions']}  "
          f"contexts={stats['num_contexts']}  "
          f"impossible={stats['num_impossible']} ({stats['impossible_pct']}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in args.models:
        print(f"\n=== {kind} ===", flush=True)
        predictor = build_predictor(kind, args.max_answer_len)
        print(f"  max_answer_len={getattr(predictor, 'max_answer_len', '-')}")

        result = run_evaluation(predictor, examples, split=args.split)
        overall, answerable, impossible = (result["overall"], result["answerable_only"],
                                           result["impossible_only"])
        print(f"  overall     EM {overall['EM']:6.2f}  F1 {overall['F1']:6.2f}  "
              f"(n={overall['count']})")
        print(f"  answerable  EM {answerable['EM']:6.2f}  F1 {answerable['F1']:6.2f}  "
              f"(n={answerable['count']})")
        print(f"  impossible  EM {impossible['EM']:6.2f}"
              f"                    (n={impossible['count']})")
        print(f"  latency     {result['avg_latency_ms']} ms/câu   device={result['device']}")

        path = out_dir / f"eval_{kind}_{args.split}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {path}")
