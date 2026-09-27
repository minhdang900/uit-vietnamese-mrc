"""Chạy đánh giá một hoặc nhiều model và ghi kết quả có provenance.

Đặt trong ``src/`` chứ không phải ``scripts/`` vì nó chứa quyết định cần test
(lấy mẫu, chọn ``max_answer_len``), không chỉ là nối dây. ``scripts/run_eval.py``
là vỏ mỏng gọi vào đây.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mrc.data import assert_gradeable, compute_stats, load_squad_file, reproducible_subset
from mrc.evaluate import _git_commit, run_evaluation

__all__ = [
    "subset", "max_answer_len_for", "build_predictor", "parse_args",
    "resolve_limit", "main", "MAX_ANSWER_LEN", "DEFAULT_MAX_ANSWER_LEN",
    "load_selection", "resolve_inference_config", "INVOCATION_LOG",
]

#: Thư mục chứa model fine-tuned, tương đối với thư mục chạy (gốc repo).
MODELS_DIR = Path("models")

#: Cấu hình cửa sổ mặc định — chính là cấu hình mà các checkpoint v1 được huấn luyện.
DEFAULT_MAX_LENGTH = 384
DEFAULT_DOC_STRIDE = 128

#: Mỗi lần chạy CLI để lại một dòng ở đây, kể cả lần bị từ chối. Headline chỉ được
#: chấm MỘT lần; nhật ký này là bằng chứng cho điều đó.
INVOCATION_LOG = "eval_invocations.jsonl"

#: Model không có gì để chọn (không huấn luyện, không τ).
NON_TRANSFORMER_KINDS = ("baseline", "empty")

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


def load_selection(kind: str) -> dict | None:
    """``models/{kind}/selection.json`` do ``finetune.py`` ghi, hoặc ``None`` (checkpoint v1)."""
    path = MODELS_DIR / kind / "selection.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_inference_config(
    kind: str,
    max_answer_len: int | None = None,
    null_threshold: float | None = None,
    max_length: int | None = None,
    doc_stride: int | None = None,
) -> dict:
    """Cấu hình suy luận HIỆU LỰC cho ``kind``, kèm nguồn của từng giá trị.

    Thứ tự ưu tiên: cờ CLI tường minh > ``selection.json`` > bảng legacy. Bảng
    legacy tra ``MAX_ANSWER_LEN`` theo tên CHÍNH XÁC, nên ``visobert-dev`` sẽ bị
    chấm với 30 token nếu không có ``selection.json`` — lý do file đó mang theo
    cả ``max_length``/``doc_stride``/``max_answer_len`` chứ không chỉ ``tau``.
    """
    sel = load_selection(kind)
    legacy = {
        "max_length": DEFAULT_MAX_LENGTH,
        "doc_stride": DEFAULT_DOC_STRIDE,
        "max_answer_len": MAX_ANSWER_LEN.get(kind, DEFAULT_MAX_ANSWER_LEN),
        "tau": 0.0,
    }
    cli = {"max_length": max_length, "doc_stride": doc_stride,
           "max_answer_len": max_answer_len, "tau": null_threshold}

    cfg: dict = {}
    source: dict = {}
    for key in legacy:
        if cli[key] is not None:
            cfg[key], source[key] = cli[key], "cli"
        elif sel is not None and sel.get(key) is not None:
            cfg[key], source[key] = sel[key], "selection.json"
        else:
            cfg[key], source[key] = legacy[key], "legacy"
    cfg["source"] = source
    if sel is not None:
        cfg["selected_on"] = sel.get("selected_on")
        cfg["selection_epoch"] = sel.get("epoch")
    elif kind in NON_TRANSFORMER_KINDS or kind == "xlmr":
        cfg["selected_on"] = None
    else:
        # Checkpoint v1: epoch chọn trên 300 câu ngẫu nhiên của CHÍNH validation.
        cfg["selected_on"] = "validation_subset_300"
    return cfg


def build_predictor(
    kind: str,
    max_answer_len: int | None = None,
    null_threshold: float | None = None,
    max_length: int | None = None,
    doc_stride: int | None = None,
):
    """Dựng predictor theo tên. Model fine-tuned chưa có trên đĩa thì FAIL TO ỒN."""
    if kind == "baseline":
        from mrc.baseline_tfidf import TfidfRetriever

        return TfidfRetriever()
    if kind == "empty":
        from mrc.predictor import EmptyPredictor

        return EmptyPredictor()

    from mrc import transformer_qa

    cfg = resolve_inference_config(kind, max_answer_len, null_threshold,
                                   max_length, doc_stride)
    kwargs = {"max_length": cfg["max_length"], "doc_stride": cfg["doc_stride"],
              "max_answer_len": cfg["max_answer_len"], "null_threshold": cfg["tau"]}
    if kind == "xlmr":
        return transformer_qa.TransformerQA("deepset/xlm-roberta-base-squad2",
                                            name="XLM-R (squad2, zero-shot)", **kwargs)

    path = MODELS_DIR / kind
    if not path.exists():
        raise SystemExit(
            f"Chưa có model fine-tuned tại {path}. "
            f"Chạy: python scripts/finetune.py --model <hf-name> --out models/{kind}"
        )
    return transformer_qa.TransformerQA(str(path), name=f"{kind} (fine-tuned)", **kwargs)


def checkpoint_for(kind: str) -> str | None:
    if kind in NON_TRANSFORMER_KINDS:
        return None
    if kind == "xlmr":
        return "deepset/xlm-roberta-base-squad2"
    return str(MODELS_DIR / kind)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Đánh giá model MRC trên UIT-ViQuAD 2.0")
    ap.add_argument("--models", nargs="+", default=["baseline"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="dùng toàn bộ split")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-answer-len", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--doc-stride", type=int, default=None)
    ap.add_argument("--null-threshold", type=float, default=None,
                    help="τ tường minh; mặc định đọc models/{kind}/selection.json, rồi 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preds", action=argparse.BooleanOptionalAction, default=True,
                    help="ghi preds_{kind}_{split}.jsonl (một dòng mỗi câu hỏi)")
    ap.add_argument("--force", action="store_true",
                    help="cho phép ghi đè eval JSON đã có — chỉ khi chạy lại vì bug code")
    ap.add_argument("--reason", default=None, help="bắt buộc đi kèm --force")
    return ap.parse_args(argv)


def resolve_limit(args: argparse.Namespace) -> int | None:
    """``--full`` thắng ``--limit``."""
    return None if args.full else args.limit


def _log_invocation(out_dir: Path, argv, args, refused: list[str]) -> None:
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(argv),
        "commit": _git_commit(),
        "run_ids": list(args.models),
        "split": args.split,
        "forced": bool(args.force),
        "reason": args.reason,
        "refused": refused,
    }
    with (out_dir / INVOCATION_LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.force and not args.reason:
        raise SystemExit("--force phải kèm --reason: lý do chạy lại được ghi vào nhật ký.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chấm-một-lần: kiểm TRƯỚC khi tải dữ liệu hay model, và ghi nhật ký cả khi từ chối.
    existing = [str(out_dir / f"eval_{k}_{args.split}.json") for k in args.models
                if (out_dir / f"eval_{k}_{args.split}.json").exists()]
    refused = existing if not args.force else []
    _log_invocation(out_dir, raw_argv, args, refused)
    if refused:
        raise SystemExit(
            f"Từ chối ghi đè {refused}: mỗi run chỉ được chấm trên split này MỘT lần. "
            "Nếu chạy lại vì bug code, dùng --force --reason '<lý do>'."
        )

    examples = load_squad_file(Path(args.data_dir) / f"viquad2_{args.split}.json")
    # Cổng: từ chối split không chấm được TRƯỚC khi tốn thời gian chạy model.
    assert_gradeable(examples)

    examples = subset(examples, resolve_limit(args), seed=args.seed)
    stats = compute_stats(examples)
    print(f"split={args.split}  n={stats['num_questions']}  "
          f"contexts={stats['num_contexts']}  "
          f"impossible={stats['num_impossible']} ({stats['impossible_pct']}%)")

    for kind in args.models:
        print(f"\n=== {kind} ===", flush=True)
        overrides = dict(max_answer_len=args.max_answer_len,
                         null_threshold=args.null_threshold,
                         max_length=args.max_length, doc_stride=args.doc_stride)
        predictor = build_predictor(kind, **overrides)
        cfg = (None if kind in NON_TRANSFORMER_KINDS
               else resolve_inference_config(kind, **overrides))
        if cfg:
            print(f"  max_length={cfg['max_length']} doc_stride={cfg['doc_stride']} "
                  f"max_answer_len={cfg['max_answer_len']} tau={cfg['tau']} "
                  f"(nguồn: {cfg['source']})")

        preds_path = out_dir / f"preds_{kind}_{args.split}.jsonl" if args.preds else None
        result = run_evaluation(
            predictor, examples, split=args.split, preds_path=preds_path, run_id=kind,
            checkpoint=checkpoint_for(kind),
            selected_on=cfg.get("selected_on") if cfg else None,
            inference_config=cfg,
        )
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
