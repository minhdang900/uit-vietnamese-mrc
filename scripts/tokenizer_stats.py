"""Đo mức độ phân mảnh token của mBERT so với ViSoBERT — Chương 7.2, N2.

Kế thừa ``06_BaoCao_T11/05_BANG_CHUNG/tokenizer_stats.py`` (đường dẫn cứng, chỉ in
stdout). Thêm ``--results-dir``, ghi ``results/tokenizer_stats.json``.

Bảng số đo hai điều báo cáo v1 dễ hiểu lầm (N2, Phụ lục
``ban_ky_thuat_truoc_khi_viet_lai.md``):

1. **"Sức chứa nhỏ hơn 45%" là về VOCAB, không phải về TÍNH TOÁN.** mBERT và
   ViSoBERT có cùng kiến trúc encoder (12 lớp, hidden 768, FFN 3072) — thân
   Transformer ``encoder_body_params`` bằng NHAU (~85M). Toàn bộ chênh lệch
   177,3M so với 97,0M tổng tham số nằm ở ma trận embedding từ vựng
   (``embedding_params``: 91,8M so với 11,5M).
2. **Vocab và số hàng embedding không phải luôn bằng nhau.** ViSoBERT có
   ``vocab_size`` báo cáo 15.002 nhưng ma trận embedding có 15.004 hàng (2 hàng
   dự phòng không thuộc vocab công bố) — bẫy nếu ai tính tham số bằng tay từ
   ``vocab_size``.

    python scripts/tokenizer_stats.py
"""

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import math
import statistics as st
from datetime import datetime, timezone

from mrc.data import load_squad_file
from mrc.evaluate import _git_commit
from mrc.windowing import make_windows

_SAMPLE_SENTENCE = "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam."
_LONG_CTX_THRESHOLD = 357


def _param_breakdown(model_dir: _Path) -> dict:
    """Tổng tham số + tách embedding-từ-vựng khỏi phần thân Transformer.

    ``encoder_body_params`` = tổng − embedding từ vựng − position embedding −
    token-type embedding, để tách rời "sức chứa tính toán" (thân, như nhau giữa
    hai model) khỏi "dung lượng từ vựng" (embedding, khác nhau — xem N2).
    """
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        shapes = {k: f.get_slice(k).get_shape() for k in keys}

    total = sum(math.prod(shape) for shape in shapes.values())
    word_emb_keys = [k for k in keys if "word_embeddings" in k]
    other_emb_keys = [k for k in keys if "position_embeddings" in k or "token_type_embeddings" in k]

    embedding_rows, hidden_size = shapes[word_emb_keys[0]] if word_emb_keys else (0, 0)
    embedding_params = sum(math.prod(shapes[k]) for k in word_emb_keys)
    other_embedding_params = sum(math.prod(shapes[k]) for k in other_emb_keys)

    return {
        "total_params": total,
        "embedding_rows": embedding_rows,
        "hidden_size": hidden_size,
        "embedding_params": embedding_params,
        "encoder_body_params": total - embedding_params - other_embedding_params,
    }


def compute_tokenizer_stats(
    model_dirs: dict[str, str],
    contexts: list[str],
    train_examples: list,
    val_examples: list,
    max_length: int = 384,
    doc_stride: int = 128,
    n_train_answers: int = 4000,
) -> dict:
    from transformers import AutoTokenizer

    out: dict = {}
    for name, model_dir in model_dirs.items():
        model_dir = _Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

        ctx_lens = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in contexts]
        words = sum(len(c.split()) for c in contexts)
        answer_lens = sorted(
            len(tokenizer(e.answers[0], add_special_tokens=False)["input_ids"])
            for e in train_examples[:n_train_answers] if e.answers
        )

        # Cửa sổ phụ thuộc CẢ context lẫn câu hỏi (ngân sách trừ đi độ dài câu
        # hỏi), nên hai câu cùng context có thể cho số cửa sổ khác nhau. Một
        # PARAGRAPH (context) được tính là ">2 cửa sổ" nếu BẤT KỲ câu hỏi nào
        # của nó cần >2 cửa sổ — đây là định nghĩa G5 dùng (khớp
        # "questions_gt2_windows" ở mức câu hỏi, gộp lên mức context).
        gt2_windows_qids: set[str] = set()
        gt2_windows_contexts: set[str] = set()
        for e in val_examples:
            n_windows = len(list(make_windows(
                e.question, e.context, tokenizer,
                max_length=max_length, doc_stride=doc_stride,
            )))
            if n_windows > 2:
                gt2_windows_qids.add(e.qid)
                gt2_windows_contexts.add(e.context)
        gt2_windows = len(gt2_windows_qids)
        ctx_gt2_windows = len(gt2_windows_contexts)

        row = {
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "sample_sentence_tokens": len(tokenizer.tokenize(_SAMPLE_SENTENCE)),
            "val_ctx_mean_tokens": round(st.mean(ctx_lens), 1) if ctx_lens else 0.0,
            "val_ctx_tokens_per_word": round(sum(ctx_lens) / words, 2) if words else 0.0,
            "val_ctx_over_357": sum(1 for x in ctx_lens if x > _LONG_CTX_THRESHOLD),
            "val_ctx_total": len(ctx_lens),
            "questions_gt2_windows": gt2_windows,
            "val_ctx_gt2_windows": ctx_gt2_windows,
            "questions_gt2_windows_config": {"max_length": max_length, "doc_stride": doc_stride},
            "train_answers_sampled": len(answer_lens),
            "train_answers_p95": answer_lens[int(0.95 * len(answer_lens))] if answer_lens else None,
            "train_answers_over_30_pct": round(
                100 * sum(1 for x in answer_lens if x > 30) / len(answer_lens), 1
            ) if answer_lens else 0.0,
        }
        row.update(_param_breakdown(model_dir))
        out[name] = row

    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["mbert", "visobert"])
    p.add_argument("--models-dir", default="models")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = _Path(args.data_dir)

    val_examples = load_squad_file(data_dir / "viquad2_validation.json")
    train_examples = load_squad_file(data_dir / "viquad2_train.json")
    contexts = list({e.context for e in val_examples})

    model_dirs = {m: str(_Path(args.models_dir) / m) for m in args.models}
    payload = compute_tokenizer_stats(
        model_dirs, contexts, train_examples, val_examples,
        max_length=args.max_length, doc_stride=args.doc_stride,
    )
    payload["commit"] = _git_commit()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results_dir = _Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "tokenizer_stats.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
