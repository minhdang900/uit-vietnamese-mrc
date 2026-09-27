"""Kiểm khứ hồi ký tự: nhãn token của huấn luyện có cắt lại ĐÚNG đáp án vàng không?

Với mỗi câu answerable của TRAIN (không bao giờ validation): cắt cửa sổ như lúc
huấn luyện, tìm cặp token bao đáp án (``_locate_answer_tokens``), rồi cắt lại
context theo offset của hai token đó. Nếu chuỗi cắt ra khác gold, model đang được
dạy một span khác với đáp án — trần EM của nó thấp đi tương ứng.

Hai tỉ lệ cho mỗi tokenizer:

* ``exact`` — trùng từng ký tự;
* ``normalized`` — trùng sau ``normalize_answer`` (bỏ dấu câu, chữ thường…), tức
  đúng thước đo EM. Đây là tỉ lệ dùng cho cổng P6: dấu câu dính vào từ (PhoBERT
  tách theo khoảng trắng) làm lệch ``exact`` nhưng không đổi EM.

Cổng P6 (quyết định của người dùng): PhoBERT ``normalized`` ≥ 99 % VÀ cách mBERT
không quá 0,5 điểm.

    python scripts/roundtrip_audit.py            # -> results/roundtrip_train.json
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file
from mrc.evaluate import _git_commit
from mrc.features import _locate_answer_tokens
from mrc.normalize import normalize_answer
from mrc.windowing import decode_span, make_windows

#: (khoá, tên HF, max_length, doc_stride) — đúng cấu hình huấn luyện của từng model.
TOKENIZERS = (
    ("mbert", "bert-base-multilingual-cased", 384, 128),
    ("visobert", "uitnlp/visobert", 384, 128),
    ("phobert", "vinai/phobert-base-v2", 256, 64),
)

GATE_MIN_NORMALIZED = 99.0
GATE_MAX_GAP_TO_MBERT = 0.5
PUNCT = set(string.punctuation) | set("“”‘’…–—")


def categorize(context: str, start: int, gold: str, got: str | None) -> str:
    """Vì sao ``got`` khác ``gold``. Thứ tự: không cửa sổ > gold cắt giữa từ > dấu câu."""
    if got is None:
        return "no_containing_window"
    end = start + len(gold)
    if (start > 0 and context[start - 1].isalnum() and context[start].isalnum()) or \
       (end < len(context) and context[end].isalnum() and context[end - 1].isalnum()):
        return "gold_cuts_inside_word"
    if got.startswith(gold) and set(got[len(gold):]) <= PUNCT:
        return "trailing_punct_attached"
    if got.endswith(gold) and set(got[: len(got) - len(gold)]) <= PUNCT:
        return "leading_punct_attached"
    return "other"


def audit(examples, tokenizer, max_length: int, doc_stride: int, n_examples: int = 10) -> dict:
    """Tỉ lệ khứ hồi exact/normalized trên các câu answerable của ``examples``."""
    t0 = time.time()
    n = exact = normalized = 0
    categories: Counter = Counter()
    samples: dict[str, list] = {}
    for ex in examples:
        if ex.is_impossible or not ex.answers:
            continue
        n += 1
        gold = ex.answers[0]
        s0, s1 = ex.answer_start, ex.answer_start + len(gold)
        got = None
        for w in make_windows(ex.question, ex.context, tokenizer,
                              max_length=max_length, doc_stride=doc_stride):
            cls = w.input_ids.index(tokenizer.cls_token_id)
            a, b = _locate_answer_tokens(w.offset_mapping, s0, s1, cls)
            if a == cls:
                continue
            got = decode_span(ex.context, w.offset_mapping[a][0], w.offset_mapping[b][1])
            break
        if got == gold:
            exact += 1
            normalized += 1
            continue
        if got is not None and normalize_answer(got) == normalize_answer(gold):
            normalized += 1
        cat = categorize(ex.context, s0, gold, got)
        categories[cat] += 1
        bucket = samples.setdefault(cat, [])
        if len(bucket) < n_examples:
            bucket.append({"qid": ex.qid, "gold": gold, "got": got})
    return {
        "max_length": max_length,
        "doc_stride": doc_stride,
        "answerable": n,
        "exact": exact,
        "normalized": normalized,
        "exact_rate": round(100.0 * exact / n, 3) if n else 0.0,
        "normalized_rate": round(100.0 * normalized / n, 3) if n else 0.0,
        "failure_categories": dict(categories.most_common()),
        "failure_examples": samples,
        "seconds": round(time.time() - t0, 1),
    }


def p6_gate(results: dict) -> dict:
    pho, mb = results["phobert"]["normalized_rate"], results["mbert"]["normalized_rate"]
    passed = pho >= GATE_MIN_NORMALIZED and mb - pho <= GATE_MAX_GAP_TO_MBERT
    return {"rule": f"phobert normalized >= {GATE_MIN_NORMALIZED} AND "
                    f"mbert - phobert <= {GATE_MAX_GAP_TO_MBERT}",
            "phobert_normalized": pho, "mbert_normalized": mb,
            "gap": round(mb - pho, 3), "passed": passed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)

    from mrc.tokenization import load_fast_tokenizer

    train = load_squad_file(Path(args.data_dir) / "viquad2_train.json")
    out = {"split": "train", "commit": _git_commit(),
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tokenizers": {}}
    for key, name, max_length, stride in TOKENIZERS:
        tok = load_fast_tokenizer(name)
        r = audit(train, tok, max_length, stride)
        out["tokenizers"][key] = {"name": name, "tokenizer_class": type(tok).__name__, **r}
        print(f"{key:9s} exact {r['exact_rate']:7.3f}%  normalized {r['normalized_rate']:7.3f}%"
              f"  (n={r['answerable']}, {r['max_length']}/{r['doc_stride']})  "
              f"{r['failure_categories']}")
    out["p6_gate"] = p6_gate(out["tokenizers"])
    print(f"P6 gate: {'ĐẠT' if out['p6_gate']['passed'] else 'KHÔNG ĐẠT'} — {out['p6_gate']}")

    path = Path(args.results_dir) / "roundtrip_train.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
