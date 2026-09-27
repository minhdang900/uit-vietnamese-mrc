"""Bảng số cửa sổ ``make_windows`` sinh ra theo độ dài context — N4, ADR-003.

ADR-003 (``docs/ARCHITECTURE.md``) ghi một bảng "số window sinh ra" cho context
210/420/700/1400 token, với caption nói đo tại ``max_length=384``. Số đo lại
27/09/2026 (``00_STUDY_PACK_CS116/09_bang_chung/ket_qua_do_2026-09-27.txt``) cho
thấy các số 2/5/8/16 khớp công thức ở ``max_length=128, doc_stride=32`` (đúng
tham số dùng trong ``tests/test_windowing.py``), KHÔNG khớp 384/128: ở 384/128,
context 700 token chỉ cần ~3 cửa sổ.

``n_windows_for_length`` là CÙNG công thức bước trượt mà ``mrc.windowing.make_windows``
dùng (xem ``windowing.py:363-401``: ``step = max(1, budget - doc_stride)``, cửa sổ
cuối khi ``tok_end >= n_ctx_tokens``), nhưng làm việc trên SỐ TOKEN thay vì token
thật — nên test được bằng số nguyên, không cần tokenizer.

    python scripts/window_table.py
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

from mrc.evaluate import _git_commit

#: Độ dài context (SỐ TOKEN) mà bảng ADR-003 liệt kê.
CONTEXT_LENGTHS = (210, 420, 700, 1400)
#: (max_length, doc_stride) — cấu hình ADR-003 gốc (khớp tests/test_windowing.py)
#: và cấu hình huấn luyện thật của dự án.
CONFIGS = ((128, 32), (384, 128))


def n_windows_for_length(
    n_ctx_tokens: int, max_length: int, doc_stride: int,
    n_question_tokens: int = 0, n_special: int = 3,
) -> int:
    """Số cửa sổ ``make_windows`` sinh ra cho một context dài ``n_ctx_tokens`` token.

    Mô phỏng ĐÚNG vòng lặp của ``make_windows`` (cùng công thức ``step``, cùng
    điều kiện dừng), chỉ thay việc cắt chuỗi thật bằng số học trên số token —
    hai cách tính cho cùng kết quả vì ``make_windows`` chỉ dùng độ dài (số token)
    của context và câu hỏi để quyết định biên cửa sổ, không dùng nội dung ký tự.

    ``n_special`` mặc định 3 (``[CLS] question [SEP] context [SEP]``, cặp câu).
    """
    budget = max_length - n_question_tokens - n_special
    if budget < 1:
        raise ValueError(
            f"n_question_tokens={n_question_tokens} không còn chỗ cho context "
            f"trong max_length={max_length}."
        )
    if n_ctx_tokens <= 0:
        return 0

    step = max(1, budget - doc_stride)
    count = 0
    tok_start = 0
    while True:
        tok_end = min(tok_start + budget, n_ctx_tokens)
        count += 1
        if tok_end >= n_ctx_tokens:
            break
        tok_start += step
    return count


def build_table(
    context_lengths=CONTEXT_LENGTHS, configs=CONFIGS, n_question_tokens: int = 3,
) -> dict:
    return {
        f"{max_length}_{doc_stride}": {
            str(n): n_windows_for_length(n, max_length, doc_stride, n_question_tokens)
            for n in context_lengths
        }
        for max_length, doc_stride in configs
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-question-tokens", type=int, default=3,
                    help="độ dài câu hỏi giả định (token) dùng cho công thức; "
                         "2-4 cho cùng kết quả với ADR-003 (câu hỏi ViQuAD ngắn)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = {
        "context_lengths": list(CONTEXT_LENGTHS),
        "n_question_tokens_assumed": args.n_question_tokens,
        "n_special_tokens": 3,
        "windows": build_table(n_question_tokens=args.n_question_tokens),
        "commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    results_dir = _Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "window_counts.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
