"""Thống kê dữ liệu UIT-ViQuAD 2.0 — độ dài, chồng lấp split, mẫu đánh giá, wh-word.

Kế thừa ``06_BaoCao_T11/05_BANG_CHUNG/stats.py`` (đường dẫn cứng, chỉ in stdout +
ghi ra đường dẫn truyền ở ``sys.argv[1]``, không provenance). Bản này thêm
``--results-dir``, mặc định ghi ``results/data_stats.json``, và kèm
``commit``/``timestamp`` để số liệu truy vết được (bất biến #3).

    python scripts/data_stats.py
"""

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import re
import statistics as st
from collections import Counter
from datetime import datetime, timezone

from mrc.data import compute_stats, load_squad_file, reproducible_subset
from mrc.evaluate import _git_commit
from mrc.tagging import tag_examples

_WH_WORDS = ["bao nhiêu", "khi nào", "năm nào", "ở đâu", "tại sao", "vì sao",
             "như thế nào", "ai", "gì", "nào", "mấy"]


def _wh_word(question: str) -> str:
    """Từ để hỏi đầu tiên khớp trong câu, hoặc ``"khác"``."""
    ql = question.lower()
    for word in _WH_WORDS:
        if re.search(r"\b" + word + r"\b", ql):
            return word
    return "khác"


def compute_data_stats(splits: dict[str, list], subset_n: int = 500, subset_seed: int = 42) -> dict:
    """Thống kê đầy đủ dùng cho bảng dữ liệu của báo cáo. Hàm THUẦN, không I/O."""
    out: dict = {}
    for split_name, examples in splits.items():
        stats = compute_stats(examples)
        ctx_words = [len(c.split()) for c in {e.context for e in examples}]
        q_words = [len(e.question.split()) for e in examples]
        a_words = sorted(len(e.answers[0].split()) for e in examples if e.answers)
        stats.update(
            ctx_words_mean=round(st.mean(ctx_words), 1) if ctx_words else 0.0,
            ctx_words_median=st.median(ctx_words) if ctx_words else 0,
            ctx_words_max=max(ctx_words) if ctx_words else 0,
            q_words_mean=round(st.mean(q_words), 1) if q_words else 0.0,
            a_words_mean=round(st.mean(a_words), 2) if a_words else None,
            a_words_median=st.median(a_words) if a_words else None,
            a_words_p95=a_words[int(0.95 * len(a_words))] if a_words else None,
        )
        out[split_name] = stats

    names = list(splits)
    for a, b in ((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))):
        shared = {e.context for e in splits[a]} & {e.context for e in splits[b]}
        shared_titles = {e.title for e in splits[a]} & {e.title for e in splits[b]}
        out[f"shared_ctx_{a}_{b}"] = len(shared)
        out[f"shared_titles_{a}_{b}"] = len(shared_titles)

    if "validation" in splits:
        subset = reproducible_subset(splits["validation"], subset_n, seed=subset_seed)
        tags = tag_examples(subset)
        out["subset"] = {
            "n": len(subset),
            "seed": subset_seed,
            "impossible": sum(e.is_impossible for e in subset),
            "impossible_pct": round(100 * sum(e.is_impossible for e in subset) / len(subset), 2) if subset else 0.0,
            "contexts": len({e.context for e in subset}),
            "articles": len({e.title for e in subset}),
            "length_buckets": dict(Counter(t["length_bucket"] for t in tags.values())),
            "qtype_answerable": dict(Counter(
                tags[e.qid]["question_type"] for e in subset if not e.is_impossible
            )),
        }
        out["wh_validation"] = Counter(_wh_word(e.question) for e in splits["validation"]).most_common()

    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = _Path(args.data_dir)
    splits = {sp: load_squad_file(data_dir / f"viquad2_{sp}.json") for sp in args.splits}

    payload = compute_data_stats(splits)
    payload["commit"] = _git_commit()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results_dir = _Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "data_stats.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=list) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=list))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
