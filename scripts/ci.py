"""Khoảng tin cậy + phân rã chênh lệch EM/F1 trên mẫu đánh giá n=500.

Kế thừa ``06_BaoCao_T11/05_BANG_CHUNG/ci.py`` (in ra stdout, không ``--results-dir``,
không JSON) — vi phạm "mọi số ⟶ một file JSON" (bất biến #3, xem
``src/reporting/assets.py``). Bản này ghi JSON, đọc từ ``--results-dir`` (ưu tiên
``<results-dir>/history/n500/`` một khi n=500 được dời sang đó ở P2; trong lúc đó
đọc thẳng ``<results-dir>/eval_*_validation.json``).

Thân bài toán học GIỮ NGUYÊN cách tính CI xấp xỉ chuẩn (Wald) của bản gốc, và
THÊM Wilson — chính xác hơn khi ``p`` gần 0/1 hoặc ``n`` nhỏ (N5, Phụ lục
``ban_ky_thuat_truoc_khi_viet_lai.md``: Wald ±0,78 cho EM 0,80 trên n=500 là xấp xỉ
kém gần 0; Wilson 95% cho khoảng ``[0,31%; 2,04%]`` mà kết luận không đổi). Khi
bước 2.3 chuyển pipeline sang ``mrc.stats``, module này trở thành một vỏ mỏng gọi
qua đó; cho tới lúc đó công thức nằm thẳng ở đây.

    python scripts/ci.py --results-dir results
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
from datetime import datetime, timezone

from mrc.evaluate import _git_commit

Z_95 = 1.96

#: Nhãn hiển thị + thứ tự cho các model trong bảng CI. Khoá phải khớp
#: ``eval_{khoá}_validation.json``.
MODEL_LABELS = {"baseline": "TF-IDF", "visobert": "ViSoBERT", "xlmr": "XLM-R", "mbert": "mBERT"}


def wald_half_width(pct: float, n: int) -> float:
    """Nửa khoảng tin cậy Wald 95% cho một tỉ lệ phần trăm ``pct`` trên ``n`` mẫu."""
    if n <= 0:
        return 0.0
    p = pct / 100
    return 100 * Z_95 * math.sqrt(p * (1 - p) / n)


def wilson_interval(pct: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95% — ổn định hơn Wald khi ``p`` gần biên hoặc ``n`` nhỏ.

    Công thức chuẩn (Wilson, 1927): trung tâm và biên độ đều co theo
    ``1 / (1 + z^2/n)``, nên khoảng không bao giờ vượt ra ngoài ``[0, 100]`` như
    Wald có thể làm khi ``p`` gần 0 hoặc 100.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = pct / 100
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    adjust = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, (center - adjust) / denom)
    hi = min(1.0, (center + adjust) / denom)
    return (round(100 * lo, 2), round(100 * hi, 2))


def diff_test(pa: float, pb: float, na: int, nb: int | None = None) -> dict:
    """Kiểm định z hai tỉ lệ độc lập (thận trọng hơn kiểm định cặp — xem W5)."""
    nb = nb or na
    a, b = pa / 100, pb / 100
    se = math.sqrt(a * (1 - a) / na + b * (1 - b) / nb)
    z = (a - b) / se if se > 0 else 0.0
    return {
        "diff": round(pa - pb, 2),
        "wald_half_width": round(100 * Z_95 * se, 2),
        "z": round(z, 2),
    }


def _history_dir(results_dir: _Path) -> _Path:
    history = results_dir / "history" / "n500"
    return history if history.is_dir() else results_dir


def load_eval_results(results_dir: _Path) -> dict[str, dict]:
    """``{khoá model: eval JSON đã parse}`` cho các model có mặt."""
    source = _history_dir(results_dir)
    out = {}
    for key in MODEL_LABELS:
        path = source / f"eval_{key}_validation.json"
        if path.is_file():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def compute_ci(results: dict[str, dict]) -> dict:
    """Bảng CI theo model + các phép so sánh mBERT/XLM-R/ViSoBERT/baseline."""
    per_model = {}
    for key, label in MODEL_LABELS.items():
        if key not in results:
            continue
        r = results[key]
        overall, ans, imp = r["overall"], r["answerable_only"], r["impossible_only"]
        n_total, n_ans, n_imp = overall["count"], ans["count"], imp["count"]
        row = {
            "model": label,
            "n": n_total,
            "EM": overall["EM"], "F1": overall["F1"],
            "EM_wald_half": round(wald_half_width(overall["EM"], n_total), 2),
            "EM_wilson": wilson_interval(overall["EM"], n_total),
            "ans_EM": ans["EM"], "ans_F1": ans["F1"], "ans_n": n_ans,
            "ans_EM_wald_half": round(wald_half_width(ans["EM"], n_ans), 2),
            "imp_EM": imp["EM"], "imp_n": n_imp,
            "imp_EM_wald_half": round(wald_half_width(imp["EM"], n_imp), 2),
            "n_ans_correct": round(ans["EM"] * n_ans / 100) if n_ans else 0,
            "n_imp_correct": round(imp["EM"] * n_imp / 100) if n_imp else 0,
            "ans_gap_F1_EM": round(ans["F1"] - ans["EM"], 2),
            "overall_gap_F1_EM": round(overall["F1"] - overall["EM"], 2),
            "avg_latency_ms": r.get("avg_latency_ms"),
        }
        qtype = r.get("by_question_type", {})
        single, multi = qtype.get("single-sentence"), qtype.get("multi-sentence")
        if single and multi:
            row["single_EM_wald_half"] = round(wald_half_width(single["EM"], single["count"]), 2)
            row["multi_EM_wald_half"] = round(wald_half_width(multi["EM"], multi["count"]), 2)
            row["single_multi_F1_gap"] = round(single["F1"] - multi["F1"], 2)
            row["single_multi_EM"] = diff_test(single["EM"], multi["EM"], single["count"], multi["count"])
        per_model[key] = row

    comparisons = {}
    if "mbert" in results and "xlmr" in results:
        m, x = results["mbert"], results["xlmr"]
        n_ans = m["answerable_only"]["count"]
        n_imp = m["impossible_only"]["count"]
        n_total = m["overall"]["count"]
        comparisons["mbert_vs_xlmr"] = {
            "overall_EM": diff_test(m["overall"]["EM"], x["overall"]["EM"], n_total),
            "answerable_EM": diff_test(m["answerable_only"]["EM"], x["answerable_only"]["EM"], n_ans),
            "impossible_EM": diff_test(m["impossible_only"]["EM"], x["impossible_only"]["EM"], n_imp),
            "F1_contrib_answerable": round(
                n_ans / n_total * (m["answerable_only"]["F1"] - x["answerable_only"]["F1"]), 2),
            "F1_contrib_impossible": round(
                n_imp / n_total * (m["impossible_only"]["EM"] - x["impossible_only"]["EM"]), 2),
            "EM_contrib_answerable": round(
                n_ans / n_total * (m["answerable_only"]["EM"] - x["answerable_only"]["EM"]), 2),
            "EM_contrib_impossible": round(
                n_imp / n_total * (m["impossible_only"]["EM"] - x["impossible_only"]["EM"]), 2),
        }
    if "mbert" in results and "visobert" in results:
        m, v = results["mbert"], results["visobert"]
        comparisons["mbert_vs_visobert_answerable_EM"] = diff_test(
            m["answerable_only"]["EM"], v["answerable_only"]["EM"],
            m["answerable_only"]["count"], v["answerable_only"]["count"],
        )

    return {"per_model": per_model, "comparisons": comparisons}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = _Path(args.results_dir)

    evals = load_eval_results(results_dir)
    if not evals:
        print(f"Không tìm thấy eval_*_validation.json trong {results_dir} "
              f"(hoặc {results_dir / 'history' / 'n500'}). Chạy scripts/run_eval.py trước.")
        return 1

    payload = compute_ci(evals)
    payload["source_dir"] = str(_history_dir(results_dir))
    payload["commit"] = _git_commit()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ci_n500.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
