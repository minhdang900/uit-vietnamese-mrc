"""Sinh ``results/stats_{split}.json``: CI Wilson/cluster-bootstrap + McNemar (P2.2).

Đọc MỌI ``preds_{run_id}_{split}.jsonl`` có mặt trong ``--results-dir`` (không
chạy lại model — dữ liệu thô là bản ghi từng câu hỏi do ``mrc.evaluate.run_evaluation``
ghi ra, xem ``build_records``). Với mỗi run: EM/F1 tổng, Wilson CI của EM (chỉ
EM — F1 không phải tỉ lệ nhị phân), cluster-bootstrap CI của EM và F1 (cụm =
``paragraph_id``, đã có sẵn trong preds JSONL), tách HasAns/NoAns, tỉ lệ dự đoán
rỗng. Với mỗi cặp model trong ``_PAIR_CANDIDATES`` mà CẢ HAI run đều có mặt:
McNemar chính xác hai phía, hiệu EM, và CI bootstrap-cặp-theo-cụm của hiệu EM/F1.

Hợp đồng đầu ra là ``src/reporting/numbers.py`` (module docstring, viết bởi
agent ``exec-numbers``) — file này PHẢI khớp chính xác từng khoá. Đổi khoá ở
đây mà không đổi bên đó (hoặc ngược lại) là gãy ``make_numbers.py`` một cách
âm thầm (macro biến mất, không lỗi) hoặc ồn ào (``KeyError``).

    python scripts/compute_stats.py --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path as _Path

# Đặt sys.path TRƯỚC mọi import của dự án (xem scripts/run_eval.py).
_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from mrc.evaluate import _git_commit, read_jsonl
from mrc.stats import cluster_bootstrap_ci, empty_rate, mcnemar_exact, paired_cluster_bootstrap_diff, wilson_ci

__all__ = ["compute_run_stats", "compute_pair_stats", "build_stats", "discover_pred_files"]

#: Cặp so sánh xét tới (a, b) — PHẢI khớp ``src/reporting/numbers.py::_PAIRS``
#: (cùng khoá run_id). Cặp mà một trong hai run chưa có preds JSONL bị bỏ qua
#: lặng lẽ (phase chưa chạy), không phải lỗi.
_PAIR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("mbert", "xlmr"),
    ("mbert", "visobert"),
    ("xlmr", "baseline"),
    ("mbert-dev", "xlmr"),
    ("mbert-dev", "visobert-dev"),
    ("mbert-dev", "mbert"),
)

#: Cụm dùng cho mọi bootstrap — đoạn văn (đã băm sẵn trong preds JSONL bởi
#: ``mrc.evaluate.paragraph_id``), không phải câu hỏi.
CLUSTER_KEY = "paragraph_id"


def discover_pred_files(results_dir: _Path, split: str) -> dict[str, _Path]:
    """``{run_id: đường dẫn}`` cho mọi ``preds_*_{split}.jsonl`` có mặt."""
    suffix = f"_{split}.jsonl"
    out = {}
    for path in sorted(results_dir.glob(f"preds_*{suffix}")):
        stem = path.name[len("preds_"):-len(suffix)]
        if stem:
            out[stem] = path
    return out


def _subset(records: list[dict], predicate) -> dict | None:
    idx = [i for i, r in enumerate(records) if predicate(r)]
    if not idx:
        return None
    n_sub = len(idx)
    em = np.array([records[i]["em"] for i in idx], dtype=float) * 100.0
    k_sub = int(round(em.sum() / 100.0))
    out = {"n": n_sub, "EM": round(float(em.mean()), 4), "EM_wilson": list(wilson_ci(k_sub, n_sub))}
    return out


def compute_run_stats(records: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Thống kê một run: xem module docstring cho công thức từng trường."""
    n = len(records)
    if n == 0:
        raise ValueError("compute_run_stats: preds JSONL rỗng")

    em = np.array([r["em"] for r in records], dtype=float) * 100.0
    f1 = np.array([r["f1"] for r in records], dtype=float) * 100.0
    clusters = [r[CLUSTER_KEY] for r in records]
    k_em = int(round(em.sum() / 100.0))

    out = {
        "n": n,
        "EM": round(float(em.mean()), 4),
        "F1": round(float(f1.mean()), 4),
        "empty_rate": round(empty_rate({r["qid"]: r["pred"] for r in records}), 4),
        "EM_wilson": list(wilson_ci(k_em, n)),
        "EM_cluster_ci": list(cluster_bootstrap_ci(em, clusters, n_boot, seed)),
        "F1_cluster_ci": list(cluster_bootstrap_ci(f1, clusters, n_boot, seed)),
    }
    has_ans = _subset(records, lambda r: not r["is_impossible"])
    if has_ans is not None:
        idx = [i for i, r in enumerate(records) if not r["is_impossible"]]
        has_ans["F1"] = round(float(f1[idx].mean()), 4)
        out["has_ans"] = has_ans
    no_ans = _subset(records, lambda r: r["is_impossible"])
    if no_ans is not None:
        out["no_ans"] = no_ans
    return out


def compute_pair_stats(records_a: list[dict], records_b: list[dict],
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """So sánh cặp (a, b): McNemar hai phía + CI bootstrap-cặp-theo-cụm của hiệu EM/F1.

    ``a``/``b`` phải chấm CÙNG một tập câu hỏi (kiểm bằng tập qid); được ghép
    lại theo thứ tự qid của ``records_a`` trước khi gọi
    :func:`mrc.stats.paired_cluster_bootstrap_diff`, nên hai run có thể đến từ
    hai file JSONL với thứ tự dòng khác nhau mà vẫn ghép đúng cặp câu hỏi.
    """
    by_a = {r["qid"]: r for r in records_a}
    by_b = {r["qid"]: r for r in records_b}
    if set(by_a) != set(by_b):
        missing = set(by_a) ^ set(by_b)
        raise ValueError(
            f"compute_pair_stats: hai run không chấm cùng tập câu hỏi — lệch "
            f"{len(missing)} qid, ví dụ {sorted(missing)[:5]}"
        )
    order = [r["qid"] for r in records_a]

    em_a = {q: by_a[q]["em"] * 100.0 for q in order}
    em_b = {q: by_b[q]["em"] * 100.0 for q in order}
    f1_a = {q: by_a[q]["f1"] * 100.0 for q in order}
    f1_b = {q: by_b[q]["f1"] * 100.0 for q in order}
    clusters = {q: by_a[q][CLUSTER_KEY] for q in order}

    b01 = sum(1 for q in order if by_a[q]["em"] == 0.0 and by_b[q]["em"] == 1.0)
    b10 = sum(1 for q in order if by_a[q]["em"] == 1.0 and by_b[q]["em"] == 0.0)

    return {
        "b01": b01,
        "b10": b10,
        "p_mcnemar": mcnemar_exact(b01, b10, alternative="two-sided"),
        "dEM": round(float(np.mean(list(em_a.values())) - np.mean(list(em_b.values()))), 4),
        "dEM_cluster_ci": list(paired_cluster_bootstrap_diff(em_a, em_b, clusters, n_boot, seed)),
        "dF1_cluster_ci": list(paired_cluster_bootstrap_diff(f1_a, f1_b, clusters, n_boot, seed)),
    }


def build_stats(results_dir: _Path, split: str = "validation", n_boot: int = 2000,
                seed: int = 0) -> dict:
    """``stats_{split}.json`` đầy đủ — xem module docstring cho hợp đồng."""
    pred_files = discover_pred_files(results_dir, split)
    if not pred_files:
        raise FileNotFoundError(
            f"Không thấy preds_*_{split}.jsonl nào trong {results_dir} — chạy "
            f"scripts/run_eval.py (với --preds, mặc định bật) trước."
        )

    records_by_run = {run_id: read_jsonl(path) for run_id, path in pred_files.items()}
    runs = {run_id: compute_run_stats(recs, n_boot, seed)
           for run_id, recs in records_by_run.items()}

    pairs = []
    for a, b in _PAIR_CANDIDATES:
        if a in records_by_run and b in records_by_run:
            pairs.append({"a": a, "b": b,
                         **compute_pair_stats(records_by_run[a], records_by_run[b], n_boot, seed)})

    first = next(iter(records_by_run.values()))
    n_clusters = len({r[CLUSTER_KEY] for r in first})

    return {
        "split": split,
        "commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "B": n_boot,
        "seed": seed,
        "cluster": CLUSTER_KEY,
        "n_clusters": n_clusters,
        "runs": runs,
        "pairs": pairs,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--results-dir", default=str(_ROOT / "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="mặc định <results-dir>/stats_<split>.json")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = _Path(args.results_dir)
    try:
        payload = build_stats(results_dir, args.split, args.n_boot, args.seed)
    except FileNotFoundError as e:
        print(e)
        return 1

    out = _Path(args.out) if args.out else results_dir / f"stats_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  {out}  ({len(payload['runs'])} run, {len(payload['pairs'])} cặp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
