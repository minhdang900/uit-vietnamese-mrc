"""Hình dữ liệu cho báo cáo: thành phần nhãn theo split, độ dài context/đáp án.

Kế thừa ``06_BaoCao_T11/05_BANG_CHUNG/figs.py`` (đường dẫn cứng ``sys.argv[1]``,
không JSON đi kèm). Ghi PNG vào ``--results-dir/figures/`` (``results/figures/``
mặc định, khớp nơi ``reporting.figures.make_all_figures`` cũng ghi vào) và một
manifest JSON nhỏ có provenance + số liệu đã dùng để vẽ, để hình có thể kiểm tra
lại bằng số thay vì chỉ nhìn bằng mắt.

    python scripts/figs_data.py
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

from mrc.data import load_squad_file
from mrc.evaluate import _git_commit

SPLIT_NAMES = ("train", "validation", "test")


def _composition(splits: dict[str, list]) -> dict:
    return {
        sp: {
            "answerable": sum(bool(e.answers) for e in splits[sp]),
            "impossible": sum(e.is_impossible for e in splits[sp]),
            "blind": sum((not e.answers) and (not e.is_impossible) for e in splits[sp]),
        }
        for sp in SPLIT_NAMES if sp in splits
    }


def make_figures(splits: dict[str, list], out_dir: _Path) -> dict:
    """Vẽ hai hình + trả về số liệu đã dùng (để ghi manifest JSON)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                          "axes.spines.right": False, "figure.dpi": 200})
    out_dir.mkdir(parents=True, exist_ok=True)
    composition = _composition(splits)

    # Fig A: thành phần nhãn
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    names = [sp for sp in SPLIT_NAMES if sp in splits]
    ans = [composition[sp]["answerable"] for sp in names]
    imp = [composition[sp]["impossible"] for sp in names]
    blind = [composition[sp]["blind"] for sp in names]
    y = range(len(names))
    ax.barh(y, ans, color="#4C72B0", label="answerable (có gold)")
    ax.barh(y, imp, left=ans, color="#DD8452", label="impossible (gold = rỗng)")
    ax.barh(y, blind, left=[a + i for a, i in zip(ans, imp)], color="#BBBBBB",
            hatch="//", edgecolor="white", label="blind (gold bị lược bỏ)")
    for i, sp in enumerate(names):
        total = ans[i] + imp[i] + blind[i]
        ax.text(total + 300, i, f"{total:,}".replace(",", "."), va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Số câu hỏi"); ax.set_xlim(0, 33000)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Thành phần nhãn theo split — UIT-ViQuAD 2.0", fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / "dataset_composition.png"); plt.close(fig)

    # Fig B: phân phối độ dài
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    train = splits.get("train", [])
    ctx = [len(c.split()) for c in {e.context for e in train}]
    axes[0].hist(ctx, bins=60, range=(0, 600), color="#4C72B0")
    for b in (100, 200, 300):
        axes[0].axvline(b, color="#C44E52", ls="--", lw=0.9)
    axes[0].set_xlabel("Độ dài context (từ)"); axes[0].set_ylabel("Số context")
    axes[0].set_title(f"Context — train ({len(ctx):,})".replace(",", "."), fontsize=10)

    al = [len(e.answers[0].split()) for e in train if e.answers]
    p95 = sorted(al)[int(0.95 * len(al))] if al else 0
    axes[1].hist(al, bins=40, range=(0, 40), color="#55A868")
    if al:
        axes[1].axvline(p95, color="#C44E52", ls="--", lw=0.9, label="p95")
    axes[1].set_xlabel("Độ dài đáp án vàng (âm tiết)"); axes[1].set_ylabel("Số câu")
    axes[1].set_title(f"Đáp án — train ({len(al):,})".replace(",", "."), fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "length_distributions.png"); plt.close(fig)

    return {
        "composition": composition,
        "train_ctx_over_300": sum(1 for c in ctx if c > 300),
        "train_ctx_total": len(ctx),
        "train_answer_words_p95": p95,
        "train_answers_total": len(al),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--results-dir", default="results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = _Path(args.data_dir)
    splits = {sp: load_squad_file(data_dir / f"viquad2_{sp}.json") for sp in SPLIT_NAMES}

    results_dir = _Path(args.results_dir)
    figures_dir = results_dir / "figures"
    summary = make_figures(splits, figures_dir)
    summary["commit"] = _git_commit()
    summary["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "figs_data.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {figures_dir}/dataset_composition.png, {figures_dir}/length_distributions.png, {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
