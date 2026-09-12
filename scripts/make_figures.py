"""Sinh toàn bộ hình cho báo cáo TỪ results/*.json.

Không có hình nào được vẽ tay: mọi hình tái tạo được từ artifact do code sinh.
Nếu results/ thiếu, script FAIL TO ỒN thay vì vẽ hình rỗng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

BUCKET_ORDER = ["<100", "100-200", "200-300", "300+"]


def load_evals(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy eval_*.json trong {results_dir}. "
            "Chạy scripts/run_eval.py trước — hình phải sinh từ kết quả thật."
        )
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def fig_model_comparison(evals: list[dict], out: Path) -> None:
    """EM/F1 theo model, có ghi số trên cột và ghi n vào tiêu đề."""
    evals = sorted(evals, key=lambda r: r["overall"]["EM"])
    names = [r["model"] for r in evals]
    em = [r["overall"]["EM"] for r in evals]
    f1 = [r["overall"]["F1"] for r in evals]

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], em, w, label="EM", color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], f1, w, label="F1", color="#DD8452")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("Điểm (%)")
    n = evals[0]["n"]
    ax.set_title(f"So sánh mô hình — UIT-ViQuAD 2.0 validation (n={n})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _grouped_bars(evals, key, order, title, out, note=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    groups = order or sorted(
        {g for r in evals for g in r[key] if not g.startswith("_")}
    )
    width = 0.8 / max(1, len(evals))
    for i, r in enumerate(evals):
        vals, labels = [], []
        for g in groups:
            d = r[key].get(g)
            vals.append(d["F1"] if d else 0.0)
            # Ghi n NGAY TRÊN nhãn trục, và đánh dấu * nếu nhóm không đáng tin
            labels.append(f"{g}\nn={d['count']}{'*' if d and d.get('unreliable') else ''}"
                          if d else f"{g}\nn=0")
        pos = [j + i * width - 0.4 + width / 2 for j in range(len(groups))]
        bars = ax.bar(pos, vals, width, label=r["model"][:28])
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1 (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if note:
        fig.text(0.01, -0.04, note, fontsize=7, style="italic", wrap=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_by_length(evals, out):
    _grouped_bars(
        evals, "by_context_length", BUCKET_ORDER,
        "F1 theo độ dài context (số từ)", out,
        note="* = nhóm có n < 30, quá ít mẫu để kết luận.",
    )


def fig_by_qtype(evals, out):
    note = next((r["by_question_type"].get("_note") for r in evals
                 if r["by_question_type"].get("_note")), None)
    _grouped_bars(
        evals, "by_question_type", ["single-sentence", "multi-sentence"],
        "F1 theo phạm vi suy luận của câu hỏi", out,
        note=(note[:200] + "…") if note else None,
    )


def fig_training_curves(results_dir: Path, out: Path) -> bool:
    files = sorted(results_dir.glob("training_curve_*.json"))
    files = [f for f in files if "smoke" not in f.name]
    if not files:
        return False
    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 4), squeeze=False)
    for ax1, f in zip(axes[0], files):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = d["curve"]
        ep = [r["epoch"] for r in c]
        ax1.plot(ep, [r["train_loss"] for r in c], "o-", color="#C44E52", label="train loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="#C44E52")
        ax1.set_xticks(ep)
        ax2 = ax1.twinx()
        ax2.plot(ep, [r["val_f1"] for r in c], "s-", color="#4C72B0", label="val F1")
        ax2.plot(ep, [r["val_em"] for r in c], "^--", color="#55A868", label="val EM")
        ax2.set_ylabel("Validation (%)", color="#4C72B0"); ax2.grid(False)
        best = max(c, key=lambda r: r["val_f1"])
        ax1.set_title(f"{d['model']}\nbest epoch = {best['epoch']} (val F1 {best['val_f1']})",
                      fontsize=9)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_answerable_split(evals, out):
    """EM trên câu answerable vs impossible — hai kỹ năng khác nhau."""
    evals = sorted(evals, key=lambda r: r["overall"]["EM"])
    names = [r["model"][:26] for r in evals]
    ans = [r["answerable_only"]["EM"] for r in evals]
    imp = [r["impossible_only"]["EM"] for r in evals]
    x = range(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar([i - w/2 for i in x], ans, w, label="answerable (tìm đúng span)", color="#4C72B0")
    b2 = ax.bar([i + w/2 for i in x], imp, w, label="impossible (biết trả lời rỗng)", color="#937860")
    ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("EM (%)")
    ax.set_title("EM tách theo loại câu hỏi — metric tổng trộn hai kỹ năng")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evals = load_evals(results_dir)
    print(f"Đọc {len(evals)} file kết quả: {[r['model'] for r in evals]}")

    fig_model_comparison(evals, out_dir / "model_comparison.png")
    fig_by_length(evals, out_dir / "f1_by_context_length.png")
    fig_by_qtype(evals, out_dir / "f1_by_question_type.png")
    fig_answerable_split(evals, out_dir / "em_answerable_vs_impossible.png")
    made_curve = fig_training_curves(results_dir, out_dir / "training_curves.png")

    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}  ({p.stat().st_size // 1024} KB)")
    if not made_curve:
        print("  (chưa có training_curve_*.json — chạy scripts/finetune.py để có đường cong)")


if __name__ == "__main__":
    main()
