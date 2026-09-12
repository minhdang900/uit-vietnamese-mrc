"""Sinh hình và bảng cho báo cáo TỪ ``results/*.json``.

Nguyên tắc: mọi hình tái tạo được từ artifact do code sinh. Thiếu results thì FAIL
TO ỒN thay vì vẽ hình rỗng — một hình trống trong báo cáo trông y hệt một hình thật
cho tới khi ai đó nhìn kỹ.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "collect_results", "order_length_buckets", "unreliable_groups",
    "format_results_table", "make_all_figures", "BUCKET_ORDER", "MIN_RELIABLE",
]

#: Thứ tự theo ĐỘ DÀI, không theo bảng chữ cái. Sắp xếp chữ cái đặt "<100" sau
#: "300+" và biểu đồ mất hết ý nghĩa.
BUCKET_ORDER = ("<100", "100-200", "200-300", "300+")
MIN_RELIABLE = 30


def collect_results(results_dir: str | Path) -> list[dict]:
    """Đọc mọi ``eval_*.json``, sắp xếp theo EM tăng dần."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(
            f"Không có eval_*.json trong {results_dir}. Chạy scripts/run_eval.py trước "
            "— hình phải sinh từ kết quả thật, không vẽ tay."
        )
    results = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return sorted(results, key=lambda r: r["overall"]["EM"])


def order_length_buckets(labels) -> list[str]:
    """Sắp bucket theo độ dài; bỏ nhãn lạ."""
    return [b for b in BUCKET_ORDER if b in set(labels)]


def unreliable_groups(groups: dict) -> list[str]:
    """Tên các nhóm có quá ít mẫu để kết luận (bỏ qua khoá ``_note``)."""
    return [
        name for name, value in groups.items()
        if not name.startswith("_") and isinstance(value, dict)
        and (value.get("unreliable") or value.get("count", 0) < MIN_RELIABLE)
    ]


def format_results_table(results: list[dict]) -> list[dict]:
    """Bảng phẳng cho báo cáo. Tách answerable/impossible vì metric tổng trộn hai
    kỹ năng: tìm đúng span, và biết khi nào nên trả lời rỗng."""
    return [
        {
            "model": r["model"],
            "EM": round(r["overall"]["EM"], 2),
            "F1": round(r["overall"]["F1"], 2),
            "answerable_EM": round(r["answerable_only"]["EM"], 2),
            "answerable_F1": round(r["answerable_only"]["F1"], 2),
            "impossible_EM": round(r["impossible_only"]["EM"], 2),
            "latency_ms": r.get("avg_latency_ms", 0.0),
            "n": r["overall"]["count"],
        }
        for r in results
    ]


# ── vẽ hình (cần matplotlib; tách khỏi phần logic trên) ───────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _bar_pair(ax, labels, left, right, left_name, right_name, fmt="%.2f"):
    x = range(len(labels))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], left, w, label=left_name, color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], right, w, label=right_name, color="#DD8452")
    ax.bar_label(b1, fmt=fmt, fontsize=8)
    ax.bar_label(b2, fmt=fmt, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)


def make_all_figures(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Vẽ toàn bộ hình. Trả về danh sách đường dẫn đã ghi."""
    plt = _plt()
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(results_dir)
    written: list[Path] = []

    # 1. So sánh model
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bar_pair(ax, [r["model"].replace(" (", "\n(") for r in results],
              [r["overall"]["EM"] for r in results],
              [r["overall"]["F1"] for r in results], "EM", "F1")
    ax.set_ylabel("Điểm (%)")
    ax.set_title(f"So sánh mô hình — UIT-ViQuAD 2.0 validation (n={results[0]['n']})")
    fig.tight_layout(); p = out_dir / "model_comparison.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); written.append(p)

    # 2. EM tách answerable vs impossible
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bar_pair(ax, [r["model"][:26] for r in results],
              [r["answerable_only"]["EM"] for r in results],
              [r["impossible_only"]["EM"] for r in results],
              "answerable (tìm đúng span)", "impossible (biết trả lời rỗng)", "%.1f")
    ax.set_ylabel("EM (%)")
    ax.set_title("EM tách theo loại câu hỏi — metric tổng trộn hai kỹ năng")
    fig.tight_layout(); p = out_dir / "em_answerable_vs_impossible.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); written.append(p)

    # 3 & 4. Breakdown
    for key, order, title, fname in (
        ("by_context_length", None, "F1 theo độ dài context (số từ)",
         "f1_by_context_length.png"),
        ("by_question_type", ["single-sentence", "multi-sentence"],
         "F1 theo phạm vi suy luận của câu hỏi", "f1_by_question_type.png"),
    ):
        groups = order or order_length_buckets(
            {g for r in results for g in r[key] if not g.startswith("_")}
        )
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        width = 0.8 / max(1, len(results))
        labels = []
        for i, r in enumerate(results):
            vals, labels = [], []
            for g in groups:
                d = r[key].get(g)
                vals.append(d["F1"] if d else 0.0)
                mark = "*" if d and d.get("unreliable") else ""
                labels.append(f"{g}\nn={d['count']}{mark}" if d else f"{g}\nn=0")
            pos = [j + i * width - 0.4 + width / 2 for j in range(len(groups))]
            ax.bar_label(ax.bar(pos, vals, width, label=r["model"][:28]),
                         fmt="%.1f", fontsize=7)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("F1 (%)"); ax.set_title(title); ax.legend(fontsize=8)
        fig.text(0.01, -0.04, "* = nhóm có n < 30, quá ít mẫu để kết luận.",
                 fontsize=7, style="italic")
        fig.tight_layout(); p = out_dir / fname
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); written.append(p)

    # 5. Đường cong huấn luyện
    curves = [f for f in sorted(results_dir.glob("training_curve_*.json"))
              if "smoke" not in f.name]
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4), squeeze=False)
        for ax1, f in zip(axes[0], curves):
            d = json.loads(f.read_text(encoding="utf-8"))
            c = d["curve"]; ep = [r["epoch"] for r in c]
            ax1.plot(ep, [r["train_loss"] for r in c], "o-", color="#C44E52",
                     label="train loss")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="#C44E52")
            ax1.set_xticks(ep)
            ax2 = ax1.twinx()
            ax2.plot(ep, [r["val_f1"] for r in c], "s-", color="#4C72B0", label="val F1")
            ax2.plot(ep, [r["val_em"] for r in c], "^--", color="#55A868", label="val EM")
            ax2.set_ylabel("Validation (%)", color="#4C72B0"); ax2.grid(False)
            best = d.get("best_epoch") or max(c, key=lambda r: r["val_f1"])["epoch"]
            ax1.set_title(f"{d.get('config', {}).get('model', f.stem)}\n"
                          f"best epoch = {best}", fontsize=9)
            lines = ax1.get_lines() + ax2.get_lines()
            ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
        fig.tight_layout(); p = out_dir / "training_curves.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); written.append(p)

    return written
