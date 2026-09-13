"""Màn hình Huấn luyện — đường cong loss và val EM/F1 qua từng epoch."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.catalog import MODEL_BY_ID
from demo.results import load_training_curves
from demo.vi import integer, number

#: Màu và kiểu nét của từng model, dùng chung cho cả hai biểu đồ và chú giải.
_STYLE = {
    "mbert": {"color": "var(--color-accent)", "soft": "var(--color-accent-400)",
              "dashed": False},
    "visobert": {"color": "var(--color-accent-2-600)", "soft": "var(--color-accent-2-400)",
                 "dashed": True},
}


def _series(curves: dict, field: str, soft: bool = False) -> list[dict]:
    return [
        {"values": [point[field] for point in curve["curve"]],
         "color": _STYLE[model_id]["soft" if soft else "color"],
         "dashed": _STYLE[model_id]["dashed"]}
        for model_id, curve in curves.items() if model_id in _STYLE
    ]


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(1000)
    curves = load_training_curves()
    if not curves:
        st.warning("Chưa có `results/training_curve_*.json`. Chạy "
                   "`python scripts/finetune.py` trước.")
        return

    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">Đường cong huấn luyện</h1>'
        '<p style="margin:0;color:var(--color-neutral-800)">Val EM/F1 tính trên mẫu '
        'ngẫu nhiên 300 câu (seed = 42), cùng hàm với script đánh giá nên so sánh được '
        'với bảng kết quả.</p></div>'
    )

    epochs = max(len(curve["curve"]) for curve in curves.values())
    losses = [point["train_loss"] for curve in curves.values() for point in curve["curve"]]
    loss_max = html.nice_ceiling(max(losses))

    st.html(html.chart_card("Train loss", html.line_chart(
        _series(curves, "train_loss"), loss_max, epochs)))
    st.html(html.chart_card("Val EM và F1", html.line_chart(
        _series(curves, "val_em") + _series(curves, "val_f1", soft=True),
        100.0, epochs, y_label="100")))

    st.html(html.chart_legend([
        {"color": _STYLE[model_id]["color"],
         "label": f"{MODEL_BY_ID[model_id].name.split(' +')[0]} · "
                  f"lr {curve['config']['lr']:g} · "
                  f"{integer(len(curve['curve']))} epoch"}
        for model_id, curve in curves.items() if model_id in _STYLE
    ]))

    table, notes = st.columns(2, gap="medium", vertical_alignment="top")

    with table:
        st.html(html.epoch_table([
            {"model": MODEL_BY_ID[model_id].name.split(" +")[0], "epoch": point["epoch"],
             "train_loss": point["train_loss"], "val_em": point["val_em"],
             "val_f1": point["val_f1"]}
            for model_id, curve in curves.items() for point in curve["curve"]
        ]))

    with notes:
        _reading(curves)
        _config(curves)


def _reading(curves: dict) -> None:
    """Đọc đường cong: dấu hiệu chưa overfit, và chữ ký của sự suy sụp."""
    paragraphs = []
    mbert = curves.get("mbert")
    if mbert:
        last = mbert["curve"][-1]
        first = mbert["curve"][0]
        if last["train_loss"] < first["train_loss"] and last["val_em"] > first["val_em"]:
            paragraphs.append(
                f"mBERT: loss giảm ({number(first['train_loss'], 4)} → "
                f"{number(last['train_loss'], 4)}), val EM tăng "
                f"({number(first['val_em'])} → {number(last['val_em'])}) — chưa overfit, "
                "thậm chí còn thiếu epoch."
            )

    viso = curves.get("visobert")
    if viso:
        collapsed = [p["epoch"] for p in viso["curve"] if p["val_em"] == p["val_f1"]]
        if collapsed:
            span = ("–".join(str(e) for e in (collapsed[0], collapsed[-1]))
                    if len(collapsed) > 1 else str(collapsed[0]))
            paragraphs.append(
                f"ViSoBERT epoch {span} có val EM bằng đúng val F1 — chữ ký của suy sụp "
                "về luôn-trả-rỗng: câu impossible được 1 điểm, câu answerable được 0, "
                "nên hai metric trùng nhau."
            )

    if paragraphs:
        st.html(html.note_card("Đọc đường cong", paragraphs, tone="accent"))


def _config(curves: dict) -> None:
    """Siêu tham số, đọc thẳng từ ``config`` trong file đường cong."""
    reference = curves.get("mbert") or next(iter(curves.values()))
    config = reference["config"]
    line = (
        f"batch {config['batch_size']} × grad accum {config['grad_accum']} · "
        f"max_length {config['max_length']} · doc_stride {config['doc_stride']} · "
        f"warmup {number(config['warmup_ratio'] * 100, 0)}% · "
        f"weight decay {number(config['weight_decay'])} · seed {config['seed']}."
    )
    viso = (curves.get("visobert") or {}).get("config", {})
    if viso.get("max_answer_len"):
        line += (f" ViSoBERT thêm max_answer_len {viso['max_answer_len']} vì vocab nhỏ "
                 "chia gold thành nhiều token hơn.")

    caveat = (curves.get("visobert") or {}).get("timing_caveat", "")
    extra = (f'<p style="margin:0;font-size:12px;color:var(--color-neutral-700);'
             f'line-height:1.5">{html.esc(caveat)}</p>') if caveat else ""

    st.html(
        '<div class="om om-card" style="gap:6px;max-width:72ch">'
        '<div class="om-kicker om-kicker-n">Cấu hình</div>'
        f'<p class="om-note" style="margin:0">{html.esc(line)}</p>{extra}</div>'
    )
