"""Màn hình Kết quả — bảng tổng, biểu đồ thanh, và hai bảng chia nhóm."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.catalog import MODEL_BY_ID
from demo.results import MissingResults, best_by, load_all_evals
from demo.vi import integer, number

#: Thứ tự kể chuyện: kém nhất trước, tốt nhất sau. Bảng đọc như một lập luận
#: chứ không như một bảng xếp hạng.
_STORY_ORDER = ("baseline", "visobert", "xlmr", "mbert")

_TAGS = {"mbert": "fine-tuned", "visobert": "fine-tuned", "xlmr": "zero-shot"}


def _rows(evals: dict) -> list[dict]:
    out = []
    for model_id in _STORY_ORDER:
        data = evals.get(model_id)
        if data is None:
            continue
        out.append({
            "id": model_id,
            "name": MODEL_BY_ID[model_id].name,
            "tag": _TAGS.get(model_id, ""),
            "EM": data["overall"]["EM"],
            "F1": data["overall"]["F1"],
            "ans_em": data["answerable_only"]["EM"],
            "ans_f1": data["answerable_only"]["F1"],
            "imp_em": data["impossible_only"]["EM"],
            "latency": data["avg_latency_ms"],
        })
    return out


def _breakdown(data: dict, section: str) -> list[dict]:
    """Đổi ``by_context_length`` / ``by_question_type`` thành hàng để vẽ."""
    return [
        {"label": label, "EM": group["EM"], "F1": group["F1"],
         "count": group["count"], "unreliable": group.get("unreliable", False)}
        for label, group in sorted(data.get(section, {}).items())
        if isinstance(group, dict) and "EM" in group
    ]


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(1000)
    try:
        evals = load_all_evals()
    except MissingResults as error:
        st.error(str(error))
        return

    best_em_model, best_em = best_by("EM", evals)
    best_f1_model, best_f1 = best_by("F1", evals)
    reference = evals[best_em_model.id]

    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">Kết quả</h1>'
        f'<p style="margin:0;color:var(--color-neutral-800)">{reference["dataset"]}, '
        f'{reference["split"]} split, cùng một mẫu ngẫu nhiên seed = 42 cho mọi model. '
        'Mọi con số đọc từ <code>results/*.json</code>.</p></div>'
    )

    answerable = reference["answerable_only"]["count"]
    impossible = reference["impossible_only"]["count"]
    st.html(html.metric_cards([
        {"kicker": "EM tốt nhất", "value": number(best_em),
         "sub": best_em_model.name, "tone": "accent"},
        {"kicker": "F1 tốt nhất", "value": number(best_f1),
         "sub": best_f1_model.name, "tone": "accent-2"},
        {"kicker": "Câu đánh giá", "value": integer(reference["n"]),
         "sub": f"{integer(answerable)} answerable · {integer(impossible)} impossible"},
        {"kicker": "Độ trễ", "value": number(reference["avg_latency_ms"], 1),
         "unit": "ms", "sub": f"{best_em_model.name} trên {reference['device']}"},
    ]))

    rows = _rows(evals)
    st.html('<div class="om"><h3 style="margin:0">Bốn model, hai kỹ năng</h3></div>')
    st.html(html.model_table(rows, best_id=best_em_model.id))

    chart, note = st.columns(2, gap="medium", vertical_alignment="top")
    with chart:
        st.html('<div class="om"><h3 style="margin:0">EM và F1 cạnh nhau</h3></div>')
        st.html(html.bars(list(reversed(rows))))
    with note:
        xlmr, mbert, viso = evals.get("xlmr"), evals["mbert"], evals.get("visobert")
        paragraphs = []
        if xlmr:
            paragraphs.append(
                "mBERT thắng XLM-R <strong style='font-weight:700'>không</strong> phải vì "
                "tìm span giỏi hơn — trên câu answerable, XLM-R zero-shot có F1 cao hơn "
                f"({number(xlmr['answerable_only']['F1'])} so với "
                f"{number(mbert['answerable_only']['F1'])}). mBERT thắng vì biết khi nào "
                "<strong style='font-weight:700'>không</strong> nên trả lời: impossible EM "
                f"{number(mbert['impossible_only']['EM'])} so với "
                f"{number(xlmr['impossible_only']['EM'])}."
            )
        if viso:
            paragraphs.append(
                "ViSoBERT thì suy sụp về “luôn trả rỗng”: impossible EM "
                f"{number(viso['impossible_only']['EM'])} nhưng answerable EM chỉ "
                f"{number(viso['answerable_only']['EM'])} — điểm tổng "
                f"{number(viso['overall']['EM'])} xấp xỉ đúng tỉ lệ impossible của tập."
            )
        st.html(html.note_card("Điều bảng tổng che mất", paragraphs))

    st.html('<div class="om"><h3 style="margin:0">Chia nhỏ theo nhóm</h3></div>')
    selected = MODEL_BY_ID[st.session_state["model"]]
    data = evals.get(selected.id)
    if data is None:
        st.warning(f"Chưa có kết quả đánh giá cho {selected.name}.")
        return

    left, right = st.columns(2, gap="medium", vertical_alignment="top")
    with left:
        st.html(
            '<div class="om om-row"><h3 style="margin:0">Theo độ dài context</h3>'
            f'<span class="om-meta">{selected.name}</span></div>'
        )
        st.html(html.breakdown_table(_breakdown(data, "by_context_length"), "Bucket"))
    with right:
        st.html(
            '<div class="om om-row"><h3 style="margin:0">Theo loại câu hỏi</h3>'
            f'<span class="om-meta">{selected.name}</span></div>'
        )
        st.html(html.breakdown_table(
            _breakdown(data, "by_question_type"), "Loại",
            data.get("by_question_type", {}).get("_note", ""),
        ))
