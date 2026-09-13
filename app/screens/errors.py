"""Màn hình Phân tích lỗi — duyệt dự đoán mẫu, xem model sai ở đâu."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.catalog import ERROR_FILTERS, MODEL_BY_ID
from demo.results import MissingResults, sample_predictions
from demo.vi import number

_PREDICATES = {
    "all": lambda s: True,
    "correct": lambda s: s["em"] == 1,
    "wrong": lambda s: s["em"] == 0,
    "impossible": lambda s: bool(s["is_impossible"]),
}


def diagnose(sample: dict) -> tuple[str, str, str]:
    """``(chuỗi cần tô, loại tô, chẩn đoán)`` cho một câu mẫu.

    Chọn tô **span của model** khi model có trả lời, và tô **gold mà model bỏ
    lỡ** khi nó im lặng. Đó là lý do màn hình này tồn tại: terracotta cho thấy
    model nói gì, sage cho thấy lẽ ra nó phải nói gì.
    """
    prediction = sample.get("prediction") or ""
    gold = [g for g in (sample.get("gold") or []) if g]
    impossible = bool(sample.get("is_impossible"))

    if prediction and impossible:
        return prediction, "pred", "Câu impossible — model vẫn trả lời một span trông hợp lý"
    if prediction and sample["em"] == 1:
        return prediction, "pred", "Khớp gold — biên span chính xác"
    if prediction:
        return prediction, "pred", "Trả lời sai span — so với gold bên trên"
    if gold:
        # Gold ngắn nhất là chuỗi dễ định vị nhất trong context.
        return min(gold, key=len), "gold", "Từ chối trên câu CÓ đáp án — hạ ngưỡng để lấy lại"
    return "", "pred", "Từ chối trên câu impossible — đúng"


#: Badge phán quyết. Màu lấy từ ``theme.{green,orange,gray}*`` trong
#: ``.streamlit/config.toml``, vốn đã được kéo về ramp sage/terracotta của hệ
#: thiết kế — nên ba badge này ra đúng cặp màu bản thiết kế quy định.
_BADGES = {
    "correct": ":green-badge[đúng]",
    "wrong": ":orange-badge[sai]",
    "impossible": ":gray-badge[impossible]",
}


def _item_label(sample: dict) -> str:
    """Nhãn một thẻ trong danh sách: badge + điểm ở hàng trên, câu hỏi ở dưới.

    Gộp vào NHÃN NÚT thay vì vẽ riêng phía trên nút, để cả thẻ là một vùng bấm
    được — không thì hàng badge trông như thuộc về thẻ nhưng bấm vào không ăn.
    """
    kind = ("impossible" if sample["is_impossible"]
            else "correct" if sample["em"] == 1 else "wrong")
    return (f'{_BADGES[kind]} &nbsp; EM {number(sample["em"])} · '
            f'F1 {number(sample["f1"])}  \n{sample["question"]}')


def _select(index: int) -> None:
    st.session_state["err_index"] = index


def _reset_index() -> None:
    st.session_state["err_index"] = 0


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(1080)
    model = MODEL_BY_ID[st.session_state["model"]]
    try:
        samples = sample_predictions(model.id)
    except MissingResults as error:
        st.error(str(error))
        return

    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">Phân tích lỗi</h1>'
        f'<p style="margin:0;color:var(--color-neutral-800)">Dự đoán mẫu của '
        f'{model.name} trên validation, kèm gold và điểm từng câu.</p></div>'
    )

    st.pills(
        "Lọc", options=[key for key, _ in ERROR_FILTERS],
        format_func=lambda key: dict(ERROR_FILTERS)[key],
        selection_mode="single", key="err_filter", on_change=_reset_index,
    )

    chosen = st.session_state.get("err_filter") or "all"
    filtered = [s for s in samples if _PREDICATES[chosen](s)]
    if not filtered:
        st.html('<div class="om om-meta">Không có câu nào khớp bộ lọc này.</div>')
        return

    # Danh sách co lại thì con trỏ phải kẹp về khoảng hợp lệ, nếu không khung
    # chi tiết sẽ trỏ vào một câu vừa bị lọc mất.
    index = min(st.session_state["err_index"], len(filtered) - 1)

    listing, detail = st.columns([2, 3], gap="medium", vertical_alignment="top")

    with listing, st.container(key="om_errlist"):
        for position, sample in enumerate(filtered):
            st.button(
                _item_label(sample),
                key=f"om_err_{sample['qid']}",
                on_click=_select, args=(position,),
                use_container_width=True,
                type="primary" if position == index else "secondary",
            )

    with detail:
        selected = filtered[index]
        text, kind, note = diagnose(selected)
        st.html(html.error_detail(selected, text, kind, note))
