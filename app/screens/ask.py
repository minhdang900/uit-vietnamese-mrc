"""Màn hình Hỏi đáp — hỏi một đoạn văn và xem model lấy đáp án từ đâu."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.catalog import EVIDENCE, MODEL_BY_ID
from demo.logic import abstains, answer, margin, meter_position, verdict
from demo.results import passages
from demo.text import sentence_around, syllable_count
from demo.vi import join, millis


@st.cache_data(show_spinner="Đang suy luận…")
def _run(model_id: str, threshold: float, context: str, question: str) -> dict | None:
    """Chạy predictor, cache theo đúng bốn thứ ảnh hưởng tới kết quả.

    Cache ở đây để đổi chế độ xem hay bấm nút không phải suy luận lại; đổi ngưỡng
    vẫn tính lại, vì ngưỡng đổi chính quyết định của model.
    """
    predictor = shell.predictor_for(MODEL_BY_ID[model_id], threshold)
    if predictor is None:
        return None
    return answer(context, question, predictor)


def _reset_question() -> None:
    """Chọn đoạn văn khác thì lấy luôn câu hỏi của đoạn đó."""
    st.session_state["question"] = None
    st.session_state["ctx_open"] = False


def _header() -> None:
    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">Hỏi một đoạn văn</h1>'
        '<p style="margin:0;color:var(--color-neutral-800);max-width:56ch">Model trích '
        'xuất đáp án từ chính đoạn văn bên dưới — không sinh chữ mới. Nếu đoạn văn '
        'không chứa câu trả lời, nó được phép từ chối.</p></div>'
    )


def _question_row(item, index: int) -> str:
    """Ô nhập + nút Trả lời + hàng chip đoạn văn. Trả về câu hỏi hiện tại."""
    with st.container(key="om_ask"):
        field, button = st.columns([5, 1.3], vertical_alignment="center")
        default = st.session_state["question"]
        typed = field.text_input(
            "Câu hỏi",
            value=item.question if default is None else default,
            placeholder="Nhập câu hỏi về đoạn văn…",
            label_visibility="collapsed",
            key=f"om_q_{index}",
        )
        if button.button("Trả lời", type="primary", use_container_width=True):
            st.session_state["ctx_open"] = False

    if typed != item.question:
        st.session_state["question"] = typed

    items = passages(st.session_state["model"])
    st.pills(
        "Đoạn văn", options=list(range(len(items))),
        format_func=lambda i: items[i].chip,
        selection_mode="single", key="passage", on_change=_reset_question,
    )
    return typed.strip()


def _evidence_row() -> str:
    head, control = st.columns([1, 3], vertical_alignment="center")
    head.html('<div class="om"><h2 style="margin:0;white-space:nowrap">Đáp án</h2></div>')
    with control:
        st.segmented_control(
            "Chế độ xem", options=[key for key, _ in EVIDENCE],
            format_func=lambda key: dict(EVIDENCE)[key],
            key="evidence", label_visibility="collapsed",
        )
    return st.session_state["evidence"] or EVIDENCE[0][0]


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(880)
    items = passages(st.session_state["model"])

    # st.pills bỏ chọn được, khi đó giá trị là None — quay về đoạn đầu tiên.
    selected = st.session_state.get("passage")
    index = 0 if selected is None else min(selected, len(items) - 1)
    item = items[index]

    _header()
    question = _question_row(item, index)
    if not question:
        st.html('<div class="om om-meta">Nhập một câu hỏi để bắt đầu.</div>')
        return

    result = _run(st.session_state["model"], st.session_state["threshold"],
                  item.context, question)
    if result is None:
        return

    threshold = st.session_state["threshold"]
    null_delta = result["null_delta"]

    # Model nào báo biên độ thì quyết định tính lại từ biên độ, nên thanh trượt
    # phản ứng tức thì; model không báo (baseline TF-IDF) thì kết quả trả về CHÍNH
    # LÀ quyết định, và thanh đo nói rõ là nó không có biên độ để vẽ.
    abstain = abstains(null_delta, threshold) if null_delta is not None \
        else not result["found"]
    span = None if abstain else result["span"]
    reading = margin(null_delta, threshold)

    evidence = _evidence_row()

    # Nhãn gold và cờ impossible thuộc về CÂU HỎI GỐC của đoạn văn. Người dùng gõ
    # câu khác thì không còn nhãn nào đối chiếu được — nói "gold: 1999" cạnh một
    # câu hỏi về diện tích là nói sai về dữ liệu.
    preset = question == item.question.strip()
    kicker, explain = verdict(abstain, item.impossible if preset else None)

    if evidence != "incontext":
        st.html(html.answer_card(
            "Model từ chối trả lời" if abstain else result["answer"],
            kicker, explain, abstain,
        ))

    if evidence == "confidence":
        st.html(html.confidence(result["start_prob"], result["end_prob"],
                                result["top_k"]))

    st.html(html.meter(reading, threshold, meter_position(reading), abstain))

    meta = f"{syllable_count(item.context)} âm tiết · {item.title}"

    if evidence == "compactview" and not st.session_state["ctx_open"]:
        sentence, offset = sentence_around(item.context, span)
        inner = None if offset < 0 else (offset, offset + len(result["answer"]))
        st.html(html.passage(sentence, inner, "Câu chứa đáp án", meta))
        if st.button("Xem đoạn văn đầy đủ", type="tertiary"):
            st.session_state["ctx_open"] = True
            st.rerun()
    else:
        heading = ("Đáp án được tô sáng trong đoạn văn" if evidence == "incontext"
                   else "Đoạn văn")
        st.html(html.passage(item.context, span, heading, meta))
        if evidence == "compactview" and st.button("Ẩn đoạn văn đầy đủ", type="tertiary"):
            st.session_state["ctx_open"] = False
            st.rerun()

    model = shell.current_model()
    device = getattr(shell.load_predictor(model.id), "device", "cpu")
    st.html(html.footer([
        f"{model.name} · {model.note}",
        f"suy luận {millis(result['latency_ms'])}",
        f"thiết bị {device}",
        f"gold: {join(list(item.gold), empty='rỗng (impossible)')}" if preset
        else "gold: không có — câu tự nhập",
    ]))
