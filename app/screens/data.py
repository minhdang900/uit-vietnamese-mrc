"""Màn hình Dữ liệu — ba split đo trực tiếp, và các đoạn văn dùng trong demo."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.results import SPLIT, dataset_stats, passages
from demo.vi import integer


def _use(index: int) -> None:
    """Nạp đoạn văn này vào màn hình Hỏi đáp RỒI chuyển sang đó.

    Chỉ đặt state mà không chuyển màn hình thì nút trông như bấm không ăn: người
    dùng vẫn đứng ở màn hình Dữ liệu và không thấy gì thay đổi.
    """
    st.session_state["passage"] = index
    st.session_state["question"] = None
    st.session_state["ctx_open"] = False
    shell.goto("ask")


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(1000)
    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">Dữ liệu</h1>'
        '<p style="margin:0;color:var(--color-neutral-800)">UIT-ViQuAD 2.0 định dạng '
        'SQuAD 2.0. Số liệu đo trực tiếp từ file đã tải, không chép từ tài liệu.</p></div>'
    )

    stats = dataset_stats()
    if not stats:
        st.warning("Chưa có `data/raw/`. Chạy `python scripts/fetch_data.py` trước.")
    else:
        st.html(html.split_table(stats, active=SPLIT))
        test = stats.get("test")
        if test and test["num_gradeable"] == 0:
            st.html(html.note_card(
                "Test split là blind set",
                [
                    f"Cả {integer(test['num_questions'])} câu test có "
                    "<code>answers.text</code> rỗng trong khi <code>is_impossible = "
                    "False</code> — hai điều này mâu thuẫn nếu coi là nhãn thật, nên "
                    "đây là placeholder cho leaderboard.",
                    "<code>assert_gradeable()</code> từ chối chấm split này: không có "
                    "nó, một model luôn trả chuỗi rỗng sẽ đạt EM 100% và con số đó "
                    "trông hoàn toàn hợp lý trong báo cáo.",
                ],
                tone="accent",
            ))

    st.html('<div class="om"><h3 style="margin:0">Đoạn văn trong mẫu đánh giá</h3></div>')
    items = passages(st.session_state["model"])
    for index, item in enumerate(items):
        with st.container(key=f"om_pass_{item.key}"):
            head, action = st.columns([3, 1.25], vertical_alignment="center")
            head.html(html.passage_head(item))
            action.button("Dùng đoạn này", key=f"om_use_{item.key}",
                          on_click=_use, args=(index,), type="secondary",
                          use_container_width=True)
            st.html(html.passage_excerpt(item))
