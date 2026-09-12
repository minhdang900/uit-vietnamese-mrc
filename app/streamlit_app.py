"""Demo web tương tác cho hệ thống đọc hiểu tiếng Việt.

Tách logic khỏi UI: hàm ``answer()`` là hàm thuần, test được mà không cần chạy
Streamlit. Phiên bản trước của dự án có ``SyntaxError`` trong app khiến demo —
tiêu chí chấm quan trọng nhất — không chạy được; ở đây có test import để chặn
tình trạng đó tái diễn.

Chạy:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time
from pathlib import Path

# ── Logic thuần, không phụ thuộc Streamlit ──────────────────────────

AVAILABLE_MODELS = {
    "XLM-R (squad2, zero-shot)": "deepset/xlm-roberta-base-squad2",
    "mBERT (fine-tuned trên ViQuAD)": "models/mbert",
    "ViSoBERT (fine-tuned trên ViQuAD)": "models/visobert",
}


def answer(context: str, question: str, predictor) -> dict:
    """Chạy một predictor và trả về đáp án kèm thông tin phụ trợ.

    Hàm thuần: nhận predictor qua tham số nên test được bằng stub.
    """
    if not context.strip() or not question.strip():
        return {"answer": "", "latency_ms": 0.0, "found": False, "span": None}

    t0 = time.perf_counter()
    out = predictor.predict(context, question)
    ms = (time.perf_counter() - t0) * 1000.0

    span = None
    if out:
        idx = context.find(out)
        if idx >= 0:
            span = (idx, idx + len(out))

    return {"answer": out, "latency_ms": round(ms, 1), "found": bool(out), "span": span}


def highlight(context: str, span: tuple[int, int] | None) -> str:
    """Bọc đáp án trong markdown highlight để hiển thị trong context."""
    if not span:
        return context
    s, e = span
    return f"{context[:s]}**:orange[{context[s:e]}]**{context[e:]}"


# ── UI ──────────────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover - UI
    import streamlit as st

    st.set_page_config(page_title="Đọc hiểu tiếng Việt — CS116 T11", page_icon="📖")
    st.title("📖 Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt")
    st.caption(
        "Extractive MRC trên UIT-ViQuAD 2.0 · CS116 đề tài T11 · "
        "Mô hình trích xuất answer span từ context, không sinh chữ mới."
    )

    installed = {
        label: path for label, path in AVAILABLE_MODELS.items()
        if not path.startswith("models/") or Path(path).exists()
    }
    if not installed:
        st.error("Chưa có model nào. Chạy `python scripts/finetune.py` trước.")
        return

    label = st.sidebar.selectbox("Mô hình", list(installed))
    null_threshold = st.sidebar.slider(
        "Ngưỡng 'không có đáp án'", -10.0, 10.0, 0.0, 0.5,
        help="Tăng ⇒ model dè dặt hơn, hay trả lời rỗng (Precision ↑). "
             "Giảm ⇒ mạnh dạn trả lời hơn (Recall ↑).",
    )

    @st.cache_resource(show_spinner="Đang tải model…")
    def load(path: str, thr: float):
        from mrc.transformer_qa import TransformerQA
        return TransformerQA(path, null_threshold=thr, name=path)

    predictor = load(installed[label], null_threshold)
    st.sidebar.success(f"device: `{predictor.device}`")

    default_ctx = (
        "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. "
        "Thành phố nằm bên bờ sông Hồng, có diện tích khoảng 3.359 km² và "
        "dân số hơn tám triệu người. Hà Nội được UNESCO công nhận là Thành phố "
        "vì hoà bình vào năm 1999."
    )
    context = st.text_area("Đoạn văn (context)", default_ctx, height=170)
    question = st.text_input("Câu hỏi", "Hà Nội được UNESCO công nhận vào năm nào?")

    if st.button("Trả lời", type="primary"):
        r = answer(context, question, predictor)
        if r["found"]:
            st.success(f"**Đáp án:** {r['answer']}")
            st.markdown("**Vị trí trong đoạn văn:**")
            st.markdown(highlight(context, r["span"]))
        else:
            st.warning(
                "Model cho rằng **đoạn văn không chứa câu trả lời**. "
                "Đây là hành vi đúng với ViQuAD 2.0 — khoảng 30% câu hỏi "
                "thuộc loại không có đáp án (unanswerable)."
            )
        st.caption(f"Thời gian suy luận: {r['latency_ms']} ms · thiết bị: {predictor.device}")


if __name__ == "__main__":  # pragma: no cover
    main()
