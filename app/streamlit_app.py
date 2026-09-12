"""Demo web cho hệ thống đọc hiểu tiếng Việt.

Chỉ là lớp UI: mọi logic nằm trong ``demo.logic``, nơi nó được test không cần
chạy Streamlit.

    streamlit run app/streamlit_app.py
"""

from pathlib import Path

from demo.logic import answer, highlight

AVAILABLE_MODELS = {
    "XLM-R (squad2, zero-shot)": "deepset/xlm-roberta-base-squad2",
    "mBERT (fine-tuned trên ViQuAD)": "models/mbert",
    "ViSoBERT (fine-tuned trên ViQuAD)": "models/visobert",
}

#: ViSoBERT chia từ nhỏ hơn nhiều (vocab 15k) nên cần giới hạn span dài hơn.
MAX_ANSWER_LEN = {"models/visobert": 64}

DEFAULT_CONTEXT = (
    "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. "
    "Thành phố nằm bên bờ sông Hồng, có diện tích khoảng 3.359 km² và "
    "dân số hơn tám triệu người. Hà Nội được UNESCO công nhận là Thành phố "
    "vì hoà bình vào năm 1999."
)


def main() -> None:  # pragma: no cover - lớp UI
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
    def load(path: str, threshold: float):
        from mrc.transformer_qa import TransformerQA

        return TransformerQA(path, null_threshold=threshold, name=path,
                             max_answer_len=MAX_ANSWER_LEN.get(path, 30))

    predictor = load(installed[label], null_threshold)
    st.sidebar.success(f"device: `{predictor.device}`")

    context = st.text_area("Đoạn văn (context)", DEFAULT_CONTEXT, height=170)
    question = st.text_input("Câu hỏi", "Hà Nội được UNESCO công nhận vào năm nào?")

    if st.button("Trả lời", type="primary"):
        result = answer(context, question, predictor)
        if result["found"]:
            st.success(f"**Đáp án:** {result['answer']}")
            st.markdown("**Vị trí trong đoạn văn:**")
            st.markdown(highlight(context, result["span"]))
        else:
            st.warning(
                "Model cho rằng **đoạn văn không chứa câu trả lời**. Đây là hành vi "
                "đúng với ViQuAD 2.0 — khoảng 30% câu hỏi thuộc loại không có đáp án "
                "(unanswerable)."
            )
        st.caption(f"Thời gian suy luận: {result['latency_ms']} ms · "
                   f"thiết bị: {predictor.device}")


if __name__ == "__main__":  # pragma: no cover
    main()
