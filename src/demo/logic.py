"""Logic của demo web, tách hoàn toàn khỏi Streamlit.

Tách ra để test được mà không chạy server. Phiên bản trước của dự án có
``SyntaxError`` trong file app khiến demo — tiêu chí chấm quan trọng nhất của môn —
không chạy được, và không có test nào phát hiện.
"""

from __future__ import annotations

import time

__all__ = ["answer", "highlight"]


def answer(context: str, question: str, predictor) -> dict:
    """Chạy predictor, trả về đáp án + vị trí span + độ trễ đo được.

    ``predictor`` được tiêm vào nên hàm này test được bằng stub, không cần model.
    """
    if not context or not context.strip() or not question or not question.strip():
        return {"answer": "", "latency_ms": 0.0, "found": False, "span": None}

    started = time.perf_counter()
    prediction = predictor.predict(context, question)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    span = None
    if prediction:
        index = context.find(prediction)
        if index >= 0:
            span = (index, index + len(prediction))

    return {
        "answer": prediction,
        "latency_ms": round(elapsed_ms, 1),
        "found": bool(prediction),
        "span": span,
    }


def highlight(context: str, span: tuple[int, int] | None) -> str:
    """Bọc span trong cú pháp markdown của Streamlit để tô sáng trong ngữ cảnh."""
    if not span:
        return context
    start, end = span
    return f"{context[:start]}**:orange[{context[start:end]}]**{context[end:]}"
