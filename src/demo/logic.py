"""Logic của demo web, tách hoàn toàn khỏi Streamlit.

Tách ra để test được mà không chạy server. Phiên bản trước của dự án có
``SyntaxError`` trong file app khiến demo — tiêu chí chấm quan trọng nhất của môn —
không chạy được, và không có test nào phát hiện.
"""

from __future__ import annotations

import time

__all__ = ["answer", "highlight", "margin", "abstains", "meter_position", "verdict"]


#: Khoá bằng chứng mà ``answer`` luôn trả về, kể cả khi predictor không cung cấp.
#: Giao diện đọc thẳng ``result[...]`` nên thiếu khoá là ``KeyError`` lúc demo
#: đang chạy — đúng thứ không được phép xảy ra trước mặt người chấm.
_EVIDENCE_DEFAULTS: dict = {
    "null_delta": None, "start_prob": None, "end_prob": None, "top_k": [],
}


def answer(context: str, question: str, predictor) -> dict:
    """Chạy predictor, trả về đáp án + vị trí span + độ trễ đo được.

    ``predictor`` được tiêm vào nên hàm này test được bằng stub, không cần model.

    Predictor nào có ``predict_detailed`` (hiện là ``TransformerQA``) thì bằng
    chứng đi kèm — biên độ null, xác suất start/end, các span xếp sau — được lấy
    luôn, vì đó là số ĐO ĐƯỢC. Predictor chỉ có ``predict`` (baseline TF-IDF, hay
    stub trong test) vẫn chạy bình thường và các khoá đó nhận ``None``: giao diện
    ẩn khối bằng chứng đi chứ không bịa ra số để lấp chỗ trống.
    """
    if not context or not context.strip() or not question or not question.strip():
        return {"answer": "", "latency_ms": 0.0, "found": False, "span": None,
                **_EVIDENCE_DEFAULTS}

    detailed = getattr(predictor, "predict_detailed", None)

    started = time.perf_counter()
    detail = detailed(context, question) if detailed else None
    prediction = detail["answer"] if detail else predictor.predict(context, question)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    span = detail["span"] if detail else None
    if span is None and prediction:
        # Predictor không nói span nằm đâu: tìm lại trong context. Chỉ đúng khi
        # đáp án là substring — đúng bất biến của extractive MRC.
        index = context.find(prediction)
        if index >= 0:
            span = (index, index + len(prediction))

    evidence = {key: detail[key] for key in _EVIDENCE_DEFAULTS} if detail \
        else dict(_EVIDENCE_DEFAULTS)

    return {
        "answer": prediction,
        "latency_ms": round(elapsed_ms, 1),
        "found": bool(prediction),
        "span": span,
        **evidence,
    }


def highlight(context: str, span: tuple[int, int] | None) -> str:
    """Bọc span trong cú pháp markdown của Streamlit để tô sáng trong ngữ cảnh."""
    if not span:
        return context
    start, end = span
    return f"{context[:start]}**:orange[{context[start:end]}]**{context[end:]}"


# ══════════════════════════════════════════════════════════════════════
# Quyết định trả lời / từ chối
#
# Cùng một phép so sánh mà ``mrc.windowing.select_best_span`` thực hiện, viết
# lại ở đây dưới dạng hàm thuần để giao diện GIẢI THÍCH được quyết định mà không
# phải chạy lại model: thanh ngưỡng kéo tới đâu, thanh đo nhảy tới đó.
# ══════════════════════════════════════════════════════════════════════

def margin(null_delta: float | None, threshold: float) -> float | None:
    """``null_delta + threshold`` — biên độ so với ranh giới quyết định.

    Âm nghĩa là model trả lời, không âm nghĩa là từ chối. Độ lớn cho biết quyết
    định chắc chắn tới đâu, và đó là thứ thanh đo trên màn hình vẽ ra.
    """
    if null_delta is None:
        return None
    return null_delta + threshold


def abstains(null_delta: float | None, threshold: float) -> bool:
    """Model có từ chối ở ngưỡng này không.

    Ranh giới thuộc về phía TỪ CHỐI (``>= 0``), khớp với phép so sánh
    ``best_score <= null_score + null_threshold`` trong ``select_best_span``.
    Lệch dấu bằng ở đây thì thanh đo sẽ báo "trả lời" đúng lúc model im lặng.
    """
    value = margin(null_delta, threshold)
    return value is not None and value >= 0.0


def meter_position(value: float | None, span: float = 10.0,
                   low: float = 2.0, high: float = 98.0) -> float:
    """Map biên độ về vị trí phần trăm trên thanh đo.

    ``−span`` về mép trái, ``+span`` về mép phải, 0 đúng giữa. Kẹp trong
    ``[low, high]`` để con trỏ không bao giờ tràn khỏi thanh: biên độ thực tế có
    thể lớn hơn nhiều lần ``span``, và một con trỏ bị cắt mất nửa trông như lỗi
    vẽ chứ không như "rất chắc chắn".
    """
    if value is None:
        return 50.0
    scaled = (value + span) / (2 * span) * 100.0
    return max(low, min(high, scaled))


def verdict(abstain: bool, impossible: bool | None) -> tuple[str, str]:
    """``(kicker, giải thích)`` cho thẻ đáp án — năm trường hợp, không phải hai.

    Đúng/sai không nằm ở chỗ model có trả lời hay không, mà ở chỗ câu hỏi CÓ đáp
    án hay không. Gộp thành "tìm thấy / không tìm thấy" là bỏ mất đúng điều mà
    ViQuAD 2.0 đem lại, và là lý do bản demo cũ chỉ hiện được một câu cảnh báo
    chung chung cho mọi lần từ chối.

    Args:
        abstain: model có từ chối ở ngưỡng hiện tại không.
        impossible: câu hỏi có thuộc nhóm impossible không. ``None`` nghĩa là
            KHÔNG BIẾT — người dùng tự gõ câu hỏi, nên không có nhãn nào của
            ViQuAD để đối chiếu. Phải tách riêng: khẳng định "đúng, câu này
            impossible" cho một câu vừa được gõ ra là bịa, và nhãn đó thuộc về
            câu hỏi gốc của đoạn văn chứ không thuộc câu vừa gõ.
    """
    if impossible is None:
        return ("Không có đáp án trong đoạn văn" if abstain else "Đáp án trích xuất",
                "Câu hỏi tự nhập nên không có đáp án vàng để đối chiếu — "
                + ("model cho rằng đoạn văn không chứa câu trả lời."
                   if abstain else
                   "span nằm nguyên trong đoạn văn, model không sinh chữ mới."))
    if abstain and impossible:
        return ("Không có đáp án trong đoạn văn",
                "Đúng: câu này thuộc nhóm impossible của ViQuAD 2.0, gold là rỗng. "
                "Khoảng 30% câu validation như vậy.")
    if abstain:
        return ("Không có đáp án trong đoạn văn",
                "Câu này thật ra có đáp án trong đoạn văn. Hạ ngưỡng từ chối để "
                "model mạnh dạn hơn.")
    if impossible:
        return ("Đáp án trích xuất",
                "Cảnh báo: câu này thuộc nhóm impossible, gold là rỗng — model đang "
                "trả lời một span trông hợp lý. Tăng ngưỡng để nó dè dặt hơn.")
    return ("Đáp án trích xuất",
            "Span nằm nguyên trong đoạn văn; model không sinh chữ mới.")
