"""Interface chung cho mọi model dự đoán đáp án.

Việc chốt interface TRƯỚC khi viết model nào cho phép harness đánh giá xử lý
baseline và transformer hoàn toàn như nhau — không có nhánh ``if isinstance(...)``
nào trong code đánh giá.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Predictor", "TimedPredictorMixin"]


@runtime_checkable
class Predictor(Protocol):
    """Bất biến cốt lõi: ``predict`` trả về một chuỗi CON của ``context``.

    Bài toán là *extractive* MRC — model chỉ ra vị trí đáp án, không sinh chữ mới.
    Bất biến này được test cho mọi implementation; nó bắt được cả lỗi tokenizer
    lệch offset lẫn model "bịa" đáp án.
    """

    name: str

    def predict(self, context: str, question: str) -> str: ...


class TimedPredictorMixin:
    """Thêm ``predict_timed`` đo latency thật, cho mọi lớp có ``predict``."""

    def predict_timed(self, context: str, question: str) -> tuple[str, float]:
        """Trả về ``(đáp án, latency_ms)``. Latency được ĐO, không ước lượng."""
        t0 = time.perf_counter()
        out = self.predict(context, question)  # type: ignore[attr-defined]
        return out, (time.perf_counter() - t0) * 1000.0
