"""Extractive QA bằng transformer có QA head.

Bất biến cốt lõi: :meth:`TransformerQA.predict` luôn trả về một chuỗi CON của
``context``, hoặc chuỗi rỗng. Bài toán là *extractive* — model chỉ ra vị trí, không
sinh chữ mới.

Hai điều kiện bắt buộc, sai là hỏng âm thầm chứ không báo lỗi:

* **Fast tokenizer.** Chỉ nó có ``return_offsets_mapping``, thứ cần để map token
  span về ký tự trong context gốc. Không có nó, cách duy nhất lấy lại chuỗi là
  ``tokenizer.decode()`` — và decode **làm mất dấu tiếng Việt** (``"hoà"`` ->
  ``"hoa"``), phá cả EM lẫn bất biến substring. PhoBERT rơi đúng vào trường hợp
  này, nên lớp này TỪ CHỐI khởi tạo thay vì chạy rồi cho kết quả sai.
* **Windowing tự cài.** transformers 5.17 giới hạn ``return_overflowing_tokens`` ở
  2 window bất kể context dài bao nhiêu, cắt mất phần đuôi mà không cảnh báo.
"""

from __future__ import annotations

from mrc.device import pick_device
from mrc.predictor import TimedPredictorMixin
from mrc.windowing import decode_span, make_windows, select_best_span

__all__ = ["TransformerQA", "DEFAULT_MODEL"]

#: XLM-R đã fine-tune trên SQuAD-2.0: dùng được ngay không cần huấn luyện, nên nó
#: vừa là một model được so sánh, vừa là phép thử pipeline — nếu nó cho EM ~ 0 thì
#: lỗi nằm ở pipeline chứ không ở model.
DEFAULT_MODEL = "deepset/xlm-roberta-base-squad2"


class TransformerQA(TimedPredictorMixin):
    """Bọc ``AutoModelForQuestionAnswering`` thành một ``Predictor``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_len: int = 30,
        null_threshold: float = 0.0,
        device: str | None = None,
        name: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_len = max_answer_len
        self.null_threshold = null_threshold
        self.device = device or pick_device()
        self.name = name or model_name
        self._torch = torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{model_name}: không có fast tokenizer ⇒ không có offset_mapping ⇒ "
                "không map được token span về ký tự gốc. Cách thay thế duy nhất là "
                "tokenizer.decode(), nhưng decode làm mất dấu tiếng Việt. "
                "Không dùng được model này cho extractive QA."
            )
        self.tokenizer = tokenizer

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _score_window(self, window) -> tuple[list[float], list[float]]:
        ids = self._torch.tensor([window.input_ids], device=self.device)
        mask = self._torch.tensor([window.attention_mask], device=self.device)
        with self._torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask)
        return (out.start_logits[0].float().cpu().tolist(),
                out.end_logits[0].float().cpu().tolist())

    def predict(self, context: str, question: str) -> str:
        """Answer span (substring của ``context``), hoặc ``""`` nếu không có đáp án."""
        if not context or not context.strip():
            return ""

        windows = make_windows(question, context, self.tokenizer,
                               max_length=self.max_length, doc_stride=self.doc_stride)
        if not windows:
            return ""

        best_score, best_span = float("-inf"), None
        for window in windows:
            start_logits, end_logits = self._score_window(window)
            span = select_best_span(
                start_logits, end_logits, window.offset_mapping,
                max_answer_len=self.max_answer_len,
                null_threshold=self.null_threshold,
            )
            if span is None:
                continue  # window này nói "không có đáp án"

            # Điểm của span, để so sánh GIỮA các window.
            start_token = next(i for i, off in enumerate(window.offset_mapping)
                               if off and off[0] == span[0])
            end_token = max(i for i, off in enumerate(window.offset_mapping)
                            if off and off[1] == span[1])
            score = start_logits[start_token] + end_logits[end_token]
            if score > best_score:
                best_score, best_span = score, span

        if best_span is None:
            return ""      # mọi window đều nói "không có đáp án"
        return decode_span(context, *best_span)
