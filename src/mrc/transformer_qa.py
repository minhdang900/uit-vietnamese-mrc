"""Extractive QA bằng transformer có QA head, chạy trên MPS.

Xử lý context dài bằng **doc-stride windowing**: context vượt ``max_length`` được
cắt thành nhiều cửa sổ chồng lấp, mỗi cửa sổ được chấm riêng, rồi lấy span tốt
nhất trên TOÀN BỘ các cửa sổ.

Hai cấu hình BẮT BUỘC, sai là hỏng âm thầm:

* ``use_fast=True`` — chỉ fast tokenizer (Rust) cung cấp ``return_offsets_mapping``,
  thứ cần để map token span về ký tự gốc. Không có nó phải dùng ``decode()``, và
  ``decode()`` làm mất dấu tiếng Việt.
* ``truncation="only_second"`` — chỉ cắt CONTEXT (đối số thứ hai), giữ nguyên
  QUESTION. Dùng ``truncation=True`` sẽ cắt cả câu hỏi khi input dài.
"""

from __future__ import annotations

from mrc.device import pick_device
from mrc.predictor import TimedPredictorMixin
from mrc.windowing import decode_span, make_windows, select_best_span

__all__ = ["TransformerQA"]

#: Checkpoint mặc định: XLM-R đã fine-tune trên SQuAD-2.0. Dùng được ngay không
#: cần huấn luyện, nên nó vừa là một model được so sánh, vừa là phép kiểm tra
#: pipeline (nếu nó cho EM ~ 0 thì lỗi nằm ở pipeline, không ở model).
DEFAULT_MODEL = "deepset/xlm-roberta-base-squad2"


class TransformerQA(TimedPredictorMixin):
    """Bọc một ``AutoModelForQuestionAnswering`` thành ``Predictor``."""

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"{model_name} không có fast tokenizer. Không lấy được "
                "offset_mapping, nên không thể map span về ký tự gốc một cách an toàn."
            )
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context: str, question: str) -> str:
        """Trả về answer span (substring của ``context``), hoặc ``""`` nếu không có."""
        if not context or not context.strip():
            return ""

        windows = make_windows(
            question, context, self.tokenizer,
            max_length=self.max_length, doc_stride=self.doc_stride,
        )
        if not windows:
            return ""

        best: tuple[float, tuple[int, int]] | None = None
        for win in windows:
            ids = self._torch.tensor([win.input_ids], device=self.device)
            mask = self._torch.tensor([win.attention_mask], device=self.device)
            with self._torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask)

            start_logits = out.start_logits[0].float().cpu().tolist()
            end_logits = out.end_logits[0].float().cpu().tolist()

            span = select_best_span(
                start_logits, end_logits, win.offset_mapping,
                max_answer_len=self.max_answer_len,
                null_threshold=self.null_threshold,
            )
            if span is None:
                continue

            # Điểm của span, để so sánh GIỮA các window.
            s_tok = next(i for i, o in enumerate(win.offset_mapping) if o and o[0] == span[0])
            e_tok = max(i for i, o in enumerate(win.offset_mapping) if o and o[1] == span[1])
            score = start_logits[s_tok] + end_logits[e_tok]
            if best is None or score > best[0]:
                best = (score, span)

        if best is None:
            return ""      # mọi window đều nói "không có đáp án"
        return decode_span(context, *best[1])
