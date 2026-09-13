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
from mrc.windowing import decode_span, make_windows, score_spans

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
        return self.predict_detailed(context, question)["answer"]

    def predict_detailed(
        self, context: str, question: str, top_k: int = 3
    ) -> dict:
        """Như :meth:`predict`, nhưng giữ lại bằng chứng thay vì vứt đi.

        ``predict`` chỉ trả về chuỗi đáp án; mọi thứ giải thích *vì sao* model
        chọn chuỗi đó — biên độ so với null, xác suất start/end, các span xếp sau —
        đã được tính trong :func:`mrc.windowing.score_spans` rồi bị bỏ. Demo cần
        đúng những số đó, và lấy từ đây thì chúng là số ĐO ĐƯỢC chứ không phải số
        minh hoạ viết tay.

        Returns:
            dict với các khoá:

            ``answer``
                Span thắng cuộc, hoặc ``""`` nếu model từ chối.
            ``span``
                ``(start_char, end_char)`` trong ``context``, hoặc ``None``.
            ``found``
                ``bool``, đáp án có rỗng hay không.
            ``null_delta``
                ``null_score − best_score`` của cửa sổ tự tin nhất. Model trả lời
                khi và chỉ khi ``null_delta + null_threshold < 0``, nên đây là
                đại lượng mà thanh ngưỡng của demo so sánh. ``None`` nếu không có
                cửa sổ nào chấm được.
            ``start_prob`` / ``end_prob``
                Softmax của logit start/end trên các token thuộc context, đọc tại
                hai đầu của span thắng cuộc. ``None`` nếu không có bằng chứng.
            ``top_k``
                ``[{"text", "score", "prob"}]`` — các span xếp sau span thắng cuộc,
                giảm dần theo điểm.

        Note:
            Với context nhiều cửa sổ, ``null_delta`` lấy GIÁ TRỊ NHỎ NHẤT trên các
            cửa sổ, vì model trả lời nếu *có* một cửa sổ vượt ngưỡng. Nhờ vậy dấu
            của ``null_delta + null_threshold`` khớp chính xác với quyết định mà
            :func:`mrc.windowing.select_best_span` đưa ra từng cửa sổ.
        """
        empty: dict = {
            "answer": "", "span": None, "found": False, "null_delta": None,
            "start_prob": None, "end_prob": None, "top_k": [],
        }
        if not context or not context.strip():
            return empty

        windows = make_windows(question, context, self.tokenizer,
                               max_length=self.max_length, doc_stride=self.doc_stride)
        if not windows:
            return empty

        best_score, best_span, best_scores = float("-inf"), None, None
        evidence, evidence_delta = None, None

        for window in windows:
            start_logits, end_logits = self._score_window(window)
            scores = score_spans(
                start_logits, end_logits, window.offset_mapping,
                max_answer_len=self.max_answer_len, top_k=top_k,
            )
            if scores is None or scores.best is None:
                continue

            # Cửa sổ tự tin nhất, kể cả khi nó từ chối: thanh đo của demo vẫn phải
            # hiện biên độ để người xem thấy mình đang cách ranh giới bao xa.
            delta = scores.null_delta
            if delta is not None and (evidence_delta is None or delta < evidence_delta):
                evidence, evidence_delta = scores, delta

            if scores.null_score is not None:
                if scores.best.score <= scores.null_score + self.null_threshold:
                    continue  # window này nói "không có đáp án"

            if scores.best.score > best_score:
                best_score = scores.best.score
                best_span = (scores.best.start_char, scores.best.end_char)
                best_scores = scores

        shown = best_scores or evidence
        detail: dict = {
            "answer": "", "span": None, "found": False,
            "null_delta": evidence_delta,
            "start_prob": shown.start_prob if shown else None,
            "end_prob": shown.end_prob if shown else None,
            "top_k": [
                {
                    "text": decode_span(context, s.start_char, s.end_char),
                    "score": s.score,
                    "prob": s.prob,
                }
                for s in (shown.candidates[1:] if shown else ())
            ],
        }

        if best_span is None:
            return detail  # mọi window đều nói "không có đáp án"

        detail["span"] = best_span
        detail["answer"] = decode_span(context, *best_span)
        detail["found"] = bool(detail["answer"])
        return detail
