"""Chọn answer span từ logits và map token → ký tự gốc.

Đây là phần dễ sai âm thầm nhất của extractive QA. Ba lỗi kinh điển, cả ba KHÔNG
crash mà chỉ làm F1 thấp một cách khó hiểu:

1. **Span lấy từ vùng question.** Input của model là ``[CLS] question [SEP] context
   [SEP]``; nếu không lọc theo ``offset_mapping``, token có logit cao nhất có thể
   nằm trong câu hỏi.
2. **Span đảo ngược.** ``argmax(start_logits)`` và ``argmax(end_logits)`` độc lập
   nhau, nên ``end`` có thể đứng trước ``start``.
3. **Dùng ``tokenizer.decode()`` để lấy lại chuỗi.** Decode làm mất dấu tiếng Việt
   và thay đổi khoảng trắng — ``"hoà"`` có thể ra ``"hoa"``. Cách duy nhất đúng là
   cắt chuỗi gốc theo offset ký tự.

Mọi hàm ở đây là hàm THUẦN (không cần tokenizer, không cần model), nên test chạy
trong vài millisecond.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "select_best_span",
    "score_spans",
    "decode_span",
    "make_windows",
    "Window",
    "ScoredSpan",
    "SpanScores",
]

#: Offset của token không thuộc context (``[CLS]``, ``[SEP]``, token của question)
#: được biểu diễn bằng ``None``. Chỉ token có offset mới là ứng viên đáp án.
Offset = tuple[int, int] | None


@dataclass(frozen=True)
class ScoredSpan:
    """Một ứng viên đáp án đã map về ký tự trong context GỐC.

    Attributes:
        start_char: offset ký tự bắt đầu trong context gốc.
        end_char: offset ký tự kết thúc.
        score: ``start_logit + end_logit`` — thang logit, so sánh được giữa các
            span trong CÙNG một cửa sổ.
        prob: softmax của ``score`` trên TOÀN BỘ cặp (start, end) hợp lệ của cửa
            sổ, cộng thêm null. Đây là con số đưa lên UI: nó nằm trong [0, 1] và
            cộng lại bằng 1, khác với logit thô vốn không đọc được.
    """

    start_char: int
    end_char: int
    score: float
    prob: float


@dataclass(frozen=True)
class SpanScores:
    """Toàn bộ bằng chứng QA head sinh ra cho MỘT cửa sổ.

    ``select_best_span`` chỉ trả về span thắng cuộc và vứt phần còn lại đi. Demo
    cần phần còn lại — biên độ so với null và các span xếp sau là thứ giải thích
    được *vì sao* model trả lời hay từ chối, nên chúng được giữ lại ở đây thay vì
    tính lại bằng số viết tay.

    Attributes:
        candidates: các span tốt nhất, giảm dần theo ``score``, đã khử trùng lặp
            theo khoảng ký tự. Phần tử đầu là span thắng cuộc.
        null_score: ``start_logits[cls] + end_logits[cls]`` — điểm của phương án
            "không có đáp án". ``None`` nếu cửa sổ không có token ``[CLS]``.
        start_prob: softmax của ``start_logits`` trên các token THUỘC CONTEXT, lấy
            tại token bắt đầu của span thắng cuộc.
        end_prob: tương tự cho ``end_logits``.
    """

    candidates: tuple[ScoredSpan, ...]
    null_score: float | None
    start_prob: float
    end_prob: float

    @property
    def best(self) -> ScoredSpan | None:
        """Span điểm cao nhất, hoặc ``None`` nếu cửa sổ không có ứng viên nào."""
        return self.candidates[0] if self.candidates else None

    @property
    def null_delta(self) -> float | None:
        """``null_score − best.score``.

        Dấu của nó là quyết định: ``null_delta < −null_threshold`` thì cửa sổ này
        trả lời, ngược lại nó từ chối. Đây chính là "biên độ" trên thanh đo của
        demo, và là lý do giá trị này được trả về thay vì bị vứt đi.
        """
        if self.null_score is None or self.best is None:
            return None
        return self.null_score - self.best.score


def _softmax_at(logits: Sequence[float], indices: Sequence[int], pick: int) -> float:
    """Softmax của ``logits`` giới hạn trong ``indices``, đọc tại ``pick``.

    Giới hạn trong ``indices`` (các token thuộc context) chứ không trên cả chuỗi:
    token của question và token đặc biệt không phải ứng viên đáp án, để chúng vào
    mẫu số sẽ làm loãng xác suất theo độ dài câu hỏi.
    """
    if not indices:
        return 0.0
    ceiling = max(logits[i] for i in indices)
    total = sum(math.exp(logits[i] - ceiling) for i in indices)
    if total <= 0.0:
        return 0.0
    return math.exp(logits[pick] - ceiling) / total


def score_spans(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offset_mapping: Sequence[Offset],
    max_answer_len: int = 30,
    cls_index: int = 0,
    top_k: int = 3,
) -> SpanScores | None:
    """Chấm mọi cặp ``(start, end)`` hợp lệ, giữ lại top-k và điểm null.

    Hàm THUẦN: không tokenizer, không model, không ngưỡng. Quyết định trả lời hay
    từ chối thuộc về :func:`select_best_span`, hàm này chỉ cung cấp bằng chứng.

    Args:
        start_logits: logit vị trí bắt đầu, một giá trị mỗi token.
        end_logits: logit vị trí kết thúc.
        offset_mapping: ``(start_char, end_char)`` cho token thuộc context,
            ``None`` cho token khác.
        max_answer_len: giới hạn độ dài span tính theo SỐ TOKEN.
        cls_index: vị trí token ``[CLS]``, dùng để tính null score.
        top_k: số span giữ lại sau span thắng cuộc.

    Returns:
        ``SpanScores``, hoặc ``None`` nếu cửa sổ không có token context nào.

    Note:
        ``prob`` được chuẩn hoá trên TOÀN BỘ cặp hợp lệ cộng null, tính bằng
        log-sum-exp chạy dòng — không dựng mảng 147.456 phần tử cho cửa sổ 384
        token, và không tràn số khi logit lớn.
    """
    if not start_logits or not end_logits or not offset_mapping:
        return None

    candidates = [i for i, off in enumerate(offset_mapping) if off is not None]
    if not candidates:
        return None

    has_null = cls_index < len(start_logits) and cls_index < len(end_logits)
    null_score = (start_logits[cls_index] + end_logits[cls_index]) if has_null else None

    # Heap nhỏ giữ top-k, kèm số thứ tự để hoà điểm thì span sinh TRƯỚC thắng —
    # đúng quy tắc của vòng lặp gốc trong select_best_span.
    keep = max(1, top_k) * 4 + 4
    heap: list[tuple[float, int, int, int]] = []
    order = 0

    best_score = float("-inf")
    best_pair: tuple[int, int] | None = None

    # log-sum-exp chạy dòng.
    ceiling = float("-inf")
    total = 0.0

    for start_idx in candidates:
        for end_idx in candidates:
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_len:
                continue
            score = start_logits[start_idx] + end_logits[end_idx]

            if score > best_score:
                best_score, best_pair = score, (start_idx, end_idx)

            if score > ceiling:
                total *= math.exp(ceiling - score) if ceiling > float("-inf") else 0.0
                ceiling = score
            total += math.exp(score - ceiling)

            entry = (score, -order, start_idx, end_idx)
            order += 1
            if len(heap) < keep:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    if best_pair is None:
        return None

    if null_score is not None:
        if null_score > ceiling:
            total *= math.exp(ceiling - null_score) if ceiling > float("-inf") else 0.0
            ceiling = null_score
        total += math.exp(null_score - ceiling)

    log_z = ceiling + math.log(total) if total > 0.0 else ceiling

    # Khử trùng lặp theo KHOẢNG KÝ TỰ: nhiều cặp token khác nhau có thể trỏ về
    # cùng một chuỗi, và một danh sách "span xếp sau" lặp lại chính nó thì vô dụng.
    seen: set[tuple[int, int]] = set()
    ranked: list[ScoredSpan] = []
    for score, _, start_idx, end_idx in sorted(heap, reverse=True):
        start_char = offset_mapping[start_idx][0]  # type: ignore[index]
        end_char = offset_mapping[end_idx][1]  # type: ignore[index]
        if (start_char, end_char) in seen:
            continue
        seen.add((start_char, end_char))
        ranked.append(
            ScoredSpan(start_char, end_char, score, math.exp(score - log_z))
        )
        if len(ranked) >= max(1, top_k) + 1:
            break

    return SpanScores(
        candidates=tuple(ranked),
        null_score=null_score,
        start_prob=_softmax_at(start_logits, candidates, best_pair[0]),
        end_prob=_softmax_at(end_logits, candidates, best_pair[1]),
    )


def select_best_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offset_mapping: Sequence[Offset],
    max_answer_len: int = 30,
    null_threshold: float = 0.0,
    cls_index: int = 0,
) -> tuple[int, int] | None:
    """Tìm cặp ``(start_char, end_char)`` tốt nhất, hoặc ``None`` nếu "không có đáp án".

    Args:
        start_logits: logit vị trí bắt đầu, một giá trị mỗi token.
        end_logits: logit vị trí kết thúc.
        offset_mapping: ``(start_char, end_char)`` cho token thuộc context,
            ``None`` cho token không thuộc context.
        max_answer_len: giới hạn độ dài span tính theo SỐ TOKEN. Chặn trường hợp
            model trả về cả đoạn văn.
        null_threshold: ngưỡng quyết định "không có đáp án". Span được chọn chỉ khi
            ``span_score > null_score + null_threshold``. Tăng ngưỡng ⇒ model dè
            dặt hơn (Precision ↑), giảm ⇒ mạnh dạn trả lời hơn (Recall ↑).
        cls_index: vị trí token ``[CLS]``, dùng để tính null score.

    Returns:
        ``(start_char, end_char)`` để cắt context, hoặc ``None`` cho câu impossible.

    Note:
        32,4% câu trong ViQuAD 2.0 train là impossible, nên nhánh trả về ``None``
        KHÔNG phải trường hợp biên hiếm gặp — nó là một phần ba dữ liệu.
    """
    scores = score_spans(
        start_logits, end_logits, offset_mapping,
        max_answer_len=max_answer_len, cls_index=cls_index, top_k=1,
    )
    if scores is None or scores.best is None:
        return None

    # So với null score: model nói "không có đáp án" bằng cách dồn xác suất về [CLS].
    if scores.null_score is not None:
        if scores.best.score <= scores.null_score + null_threshold:
            return None

    return scores.best.start_char, scores.best.end_char


def decode_span(context: str, start_char: int, end_char: int) -> str:
    """Cắt ``context[start_char:end_char]`` sau khi kiểm tra hợp lệ.

    KHÔNG dùng ``tokenizer.decode()``: cắt trực tiếp chuỗi gốc là cách duy nhất
    bảo toàn dấu tiếng Việt và khoảng trắng nguyên bản, và là điều đảm bảo bất biến
    "đáp án luôn là substring của context".

    Raises:
        ValueError: nếu khoảng không hợp lệ. Fail to ồn thay vì trả chuỗi rỗng —
            một span sai là bug cần sửa, không phải "không tìm thấy đáp án".
    """
    if start_char < 0 or end_char < 0:
        raise ValueError(f"Offset không thể âm: ({start_char}, {end_char})")
    if end_char < start_char:
        raise ValueError(f"end_char < start_char: ({start_char}, {end_char})")
    if end_char > len(context):
        raise ValueError(
            f"end_char={end_char} vượt quá độ dài context ({len(context)})"
        )
    return context[start_char:end_char]


@dataclass(frozen=True)
class Window:
    """Một cửa sổ của context đã tokenize, sẵn sàng đưa vào model.

    Attributes:
        input_ids: token id của ``[CLS] question [SEP] context_chunk [SEP]``.
        attention_mask: mask tương ứng.
        offset_mapping: với token thuộc context, là ``(start_char, end_char)``
            **TUYỆT ĐỐI** trong context GỐC (không phải trong chunk); với token
            khác là ``None``.
        context_char_start: vị trí ký tự nơi chunk này bắt đầu trong context gốc.
        context_char_end: vị trí ký tự nơi chunk này kết thúc.
    """

    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[Offset]
    context_char_start: int
    context_char_end: int


def make_windows(
    question: str,
    context: str,
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> list[Window]:
    """Cắt context thành các cửa sổ chồng lấp, offset trả về là TUYỆT ĐỐI.

    Tự cài thay vì dùng ``return_overflowing_tokens`` của transformers, vì phiên
    bản 5.17.0 giới hạn số window ở 2 bất kể context dài bao nhiêu: context 420,
    700 và 1400 token đều chỉ sinh 2 window (đáng lẽ 5, 8, 16). Phần đuôi context
    bị cắt ÂM THẦM — không exception, không cảnh báo. Đáp án nằm ở cuối đoạn văn
    sẽ không bao giờ được tìm thấy.

    Args:
        question: câu hỏi, KHÔNG bị cắt.
        context: đoạn văn, được cắt thành cửa sổ nếu cần.
        tokenizer: fast tokenizer (cần ``return_offsets_mapping``).
        max_length: độ dài tối đa của MỘT cửa sổ, tính cả question và token đặc biệt.
        doc_stride: số token chồng lấp giữa hai cửa sổ liền kề. Chồng lấp là cần
            thiết để đáp án nằm ở ranh giới không bị chia đôi.

    Returns:
        Danh sách ``Window``. Rỗng nếu context rỗng.
    """
    if not context or not context.strip():
        return []

    # Offset của từng token context trong chuỗi GỐC.
    ctx_enc = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    ctx_offsets = ctx_enc["offset_mapping"]
    n_ctx_tokens = len(ctx_offsets)
    if n_ctx_tokens == 0:
        return []

    # Ngân sách token còn lại cho context sau khi trừ question và token đặc biệt.
    n_question = len(tokenizer(question, add_special_tokens=False)["input_ids"])
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    budget = max_length - n_question - n_special
    if budget < 1:
        raise ValueError(
            f"Câu hỏi dài {n_question} token, không còn chỗ cho context trong "
            f"max_length={max_length}. Tăng max_length."
        )

    step = max(1, budget - doc_stride)

    windows: list[Window] = []
    tok_start = 0
    while True:
        tok_end = min(tok_start + budget, n_ctx_tokens)
        char_start = ctx_offsets[tok_start][0]
        char_end = ctx_offsets[tok_end - 1][1]
        chunk = context[char_start:char_end]

        enc = tokenizer(
            question,
            chunk,
            truncation="only_second",     # chỉ cắt context, giữ nguyên question
            max_length=max_length,
            return_offsets_mapping=True,
            padding=False,
        )
        seq_ids = enc.sequence_ids(0) if hasattr(enc, "sequence_ids") else None
        offsets: list[Offset] = []
        for i, off in enumerate(enc["offset_mapping"]):
            is_context = (seq_ids[i] == 1) if seq_ids is not None else False
            # Dịch offset của chunk về hệ toạ độ của context GỐC.
            offsets.append((off[0] + char_start, off[1] + char_start) if is_context else None)

        windows.append(
            Window(
                input_ids=list(enc["input_ids"]),
                attention_mask=list(enc["attention_mask"]),
                offset_mapping=offsets,
                context_char_start=char_start,
                context_char_end=char_end,
            )
        )

        if tok_end >= n_ctx_tokens:
            break
        tok_start += step

    return windows
