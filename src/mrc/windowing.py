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

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["select_best_span", "decode_span", "make_windows", "Window"]

#: Offset của token không thuộc context (``[CLS]``, ``[SEP]``, token của question)
#: được biểu diễn bằng ``None``. Chỉ token có offset mới là ứng viên đáp án.
Offset = tuple[int, int] | None


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
    if not start_logits or not end_logits or not offset_mapping:
        return None

    # Chỉ token thuộc context mới là ứng viên (loại [CLS], [SEP], question).
    candidates = [i for i, off in enumerate(offset_mapping) if off is not None]
    if not candidates:
        return None

    best_score = float("-inf")
    best: tuple[int, int] | None = None
    for start_idx in candidates:
        for end_idx in candidates:
            if end_idx < start_idx:
                continue  # lỗi #2: span đảo ngược
            if end_idx - start_idx + 1 > max_answer_len:
                continue  # span quá dài
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best = (start_idx, end_idx)

    if best is None:
        return None

    # So với null score: model nói "không có đáp án" bằng cách dồn xác suất về [CLS].
    if cls_index < len(start_logits) and cls_index < len(end_logits):
        null_score = start_logits[cls_index] + end_logits[cls_index]
        if best_score <= null_score + null_threshold:
            return None

    start_char = offset_mapping[best[0]][0]  # type: ignore[index]
    end_char = offset_mapping[best[1]][1]  # type: ignore[index]
    return start_char, end_char


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
