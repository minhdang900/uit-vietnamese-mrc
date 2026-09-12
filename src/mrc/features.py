"""Chuyển ``Example`` thành feature huấn luyện cho QA head.

Nhiệm vụ chính: dịch vị trí đáp án từ KÝ TỰ (``answer_start`` của dataset) sang
TOKEN (``start_position``/``end_position`` mà model dự đoán).

Đây là chỗ sai âm thầm nguy hiểm nhất trong toàn pipeline. Nếu map lệch, model học
nhãn sai mà **loss vẫn giảm bình thường** — không exception, không cảnh báo, chỉ là
nó học nhầm thứ. Vì vậy bất biến được test không phải "chạy không lỗi" mà là *giải
mã nhãn token phải ra lại đúng đáp án vàng*.

Quy ước SQuAD-2.0: câu impossible, và window không chứa đáp án, đều gán nhãn về
``[CLS]``. Đó là cách model học nói "ở đây không có đáp án" — và với ViQuAD 2.0,
nơi 32,4% câu train là impossible, nhánh này chiếm một phần ba dữ liệu.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrc.data import Example
from mrc.windowing import make_windows

__all__ = ["prepare_train_features", "prepare_eval_features"]


def _pad(values: list, pad_value, target: int) -> list:
    return list(values) + [pad_value] * (target - len(values))


def _locate_answer_tokens(
    offsets: list, start_char: int, end_char: int, cls_index: int
) -> tuple[int, int]:
    """Tìm cặp token bao trọn khoảng ký tự ``[start_char, end_char)``.

    Trả về ``(cls_index, cls_index)`` nếu đáp án KHÔNG nằm trọn trong window —
    tức "window này không chứa đáp án", cùng quy ước với câu impossible.
    """
    context_positions = [i for i, off in enumerate(offsets) if off is not None]
    if not context_positions:
        return cls_index, cls_index

    first, last = context_positions[0], context_positions[-1]
    if not (offsets[first][0] <= start_char and offsets[last][1] >= end_char):
        return cls_index, cls_index

    # Token đầu tiên kết thúc SAU khi đáp án bắt đầu = token chứa start_char.
    start_token = next((i for i in context_positions if offsets[i][1] > start_char), None)
    # Token cuối cùng bắt đầu TRƯỚC khi đáp án kết thúc = token chứa end_char.
    end_token = next((i for i in reversed(context_positions) if offsets[i][0] < end_char), None)

    if start_token is None or end_token is None or start_token > end_token:
        return cls_index, cls_index
    return start_token, end_token


def prepare_train_features(
    examples: Sequence[Example],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict:
    """Tokenize và gán nhãn vị trí theo token, sẵn sàng cho ``QADataset``.

    Dùng :func:`mrc.windowing.make_windows` (tự cài) thay vì
    ``return_overflowing_tokens``, vì transformers 5.17 giới hạn overflow ở 2
    window bất kể context dài bao nhiêu — phần đuôi bị cắt âm thầm và đáp án nằm
    cuối đoạn văn sẽ không bao giờ được gán nhãn.
    """
    out: dict[str, list] = {
        "input_ids": [], "attention_mask": [],
        "start_positions": [], "end_positions": [], "example_index": [],
    }
    pad_id = tokenizer.pad_token_id

    for example_index, example in enumerate(examples):
        for window in make_windows(example.question, example.context, tokenizer,
                                   max_length=max_length, doc_stride=doc_stride):
            input_ids = _pad(window.input_ids, pad_id, max_length)
            attention = _pad(window.attention_mask, 0, max_length)
            offsets = _pad(window.offset_mapping, None, max_length)
            cls_index = input_ids.index(tokenizer.cls_token_id)

            if not example.answers or example.answer_start < 0:
                start_pos = end_pos = cls_index          # câu impossible
            else:
                answer = example.answers[0]
                start_pos, end_pos = _locate_answer_tokens(
                    offsets, example.answer_start,
                    example.answer_start + len(answer), cls_index,
                )

            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attention)
            out["start_positions"].append(start_pos)
            out["end_positions"].append(end_pos)
            out["example_index"].append(example_index)

    return out


def prepare_eval_features(
    examples: Sequence[Example],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict:
    """Như trên nhưng giữ ``offset_mapping`` (toạ độ TUYỆT ĐỐI) thay vì nhãn.

    Offset tuyệt đối cho phép cắt đáp án ra từ chuỗi context GỐC, bảo toàn dấu
    tiếng Việt — điều ``tokenizer.decode()`` không làm được.
    """
    out: dict[str, list] = {
        "input_ids": [], "attention_mask": [], "offset_mapping": [], "example_index": [],
    }
    pad_id = tokenizer.pad_token_id

    for example_index, example in enumerate(examples):
        for window in make_windows(example.question, example.context, tokenizer,
                                   max_length=max_length, doc_stride=doc_stride):
            out["input_ids"].append(_pad(window.input_ids, pad_id, max_length))
            out["attention_mask"].append(_pad(window.attention_mask, 0, max_length))
            out["offset_mapping"].append(_pad(window.offset_mapping, None, max_length))
            out["example_index"].append(example_index)

    return out
