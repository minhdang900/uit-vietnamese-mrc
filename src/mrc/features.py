"""Chuyển ``Example`` thành feature huấn luyện cho QA head.

Đây là bước dễ sai nhất của fine-tuning extractive QA: phải map vị trí đáp án
theo KÝ TỰ (``answer_start``) sang vị trí theo TOKEN (``start_position``,
``end_position``). Nếu map lệch, model học nhãn sai mà **không có lỗi nào được
báo** — loss vẫn giảm, chỉ là nó học sai thứ.

Quy ước cho câu impossible: cả ``start_position`` và ``end_position`` trỏ về
``[CLS]`` (index 0). Đó là cách model học nói "không có đáp án".
"""

from __future__ import annotations

from collections.abc import Sequence

from mrc.data import Example
from mrc.windowing import make_windows

__all__ = ["prepare_train_features", "prepare_eval_features"]


def prepare_train_features(
    examples: Sequence[Example],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict:
    """Tokenize + gán ``start_position``/``end_position`` theo token.

    Dùng ``make_windows`` (tự cài) thay vì ``return_overflowing_tokens``, vì
    transformers 5.17.0 giới hạn overflow ở 2 window và làm mất phần đuôi context
    một cách âm thầm — đáp án ở cuối đoạn văn sẽ không bao giờ được gán nhãn.

    Một window có thể KHÔNG chứa đáp án; nhãn của nó trỏ về ``[CLS]``, tức
    "window này không có đáp án". Đó cũng là nhãn của câu impossible.
    """
    out: dict[str, list] = {
        "input_ids": [], "attention_mask": [],
        "start_positions": [], "end_positions": [], "example_index": [],
    }

    for ex_i, ex in enumerate(examples):
        for win in make_windows(ex.question, ex.context, tokenizer,
                                max_length=max_length, doc_stride=doc_stride):
            # Pad thủ công để mọi feature cùng độ dài (Trainer cần tensor đều).
            pad_id = tokenizer.pad_token_id
            n_pad = max_length - len(win.input_ids)
            input_ids = list(win.input_ids) + [pad_id] * n_pad
            attn = list(win.attention_mask) + [0] * n_pad
            offsets = list(win.offset_mapping) + [None] * n_pad

            cls_index = input_ids.index(tokenizer.cls_token_id)

            if not ex.answers or ex.answer_start < 0:
                start_pos = end_pos = cls_index          # câu impossible
            else:
                start_char = ex.answer_start
                end_char = start_char + len(ex.answers[0])
                ctx_idx = [i for i, o in enumerate(offsets) if o is not None]
                if not ctx_idx:
                    start_pos = end_pos = cls_index
                else:
                    lo, hi = offsets[ctx_idx[0]][0], offsets[ctx_idx[-1]][1]
                    if not (lo <= start_char and hi >= end_char):
                        # đáp án không nằm trong window này
                        start_pos = end_pos = cls_index
                    else:
                        s_tok = next(
                            (i for i in ctx_idx if offsets[i][1] > start_char), cls_index
                        )
                        e_tok = next(
                            (i for i in reversed(ctx_idx) if offsets[i][0] < end_char), cls_index
                        )
                        start_pos, end_pos = s_tok, e_tok
                        if start_pos > end_pos:
                            start_pos = end_pos = cls_index

            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn)
            out["start_positions"].append(start_pos)
            out["end_positions"].append(end_pos)
            out["example_index"].append(ex_i)

    return out


def prepare_eval_features(
    examples: Sequence[Example],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict:
    """Tokenize cho đánh giá; giữ ``offset_mapping`` TUYỆT ĐỐI ở vùng context."""
    out: dict[str, list] = {
        "input_ids": [], "attention_mask": [], "offset_mapping": [], "example_index": [],
    }
    for ex_i, ex in enumerate(examples):
        for win in make_windows(ex.question, ex.context, tokenizer,
                                max_length=max_length, doc_stride=doc_stride):
            pad_id = tokenizer.pad_token_id
            n_pad = max_length - len(win.input_ids)
            out["input_ids"].append(list(win.input_ids) + [pad_id] * n_pad)
            out["attention_mask"].append(list(win.attention_mask) + [0] * n_pad)
            out["offset_mapping"].append(list(win.offset_mapping) + [None] * n_pad)
            out["example_index"].append(ex_i)
    return out
