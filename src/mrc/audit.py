"""Đo tỉ lệ nhãn null trong dữ liệu HUẤN LUYỆN (Phụ lục A, phát hiện N1).

Báo cáo v1 giải thích "yếu tố phân mảnh token" ở PHÍA SUY LUẬN ("mỗi cửa sổ thêm
vào là một cơ hội nữa để điểm không-trả-lời thắng") — nhưng theo
``transformer_qa.py``, mô hình chỉ trả rỗng khi **mọi** cửa sổ đều chọn null, nên
càng nhiều cửa sổ thì lúc suy luận càng DỄ trả lời, không phải dễ trả rỗng. Cơ chế
thật nằm ở phía HUẤN LUYỆN: mỗi cửa sổ không chứa trọn đáp án được gán nhãn
``[CLS]`` (xem ``mrc.features.prepare_train_features``), nên tokenizer phân mảnh
token nhiều hơn (nhiều cửa sổ hơn cho cùng một câu) ⇒ tỉ lệ nhãn null trong dữ
liệu huấn luyện cao hơn. Hàm ở đây đo trực tiếp tỉ lệ đó, tái dùng đúng logic gán
nhãn của pipeline thật (``make_windows`` + ``_locate_answer_tokens``) thay vì viết
lại một phép đếm riêng có thể lệch khỏi những gì model thực sự học.

Hàm THUẦN (không phụ thuộc file trên đĩa) để test chạy nhanh với tokenizer giả.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrc.data import Example
from mrc.features import _locate_answer_tokens
from mrc.windowing import make_windows

__all__ = ["null_label_rates"]


def null_label_rates(
    examples: Sequence[Example],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict:
    """Đếm nhãn ``[CLS]`` (null) trên mọi cửa sổ huấn luyện sinh ra từ ``examples``.

    Mỗi cửa sổ rơi vào đúng MỘT trong ba trường hợp, cùng quy ước với
    ``prepare_train_features``:

    - ``cls_impossible``: câu gốc ``is_impossible`` (không có đáp án để tìm).
    - ``cls_answer_outside``: câu answerable nhưng đáp án KHÔNG nằm trọn trong
      cửa sổ này (bị cắt sang cửa sổ khác) — vẫn gán nhãn ``[CLS]`` khi huấn luyện.
    - ``positive``: cửa sổ chứa trọn đáp án, gán nhãn vị trí thật.

    ``null_frac`` = ``(cls_impossible + cls_answer_outside) / features`` — đây là
    tỉ lệ nhãn ``[CLS]`` thật sự nhìn thấy trong lúc huấn luyện, khác với tỉ lệ
    câu impossible ở mức CÂU HỎI (``compute_stats``).

    Returns:
        ``{questions, features, cls_impossible, cls_answer_outside, positive,
        null_frac, q_multiwindow}``. ``null_frac`` tính theo phần trăm, làm tròn
        2 chữ số thập phân; ``0.0`` nếu không sinh được cửa sổ nào.
    """
    total = cls_impossible = cls_answer_outside = positive = 0
    q_multiwindow = 0

    for example in examples:
        windows = list(
            make_windows(
                example.question, example.context, tokenizer,
                max_length=max_length, doc_stride=doc_stride,
            )
        )
        if len(windows) > 1:
            q_multiwindow += 1

        for window in windows:
            total += 1
            if not example.answers or example.answer_start < 0:
                cls_impossible += 1
                continue

            cls_index = window.input_ids.index(tokenizer.cls_token_id)
            answer = example.answers[0]
            start_token, _ = _locate_answer_tokens(
                list(window.offset_mapping),
                example.answer_start,
                example.answer_start + len(answer),
                cls_index,
            )
            if start_token == cls_index:
                cls_answer_outside += 1
            else:
                positive += 1

    return {
        "questions": len(examples),
        "features": total,
        "cls_impossible": cls_impossible,
        "cls_answer_outside": cls_answer_outside,
        "positive": positive,
        "null_frac": round(100 * (cls_impossible + cls_answer_outside) / total, 2) if total else 0.0,
        "q_multiwindow": q_multiwindow,
    }
