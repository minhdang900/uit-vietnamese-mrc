"""Chuẩn hoá chuỗi đáp án trước khi so khớp (EM / token-F1).

Theo quy ước SQuAD, có BA điều chỉnh cho tiếng Việt — mỗi điều được pin bằng
test trong ``tests/test_normalize.py``:

A. KHÔNG loại bỏ mạo từ. SQuAD tiếng Anh loại ``a/an/the``; tiếng Việt không có
   mạo từ, và các loại từ/lượng từ gần nghĩa (``các``, ``con``, ``những``) CÓ THỂ
   là phần của đáp án đúng.
B. Token hoá theo KHOẢNG TRẮNG (âm tiết), không dùng word segmentation. Khớp với
   eval chính thức của ViQuAD nên kết quả so sánh được với công trình khác, và
   tránh việc segmenter sai làm nhiễu metric.
C. GIỮ dấu tiếng Việt. ``"hoà"`` != ``"hoa"``; strip diacritics sẽ làm sai EM ở
   hàng nghìn câu.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize_answer", "tokenize"]


def _remove_punctuation(text: str) -> str:
    """Bỏ mọi ký tự thuộc nhóm Unicode punctuation (category ``P*``).

    Dùng ``unicodedata`` thay vì ``string.punctuation`` để bắt được cả dấu câu
    Unicode hay gặp trong văn bản Wikipedia (``–``, ``“``, ``”``, ``…``).
    Chữ cái (category ``L*``) không bị ảnh hưởng, nên dấu tiếng Việt an toàn.
    """
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def normalize_answer(s: str | None) -> str:
    """Hạ chữ thường, bỏ dấu câu, gộp khoảng trắng. Giữ nguyên dấu tiếng Việt."""
    if not s:
        return ""
    return " ".join(_remove_punctuation(s.lower()).split())


def tokenize(s: str | None) -> list[str]:
    """Chuẩn hoá rồi tách theo khoảng trắng thành danh sách âm tiết.

    Trả về ``list`` (không phải ``set``) để bag-of-tokens đếm đúng token lặp.
    """
    normalized = normalize_answer(s)
    return normalized.split() if normalized else []
