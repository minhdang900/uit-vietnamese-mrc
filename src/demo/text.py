"""Thao tác chuỗi cho phần hiển thị: đếm âm tiết, cắt câu, trích đoạn.

Tách khỏi tầng view để test được, và vì hai trong ba hàm ở đây mã hoá một quyết
định của dự án chứ không phải tiện ích vô thưởng vô phạt:

* :func:`syllable_count` đếm theo khoảng trắng — trùng với đơn vị mà bảng "theo
  độ dài context" trong ``results/*.json`` dùng để chia bucket. Đếm kiểu khác thì
  con số trên màn hình không khớp với con số trong file.
* :func:`sentence_around` là thứ chế độ "Gọn" hiển thị, và cũng xấp xỉ hành vi
  của baseline TF-IDF (truy hồi nguyên một câu) trên màn hình So sánh.

Bucket độ dài KHÔNG được cài lại ở đây: nó dùng thẳng ``mrc.tagging.length_bucket``,
hàm đã sinh ra nhãn nhóm trong ``results/*.json``. Cài lại là mời một phiên bản
thứ hai của cùng một quy tắc biên, rồi một ngày nào đó màn hình và file lệch nhau.
"""

from __future__ import annotations

from mrc.tagging import length_bucket  # re-export có chủ đích

__all__ = ["syllable_count", "length_bucket", "sentence_around", "excerpt"]


def syllable_count(text: str) -> int:
    """Số âm tiết = số token tách theo khoảng trắng.

    Tiếng Việt viết rời từng âm tiết, nên đây là đơn vị tự nhiên để nói "đoạn văn
    dài bao nhiêu" mà không cần tokenizer.
    """
    return len(text.split())


def sentence_around(context: str, span: tuple[int, int] | None) -> tuple[str, int]:
    """Câu chứa ``span``, kèm offset của span TRONG câu đó.

    Cắt theo ``". "`` chứ không dùng thư viện tách câu: context của ViQuAD là văn
    xuôi Wikipedia, và một phụ thuộc thêm chỉ để làm đẹp một chế độ xem thì không
    đáng. Chấp nhận cắt nhầm ở "3.359 km²" — con số vẫn nằm gọn trong câu trả về.

    Returns:
        ``(câu, offset)``. Không có span thì trả về cả context và ``-1``.
    """
    if span is None:
        return context, -1

    start, end = span
    left = context.rfind(". ", 0, start)
    left = 0 if left < 0 else left + 2
    right = context.find(". ", end)
    right = len(context) if right < 0 else right + 1
    return context[left:right], start - left


def excerpt(text: str, limit: int = 190, ellipsis: str = "…") -> str:
    """Cắt ``text`` về ``limit`` ký tự, không cắt giữa từ."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut if space < 0 else cut[:space]).rstrip(" ,.;:") + ellipsis
