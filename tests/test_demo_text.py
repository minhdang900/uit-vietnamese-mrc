"""Đếm âm tiết, cắt câu, trích đoạn — phần chuỗi của tầng hiển thị.

ĐẶC TẢ: đơn vị độ dài là âm tiết (token tách theo khoảng trắng), trùng với đơn vị
mà ``results/*.json`` dùng để chia bucket; chế độ "Gọn" hiển thị đúng câu chứa
span kèm offset của span trong câu đó.
"""
from demo.text import excerpt, length_bucket, sentence_around, syllable_count


# ── đếm âm tiết ──────────────────────────────────────────────────────
def test_syllable_count_counts_whitespace_tokens():
    assert syllable_count("Hà Nội là thủ đô") == 5


def test_syllable_count_ignores_repeated_whitespace():
    assert syllable_count("Hà   Nội\nlà") == 3


def test_syllable_count_of_empty_text_is_zero():
    assert syllable_count("   ") == 0


# ── bucket độ dài dùng chung với mrc.tagging ─────────────────────────
def test_length_bucket_is_the_one_that_produced_the_json():
    """Cùng một hàm với ``mrc.tagging`` — không phải bản sao thứ hai của quy tắc."""
    from mrc.tagging import length_bucket as source

    assert length_bucket is source


def test_length_bucket_upper_bounds_are_closed():
    assert length_bucket(200) == "100-200"
    assert length_bucket(201) == "200-300"


# ── câu chứa span ────────────────────────────────────────────────────
CTX = "Câu đầu tiên. Hà Nội là thủ đô của Việt Nam. Câu thứ ba."


def test_sentence_around_returns_only_the_containing_sentence():
    sentence, _ = sentence_around(CTX, (14, 20))
    assert sentence.startswith("Hà Nội") and "Câu đầu tiên" not in sentence


def test_sentence_around_offset_points_at_the_span_inside_the_sentence():
    sentence, offset = sentence_around(CTX, (14, 20))
    assert sentence[offset:offset + 6] == "Hà Nội"


def test_sentence_around_handles_a_span_in_the_first_sentence():
    sentence, offset = sentence_around(CTX, (0, 3))
    assert sentence[offset:offset + 3] == "Câu"


def test_sentence_around_handles_a_span_in_the_last_sentence():
    index = CTX.index("thứ ba")
    sentence, offset = sentence_around(CTX, (index, index + 6))
    assert sentence[offset:offset + 6] == "thứ ba"


def test_sentence_around_without_a_span_returns_the_whole_context():
    """Model từ chối thì không có câu nào để cắt — trả về nguyên văn, offset −1."""
    sentence, offset = sentence_around(CTX, None)
    assert sentence == CTX and offset == -1


# ── trích đoạn ───────────────────────────────────────────────────────
def test_excerpt_leaves_short_text_untouched():
    assert excerpt("ngắn", 190) == "ngắn"


def test_excerpt_respects_the_limit():
    assert len(excerpt("x " * 300, 50)) <= 51


def test_excerpt_does_not_cut_a_word_in_half():
    assert not excerpt("Hà Nội là thủ đô của Việt Nam", 12).rstrip("…").endswith("th")


def test_excerpt_marks_that_it_was_cut():
    assert excerpt("Hà Nội là thủ đô của Việt Nam", 12).endswith("…")
