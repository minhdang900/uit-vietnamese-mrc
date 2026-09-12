"""Phase 1a — chuẩn hoá chuỗi trước khi so khớp.

Ba quyết định thiết kế đặc thù tiếng Việt được PIN bằng test ở đây, để người sau
không "sửa cho giống SQuAD tiếng Anh" và làm sai metric một cách âm thầm.
"""
from mrc.normalize import normalize_answer, tokenize


# ── chuẩn hoá cơ bản ────────────────────────────────────────────────
def test_lowercases():
    assert normalize_answer("Hà Nội") == "hà nội"


def test_strips_trailing_punctuation():
    assert normalize_answer("Hà Nội.") == "hà nội"


def test_strips_surrounding_brackets():
    assert normalize_answer("(Hà Nội)") == "hà nội"


def test_strips_internal_punctuation():
    assert normalize_answer("Hà Nội, Việt Nam") == "hà nội việt nam"


def test_collapses_whitespace_and_trims():
    assert normalize_answer("  Hà   Nội \n") == "hà nội"


def test_empty_string_stays_empty():
    assert normalize_answer("") == ""


def test_none_is_treated_as_empty():
    assert normalize_answer(None) == ""


# ── QUYẾT ĐỊNH C: giữ dấu tiếng Việt ───────────────────────────────
def test_preserves_vietnamese_diacritics():
    # "hoà" != "hoa" — strip dấu sẽ làm EM sai ở hàng nghìn câu
    assert normalize_answer("Hoà Bình") == "hoà bình"


def test_does_not_ascii_fold():
    assert normalize_answer("Đà Nẵng") == "đà nẵng"
    assert "ẵ" in normalize_answer("Đà Nẵng")


# ── QUYẾT ĐỊNH A: KHÔNG loại bỏ loại từ / lượng từ ─────────────────
def test_does_not_remove_vietnamese_classifiers():
    # SQuAD tiếng Anh loại a/an/the. Tiếng Việt không có mạo từ;
    # "các", "con", "những" là loại từ và CÓ THỂ là phần của đáp án đúng.
    assert normalize_answer("các tỉnh miền Bắc") == "các tỉnh miền bắc"
    assert normalize_answer("con sông Hồng") == "con sông hồng"
    assert normalize_answer("những năm 1990") == "những năm 1990"


# ── QUYẾT ĐỊNH B: token hoá theo khoảng trắng (âm tiết) ────────────
def test_tokenize_splits_on_whitespace_into_syllables():
    # Khớp eval chính thức ViQuAD: "Hà Nội" là 2 token, không phải 1 từ "Hà_Nội"
    assert tokenize("Hà Nội") == ["hà", "nội"]


def test_tokenize_applies_normalization_first():
    assert tokenize("  Hà   Nội. ") == ["hà", "nội"]


def test_tokenize_empty_gives_empty_list():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_keeps_duplicate_tokens():
    # bag-of-tokens phải đếm được token lặp -> KHÔNG dùng set()
    assert tokenize("a a b") == ["a", "a", "b"]


def test_tokenize_keeps_digits_together():
    assert tokenize("năm 1990") == ["năm", "1990"]
