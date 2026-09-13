"""Định dạng số theo quy ước tiếng Việt.

ĐẶC TẢ: dấu PHẨY thập phân, dấu CHẤM hàng nghìn, ngưỡng và biên độ luôn có dấu.

Pin bằng test vì đây là quy ước ngược hoàn toàn với mặc định của Python: một chỗ
quên gọi là bảng có "50.80" nằm cạnh "28.454" và người đọc không biết cái nào là
phần thập phân.
"""
import pytest

from demo import vi


# ── thập phân dùng dấu phẩy ──────────────────────────────────────────
def test_number_uses_a_comma_decimal_separator():
    assert vi.number(50.8) == "50,80"


def test_number_keeps_the_requested_precision():
    assert vi.number(2.1316, 4) == "2,1316"


def test_number_does_not_round_data_away():
    """59,4886 hiện 59,49 — làm tròn HIỂN THỊ, giá trị gốc không bị đụng tới."""
    assert vi.number(59.4886) == "59,49"


def test_number_shows_a_dash_for_missing_data():
    assert vi.number(None) == "—"


# ── hàng nghìn dùng dấu chấm ─────────────────────────────────────────
def test_integer_uses_a_period_thousands_separator():
    assert vi.integer(28454) == "28.454"


def test_integer_leaves_small_numbers_alone():
    assert vi.integer(500) == "500"


def test_integer_handles_more_than_one_group():
    assert vi.integer(1234567) == "1.234.567"


# ── ngưỡng và biên độ luôn mang dấu ──────────────────────────────────
def test_signed_marks_zero_as_positive():
    """Ngưỡng 0 phải hiện "+0,0": nó là một điểm trên trục, không phải "chưa đặt"."""
    assert vi.signed(0.0) == "+0,0"


def test_signed_uses_a_real_minus_sign_not_a_hyphen():
    """U+2212 rộng bằng dấu cộng trong font tabular, nên cột số không bị lệch."""
    assert vi.signed(-1.4) == "−1,4"
    assert "-" not in vi.signed(-1.4)


def test_signed_marks_positive_values():
    assert vi.signed(2.5) == "+2,5"


# ── đơn vị ───────────────────────────────────────────────────────────
def test_percent_does_not_multiply_by_a_hundred():
    """Nhận số ĐÃ ở thang phần trăm; nhân thêm 100 là sai 100 lần."""
    assert vi.percent(32.4) == "32,4%"


def test_millis_formats_a_latency():
    assert vi.millis(12.264) == "12,3 ms"


# ── nối đáp án vàng ──────────────────────────────────────────────────
def test_join_separates_multiple_gold_answers():
    assert vi.join(["a", "b"]) == "a  ·  b"


def test_join_says_what_an_empty_gold_means():
    """Gold rỗng = câu impossible, không phải "thiếu dữ liệu"."""
    assert vi.join([], empty="rỗng (impossible)") == "rỗng (impossible)"


@pytest.mark.parametrize("value", [0, 0.0, -0.0])
def test_number_handles_every_flavour_of_zero(value):
    assert vi.number(value) == "0,00"


def test_number_never_renders_a_negative_zero():
    """Một phép tính ra đúng 0 có thể cho -0.0; "−0,00" trong bảng EM trông như lỗi."""
    assert vi.number(-0.0) == "0,00"


def test_number_uses_a_real_minus_for_negative_values():
    assert vi.number(-1.5) == vi.MINUS + "1,50"
    assert "-" not in vi.number(-1.5)


def test_integer_uses_a_real_minus_for_negative_values():
    assert "-" not in vi.integer(-1234)
