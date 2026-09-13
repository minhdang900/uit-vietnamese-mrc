"""Định dạng số theo quy ước tiếng Việt.

Toàn bộ giao diện dùng **dấu phẩy thập phân** ("50,80") và **dấu chấm hàng nghìn**
("28.454"). Python mặc định làm ngược lại, nên mọi con số đi lên màn hình phải đi
qua module này — nếu không, một bảng sẽ có "50.80" cạnh "28,454" và người đọc
không biết đâu là phần thập phân.

Các hàm ở đây CHỈ định dạng: chúng không làm tròn dữ liệu về "cho đẹp" và không
bịa số. Bất biến đầu tiên của dự án là mọi con số trên báo cáo đều đọc từ
``results/*.json``; ``decimals`` chỉ quyết định hiển thị bao nhiêu chữ số.
"""

from __future__ import annotations

__all__ = ["number", "integer", "signed", "percent", "millis", "join", "MINUS"]

#: Dấu trừ thực sự (U+2212), không phải hyphen. Trong font số dạng tabular nó có
#: cùng bề rộng với dấu cộng, nên cột số âm/dương không bị lệch một pixel.
MINUS = "−"


def number(value: float | int | None, decimals: int = 2, dash: str = "—") -> str:
    """``50.8`` → ``"50,80"``. ``None`` → ``dash``.

    Hai chi tiết nhỏ nhưng lộ ngay trên màn hình:

    * ``-0.0`` được kéo về ``0.0``. Số âm không dùng tới trong metric vẫn xuất
      hiện dưới dạng zero âm khi một phép tính ra đúng 0, và ``"-0,00"`` trong
      bảng EM trông như một lỗi.
    * Dấu âm là U+2212 chứ không phải hyphen, để cột số dạng tabular thẳng hàng.
    """
    if value is None:
        return dash
    value = float(value)
    if value == 0:
        value = 0.0
    return f"{value:.{decimals}f}".replace(".", ",").replace("-", MINUS)


def integer(value: int | float | None, dash: str = "—") -> str:
    """``28454`` → ``"28.454"``. ``None`` → ``dash``."""
    if value is None:
        return dash
    return f"{int(value):,}".replace(",", ".").replace("-", MINUS)


def signed(value: float | None, decimals: int = 1, dash: str = "—") -> str:
    """``0.0`` → ``"+0,0"``; ``-1.4`` → ``"−1,4"`` (dấu trừ thật).

    Ngưỡng từ chối và biên độ quyết định luôn hiện dấu: với hai đại lượng này
    *hướng* mới là thông tin, còn độ lớn chỉ là thứ yếu.
    """
    if value is None:
        return dash
    sign = MINUS if value < 0 else "+"
    return sign + number(abs(value), decimals)


def percent(value: float | None, decimals: int = 1, dash: str = "—") -> str:
    """``32.4`` → ``"32,4%"``. Nhận số ĐÃ ở thang phần trăm, không nhân 100."""
    if value is None:
        return dash
    return number(value, decimals) + "%"


def millis(value: float | None, decimals: int = 1, dash: str = "—") -> str:
    """``12.264`` → ``"12,3 ms"``."""
    if value is None:
        return dash
    return number(value, decimals) + " ms"


def join(values: list[str], separator: str = "  ·  ", empty: str = "") -> str:
    """Nối danh sách đáp án vàng. Danh sách rỗng → ``empty``, không phải ``""``.

    Gold rỗng mang nghĩa "câu impossible" chứ không phải "thiếu dữ liệu", nên nơi
    gọi luôn truyền một câu chữ nói rõ điều đó.
    """
    return separator.join(values) if values else empty
