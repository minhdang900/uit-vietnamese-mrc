"""Bắt số KẾT QUẢ bị gõ tay vào văn bản (luật C7 của kế hoạch v2).

Mỗi kết quả (một float trong ``results/``) nở ra thành mọi cách viết mà người ta
có thể gõ: ``50,80``, ``50.80``, ``50,8``, ``50{,}80`` (sau chuẩn hoá) và bản
làm tròn 1 chữ số lẻ ``12,3`` cho ``12,264``. Chỉ giữ cách viết có **≥ 3 chữ số
có nghĩa** và có dấu thập phân — ``0,5``, ``1,3``, ``95`` không bao giờ vào,
vì chặn chúng chỉ sinh báo động giả (năm, số trang, siêu tham số).

So khớp theo **nguyên token**: ``50,80`` không kích hoạt trên ``150,80`` hay
``50,801``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["Hit", "AllowEntry", "LiteralCollision", "MAX_ALLOWLIST",
           "significant_digits", "variants", "normalise", "find_literals", "scannable_text",
           "check_collisions", "load_allowlist", "unallowed"]

MIN_SIGNIFICANT = 3
MAX_ALLOWLIST = 10


class LiteralCollision(ValueError):
    """Một kết quả trùng giá trị với một số KHÔNG phải kết quả (thống kê dữ liệu,
    siêu tham số). Cách giải: đưa số trong văn xuôi vào macro — không nới test."""


@dataclass(frozen=True)
class Hit:
    text: str        # đúng chuỗi đã khớp (sau chuẩn hoá)
    value: float     # kết quả mà nó là một cách viết
    source: str      # kết quả đó đến từ đâu (file:đường dẫn json)
    line: int


@dataclass(frozen=True)
class AllowEntry:
    file: str
    literal: str
    justification: str


def significant_digits(s: str) -> int:
    """``"50,80"`` → 4, ``"0,80"`` → 2: bỏ dấu, dấu phân cách và số 0 đầu."""
    return len(re.sub(r"\D", "", s).lstrip("0"))


def _forms(value: float) -> set[str]:
    """Các cách viết dấu chấm của ``|value|``: 2 chữ số lẻ, bỏ số 0 cuối, 1 chữ số lẻ."""
    v = abs(value)
    two = f"{v:.2f}"
    forms = {two, two.rstrip("0"), f"{v:.1f}", f"{v:.4f}".rstrip("0")}
    if v < 1:   # p-value, tỉ lệ nhỏ: thêm 3–4 chữ số lẻ (không đệm số 0 cuối)
        forms |= {f"{v:.3f}".rstrip("0"), f"{v:.4f}".rstrip("0")}
    return {f for f in forms if "." in f and not f.endswith(".")
            and significant_digits(f) >= MIN_SIGNIFICANT}


def variants(value: float) -> set[str]:
    """Mọi chuỗi đếm là "gõ tay kết quả này".

    * ``int`` (đếm, n, epoch) → rỗng.
    * ``float`` mang giá trị nguyên (``52.0``: EM trên 300 câu) → chỉ dạng có
      phần thập phân cố định ``52,00``/``52,0`` — không bao giờ ``52`` trần.
    * còn lại → rỗng nếu bản thân giá trị có < 3 chữ số có nghĩa (``1.3``:
      ``1,30`` chỉ là đệm số 0, không thêm độ chính xác).
    """
    if isinstance(value, (bool, int)):
        return set()
    if float(value).is_integer():
        forms = {f for f in (f"{abs(value):.2f}", f"{abs(value):.1f}")
                 if significant_digits(f) >= MIN_SIGNIFICANT}
        return {v for f in forms
                for v in (f, f.replace(".", ","), f.replace(".", "{,}"))}
    shortest = f"{abs(value):.4f}".rstrip("0")
    if significant_digits(shortest) < MIN_SIGNIFICANT:
        return set()
    out: set[str] = set()
    for form in _forms(value):
        out |= {form, form.replace(".", ","), form.replace(".", "{,}")}
    return out


def normalise(text: str) -> str:
    """``{,}`` → ``,``; ``\\,`` (khoảng mảnh) → rỗng; ``~`` → khoảng trắng."""
    return text.replace("{,}", ",").replace("\\,", "").replace("~", " ")


#: Độ dài/hệ số dàn trang của LaTeX (``1.15cm``, ``\\arraystretch}{1.15}``) không
#: phải kết quả — nếu không loại, chúng va với các kết quả nhỏ như 1,15.
_TEX_LAYOUT_AFTER = re.compile(r"\s*(?:cm|mm|pt|em|ex|in|bp|sp|\\[a-z]*(?:width|height))\b")
_TEX_LAYOUT_BEFORE = re.compile(r"stretch\}?\{\s*$")


def _pattern(variant: str) -> str:
    return rf"(?<![\d.,]){re.escape(variant)}(?![\d]|[.,]\d)"


def find_literals(text: str, literals: Mapping[float, str]) -> list[Hit]:
    """Mọi lần một kết quả xuất hiện trong ``text``, theo thứ tự vị trí.

    ``literals``: giá trị → nguồn. Biến thể dài được thử trước để ``50,80`` không
    bị báo hai lần (một lần như ``50,80``, một lần như ``50,8``).
    """
    text = normalise(text)
    table: dict[str, tuple[float, str]] = {}
    for value, source in literals.items():
        for v in variants(value):
            table.setdefault(normalise(v), (value, source))
    if not table:
        return []
    alternation = "|".join(_pattern(v) for v in sorted(table, key=len, reverse=True))
    hits = []
    for m in re.finditer(alternation, text):
        if _TEX_LAYOUT_AFTER.match(text, m.end()) or _TEX_LAYOUT_BEFORE.search(
                text, 0, m.start()):
            continue
        value, source = table[m.group()]
        hits.append(Hit(m.group(), value, source, text.count("\n", 0, m.start()) + 1))
    return hits


def check_collisions(results: Mapping[float, str], known: Mapping[float, str]) -> None:
    """Gãy nếu cách viết của một kết quả trùng cách viết của một số không-phải-kết-quả."""
    known_forms: dict[str, tuple[float, str]] = {}
    for value, source in known.items():
        for v in variants(value):
            known_forms.setdefault(v, (value, source))
    clashes = []
    for value, source in results.items():
        for v in sorted(variants(value)):
            if v in known_forms:
                other, other_src = known_forms[v]
                clashes.append(f"  {v!r}: kết quả {value} ({source}) "
                               f"== không-phải-kết-quả {other} ({other_src})")
                break
    if clashes:
        raise LiteralCollision(
            "Kết quả trùng số liệu không phải kết quả — test chữ không phân biệt "
            "được hai thứ này. Đưa số trong văn xuôi vào macro (MACRO_SPEC):\n"
            + "\n".join(clashes))


def load_allowlist(path: str | Path) -> list[AllowEntry]:
    """Đọc ``file:literal  # lý do``. Dòng trống/bắt đầu bằng ``#`` bị bỏ qua."""
    path = Path(path)
    if not path.is_file():
        return []
    entries = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body, _, why = line.partition("#")
        file, sep, literal = body.strip().rpartition(":")
        if not sep or not file or not literal:
            raise ValueError(f"{path}:{n}: cần dạng 'file:literal  # justification'")
        if not why.strip():
            raise ValueError(f"{path}:{n}: thiếu justification cho {literal!r}")
        entries.append(AllowEntry(file, normalise(literal), why.strip()))
    if len(entries) > MAX_ALLOWLIST:
        raise ValueError(f"{path}: {len(entries)} mục > {MAX_ALLOWLIST} — allowlist "
                         "là ngoại lệ, không phải cách né test")
    return entries


def unallowed(hits: Mapping[str, Iterable[Hit]], entries: Iterable[AllowEntry]
              ) -> tuple[list[tuple[str, Hit]], list[AllowEntry]]:
    """Tách hit thành (vi phạm, mục allowlist cũ). Mục cũ = không còn khớp hit nào
    trong file của nó — phải xoá, để allowlist không thành chỗ chứa rác."""
    entries = list(entries)
    allowed = {(e.file, e.literal) for e in entries}
    used: set[tuple[str, str]] = set()
    offenders = []
    for file, file_hits in hits.items():
        for hit in file_hits:
            key = (file, hit.text)
            if key in allowed:
                used.add(key)
            else:
                offenders.append((file, hit))
    stale = [e for e in entries if (e.file, e.literal) not in used]
    return offenders, stale


# ── chỉ quét chữ người đọc THẤY ─────────────────────────────────────────────
# Mã dựng deck đầy số dàn trang (``fontSize: 11.5``, ``card(…, 3.05)``) trùng
# ngẫu nhiên với kết quả. Số người xem thấy nằm trong CHUỖI (JS) hoặc nội dung
# thẻ (HTML); số trần trong mã là toạ độ. Phần bị bỏ được thay bằng khoảng trắng,
# giữ nguyên xuống dòng, để số dòng trong thông báo lỗi vẫn đúng.

_REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")


def _blank(chunk: str) -> str:
    return re.sub(r"[^\n]", " ", chunk)


def _js_visible(src: str) -> str:
    out: list[str] = []
    i, n, prev = 0, len(src), ""
    while i < n:
        c = src[i]
        if src.startswith("//", i) or src.startswith("/*", i):
            if src[i + 1] == "/":
                end = src.find("\n", i)
            else:
                end = src.find("*/", i + 2)
                end = end + 2 if end >= 0 else -1
            end = n if end < 0 else end
            out.append(_blank(src[i:end]))
            i = end
        elif c in "\"'`":
            j, depth, keep = i + 1, 0, [" "]
            while j < n:
                d = src[j]
                if d == "\\":
                    keep.append(" " if depth else src[j:j + 2])
                    j += 2
                    continue
                if c == "`" and src.startswith("${", j) and depth == 0:
                    depth, j = 1, j + 2
                    keep.append("  ")
                    continue
                if depth:
                    depth += {"{": 1, "}": -1}.get(d, 0)
                    keep.append(d if d == "\n" else " ")
                elif d == c or (d == "\n" and c != "`"):
                    break
                else:
                    keep.append(d)
                j += 1
            keep.append(" " if j < n and src[j] != "\n" else "")
            out.append("".join(keep))
            i = j + 1 if j < n and src[j] != "\n" else j
            prev = c
        elif c == "/" and (prev in _REGEX_BEFORE or prev == ""):
            j, in_class = i + 1, False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            j += 1
            while j < n and src[j].isalpha():
                j += 1
            out.append(_blank(src[i:j]))
            i, prev = j, "/"
        else:
            out.append(c if c == "\n" else " ")
            if not c.isspace():
                prev = c
            i += 1
    return "".join(out)


def _html_visible(src: str) -> str:
    def script(m: re.Match) -> str:
        return _blank(m.group(1)) + _js_visible(m.group(2)) + _blank(m.group(3))

    src = re.sub(r"(<script\b[^>]*>)(.*?)(</script>)", script, src, flags=re.S | re.I)
    src = re.sub(r"<style\b.*?</style>", lambda m: _blank(m.group()), src, flags=re.S | re.I)
    return re.sub(r"<[^>]*>", lambda m: _blank(m.group()), src)


def scannable_text(name: str | Path, text: str) -> str:
    """Phần văn bản của ``name`` cần soi: chuỗi (``.js``), chữ hiển thị + chuỗi
    trong ``<script>`` (``.html``), nguyên văn với mọi loại tệp khác."""
    suffix = Path(name).suffix.lower()
    if suffix in (".js", ".mjs", ".cjs"):
        return _js_visible(text)
    if suffix in (".html", ".htm"):
        return _html_visible(text)
    return text
