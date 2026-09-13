"""Dựng HTML cho các khối mà widget Streamlit không diễn đạt được.

Ranh giới: mọi thứ ở đây là **hàm thuần** ``dữ liệu → chuỗi HTML``. Không import
``streamlit``, không đọc file, không gọi model. Nhờ vậy từng khối của giao diện
test được bằng một lời gọi hàm, thay vì phải dựng server rồi soi bằng mắt — đúng
lý do ``demo.logic`` được tách ra từ đầu.

Ngược lại, mọi thứ CÓ tương tác (ô nhập, nút, chip, slider) là widget Streamlit
thật ở tầng ``app/``, không phải HTML giả ở đây: HTML bơm vào không bắt được sự
kiện, và một cái nút bấm không ăn thì tệ hơn hẳn một cái nút trông hơi khác.

Mọi chuỗi đến từ dữ liệu đều đi qua :func:`demo.theme.esc`.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Iterable, Sequence

from demo import vi
from demo.theme import attr, esc, mark, resolve

__all__ = [
    "block", "brand", "team", "provenance", "answer_card", "confidence",
    "meter", "passage", "footer", "metric_cards", "model_table", "bars",
    "breakdown_table", "note_card", "error_detail", "split_table",
    "passage_head", "passage_excerpt", "compare_row", "team_initial",
    "team_caption", "line_chart", "epoch_table", "column_winners",
    "verdict_tag", "nice_ceiling", "chart_card", "chart_legend", "svg_image",
    "esc", "mark",
]


def block(*parts: str, gap: str = "var(--space-3)", extra: str = "") -> str:
    """Bọc các mảnh HTML trong một cột dọc mang lớp gốc ``om``."""
    inner = "".join(part for part in parts if part)
    return (f'<div class="om {extra}" style="display:flex;flex-direction:column;'
            f'gap:{gap}">{inner}</div>')


# ── sidebar ──────────────────────────────────────────────────────────

def brand(lines: Sequence[str], subtitle: str, glyph: str = "Đ",
          tagline: str = "") -> str:
    """Ô thương hiệu: chữ cái trong đĩa tròn accent + tên, mô tả, dòng môn học.

    Nhận danh sách dòng chứ không nhận một chuỗi có sẵn ``<br>``: nơi gọi truyền
    dữ liệu, còn thẻ do hàm này sinh ra — nếu không thì nơi gọi phải tự nhớ khi
    nào được escape, khi nào không.

    ``tagline`` nằm ngay dưới tiêu đề, trong cùng cột với nó. Tên ứng dụng là
    tiếng Anh còn giao diện là tiếng Việt, nên câu mô tả tiếng Việt cần đứng cạnh
    tên — nhồi nó vào dòng môn học phía dưới thì dòng đó dài quá và xuống hàng
    giữa chừng "CS116 · T11".
    """
    title = "<br>".join(esc(line) for line in lines)
    caption = (
        f'<div style="font-size:11px;color:var(--color-neutral-700);'
        f'line-height:1.2;margin-top:2px">{esc(tagline)}</div>'
        if tagline else ""
    )
    return (
        '<div class="om">'
        '<div style="display:flex;align-items:center;gap:10px">'
        '<div style="width:34px;height:34px;flex:none;border-radius:999px;'
        'background:var(--color-accent);color:var(--color-bg);display:grid;'
        'place-items:center;font-family:var(--font-heading);font-weight:800;'
        f'font-size:16px">{esc(glyph)}</div>'
        '<div style="min-width:0">'
        '<div style="font-family:var(--font-heading);font-weight:800;font-size:17px;'
        f'line-height:1.1">{title}</div>'
        f'{caption}'
        '</div>'
        '</div>'
        f'<div class="om-label" style="margin-top:var(--space-2)">{esc(subtitle)}</div>'
        '</div>'
    )


#: Nền/chữ cho ba ô chữ cái đầu, luân phiên accent → sage → trung tính.
_AVATAR_INKS = (
    ("var(--color-accent-200)", "var(--color-accent-800)"),
    ("var(--color-accent-2-200)", "var(--color-accent-2-800)"),
    ("var(--color-neutral-300)", "var(--color-neutral-800)"),
)


def team_initial(name: str) -> str:
    """Chữ cái đầu cho ô tròn avatar — lấy từ HỌ, không phải tên gọi.

    Tên gọi tiếng Việt trùng chữ đầu rất thường xuyên (Lâm, Tấn, Thi, Thu → ba
    chữ T trong cùng một nhóm bốn người), còn họ thì phân biệt được.
    """
    stripped = name.strip()
    return stripped[0].upper() if stripped else "?"


def team_caption(member: dict) -> str:
    """Dòng phụ dưới tên: MSSV, kèm phân công nếu đã chốt."""
    parts = [part for part in (member.get("mssv"), member.get("role")) if part]
    return " · ".join(parts)


def team(members: Sequence[dict], course: str) -> str:
    """Danh sách thành viên nhóm."""
    rows = []
    for index, member in enumerate(members):
        background, ink = _AVATAR_INKS[index % len(_AVATAR_INKS)]
        rows.append(
            '<div class="om-people">'
            f'<div class="om-avatar" style="background:{background};color:{ink}">'
            f'{esc(team_initial(member["name"]))}</div>'
            '<div style="min-width:0">'
            f'<div style="font-size:13px;line-height:1.25">{esc(member["name"])}</div>'
            f'<div class="om-meta">{esc(team_caption(member))}</div>'
            '</div></div>'
        )
    return (f'<div class="om"><div class="om-label" style="margin-bottom:var(--space-2)">'
            f'Nhóm thực hiện</div>{"".join(rows)}'
            f'<div class="om-meta" style="padding-top:var(--space-1)">{esc(course)}</div></div>')


def provenance(prov, device_note: str | None = None) -> str:
    """Ba dòng xuất xứ ở chân sidebar: thiết bị, split, commit."""
    device = esc(prov.device)
    suffix = f" · {esc(device_note)}" if device_note else ""
    return (
        '<div class="om om-prov" style="display:flex;flex-direction:column;gap:3px;'
        'font-size:11.5px;line-height:1.3;color:var(--color-neutral-800)">'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span class="om-dot" style="background:var(--color-accent-2-600)"></span>'
        f'<span>thiết bị <strong style="font-weight:700">{device}</strong>{suffix}</span></div>'
        f'<div>{esc(prov.dataset)} · {esc(prov.split)} · n = {vi.integer(prov.n)}</div>'
        f'<div>commit {esc(prov.commit)} · {esc(prov.date)}</div>'
        '</div>'
    )


# ── màn hình Hỏi đáp ─────────────────────────────────────────────────

def answer_card(display: str, kicker: str, explain: str, abstain: bool) -> str:
    """Thẻ đáp án: kicker nhỏ, đáp án cỡ 34px, một đoạn giải thích.

    Mực chữ mang thông tin: accent-700 khi model trả lời, trung tính khi nó từ
    chối. Người xem từ xa nhận ra trạng thái trước khi kịp đọc chữ.
    """
    ink = "var(--color-neutral-700)" if abstain else "var(--color-accent-700)"
    return (
        '<div class="om om-card om-card-lg" style="gap:var(--space-3)">'
        f'<div class="om-kicker">{esc(kicker)}</div>'
        f'<div class="om-display" style="font-size:34px;color:{ink}">{esc(display)}</div>'
        f'<div class="om-note" style="max-width:62ch">{esc(explain)}</div>'
        '</div>'
    )


def _bar(label: str, value: float) -> str:
    pct = max(0.0, min(1.0, value)) * 100.0
    return (
        '<div style="display:flex;flex-direction:column;gap:4px">'
        '<div style="display:flex;justify-content:space-between;font-size:13px">'
        f'<span>{esc(label)}</span>'
        f'<span class="om-num" style="color:var(--color-neutral-700)">{vi.number(value)}</span></div>'
        f'<div class="om-bar-track"><div class="om-bar-fill" style="width:{pct:.1f}%"></div></div>'
        '</div>'
    )


def confidence(start_prob: float | None, end_prob: float | None,
               alternatives: Sequence[dict], max_length: int = 384) -> str:
    """Hai thanh xác suất start/end và danh sách span xếp sau.

    Số ở đây là softmax ĐO ĐƯỢC từ QA head (xem
    ``mrc.transformer_qa.predict_detailed``), không phải minh hoạ.
    """
    if start_prob is None or end_prob is None:
        return ""

    rows = []
    for alt in alternatives:
        rows.append(
            '<div style="display:flex;align-items:baseline;justify-content:space-between;'
            'gap:var(--space-3);padding:6px 0;border-bottom:1px solid '
            'color-mix(in srgb, var(--color-text) 8%, transparent)">'
            f'<span style="font-size:13px">{esc(alt["text"])}</span>'
            f'<span class="om-num" style="font-size:12px;color:var(--color-neutral-700);'
            f'flex:none">{vi.number(alt["prob"])}</span></div>'
        )
    if not rows:
        rows.append('<div class="om-meta">Không có span nào khác vượt ngưỡng.</div>')

    return (
        '<div class="om om-grid" style="--om-min:240px">'
        '<div class="om-card" style="gap:var(--space-3)">'
        '<div class="om-label">Điểm start / end</div>'
        f'{_bar("start", start_prob)}{_bar("end", end_prob)}'
        f'<div class="om-meta">Softmax trên logit của QA head, giới hạn trong cửa sổ '
        f'{max_length} token.</div></div>'
        '<div class="om-card" style="gap:var(--space-2)">'
        f'<div class="om-label">Span xếp sau</div>{"".join(rows)}</div>'
        '</div>'
    )


def meter(margin_value: float | None, threshold: float, position: float,
          abstain: bool) -> str:
    """Thanh "trả lời hay từ chối" với vạch ranh giới và con trỏ biên độ."""
    ink = "var(--color-neutral-700)" if abstain else "var(--color-accent-700)"
    reading = (f'biên độ {vi.signed(margin_value)} · ngưỡng {vi.signed(threshold)}'
               if margin_value is not None
               else f'ngưỡng {vi.signed(threshold)} · model này không báo biên độ')
    return (
        '<div class="om om-card" style="gap:var(--space-3)">'
        '<div class="om-row"><div class="om-label">Trả lời hay từ chối</div>'
        f'<div class="om-num" style="font-size:13px;color:var(--color-neutral-800)">'
        f'{esc(reading)}</div></div>'
        '<div class="om-meter"><div class="om-meter-track"></div>'
        '<div class="om-meter-line"></div>'
        f'<div class="om-meter-dot" style="left:{position:.1f}%;background:{ink}"></div></div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;'
        'color:var(--color-neutral-800)">'
        '<span>nghiêng về trả lời</span><span>ranh giới quyết định</span>'
        '<span>nghiêng về từ chối</span></div>'
        '</div>'
    )


def passage(context: str, span: tuple[int, int] | None, heading: str,
            meta: str, kind: str = "pred", compact: bool = False) -> str:
    """Đoạn văn, tô sáng span nếu có.

    Model từ chối thì KHÔNG tô gì — in đoạn văn trơn. Tô một span rồi vẫn bảo
    "không có đáp án" là hai câu trái ngược nhau trên cùng màn hình.
    """
    if span is None:
        body = esc(context)
    else:
        start, end = span
        body = esc(context[:start]) + mark(context[start:end], kind) + esc(context[end:])
    size = " om-prose-sm" if compact else ""
    return (
        '<div class="om" style="display:flex;flex-direction:column;gap:var(--space-2)">'
        f'<div class="om-row"><div class="om-label">{esc(heading)}</div>'
        f'<div class="om-meta">{esc(meta)}</div></div>'
        f'<p class="om-prose{size}">{body}</p></div>'
    )


def footer(items: Iterable[str]) -> str:
    """Dòng chân: model, độ trễ, thiết bị, gold."""
    cells = "".join(f"<span>{esc(item)}</span>" for item in items)
    return f'<div class="om"><div class="om-foot">{cells}</div></div>'


# ── màn hình Kết quả ─────────────────────────────────────────────────

def metric_cards(cards: Sequence[dict]) -> str:
    """Hàng thẻ số lớn. Mỗi thẻ: ``{kicker, value, unit?, sub, tone?}``."""
    tones = {"accent": ("om-accent", "om-kicker"),
             "accent-2": ("om-accent-2", "om-kicker om-kicker-2"),
             "neutral": ("", "om-kicker om-kicker-n")}
    out = []
    for card in cards:
        background, kicker = tones.get(card.get("tone", "neutral"), tones["neutral"])
        unit = (f'<span style="font-size:16px"> {esc(card["unit"])}</span>'
                if card.get("unit") else "")
        out.append(
            f'<div class="om-card {background}" style="gap:4px">'
            f'<div class="{kicker}">{esc(card["kicker"])}</div>'
            f'<div class="om-display om-num" style="font-size:32px;line-height:1.1">'
            f'{esc(card["value"])}{unit}</div>'
            f'<div style="font-size:12px;color:var(--color-neutral-800)">{esc(card["sub"])}</div>'
            '</div>'
        )
    return f'<div class="om om-grid" style="--om-min:180px">{"".join(out)}</div>'


def column_winners(rows: Sequence[dict], columns: Sequence[str]) -> dict[str, str]:
    """``{cột: id của hàng thắng}`` — ai cao nhất ở từng cột.

    Tính từ dữ liệu chứ không tô đậm bằng tay: nếu ai đó huấn luyện lại và model
    khác dẫn đầu một cột, bảng tự chuyển chỗ in đậm thay vì nói sai.
    """
    winners: dict[str, str] = {}
    for column in columns:
        scored = [r for r in rows if r.get(column) is not None]
        if scored:
            winners[column] = max(scored, key=lambda r: r[column])["id"]
    return winners


def _cell(value: str, right: bool = True, strong: bool = False) -> str:
    body = f'<strong style="font-weight:700">{value}</strong>' if strong else value
    return f'<td class="{"om-right" if right else ""}">{body}</td>'


def model_table(rows: Sequence[dict], best_id: str | None = None) -> str:
    """Bảng bốn model × sáu cột, in đậm người thắng từng cột.

    ``rows``: ``{id, name, tag, EM, F1, ans_em, ans_f1, imp_em, latency}``.
    Sắp xếp tăng dần theo chất lượng ở nơi gọi, để bảng đọc như một câu chuyện
    thay vì một bảng xếp hạng.
    """
    winners = column_winners(rows, ("EM", "F1", "ans_em", "ans_f1", "imp_em"))
    body = []
    for row in rows:
        tag_class = "om-tag-accent" if row["id"] == best_id else "om-tag-neutral"
        tag = (f'<span class="om-tag {tag_class}" style="margin-left:6px">'
               f'{esc(row["tag"])}</span>') if row.get("tag") else ""
        name = esc(row["name"])
        if row["id"] == best_id:
            name = f'<strong style="font-weight:700">{name}</strong>'
        pair = (f'{vi.number(row["ans_em"])} / '
                f'{"<strong style=font-weight:700>" if winners.get("ans_f1") == row["id"] else ""}'
                f'{vi.number(row["ans_f1"])}'
                f'{"</strong>" if winners.get("ans_f1") == row["id"] else ""}')
        body.append(
            f'<tr class="{"om-win" if row["id"] == best_id else ""}">'
            f'<td>{name}{tag}</td>'
            + _cell(vi.number(row["EM"]), strong=winners.get("EM") == row["id"])
            + _cell(vi.number(row["F1"]), strong=winners.get("F1") == row["id"])
            + f'<td class="om-right">{pair}</td>'
            + _cell(vi.number(row["imp_em"]), strong=winners.get("imp_em") == row["id"])
            + _cell(vi.millis(row["latency"]))
            + '</tr>'
        )
    return (
        '<div class="om om-scroll"><table class="om-table"><thead><tr>'
        '<th>Model</th><th class="om-right">EM</th><th class="om-right">F1</th>'
        '<th class="om-right">answerable EM / F1</th>'
        '<th class="om-right">impossible EM</th><th class="om-right">Độ trễ</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def bars(rows: Sequence[dict]) -> str:
    """Hai thanh EM/F1 cho mỗi model, bề rộng = chính con số phần trăm.

    F1 dùng accent-2-**500** chứ không phải một sắc nhạt: một lần thử trước để F1
    ở accent-200 cho tương phản 1,04:1 trên nền — thanh biến mất hoàn toàn.
    """
    out = []
    for row in rows:
        out.append(
            '<div style="display:flex;flex-direction:column;gap:5px">'
            '<div style="display:flex;justify-content:space-between;font-size:13px">'
            f'<span>{esc(row["name"])}</span>'
            f'<span class="om-num" style="color:var(--color-neutral-700)">'
            f'EM {vi.number(row["EM"])} · F1 {vi.number(row["F1"])}</span></div>'
            f'<div class="om-bar-f1" style="width:{row["F1"]:.2f}%"></div>'
            f'<div class="om-bar-em" style="width:{row["EM"]:.2f}%"></div>'
            '</div>'
        )
    legend = (
        '<div class="om-legend">'
        '<span style="display:flex;align-items:center;gap:6px">'
        '<span class="om-swatch" style="background:var(--color-accent)"></span>EM</span>'
        '<span style="display:flex;align-items:center;gap:6px">'
        '<span class="om-swatch" style="background:var(--color-accent-2-500)"></span>F1</span>'
        '</div>'
    )
    return (f'<div class="om" style="display:flex;flex-direction:column;gap:var(--space-3)">'
            f'{"".join(out)}{legend}</div>')


def breakdown_table(rows: Sequence[dict], first_header: str,
                    caption: str = "") -> str:
    """Bảng EM/F1/n theo nhóm, gắn cờ nhóm quá nhỏ.

    Cờ ``n < 30`` phản chiếu ``unreliable`` trong file đánh giá và KHÔNG được bỏ:
    bucket ``<100`` của mBERT có đúng 1 câu, và "EM 0,00" ở đó không nói lên điều
    gì về model cả.
    """
    body = []
    for row in rows:
        flag = ('<span class="om-tag om-tag-outline" style="margin-left:6px">n &lt; 30</span>'
                if row.get("unreliable") else "")
        body.append(
            f'<tr><td>{esc(row["label"])}{flag}</td>'
            f'<td class="om-right">{vi.number(row["EM"])}</td>'
            f'<td class="om-right">{vi.number(row["F1"])}</td>'
            f'<td class="om-right" style="color:var(--color-neutral-700)">'
            f'{vi.integer(row["count"])}</td></tr>'
        )
    note = f'<p class="om-meta" style="margin:0;line-height:1.5">{esc(caption)}</p>' if caption else ""
    return (
        '<div class="om" style="display:flex;flex-direction:column;gap:var(--space-3)">'
        '<div class="om-scroll"><table class="om-table"><thead><tr>'
        f'<th>{esc(first_header)}</th><th class="om-right">EM</th>'
        '<th class="om-right">F1</th><th class="om-right">n</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>{note}</div>'
    )


def note_card(kicker: str, paragraphs: Sequence[str], tone: str = "accent-2",
              inset: str = "") -> str:
    """Thẻ chữ nhấn mạnh — chỗ đặt phát hiện mà bảng số không tự nói ra."""
    background = {"accent": "om-accent", "accent-2": "om-accent-2",
                  "neutral": ""}.get(tone, "om-accent-2")
    kicker_class = "om-kicker om-kicker-2" if tone == "accent-2" else "om-kicker"
    body = "".join(
        f'<p style="margin:0;font-size:14px;line-height:1.6">{text}</p>'
        for text in paragraphs
    )
    return (f'<div class="om om-card {background}" style="gap:var(--space-2);max-width:72ch">'
            f'<div class="{kicker_class}">{esc(kicker)}</div>{body}{inset}</div>')


# ── màn hình Phân tích lỗi ───────────────────────────────────────────

def error_detail(sample: dict, mark_text: str, mark_kind: str,
                 diagnosis: str) -> str:
    """Chi tiết một câu: dự đoán, gold, và context có tô sáng.

    Cặp màu là toàn bộ nội dung màn hình: **terracotta = model nói gì**,
    **sage = đáp án đúng mà model bỏ lỡ**. Nhìn màu là biết loại lỗi trước khi
    kịp đọc chữ.
    """
    context = sample["context"]
    index = context.find(mark_text) if mark_text else -1
    if index < 0:
        body = esc(context)
    else:
        body = (esc(context[:index]) + mark(mark_text, mark_kind)
                + esc(context[index + len(mark_text):]))

    gold = sample.get("gold") or []
    prediction = sample.get("prediction") or ""
    return (
        '<div class="om" style="display:flex;flex-direction:column;gap:var(--space-4)">'
        '<div>'
        f'<div class="om-label" style="margin-bottom:6px">Câu hỏi · {esc(sample["qid"])}</div>'
        f'<div class="om-display" style="font-size:22px">{esc(sample["question"])}</div></div>'
        '<div class="om-grid" style="--om-min:220px">'
        '<div class="om-card om-accent" style="gap:6px">'
        '<div class="om-kicker">Model dự đoán</div>'
        f'<div style="font-size:16px;line-height:1.5">{esc(prediction or "— rỗng —")}</div></div>'
        '<div class="om-card om-accent-2" style="gap:6px">'
        '<div class="om-kicker om-kicker-2">Gold</div>'
        f'<div style="font-size:16px;line-height:1.5">'
        f'{esc(vi.join(list(gold), empty="rỗng (impossible)"))}</div></div></div>'
        '<div class="om-card" style="gap:var(--space-2)">'
        '<div class="om-row"><div class="om-label">Context</div>'
        f'<div class="om-meta">{esc(diagnosis)}</div></div>'
        f'<p class="om-prose om-prose-sm">{body}</p></div>'
        '</div>'
    )


def verdict_tag(em: float, impossible: bool) -> str:
    """Nhãn phán quyết cho một câu mẫu."""
    if impossible:
        return '<span class="om-tag om-tag-neutral">impossible</span>'
    if em == 1:
        return '<span class="om-tag om-tag-accent-2">đúng</span>'
    return '<span class="om-tag om-tag-accent">sai</span>'


# ── màn hình Dữ liệu ─────────────────────────────────────────────────

def split_table(stats: dict[str, dict], active: str = "validation") -> str:
    """Bảng ba split. Cột "Chấm được" là cột quan trọng nhất.

    Test split có 7.301 câu nhưng 0 câu chấm được: ``answers.text`` rỗng trong
    khi ``is_impossible = False``. Không có cột này thì nó trông như một split
    dùng được, và một model luôn trả chuỗi rỗng sẽ đạt EM 100% trên đó.
    """
    body = []
    for split, row in stats.items():
        tag = ('<span class="om-tag om-tag-accent" style="margin-left:6px">đang dùng</span>'
               if split == active else "")
        impossible = (f'{vi.integer(row["num_impossible"])} · '
                      f'{vi.percent(row["impossible_pct"])}'
                      if row["num_impossible"] else vi.integer(0))
        gradeable = vi.integer(row["num_gradeable"])
        if row["num_gradeable"] == 0:
            gradeable = f'<strong style="font-weight:700">{gradeable}</strong>'
        body.append(
            f'<tr><td>{esc(split)}{tag}</td>'
            f'<td class="om-right">{vi.integer(row["num_questions"])}</td>'
            f'<td class="om-right">{vi.integer(row["num_contexts"])}</td>'
            f'<td class="om-right">{vi.integer(row["num_articles"])}</td>'
            f'<td class="om-right">{impossible}</td>'
            f'<td class="om-right">{gradeable}</td></tr>'
        )
    return (
        '<div class="om om-scroll"><table class="om-table"><thead><tr>'
        '<th>Split</th><th class="om-right">Questions</th><th class="om-right">Contexts</th>'
        '<th class="om-right">Articles</th><th class="om-right">Impossible</th>'
        f'<th class="om-right">Chấm được</th></tr></thead><tbody>{"".join(body)}</tbody>'
        '</table></div>'
    )


def passage_head(item) -> str:
    """Nửa trên thẻ đoạn văn: tiêu đề bên trái, số âm tiết và bucket bên phải.

    Tách khỏi :func:`passage_excerpt` để nơi gọi kẹp được một nút Streamlit thật
    vào giữa hai nửa — nút phải là widget mới bấm được, nên nó không thể nằm
    trong chuỗi HTML này.
    """
    return (
        '<div class="om om-row">'
        f'<div class="om-display" style="font-size:17px">{esc(item.title)}</div>'
        '<div style="display:flex;align-items:center;gap:var(--space-3)" class="om-meta">'
        f'<span>{vi.integer(item.syllables)} âm tiết</span>'
        f'<span>{esc(item.bucket)}</span></div></div>'
    )


def passage_excerpt(item) -> str:
    """Nửa dưới thẻ đoạn văn: trích đoạn đầu."""
    return f'<p class="om om-note" style="margin:0">{esc(item.excerpt)}</p>'


# ── màn hình So sánh ─────────────────────────────────────────────────

def compare_row(name: str, note: str, prediction: str, verdict_text: str,
                em: float, f1: float, muted: bool = False) -> str:
    """Một hàng so sánh: model bên trái, dự đoán ở giữa, EM/F1 bên phải."""
    ink = "var(--color-neutral-600)" if muted else "var(--color-accent-800)"
    return (
        '<div class="om" style="display:grid;grid-template-columns:minmax(160px,210px) '
        'minmax(0,1fr) 110px;gap:var(--space-4);align-items:start;'
        'padding:var(--space-3) 0;border-bottom:1px solid var(--color-divider)">'
        f'<div><div class="om-display" style="font-size:16px">{esc(name)}</div>'
        f'<div class="om-meta">{esc(note)}</div></div>'
        f'<div><div style="font-size:15px;line-height:1.6;color:{ink}">{esc(prediction)}</div>'
        f'<div class="om-meta" style="margin-top:4px">{esc(verdict_text)}</div></div>'
        f'<div class="om-num" style="font-size:13px;color:var(--color-neutral-800);'
        f'text-align:right">EM {vi.number(em)}<br>F1 {vi.number(f1)}</div></div>'
    )


# ── màn hình Huấn luyện ──────────────────────────────────────────────

#: Khung vẽ. x∈[40,540] là vùng dữ liệu, y=190 là trục hoành, y=16 là đỉnh.
_CHART = {"width": 580, "height": 220, "left": 40, "right": 540,
          "top": 16, "base": 190}


def nice_ceiling(value: float, step: float = 0.5, headroom: float = 1.10) -> float:
    """Trần trục tung: làm tròn LÊN theo ``step`` sau khi chừa khoảng hở.

    Tính từ dữ liệu thay vì ghi sẵn "3,5": huấn luyện lại với loss khác thì trục
    tự co giãn, thay vì đường vẽ trèo ra khỏi khung.
    """
    if value <= 0:
        return step
    return math.ceil(value * headroom / step) * step


def _x(index: int, count: int) -> float:
    if count <= 1:
        return (_CHART["left"] + _CHART["right"]) / 2
    span = _CHART["right"] - _CHART["left"]
    return _CHART["left"] + index * span / (count - 1)


def _y(value: float, y_max: float) -> float:
    span = _CHART["base"] - _CHART["top"]
    return _CHART["base"] - (value / y_max) * span


def line_chart(series: Sequence[dict], y_max: float, epochs: int,
               y_label: str | None = None) -> str:
    """Biểu đồ đường SVG viết tay, không thư viện.

    Vẽ tay vì đây là hai biểu đồ tĩnh với năm điểm dữ liệu: kéo cả một thư viện
    biểu đồ vào chỉ để dựng chừng đó polyline là đắt hơn nhiều so với giá trị,
    và biểu đồ tự vẽ thì dùng đúng token màu của hệ thiết kế.

    ``series``: ``{values, color, dashed?}``. ``values`` có thể ngắn hơn
    ``epochs`` — mBERT chạy 2 epoch còn ViSoBERT 3, hai đường vẫn chung một lưới
    hoành để so sánh được theo epoch.
    """
    parts = [
        f'<line x1="{_CHART["left"]}" y1="{_CHART["top"]}" x2="{_CHART["left"]}" '
        f'y2="{_CHART["base"]}" stroke="var(--color-neutral-400)" stroke-width="1"></line>',
        f'<line x1="{_CHART["left"]}" y1="{_CHART["base"]}" x2="{_CHART["right"] + 20}" '
        f'y2="{_CHART["base"]}" stroke="var(--color-neutral-400)" stroke-width="1"></line>',
        f'<text x="4" y="20" font-size="11" fill="var(--color-neutral-700)">'
        f'{esc(y_label or vi.number(y_max, 1))}</text>',
        f'<text x="14" y="194" font-size="11" fill="var(--color-neutral-700)">0</text>',
    ]

    for line in series:
        values = line["values"]
        if not values:
            continue
        points = " ".join(
            f"{_x(i, epochs):.0f},{_y(value, y_max):.0f}"
            for i, value in enumerate(values)
        )
        dash = ' stroke-dasharray="7 6"' if line.get("dashed") else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{line["color"]}" '
            f'stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round"{dash}></polyline>'
        )
        for i, value in enumerate(values):
            parts.append(
                f'<circle cx="{_x(i, epochs):.0f}" cy="{_y(value, y_max):.0f}" r="5" '
                f'fill="{line["color"]}"></circle>'
            )

    for i in range(epochs):
        parts.append(
            f'<text x="{_x(i, epochs):.0f}" y="210" font-size="11" '
            f'fill="var(--color-neutral-700)" text-anchor="middle">epoch {i + 1}</text>'
        )

    # Tài liệu SVG ĐỘC LẬP: có ``xmlns`` và có width/height thật. Thiếu xmlns thì
    # trình duyệt từ chối parse khi SVG đi qua ``data:`` URI (ảnh hỏng, không báo
    # lỗi); thiếu width/height thì kích thước nội tại bằng 0 và ảnh cũng biến mất.
    # Cỡ hiển thị do CSS của thẻ <img> quyết định, xem :func:`svg_image`.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {_CHART["width"]} {_CHART["height"]}" '
            f'width="{_CHART["width"]}" height="{_CHART["height"]}" '
            f'preserveAspectRatio="xMidYMid meet" role="img">'
            f'{"".join(parts)}</svg>')


def svg_image(svg: str, alt: str) -> str:
    """Đóng gói SVG thành ``<img>`` với ``data:`` URI.

    ``st.html`` của Streamlit LỌC BỎ thẻ ``<svg>`` — nó biến mất khỏi DOM không
    một lời cảnh báo, để lại một cái thẻ rỗng. Thẻ ``<img>`` thì được giữ, nên
    hình được nhúng qua đường đó; đổi lại, biến CSS phải phân giải trước vì ảnh
    là tài liệu riêng (xem :func:`demo.theme.resolve`).
    """
    payload = base64.b64encode(resolve(svg).encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{payload}" alt="{attr(alt)}" '
            f'style="width:100%;height:auto;display:block">')


def chart_card(label: str, svg: str) -> str:
    """Bọc một biểu đồ trong thẻ có nhãn."""
    return (f'<div class="om om-card" style="gap:var(--space-3)">'
            f'<div class="om-label">{esc(label)}</div>'
            f'{svg_image(svg, label)}</div>')


def chart_legend(entries: Sequence[dict]) -> str:
    """Chú giải: đoạn gạch màu + nhãn."""
    cells = "".join(
        '<span style="display:flex;align-items:center;gap:8px">'
        f'<span class="om-line" style="background:{entry["color"]}"></span>'
        f'{esc(entry["label"])}</span>'
        for entry in entries
    )
    return (f'<div class="om" style="display:flex;gap:var(--space-6);flex-wrap:wrap;'
            f'font-size:13px">{cells}</div>')


def epoch_table(rows: Sequence[dict]) -> str:
    """Bảng từng epoch: loss huấn luyện và EM/F1 trên tập validation."""
    body = "".join(
        f'<tr><td>{esc(row["model"])}</td>'
        f'<td class="om-right">{vi.integer(row["epoch"])}</td>'
        f'<td class="om-right">{vi.number(row["train_loss"], 4)}</td>'
        f'<td class="om-right">{vi.number(row["val_em"])}</td>'
        f'<td class="om-right">{vi.number(row["val_f1"])}</td></tr>'
        for row in rows
    )
    return (
        '<div class="om om-scroll"><table class="om-table"><thead><tr>'
        '<th>Model</th><th class="om-right">Epoch</th><th class="om-right">Train loss</th>'
        '<th class="om-right">Val EM</th><th class="om-right">Val F1</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )
