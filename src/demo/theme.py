"""Token và CSS của hệ thiết kế, cộng vài hàm dựng HTML an toàn.

Chia việc với ``.streamlit/config.toml``: file đó khai báo những token Streamlit
hiểu được (màu, font, bo góc) để mọi widget dựng sẵn tự theo; module này lo phần
Streamlit không biết — bố cục vỏ, thanh điều hướng dạng viên thuốc, và các lớp
component dùng trong những khối HTML mà demo tự vẽ.

Mọi giá trị màu/khoảng cách ở đây khai báo dưới dạng biến CSS chứ không rải số
literal, đúng như hệ thiết kế yêu cầu: đổi một token là đổi toàn bộ giao diện.
"""

from __future__ import annotations

import re
from html import escape as _escape

__all__ = ["TOKENS", "CSS", "esc", "attr", "mark", "resolve"]

#: Ramp màu, khoảng cách, bo góc và đổ bóng của hệ "Organic".
#: Đây là bản sao có chủ đích của ``reference/organic-styles.css`` — demo không
#: nạp file đó từ đĩa vì nó nằm ngoài repo, nhưng giá trị phải trùng từng con số.
TOKENS: dict[str, str] = {
    "color-bg": "#f5ead8",
    "color-surface": "#ebddc5",
    "color-text": "#201e1d",
    "color-accent": "#c67139",
    "color-accent-2": "#7a8a5e",
    "color-divider": "color-mix(in srgb, #201e1d 16%, transparent)",

    "color-neutral-100": "#f9f4ed", "color-neutral-200": "#eee7db",
    "color-neutral-300": "#dcd3c4", "color-neutral-400": "#c0b6a5",
    "color-neutral-500": "#a19786", "color-neutral-600": "#82796a",
    "color-neutral-700": "#645c50", "color-neutral-800": "#474238",
    "color-neutral-900": "#2e2b25",

    "color-accent-100": "#fff2eb", "color-accent-200": "#ffe1d0",
    "color-accent-300": "#ffc6a5", "color-accent-400": "#f6a06b",
    "color-accent-500": "#d67f48", "color-accent-600": "#b2622d",
    "color-accent-700": "#8c491a", "color-accent-800": "#643312",
    "color-accent-900": "#402310",

    "color-accent-2-100": "#f0fae1", "color-accent-2-200": "#e1eecc",
    "color-accent-2-300": "#ccdbb2", "color-accent-2-400": "#aebf92",
    "color-accent-2-500": "#8fa073", "color-accent-2-600": "#728157",
    "color-accent-2-700": "#56633f", "color-accent-2-800": "#3d472b",
    "color-accent-2-900": "#272e1b",

    "space-1": "4.4px", "space-2": "8.8px", "space-3": "13.2px",
    "space-4": "17.6px", "space-6": "26.4px", "space-8": "35.2px",

    "radius-sm": "8px", "radius-md": "16px", "radius-lg": "28px",
    "radius-card": "32.2px",

    "shadow-sm": "0 1px 2px color-mix(in srgb, #2e2b25 14%, transparent)",
    "shadow-md": "0 3px 10px color-mix(in srgb, #2e2b25 16%, transparent)",
    "shadow-lg": "0 12px 32px color-mix(in srgb, #2e2b25 22%, transparent)",

    "font-heading": '"Baloo 2", system-ui, sans-serif',
    "font-body": '"Nunito", system-ui, sans-serif',
}

_ROOT_VARS = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())

#: CSS bơm một lần cho cả phiên. Hai phần: (1) chỉnh vỏ Streamlit về đúng bố cục
#: của thiết kế, (2) các lớp ``om-*`` dùng trong HTML tự vẽ.
#:
#: Tiền tố ``om-`` (organic-mrc) để không bao giờ đụng tên với lớp của Streamlit,
#: vốn có thể đổi giữa các phiên bản.
CSS = f""":root {{
{_ROOT_VARS}
}}

/* ── kiểu chữ tiêu đề ────────────────────────────────────────────
   Khai báo ở đây chứ không phó thác cho ``theme.headingFont`` trong
   config.toml: khoá đó NẠP được font nhưng không áp lên h1-h6 do Streamlit dựng
   (đo tại chỗ: h1 vẫn ra Nunito 700), nên giọng chữ display của hệ thiết kế biến
   mất mà không có dấu hiệu nào. Cỡ chữ theo đúng thang của bản thiết kế. */
h1, h2, h3, h4, h5, h6 {{
  font-family: var(--font-heading); font-weight: 800;
  letter-spacing: -0.015em; line-height: 1.12;
}}
h1 {{ font-size: 38px; }}
h2 {{ font-size: 22px; }}
h3 {{ font-size: 25px; }}
h4 {{ font-size: 20px; }}

/* ── vỏ ứng dụng ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ width: 276px !important; min-width: 276px !important; }}
[data-testid="stSidebarUserContent"] {{ padding: var(--space-4) var(--space-4) var(--space-6); }}
/* Dải header mặc định của Streamlit ở đầu sidebar bị bỏ hẳn. Đo tại chỗ: cao
   56px và chỉ chứa một ô trống dành cho logo cùng nút thu gọn — 56px không nội
   dung ngay phía trên khối thương hiệu, trong khi cả cột đang thiếu chỗ. Bỏ đi
   thì "Vietnamese MRC" nằm đúng đỉnh cột như bản thiết kế.

   Hai hệ quả, cả hai đều có chủ đích:
   - Không còn nút thu gọn sidebar. Sidebar vốn là khung cố định 276px của bản
     thiết kế, và nút đó đã bị ẩn từ trước.
   - ``st.logo()`` sẽ KHÔNG hiện, vì nó vẽ vào chính dải này. Muốn dùng logo thì
     phải bỏ quy tắc này và tính lại chiều cao cột. */
[data-testid="stSidebarHeader"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{
  padding: var(--space-8) var(--space-6) 88px;
  max-width: 1120px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] hr {{ margin: var(--space-3) 0; border-color: var(--color-divider); }}

/* Streamlit chèn khoảng trống đều nhau giữa mọi element; thiết kế dùng nhịp
   riêng cho từng khối, nên siết mặc định lại và để từng khối tự khai báo gap. */
[data-testid="stVerticalBlock"] {{ gap: var(--space-3); }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 6px; }}

/* ── điều hướng dạng viên thuốc ──────────────────────────────────
   st.navigation chạy ở position="hidden" và thanh điều hướng được dựng lại bằng
   nút: bản thiết kế đặt khối thương hiệu TRÊN nav, còn Streamlit luôn gắn nav
   của nó lên trên mọi nội dung sidebar và không có API nào đảo thứ tự đó. */
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}

[data-testid="stSidebar"] .st-key-om_nav {{ gap: 2px; }}
.st-key-om_nav button {{
  justify-content: flex-start; padding: 7px 14px; font-size: 14px;
  font-family: var(--font-body); border: none; background: transparent;
  min-height: 0;
}}
/* Nhãn nút nằm trong một div canh giữa; kéo về trái và thêm chấm 6px của
   bản thiết kế bằng ::before, khỏi phải nhồi HTML vào nhãn nút. */
.st-key-om_nav button > div {{ justify-content: flex-start; width: 100%; line-height: 1.2; }}
.st-key-om_nav button > div::before {{
  content: ""; width: 6px; height: 6px; border-radius: 999px; flex: none;
  background: currentColor; opacity: 0.55; margin-right: 10px;
}}
.st-key-om_nav button:hover {{
  background: color-mix(in srgb, var(--color-text) 7%, transparent);
}}
.st-key-om_nav [data-testid="stBaseButton-primary"],
.st-key-om_nav [data-testid="stBaseButton-primary"]:hover {{
  background: var(--color-accent); color: var(--color-bg);
}}

/* ── chọn model: radio hoá thành thẻ có viền ─────────────────────── */
.st-key-om_models [data-testid="stRadioOption"] {{
  padding: 7px 11px; border-radius: var(--radius-md);
  border: 1px solid var(--color-divider); margin-bottom: 3px; align-items: flex-start;
}}
.st-key-om_models [data-testid="stRadioOption"]:hover {{ border-color: var(--color-accent); }}
.st-key-om_models [data-testid="stRadioOption"]:has(input:checked) {{
  border-color: var(--color-accent); background: var(--color-accent-100);
}}
.st-key-om_models [data-testid="stRadioOption"] p {{ font-size: 12px; line-height: 1.3; }}
.st-key-om_models [data-testid="stRadioOption"] strong {{
  font-family: var(--font-heading); font-weight: 800; font-size: 14px;
}}

/* ── danh sách câu mẫu: nút hoá thành thẻ ────────────────────────
   Nút được dùng vì một thẻ HTML bơm vào không bắt được cú click. Ở đây nút được
   kéo về hình dạng thẻ của bản thiết kế: chữ canh trái và xuống dòng được, thẻ
   đang chọn có viền accent chứ không bị tô đặc — tô đặc sẽ nuốt mất chính cái
   badge phán quyết đang nằm trong nhãn. */
.st-key-om_errlist button {{
  padding: 12px 14px; border-radius: var(--radius-md); height: auto;
  background: transparent; border: 1px solid var(--color-divider);
}}
.st-key-om_errlist button > div {{
  justify-content: flex-start; text-align: left; white-space: normal; width: 100%;
}}
.st-key-om_errlist button p {{ font-size: 13px; line-height: 1.45; }}
.st-key-om_errlist button:hover {{ border-color: var(--color-accent); }}
.st-key-om_errlist [data-testid="stBaseButton-primary"],
.st-key-om_errlist [data-testid="stBaseButton-primary"]:hover {{
  background: var(--color-neutral-100); color: var(--color-text);
  border-color: var(--color-accent);
}}

/* ── thanh trượt: bỏ nhãn số mặc định ────────────────────────────
   Giá trị đã hiện ở hàng nhãn phía trên dưới dạng có dấu ("+0,0"), và biên
   được nói bằng chữ ("−10 · mạnh dạn"). Để nguyên thì mỗi con số xuất hiện hai
   lần, một lần theo quy ước Việt và một lần theo quy ước Mỹ. */
.st-key-om_threshold [data-testid="stSliderThumbValue"],
.st-key-om_threshold [data-testid="stSliderTickBar"] {{ display: none; }}

/* ── input câu hỏi: viên thuốc cao 52px, ngang hàng với nút Trả lời ── */
.st-key-om_ask [data-testid="stTextInput"] input {{
  min-height: 52px; font-size: 16px; border-radius: 999px;
  background: var(--color-neutral-100); padding-inline: 18px;
}}
.st-key-om_ask [data-testid="stBaseButton-primary"] {{
  min-height: 52px; padding-inline: 18px; font-size: 15px; width: 100%;
}}

/* Nhãn nút KHÔNG được cắt bớt. Streamlit mặc định cho ellipsis khi cột hẹp, và
   "Trả lời" rút thành "Trả …" thì cái nút hết nói được nó làm gì. */
.st-key-om_ask button > div,
[class*="st-key-om_use_"] button > div {{
  white-space: nowrap; overflow: visible; text-overflow: clip;
}}
[class*="st-key-om_use_"] button {{ padding-inline: 12px; }}

/* ── chip và segmented control ───────────────────────────────────── */
button[kind="pills"], button[kind="pillsActive"],
button[kind="segmented_control"], button[kind="segmented_controlActive"] {{
  font-family: var(--font-body); font-size: 13px;
}}
button[kind="pills"] {{ background: transparent; border-color: var(--color-divider); }}
button[kind="pills"]:hover {{ border-color: var(--color-accent); }}
button[kind="pillsActive"], button[kind="segmented_controlActive"] {{
  background: var(--color-accent); border-color: var(--color-accent); color: var(--color-bg);
}}
button[kind="pillsActive"] p, button[kind="segmented_controlActive"] p {{ color: var(--color-bg); }}

/* ── thẻ đoạn văn: container Streamlit mang hình dạng thẻ ─────────
   Khớp tiền tố vì mỗi thẻ có một key riêng theo qid. Cần là container thật chứ
   không phải div bơm vào, vì nút "Dùng đoạn này" phải nằm trong thẻ. */
[class*="st-key-om_pass_"] {{
  padding: var(--space-3); border-radius: var(--radius-card);
  background: var(--color-neutral-100); gap: var(--space-2);
}}

/* ── nhịp dọc của sidebar ────────────────────────────────────────
   Sáu nhóm xếp chồng trong 276px: thương hiệu, điều hướng, model, ngưỡng, nhóm
   thực hiện, xuất xứ. Đo tại chỗ ở cửa sổ 1440×1000: nếu để nguyên mặc định,
   cột cao 1205px trong khung 929px — một phần ba nằm dưới màn hình, và phần bị
   khuất chính là tên nhóm với dòng xuất xứ. Các trị số dưới đây kéo nó về vừa
   khung, không bỏ đi thông tin nào. */
[data-testid="stSidebar"] hr {{ margin: var(--space-2) 0; }}

.om-people {{ margin-bottom: 2px; }}
.om-avatar {{ width: 26px; height: 26px; font-size: 11px; }}
.om-people > div:last-child > div:first-child {{ font-size: 12.5px; line-height: 1.2; }}

[data-testid="stSidebar"] .om-meta {{ line-height: 1.2; font-size: 11px; }}
.om-prov {{ padding-top: 2px; }}

/* ── khối HTML tự vẽ ─────────────────────────────────────────────── */
.om {{ font-family: var(--font-body); color: var(--color-text); font-size: 15px; line-height: 1.55; }}
.om * {{ box-sizing: border-box; }}
.om-stack {{ display: flex; flex-direction: column; gap: var(--space-3); }}
.om-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-3); flex-wrap: wrap; }}
.om-grid {{ display: grid; gap: var(--space-3); grid-template-columns: repeat(auto-fit, minmax(var(--om-min, 240px), 1fr)); }}

.om-card {{
  display: flex; flex-direction: column; gap: var(--space-2);
  padding: var(--space-3); border-radius: var(--radius-card);
  background: var(--color-neutral-100);
}}
.om-card-lg {{ padding: var(--space-6); box-shadow: var(--shadow-sm); }}
.om-accent {{ background: var(--color-accent-100); }}
.om-accent-2 {{ background: var(--color-accent-2-100); }}

.om-kicker {{
  font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--color-accent); font-family: var(--font-body);
}}
.om-kicker-2 {{ color: var(--color-accent-2-700); }}
.om-kicker-n {{ color: var(--color-neutral-700); }}
.om-label {{
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--color-neutral-700);
}}
.om-meta {{ font-size: 12px; color: var(--color-neutral-700); }}
.om-note {{ font-size: 13px; color: var(--color-neutral-800); line-height: 1.6; }}
.om-display {{ font-family: var(--font-heading); font-weight: 800; line-height: 1.15; }}
.om-num {{ font-variant-numeric: tabular-nums; }}

.om-prose {{ margin: 0; font-size: 16px; line-height: 1.75; text-wrap: pretty; }}
.om-prose-sm {{ font-size: 15px; }}
mark.om-mark {{
  background: var(--color-accent-200); color: var(--color-accent-800);
  padding: 1px 5px; border-radius: 7px;
}}
mark.om-mark-gold {{ background: var(--color-accent-2-200); color: var(--color-accent-2-800); }}

.om-tag {{
  display: inline-flex; align-items: center; font-size: 11px; letter-spacing: 0.02em;
  padding: 3px 10px; border-radius: 12px;
}}
.om-tag-accent {{ background: var(--color-accent-100); color: var(--color-accent-800); }}
.om-tag-accent-2 {{ background: var(--color-accent-2-100); color: var(--color-accent-2-800); }}
.om-tag-neutral {{ background: var(--color-neutral-200); color: var(--color-neutral-800); }}
.om-tag-outline {{ border: 1px solid var(--color-accent); color: var(--color-accent); }}

.om-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.om-table th {{
  text-align: left; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: color-mix(in srgb, var(--color-text) 60%, transparent);
  padding: var(--space-2); border-bottom: 1px solid var(--color-divider); font-weight: 400;
}}
.om-table td {{
  padding: var(--space-2);
  border-bottom: 1px solid color-mix(in srgb, var(--color-text) 8%, transparent);
}}
.om-table .om-right {{ text-align: right; font-variant-numeric: tabular-nums; }}
.om-table tbody tr:hover {{ background: color-mix(in srgb, var(--color-text) 4%, transparent); }}
.om-table tr.om-win {{ background: var(--color-accent-100); }}
.om-scroll {{ overflow-x: auto; }}

.om-bar-track {{ height: 8px; border-radius: 999px; background: var(--color-neutral-300); }}
.om-bar-fill {{ height: 8px; border-radius: 999px; background: var(--color-accent); }}
.om-bar-em {{ height: 12px; border-radius: 999px; background: var(--color-accent); }}
.om-bar-f1 {{ height: 12px; border-radius: 999px; background: var(--color-accent-2-500); }}

/* Thanh "trả lời hay từ chối": gradient sage → xám → terracotta, vạch quyết định
   ở giữa, con trỏ tròn tại vị trí biên độ đã map. */
.om-meter {{ position: relative; height: 38px; }}
.om-meter-track {{
  position: absolute; inset: 14px 0 auto; height: 10px; border-radius: 999px;
  background: linear-gradient(90deg, var(--color-accent-2-300), var(--color-neutral-300) 50%, var(--color-accent-300));
}}
.om-meter-line {{ position: absolute; top: 6px; bottom: 6px; left: 50%; width: 2px; background: var(--color-neutral-700); }}
.om-meter-dot {{
  position: absolute; top: 8px; width: 22px; height: 22px; margin-left: -11px;
  border-radius: 999px; box-shadow: var(--shadow-sm);
}}

.om-foot {{
  display: flex; gap: var(--space-4); flex-wrap: wrap; font-size: 12px;
  color: var(--color-neutral-700); padding-top: var(--space-2);
  border-top: 1px solid var(--color-divider);
}}

.om-dot {{ width: 7px; height: 7px; border-radius: 999px; flex: none; display: inline-block; }}
.om-avatar {{
  width: 30px; height: 30px; flex: none; border-radius: 999px; display: grid;
  place-items: center; font-family: var(--font-heading); font-weight: 800; font-size: 12px;
}}
.om-people {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}

.om-legend {{ display: flex; gap: var(--space-4); flex-wrap: wrap; font-size: 12px; color: var(--color-neutral-800); }}
.om-swatch {{ width: 14px; height: 8px; border-radius: 999px; display: inline-block; }}
.om-line {{ width: 18px; height: 3px; display: inline-block; }}

.om-inset {{
  font-size: 13px; line-height: 1.7; background: var(--color-neutral-100);
  border-radius: var(--radius-md); padding: var(--space-3);
}}

@media (max-width: 900px) {{
  [data-testid="stMainBlockContainer"] {{ padding-inline: var(--space-4); }}
}}
"""


def esc(value: object) -> str:
    """Escape HTML. Mọi chuỗi từ dữ liệu phải đi qua đây.

    Context của ViQuAD là văn bản Wikipedia tuỳ ý và câu hỏi thì người dùng gõ
    vào; một dấu ``<`` lọt vào khối HTML tự vẽ sẽ nuốt mất phần còn lại của thẻ.
    """
    return _escape(str(value), quote=True)


def attr(value: object) -> str:
    """Escape cho giá trị nằm trong ngoặc kép của thuộc tính."""
    return _escape(str(value), quote=True)


def mark(text: str, kind: str = "pred") -> str:
    """Bọc ``text`` trong ``<mark>``.

    ``kind="pred"`` (terracotta) = model nói gì; ``kind="gold"`` (sage) = đáp án
    đúng mà model bỏ lỡ. Cặp màu này là toàn bộ nội dung của màn hình phân tích
    lỗi, nên nó được đặt tên chứ không rải class ở nơi gọi.
    """
    extra = " om-mark-gold" if kind == "gold" else ""
    return f'<mark class="om-mark{extra}">{esc(text)}</mark>'


_VAR = re.compile(r"var\(\s*--([A-Za-z0-9-]+)\s*\)")


def resolve(value: str) -> str:
    """Thay ``var(--token)`` bằng giá trị thật của token.

    Cần cho SVG nhúng dưới dạng ``data:`` URI: ảnh là một tài liệu RIÊNG, không
    thấy được ``:root`` của trang, nên mọi biến CSS trong đó sẽ rỗng và hình vẽ
    ra đen thui. Token nào không có trong bảng thì giữ nguyên ``var(...)`` để lỗi
    lộ ra thay vì âm thầm thành màu mặc định.
    """
    return _VAR.sub(lambda m: TOKENS.get(m.group(1), m.group(0)), value)
