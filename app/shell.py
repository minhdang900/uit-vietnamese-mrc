"""Vỏ dùng chung: theme, state và sidebar.

Sidebar hiện trên cả sáu màn hình nên nó dựng ở đây một lần. Mọi lựa chọn của
người dùng sống trong ``st.session_state`` với khoá khai báo ở :data:`DEFAULTS`,
để màn hình nào cũng đọc được cùng một trạng thái mà không truyền tham số vòng
quanh.
"""

from __future__ import annotations

import streamlit as st

from demo import render
from demo.catalog import (BRAND_GLYPH, BRAND_SUBTITLE, BRAND_TAGLINE,
                          BRAND_TITLE, COURSE,
                          EVIDENCE, MODEL_BY_ID, MODELS, NAV, TEAM)
from demo.results import MissingResults, load_eval, provenance
from demo.theme import CSS
from demo.vi import number, signed

#: Trạng thái dùng chung và giá trị mặc định.
DEFAULTS: dict = {
    "model": "mbert",
    "threshold": 0.0,
    "evidence": EVIDENCE[0][0],
    "passage": 0,
    "question": None,     # None = dùng câu hỏi của đoạn văn đang chọn
    "ctx_open": False,
    "err_filter": "all",
    "err_index": 0,
    "asked": False,       # đã bấm "Trả lời" lần nào chưa
}


def inject_theme() -> None:
    """Bơm CSS của hệ thiết kế. Gọi một lần mỗi lần chạy lại script."""
    st.html(f"<style>{CSS}</style>")


def set_width(pixels: int) -> None:
    """Giới hạn bề rộng cột chính cho màn hình hiện tại.

    Mỗi màn hình có một bề rộng đọc riêng (880px cho Hỏi đáp, 1080px cho Phân
    tích lỗi): đo chữ chứ không phải đo màn hình, nên một dòng văn xuôi không kéo
    dài hết 1440px rồi mắt lạc dòng khi xuống hàng.
    """
    st.html(f'<style>[data-testid="stMainBlockContainer"]{{max-width:{pixels}px}}</style>')


def init_state() -> None:
    """Điền giá trị mặc định cho khoá nào chưa có."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


#: Bảng trang, do ``streamlit_app`` nạp vào. Màn hình con cần nó để chuyển màn
#: hình (nút "Dùng đoạn này" trên Dữ liệu), nhưng chúng không dựng được đối tượng
#: ``st.Page`` của riêng mình — hai đối tượng khác nhau cho cùng một màn hình thì
#: ``st.switch_page`` không nhận ra.
_PAGES: dict = {}


def register_pages(pages: dict) -> None:
    """Ghi nhớ bảng trang cho :func:`goto`."""
    _PAGES.clear()
    _PAGES.update(pages)


def goto(key: str) -> None:
    """Chuyển sang màn hình ``key``. Gọi được từ trong ``on_click``."""
    page = _PAGES.get(key)
    if page is not None:
        st.switch_page(page)


def current_model():
    """``Model`` đang được chọn trong sidebar."""
    return MODEL_BY_ID[st.session_state["model"]]


@st.cache_resource(show_spinner="Đang tải model…")
def load_predictor(model_id: str):
    """Nạp predictor, cache theo model.

    KHÔNG cache theo ngưỡng từ chối: ``null_threshold`` chỉ là một phép so sánh
    lúc suy luận, nên nó được gán lại sau khi lấy ra từ cache. Đưa nó vào khoá
    cache sẽ nạp lại checkpoint mỗi lần kéo thanh trượt — vài giây đơ máy cho một
    thay đổi lẽ ra tức thì.
    """
    model = MODEL_BY_ID[model_id]
    if model.kind == "tfidf":
        from mrc.baseline_tfidf import TfidfRetriever

        return TfidfRetriever()

    from mrc.transformer_qa import TransformerQA

    predictor = TransformerQA(model.path, name=model.id,
                              max_answer_len=model.max_answer_len)

    # Chạy nóng một lần. Lần suy luận ĐẦU TIÊN của torch tốn thêm ~185 ms để cấp
    # phát kernel MPS; con số đó bị cache lại và ngồi trên màn hình mãi mãi bên
    # cạnh "12,3 ms" của bảng kết quả, trông như demo chậm gấp 15 lần lúc đo.
    # Vẫn là độ trễ ĐO ĐƯỢC, chỉ là không đo phần khởi tạo chạy đúng một lần.
    predictor.predict("Hà Nội là thủ đô.", "Thủ đô là gì?")
    return predictor


def predictor_for(model, threshold: float):
    """Predictor đã gắn ngưỡng hiện tại, hoặc ``None`` nếu chưa nạp được."""
    if model.is_local and not (_repo() / model.path).exists():
        st.warning(
            f"Chưa có checkpoint `{model.path}`. Chạy "
            f"`python scripts/finetune.py --out {model.path}` trước, hoặc chọn "
            f"model khác trong sidebar."
        )
        return None
    try:
        predictor = load_predictor(model.id)
    except Exception as error:  # pragma: no cover - phụ thuộc môi trường
        st.error(f"Không nạp được {model.name}: {error}")
        return None

    if hasattr(predictor, "null_threshold"):
        predictor.null_threshold = threshold
    return predictor


def _repo():
    from demo.results import repo_root

    return repo_root()


def nav(pages: dict, current) -> None:
    """Thanh điều hướng: một nút mỗi màn hình, nút của màn hình đang mở được tô.

    Dựng lại bằng nút thay vì dùng thanh điều hướng sẵn của ``st.navigation``, vì
    thanh đó luôn nằm TRÊN mọi nội dung sidebar còn bản thiết kế đặt khối thương
    hiệu lên đầu. URL của từng màn hình vẫn giữ nguyên — ``st.switch_page`` đổi
    địa chỉ, nên mở thẳng ``?/ket-qua`` vẫn chạy.
    """
    with st.container(key="om_nav"):
        for key, label, _url in NAV:
            page = pages[key]
            st.button(
                label, key=f"om_nav_{key}", use_container_width=True,
                type="primary" if page.url_path == current.url_path else "secondary",
                on_click=st.switch_page, args=(page,),
            )


def _model_picker() -> None:
    """Bốn model kèm EM đo được, để chọn không phải chọn mù."""
    st.html('<div class="om om-label">Mô hình</div>')

    scores: dict[str, str] = {}
    for model in MODELS:
        try:
            scores[model.id] = number(load_eval(model.id)["overall"]["EM"])
        except (MissingResults, KeyError):
            scores[model.id] = "—"

    def label(model_id: str) -> str:
        """Tên trên một dòng, EM và ghi chú xuống dòng dưới.

        Bản thiết kế đặt EM cùng hàng với tên, canh phải. Ở bề rộng 276px của
        sidebar, "ViSoBERT + QA · EM 27,80" không đủ chỗ và xuống dòng giữa chừng
        con số — xuống dòng có chủ đích đọc gọn hơn là xuống dòng ngẫu nhiên.

        Dòng dưới dùng ``sidebar_note``: ghi chú đầy đủ làm thẻ cao ba dòng thay
        vì hai, và bốn thẻ như vậy đẩy cả nhóm thực hiện xuống dưới màn hình.
        """
        model = MODEL_BY_ID[model_id]
        return f"**{model.name}**  \nEM {scores[model_id]} · {model.sidebar_note}"

    with st.container(key="om_models"):
        st.radio(
            "Mô hình", [m.id for m in MODELS], format_func=label,
            key="model", label_visibility="collapsed",
        )


def _threshold_slider() -> None:
    """Ngưỡng từ chối — điều khiển dạy được nhiều nhất trong demo."""
    st.html(
        '<div class="om om-row" style="margin-bottom:-6px">'
        '<span class="om-label">Ngưỡng từ chối</span>'
        f'<span class="om-num" style="font-family:var(--font-heading);font-weight:800;'
        f'font-size:14px">{signed(st.session_state["threshold"])}</span></div>'
    )
    with st.container(key="om_threshold"):
        st.slider("Ngưỡng từ chối", -10.0, 10.0, step=0.5, key="threshold",
                  label_visibility="collapsed")
    st.html(
        '<div class="om" style="margin-top:-10px">'
        '<div style="display:flex;justify-content:space-between;font-size:11px;'
        'color:var(--color-neutral-700)"><span>−10 · mạnh dạn</span>'
        '<span>+10 · dè dặt</span></div>'
        '<div style="font-size:12px;color:var(--color-neutral-800);line-height:1.4;'
        'margin-top:var(--space-2)">Tăng ngưỡng, model hay trả lời rỗng hơn. '
        'Khoảng 30% câu ViQuAD 2.0 vốn không có đáp án.</div></div>'
    )


def sidebar(pages: dict, current) -> None:
    """Dựng toàn bộ sidebar, theo đúng thứ tự của bản thiết kế."""
    with st.sidebar:
        st.html(render.brand(BRAND_TITLE, BRAND_SUBTITLE, BRAND_GLYPH,
                             tagline=BRAND_TAGLINE))
        nav(pages, current)
        _model_picker()
        _threshold_slider()
        st.html('<hr style="border:0;border-top:1px solid var(--color-divider)">')
        st.html(render.team(TEAM, COURSE))
        try:
            st.html(render.provenance(provenance(st.session_state["model"])))
        except MissingResults:
            st.html('<div class="om om-meta">Chưa có file kết quả đánh giá.</div>')
