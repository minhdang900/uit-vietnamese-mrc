"""Demo web cho hệ thống đọc hiểu tiếng Việt.

Chỉ là lớp UI: mọi logic nằm trong ``demo.*``, nơi nó được test không cần chạy
Streamlit. Tệp này và mọi tệp trong ``app/`` chỉ được phép gọi widget Streamlit
và ghép các mảnh đã có test lại với nhau.

    streamlit run app/streamlit_app.py

Sáu màn hình, mỗi màn hình một URL riêng (``?/ket-qua``) để lúc thuyết trình mở
thẳng được chỗ cần thay vì bấm qua từng bước.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đặt sys.path TRƯỚC mọi import của dự án. ``streamlit run`` chỉ đưa thư mục
# ``app/`` vào path, còn pytest đưa gốc repo vào — không bối cảnh nào tự thấy
# ``src/``. Vài dòng ở đây rẻ hơn việc bắt người chấm phải ``pip install -e .``
# trước khi demo chạy được.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app import shell  # noqa: E402
from app.screens import ask, compare, data, errors, results, training  # noqa: E402
from demo.catalog import APP_TITLE, NAV  # noqa: E402

#: Hàm dựng của từng màn hình, khớp khoá với ``demo.catalog.NAV``.
SCREENS = {
    "ask": ask.render,
    "results": results.render,
    "errors": errors.render,
    "data": data.render,
    "compare": compare.render,
    "training": training.render,
}


def main() -> None:  # pragma: no cover - lớp UI
    import streamlit as st

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    shell.inject_theme()
    shell.init_state()

    # ``url_path`` chỉ truyền khi khác rỗng: trang mặc định sống ở gốc ``/`` và
    # Streamlit từ chối một url_path rỗng.
    pages = {
        key: st.Page(SCREENS[key], title=label, default=(url == ""),
                     **({"url_path": url} if url else {}))
        for key, label, url in NAV
    }
    shell.register_pages(pages)
    page = st.navigation(list(pages.values()), position="hidden")

    shell.sidebar(pages, page)
    page.run()


if __name__ == "__main__":  # pragma: no cover
    main()
