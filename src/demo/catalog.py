"""Danh mục tĩnh của demo: model nào, màn hình nào, ai làm.

Chỉ chứa thứ KHÔNG đo được — tên hiển thị, đường dẫn checkpoint, thứ tự màn hình,
tên thành viên nhóm. Mọi con số (EM, F1, độ trễ, số câu) đọc từ ``results/*.json``
qua :mod:`demo.results`; không có metric nào được viết tay ở đây.

Đây cũng là nơi duy nhất cần sửa khi nhóm muốn điền tên thật hoặc thêm một model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Model", "MODELS", "MODEL_BY_ID", "NAV", "EVIDENCE", "ERROR_FILTERS",
           "TEAM", "COURSE", "DEFAULT_CONTEXT", "DEFAULT_QUESTION",
           "BRAND_TITLE", "BRAND_SUBTITLE", "BRAND_TAGLINE", "BRAND_GLYPH",
           "APP_TITLE"]


@dataclass(frozen=True)
class Model:
    """Một hệ thống được so sánh trong báo cáo.

    Attributes:
        id: khoá dùng trong state của UI và trong tên file ``eval_<id>_*.json``.
        name: tên hiển thị.
        note: một dòng nói model này là gì — "fine-tuned trên ViQuAD" so với
            "zero-shot" là khác biệt quan trọng nhất trong bảng kết quả.
        short_note: bản rút gọn cho sidebar. Ở bề rộng 276px, ``note`` đầy đủ
            xuống dòng thành ba dòng cho ba trên bốn thẻ, đẩy 1/3 sidebar xuống
            dưới màn hình. Các màn hình rộng vẫn dùng ``note``.
        kind: ``"transformer"`` nạp qua ``mrc.transformer_qa``; ``"tfidf"`` là
            baseline truy hồi câu, không cần checkpoint.
        path: checkpoint để nạp (chỉ với ``kind="transformer"``).
        max_answer_len: giới hạn độ dài span. ViSoBERT chia từ nhỏ hơn nhiều
            (vocab 15.004) nên cùng một đáp án tốn nhiều token hơn.
    """

    id: str
    name: str
    note: str
    short_note: str = ""
    kind: str = "transformer"
    path: str | None = None
    max_answer_len: int = 30

    @property
    def sidebar_note(self) -> str:
        """Ghi chú dùng trong sidebar; lùi về ``note`` nếu chưa đặt bản ngắn."""
        return self.short_note or self.note

    @property
    def is_local(self) -> bool:
        """Checkpoint nằm trong repo (phải fine-tune xong mới có)."""
        return bool(self.path) and self.path.startswith("models/")

    @property
    def eval_file(self) -> str:
        """Tên file kết quả đánh giá tương ứng."""
        return f"eval_{self.id}_validation.json"


#: Thứ tự: tốt nhất trước. Sidebar và mọi bảng "tốt nhất" đọc từ đây.
MODELS: tuple[Model, ...] = (
    Model("mbert", "mBERT + QA", "fine-tuned trên ViQuAD", "fine-tuned",
          path="models/mbert"),
    Model("xlmr", "XLM-R squad2", "zero-shot", "zero-shot",
          path="deepset/xlm-roberta-base-squad2"),
    Model("visobert", "ViSoBERT + QA", "fine-tuned trên ViQuAD", "fine-tuned",
          path="models/visobert", max_answer_len=64),
    Model("baseline", "TF-IDF baseline", "cosine sentence retrieval",
          "cosine retrieval", kind="tfidf"),
)

MODEL_BY_ID: dict[str, Model] = {m.id: m for m in MODELS}

#: ``(khoá màn hình, nhãn, đường dẫn URL)``. URL cho phép người trình bày mở
#: thẳng một màn hình khi thuyết trình thay vì bấm qua từng bước.
#:
#: Màn hình Hỏi đáp có URL RỖNG một cách có chủ đích: Streamlit phục vụ trang mặc
#: định ở gốc ``/`` và KHÔNG đăng ký ``url_path`` của nó, nên ``/hoi-dap`` sẽ trả
#: về hộp thoại "Page not found". Để rỗng thì địa chỉ ghi ở đây là địa chỉ chạy
#: thật, thay vì một đường dẫn chỉ tồn tại trong bảng này.
NAV: tuple[tuple[str, str, str], ...] = (
    ("ask", "Hỏi đáp", ""),
    ("results", "Kết quả", "ket-qua"),
    ("errors", "Phân tích lỗi", "phan-tich-loi"),
    ("data", "Dữ liệu", "du-lieu"),
    ("compare", "So sánh model", "so-sanh"),
    ("training", "Huấn luyện", "huan-luyen"),
)

#: Bốn cách trình bày CÙNG một kết quả. Không phải bốn phép tính khác nhau.
EVIDENCE: tuple[tuple[str, str], ...] = (
    ("highlight", "Nổi bật"),
    ("incontext", "Trong ngữ cảnh"),
    ("confidence", "Độ tin cậy"),
    ("compactview", "Gọn"),
)

ERROR_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "Tất cả"),
    ("correct", "Đúng"),
    ("wrong", "Sai"),
    ("impossible", "Impossible"),
)

#: Nhóm thực hiện. Sidebar đọc thẳng từ đây, nên đây là nơi duy nhất cần sửa.
#:
#: ``role`` để trống vì phân công chưa được chốt; sidebar chỉ hiện MSSV khi đó.
#: Điền chuỗi vào ``role`` là nó tự hiện thêm sau dấu "·".
TEAM: tuple[dict[str, str], ...] = (
    {"name": "Nguyễn Quang Lâm", "mssv": "25210289",
     "email": "25210289@ms.uit.edu.vn", "role": ""},
    {"name": "Trần Trọng Tấn", "mssv": "25210334",
     "email": "25210334@ms.uit.edu.vn", "role": ""},
    {"name": "Lê Quang Thi", "mssv": "25210337",
     "email": "25210337@ms.uit.edu.vn", "role": ""},
    {"name": "Vỏ Cẩm Thu", "mssv": "25210342",
     "email": "25210342@ms.uit.edu.vn", "role": ""},
)

#: Khối thương hiệu ở đầu sidebar.
#:
#: Tiêu đề để MỘT dòng: "Vietnamese MRC" đủ ngắn để nằm cạnh đĩa tròn 34px trong
#: bề rộng 276px, khác với "Đọc hiểu tiếng Việt" trước đây phải ngắt đôi. Nhờ vậy
#: khối đầu sidebar thấp đi một dòng và cả cột dồn lên cân hơn.
BRAND_TITLE: tuple[str, ...] = ("Vietnamese MRC",)
BRAND_TAGLINE = "Đọc hiểu tiếng Việt"
BRAND_SUBTITLE = "CS116 · đề tài T11"
BRAND_GLYPH = "V"

COURSE = "CS116 · UIT, ĐHQG-HCM"

#: Context/câu hỏi mở màn, giữ nguyên từ bản demo Streamlit đầu tiên: nó ngắn,
#: đọc trên máy chiếu được, và đáp án là một con số nên khán giả thấy ngay đúng/sai.
DEFAULT_CONTEXT = (
    "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. "
    "Thành phố nằm bên bờ sông Hồng, có diện tích khoảng 3.359 km² và "
    "dân số hơn tám triệu người. Hà Nội được UNESCO công nhận là Thành phố "
    "vì hoà bình vào năm 1999."
)

DEFAULT_QUESTION = "Hà Nội được UNESCO công nhận vào năm nào?"


#: Tên hiển thị ở tab trình duyệt.
APP_TITLE = "Vietnamese MRC — CS116 T11"
