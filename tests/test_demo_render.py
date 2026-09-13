"""Các khối HTML của giao diện.

ĐẶC TẢ: hàm THUẦN ``dữ liệu → chuỗi HTML``. Không streamlit, không đọc file,
không gọi model — nên cả tầng hiển thị test được bằng lời gọi hàm.

Ba nhóm bất biến được pin ở đây, vì cả ba đều hỏng ÂM THẦM (trang vẫn hiện, chỉ
là nói sai):

1. **Escape.** Context của ViQuAD là văn bản Wikipedia tuỳ ý và câu hỏi do người
   dùng gõ; một dấu ``<`` lọt vào là nuốt phần còn lại của thẻ.
2. **Tô sáng.** Terracotta = model nói gì, sage = đáp án đúng model bỏ lỡ. Đảo
   hai màu là đảo nội dung của cả màn hình phân tích lỗi.
3. **Biểu đồ.** SVG nhúng qua ``data:`` URI phải là tài liệu độc lập; thiếu
   ``xmlns`` là ảnh hỏng, sót ``var(--token)`` là hình đen thui.
"""
import base64
import re

import pytest

from demo import render
from demo.theme import TOKENS, resolve


def _decode(img_html: str) -> str:
    return base64.b64decode(
        re.search(r'base64,([^"]+)', img_html).group(1)
    ).decode("utf-8")


# ── escape ───────────────────────────────────────────────────────────
def test_passage_escapes_markup_in_the_context():
    html = render.passage("<script>x</script>", None, "Đoạn văn", "meta")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_answer_card_escapes_the_answer():
    assert "<b>" not in render.answer_card("<b>x", "k", "e", False)


def test_brand_escapes_content_but_still_builds_its_own_line_breaks():
    html = render.brand(("<b>một", "hai"), "phụ")
    assert "&lt;b&gt;một<br>hai" in html


def test_error_detail_escapes_the_question():
    sample = {"qid": "q1", "question": "<img>", "context": "ctx",
              "gold": [], "prediction": ""}
    assert "<img>" not in render.error_detail(sample, "", "pred", "chẩn đoán")


# ── tô sáng ──────────────────────────────────────────────────────────
def test_passage_marks_the_span_inside_the_context():
    html = render.passage("Hà Nội là thủ đô", (0, 6), "Đoạn văn", "meta")
    assert "<mark" in html and "Hà Nội</mark>" in html


def test_passage_without_a_span_prints_plain_prose():
    """Model từ chối thì không tô gì: tô một span rồi bảo "không có đáp án" là
    hai câu trái ngược nhau trên cùng một màn hình."""
    assert "<mark" not in render.passage("Hà Nội", None, "Đoạn văn", "meta")


def test_passage_keeps_the_text_around_the_span():
    html = render.passage("trước Hà Nội sau", (6, 12), "h", "m")
    assert "trước" in html and "sau" in html


def test_prediction_and_gold_marks_use_different_inks():
    """Terracotta cho span model nói, sage cho gold model bỏ lỡ."""
    assert render.mark("x", "pred") != render.mark("x", "gold")
    assert "om-mark-gold" in render.mark("x", "gold")
    assert "om-mark-gold" not in render.mark("x", "pred")


def test_error_detail_falls_back_to_plain_context_when_the_mark_is_absent():
    sample = {"qid": "q", "question": "q?", "context": "abc",
              "gold": [], "prediction": ""}
    assert "<mark" not in render.error_detail(sample, "không có ở đây", "pred", "d")


# ── thẻ đáp án và thanh đo ───────────────────────────────────────────
def test_answer_card_inks_differ_between_answering_and_abstaining():
    answering = render.answer_card("1999", "k", "e", abstain=False)
    abstaining = render.answer_card("Từ chối", "k", "e", abstain=True)
    assert answering != abstaining
    assert "accent-700" in answering and "neutral-700" in abstaining


def test_meter_places_the_marker_at_the_given_position():
    assert "left:12.5%" in render.meter(-7.5, 0.0, 12.5, abstain=False)


def test_meter_says_so_when_a_model_reports_no_margin():
    """Baseline TF-IDF không có null score; thanh đo phải nói thế, không vẽ bừa."""
    assert "không báo biên độ" in render.meter(None, 0.0, 50.0, abstain=False)


# ── bảng ─────────────────────────────────────────────────────────────
ROWS = [
    {"id": "a", "name": "A", "tag": "zero-shot", "EM": 40.6, "F1": 56.84,
     "ans_em": 45.71, "ans_f1": 68.2, "imp_em": 27.34, "latency": 14.349},
    {"id": "b", "name": "B", "tag": "fine-tuned", "EM": 50.8, "F1": 59.4886,
     "ans_em": 54.57, "ans_f1": 66.6, "imp_em": 41.01, "latency": 12.264},
]


def test_column_winners_are_computed_not_hand_marked():
    winners = render.column_winners(ROWS, ("EM", "F1", "ans_f1", "imp_em"))
    assert winners == {"EM": "b", "F1": "b", "ans_f1": "a", "imp_em": "b"}


def test_column_winners_skip_columns_with_no_data():
    rows = [{"id": "a", "EM": None}, {"id": "b", "EM": None}]
    assert render.column_winners(rows, ("EM",)) == {}


def test_model_table_formats_every_number_the_vietnamese_way():
    html = render.model_table(ROWS, best_id="b")
    assert "50,80" in html and "59,49" in html and "12,3 ms" in html
    assert "50.80" not in html


def test_model_table_bolds_the_per_column_winner():
    html = render.model_table(ROWS, best_id="b")
    assert "<strong style=\"font-weight:700\">50,80</strong>" in html


def test_breakdown_table_flags_groups_that_are_too_small():
    """Cờ n < 30 phản chiếu ``unreliable`` trong eval JSON và không được bỏ:
    bucket <100 của mBERT có đúng 1 câu."""
    rows = [{"label": "<100", "EM": 0.0, "F1": 33.3, "count": 1, "unreliable": True}]
    assert "n &lt; 30" in render.breakdown_table(rows, "Bucket")


def test_breakdown_table_leaves_reliable_groups_unflagged():
    rows = [{"label": "100-200", "EM": 50.6, "F1": 59.2, "count": 405,
             "unreliable": False}]
    assert "n &lt; 30" not in render.breakdown_table(rows, "Bucket")


def test_split_table_emphasises_a_split_that_cannot_be_graded():
    """Test split có 7.301 câu nhưng 0 câu chấm được — cột đó phải nổi bật."""
    stats = {"test": {"num_questions": 7301, "num_contexts": 1241,
                      "num_articles": 48, "num_impossible": 0,
                      "impossible_pct": 0.0, "num_gradeable": 0}}
    html = render.split_table(stats)
    assert "<strong style=\"font-weight:700\">0</strong>" in html
    assert "7.301" in html


# ── biểu đồ ──────────────────────────────────────────────────────────
SERIES = [{"values": [2.1316, 1.296], "color": "var(--color-accent)", "dashed": False}]


def test_nice_ceiling_derives_the_axis_from_the_data():
    """3,0512 → trục 3,5: tính ra chứ không ghi sẵn, huấn luyện lại thì trục co giãn."""
    assert render.nice_ceiling(3.0512) == 3.5


def test_nice_ceiling_always_leaves_headroom_above_the_data():
    for value in (0.4, 0.9, 3.0512, 12.0, 87.3):
        assert render.nice_ceiling(value) > value


def test_line_chart_maps_values_to_the_designed_coordinates():
    """Toạ độ khớp bản thiết kế: loss 2,1316 trên trục 3,5 nằm ở y = 84."""
    assert "40,84" in render.line_chart(SERIES, 3.5, 3)


def test_line_chart_puts_shorter_series_on_the_shared_epoch_grid():
    """mBERT chạy 2 epoch, ViSoBERT 3 — hai đường vẫn so được theo epoch."""
    svg = render.line_chart(SERIES, 3.5, 3)
    assert "epoch 3" in svg and svg.count("<circle") == 2


def test_line_chart_dashes_only_the_series_that_ask_for_it():
    dashed = [{**SERIES[0], "dashed": True}]
    assert "stroke-dasharray" in render.line_chart(dashed, 3.5, 3)
    assert "stroke-dasharray" not in render.line_chart(SERIES, 3.5, 3)


def test_chart_image_is_a_standalone_svg_document():
    """Thiếu xmlns thì trình duyệt từ chối parse SVG trong ``data:`` URI."""
    import xml.etree.ElementTree as ET

    svg = _decode(render.svg_image(render.line_chart(SERIES, 3.5, 3), "alt"))
    assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"


def test_chart_image_resolves_css_variables_to_real_colours():
    """Ảnh là tài liệu riêng, không thấy :root của trang — biến CSS sẽ ra rỗng."""
    svg = _decode(render.svg_image(render.line_chart(SERIES, 3.5, 3), "alt"))
    assert "var(--" not in svg
    assert TOKENS["color-accent"] in svg


def test_resolve_leaves_unknown_tokens_visible():
    """Token sai phải lộ ra, không được âm thầm thành màu mặc định."""
    assert resolve("var(--khong-ton-tai)") == "var(--khong-ton-tai)"


def test_chart_image_carries_alt_text():
    assert 'alt="Train loss"' in render.svg_image(render.line_chart(SERIES, 3.5, 3),
                                                  "Train loss")


@pytest.mark.parametrize("builder", [
    lambda: render.metric_cards([{"kicker": "k", "value": "1", "sub": "s"}]),
    lambda: render.bars(ROWS),
    lambda: render.note_card("k", ["<b>an toàn</b>"]),
    lambda: render.compare_row("n", "note", "pred", "v", 1.0, 2.0),
    lambda: render.epoch_table([{"model": "m", "epoch": 1, "train_loss": 1.0,
                                 "val_em": 2.0, "val_f1": 3.0}]),
    lambda: render.footer(["a", "b"]),
    lambda: render.chart_legend([{"color": "#000", "label": "x"}]),
])
def test_every_block_returns_balanced_markup(builder):
    """Chốt thô: mỗi khối trả về HTML có mở có đóng, không rỗng."""
    html = builder()
    assert html.count("<div") == html.count("</div>")
    assert html.strip()


# ── nhóm thực hiện ───────────────────────────────────────────────────
def test_avatar_initial_comes_from_the_family_name():
    """Tên gọi tiếng Việt trùng chữ đầu rất thường xuyên — Lâm, Tấn, Thi, Thu cho
    ba chữ T trong cùng một nhóm; họ thì phân biệt được."""
    names = ["Nguyễn Quang Lâm", "Trần Trọng Tấn", "Lê Quang Thi", "Vỏ Cẩm Thu"]
    assert [render.team_initial(n) for n in names] == ["N", "T", "L", "V"]


def test_avatar_initial_survives_an_empty_name():
    assert render.team_initial("  ") == "?"


def test_team_caption_shows_the_student_id():
    assert render.team_caption({"mssv": "25210289", "role": ""}) == "25210289"


def test_team_caption_appends_a_role_once_it_is_decided():
    caption = render.team_caption({"mssv": "25210289", "role": "demo & báo cáo"})
    assert caption == "25210289 · demo & báo cáo"


def test_team_caption_has_no_dangling_separator_without_a_role():
    assert not render.team_caption({"mssv": "25210289"}).endswith("·")


def test_team_renders_every_member():
    from demo.catalog import COURSE, TEAM

    html = render.team(TEAM, COURSE)
    for member in TEAM:
        assert member["name"] in html
        assert member["mssv"] in html


def test_every_team_member_has_a_student_id_and_email():
    """Chặn việc một thành viên bị thêm vào mà thiếu trường, rồi sidebar hiện rỗng."""
    from demo.catalog import TEAM

    assert TEAM
    for member in TEAM:
        assert member["name"].strip()
        assert member["mssv"].strip()
        assert member["email"].endswith("@ms.uit.edu.vn")


def test_student_ids_are_unique():
    from demo.catalog import TEAM

    ids = [m["mssv"] for m in TEAM]
    assert len(ids) == len(set(ids))


def test_each_email_matches_its_student_id():
    from demo.catalog import TEAM

    for member in TEAM:
        assert member["email"].split("@")[0] == member["mssv"]


# ── khối thương hiệu ─────────────────────────────────────────────────
def test_brand_markup_is_balanced():
    """Khối này từng bị lệch một thẻ đóng, làm dòng mô tả nhảy ra cạnh tiêu đề
    thay vì nằm dưới. Thẻ lệch không báo lỗi — nó chỉ vẽ sai."""
    html = render.brand(("Vietnamese MRC",), "CS116 · đề tài T11", "V",
                        tagline="Đọc hiểu tiếng Việt")
    assert html.count("<div") == html.count("</div>")


def test_brand_nests_the_tagline_under_the_title():
    """Tiêu đề và mô tả phải chung một cột; nếu là anh em trong hàng ngang thì
    chúng tranh chỗ và tiêu đề bị ép xuống dòng."""
    html = render.brand(("Vietnamese MRC",), "CS116", "V", tagline="Đọc hiểu")
    column = html.index('min-width:0')
    assert column < html.index("Vietnamese MRC") < html.index("Đọc hiểu")


def test_brand_omits_the_tagline_block_when_there_is_none():
    """Không có mô tả thì không được để lại một div rỗng chiếm chỗ."""
    without = render.brand(("X",), "sub", "V")
    with_tagline = render.brand(("X",), "sub", "V", tagline="mô tả")
    assert without.count("<div") == with_tagline.count("<div") - 1
    assert "font-size:11px;color:var(--color-neutral-700)" not in without


def test_brand_uses_the_configured_glyph_and_title():
    from demo.catalog import BRAND_GLYPH, BRAND_SUBTITLE, BRAND_TAGLINE, BRAND_TITLE

    html = render.brand(BRAND_TITLE, BRAND_SUBTITLE, BRAND_GLYPH,
                        tagline=BRAND_TAGLINE)
    assert "Vietnamese MRC" in html
    assert BRAND_TAGLINE in html and BRAND_SUBTITLE in html


def test_sidebar_model_note_is_shorter_than_the_full_one():
    """Ghi chú sidebar phải ngắn hơn hoặc bằng bản đầy đủ — nếu dài hơn thì thẻ
    model lại cao ba dòng và sidebar tràn như cũ."""
    from demo.catalog import MODELS

    for model in MODELS:
        assert len(model.sidebar_note) <= len(model.note)
        assert model.sidebar_note
