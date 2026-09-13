"""Logic của demo web, tách khỏi UI Streamlit.

ĐẶC TẢ: nhận context + question + predictor, trả về đáp án kèm vị trí span để tô
sáng và độ trễ đo được.

Tách khỏi UI để test được mà không chạy Streamlit. Phiên bản trước của dự án có
SyntaxError trong app khiến demo — tiêu chí chấm quan trọng nhất — không chạy được;
test import/compile ở đây chặn điều đó tái diễn.
"""
import py_compile
from pathlib import Path

import pytest

from demo.logic import (abstains, answer, highlight, margin, meter_position,
                        verdict)

APP_DIR = Path(__file__).resolve().parents[1] / "app"

#: MỌI file trong app/, không chỉ file entry. Demo giờ là sáu màn hình; một
#: SyntaxError trong app/screens/ketqua.py cũng làm hỏng buổi demo y hệt, và test
#: chỉ soi mỗi streamlit_app.py sẽ không thấy gì.
APP_FILES = sorted(APP_DIR.rglob("*.py"))

APP_MODULES = sorted(
    ".".join(path.relative_to(APP_DIR.parent).with_suffix("").parts)
    for path in APP_FILES
)


class _Found:
    name = "stub"
    def predict(self, context, question):
        return "Hà Nội"


class _Empty:
    name = "stub-empty"
    def predict(self, context, question):
        return ""


# ── demo phải chạy được ──────────────────────────────────────────────
def test_the_app_layer_has_files_to_check():
    """Chặn trường hợp glob hỏng khiến hai test dưới lặng lẽ không kiểm gì."""
    assert len(APP_FILES) >= 6


@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: p.name)
def test_app_file_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", APP_MODULES)
def test_app_module_imports(module):
    import importlib

    importlib.import_module(module)


# ── trả lời ──────────────────────────────────────────────────────────
def test_answer_returns_the_prediction():
    assert answer("Hà Nội là thủ đô.", "Thủ đô?", _Found())["answer"] == "Hà Nội"


def test_answer_reports_measured_latency():
    assert answer("Hà Nội là thủ đô.", "Thủ đô?", _Found())["latency_ms"] >= 0.0


def test_answer_locates_the_span_in_the_context():
    ctx = "Thủ đô của Việt Nam là Hà Nội."
    start, end = answer(ctx, "Thủ đô?", _Found())["span"]
    assert ctx[start:end] == "Hà Nội"


def test_answer_marks_found_true_when_answer_present():
    assert answer("Hà Nội là thủ đô.", "Thủ đô?", _Found())["found"] is True


# ── không có đáp án ──────────────────────────────────────────────────
def test_answer_handles_the_no_answer_case():
    r = answer("Một đoạn văn.", "Câu hỏi?", _Empty())
    assert r["found"] is False and r["span"] is None


def test_answer_rejects_empty_context():
    assert answer("", "Câu hỏi?", _Found())["found"] is False


def test_answer_rejects_empty_question():
    assert answer("Đoạn văn.", "", _Found())["found"] is False


def test_answer_rejects_whitespace_only_input():
    assert answer("   ", "  ", _Found())["found"] is False


# ── tô sáng ──────────────────────────────────────────────────────────
def test_highlight_wraps_the_span():
    out = highlight("abc Hà Nội def", (4, 10))
    assert "Hà Nội" in out and ":orange[" in out


def test_highlight_without_span_returns_context_unchanged():
    assert highlight("nguyên văn", None) == "nguyên văn"


def test_highlight_preserves_text_outside_the_span():
    out = highlight("trước Hà Nội sau", (6, 12))
    assert out.startswith("trước") and out.endswith("sau")


# ══════════════════════════════════════════════════════════════════════
# Bằng chứng đi kèm đáp án
#
# ĐẶC TẢ: predictor nào có ``predict_detailed`` thì biên độ null, xác suất
# start/end và các span xếp sau được lấy luôn; predictor chỉ có ``predict`` vẫn
# chạy và các khoá đó nhận ``None``.
#
# Giao diện đọc thẳng ``result[...]`` nên KHOÁ PHẢI LUÔN CÓ MẶT — thiếu khoá là
# KeyError giữa lúc demo đang chạy trước mặt người chấm.
# ══════════════════════════════════════════════════════════════════════
EVIDENCE_KEYS = ("null_delta", "start_prob", "end_prob", "top_k")


class _Detailed:
    """Predictor báo đầy đủ bằng chứng, như ``TransformerQA``."""

    name = "stub-detailed"

    def predict(self, context, question):
        return "Hà Nội"

    def predict_detailed(self, context, question, top_k=3):
        return {"answer": "Hà Nội", "span": (0, 6), "found": True,
                "null_delta": -2.3, "start_prob": 0.94, "end_prob": 0.91,
                "top_k": [{"text": "Nội", "score": 1.0, "prob": 0.21}]}


@pytest.mark.parametrize("key", EVIDENCE_KEYS)
def test_answer_always_exposes_the_evidence_keys(key):
    assert key in answer("Hà Nội là thủ đô.", "Thủ đô?", _Found())


@pytest.mark.parametrize("key", EVIDENCE_KEYS)
def test_evidence_keys_exist_even_for_rejected_input(key):
    assert key in answer("", "", _Found())


def test_a_plain_predictor_reports_no_evidence():
    """Baseline TF-IDF không có null score; giao diện ẩn khối bằng chứng đi
    chứ không bịa số lấp chỗ trống."""
    result = answer("Hà Nội là thủ đô.", "Thủ đô?", _Found())
    assert result["null_delta"] is None and result["top_k"] == []


def test_a_detailed_predictor_passes_its_evidence_through():
    result = answer("Hà Nội là thủ đô.", "Thủ đô?", _Detailed())
    assert result["null_delta"] == -2.3
    assert result["start_prob"] == 0.94
    assert result["top_k"][0]["text"] == "Nội"


def test_a_detailed_predictor_still_reports_the_answer_and_span():
    result = answer("Hà Nội là thủ đô.", "Thủ đô?", _Detailed())
    assert result["answer"] == "Hà Nội" and result["span"] == (0, 6)


def test_latency_is_measured_for_detailed_predictors_too():
    assert answer("Hà Nội là thủ đô.", "Thủ đô?", _Detailed())["latency_ms"] >= 0.0


# ══════════════════════════════════════════════════════════════════════
# Quyết định trả lời / từ chối
#
# ĐẶC TẢ: ``abstain ⟺ null_delta + threshold >= 0`` — cùng phép so sánh mà
# ``mrc.windowing.select_best_span`` thực hiện. Lệch dấu bằng ở đây thì thanh đo
# báo "trả lời" đúng lúc model im lặng.
# ══════════════════════════════════════════════════════════════════════
def test_margin_adds_the_threshold_to_the_null_delta():
    assert margin(-2.3, 1.0) == pytest.approx(-1.3)


def test_margin_is_unknown_when_the_model_reports_no_delta():
    assert margin(None, 1.0) is None


def test_a_confident_answer_is_not_abstained():
    assert abstains(-2.3, 0.0) is False


def test_raising_the_threshold_can_flip_an_answer_into_a_refusal():
    """Bài học của demo: cùng một câu, cùng một model, chỉ khác ngưỡng."""
    assert abstains(-1.4, 0.0) is False
    assert abstains(-1.4, 2.0) is True


def test_lowering_the_threshold_can_recover_a_refused_answer():
    assert abstains(1.17, 0.0) is True
    assert abstains(1.17, -2.0) is False


def test_the_decision_boundary_belongs_to_refusal():
    """``select_best_span`` từ chối khi ``best <= null + threshold`` — dấu bằng
    thuộc phía từ chối, và thanh đo phải nói y như vậy."""
    assert abstains(0.0, 0.0) is True


def test_no_delta_means_no_decision_to_recompute():
    assert abstains(None, 5.0) is False


# ── vị trí con trỏ trên thanh đo ─────────────────────────────────────
def test_meter_puts_a_zero_margin_at_the_boundary():
    assert meter_position(0.0) == 50.0


def test_meter_leans_left_when_the_model_wants_to_answer():
    assert meter_position(-5.0) < 50.0


def test_meter_leans_right_when_the_model_wants_to_refuse():
    assert meter_position(5.0) > 50.0


@pytest.mark.parametrize("value", [-9999.0, -20.0, 20.0, 9999.0])
def test_meter_never_lets_the_marker_leave_the_bar(value):
    """Biên độ thật có thể lớn hơn thang nhiều lần; con trỏ bị cắt mất nửa trông
    như lỗi vẽ chứ không như "rất chắc chắn"."""
    assert 2.0 <= meter_position(value) <= 98.0


def test_meter_centres_itself_when_there_is_no_margin():
    assert meter_position(None) == 50.0


# ── bốn trường hợp, không phải hai ───────────────────────────────────
def test_verdict_distinguishes_all_four_labelled_cases():
    """Đúng/sai không nằm ở chỗ model có trả lời, mà ở chỗ câu hỏi CÓ đáp án."""
    cases = {verdict(a, i)[1] for a in (True, False) for i in (True, False)}
    assert len(cases) == 4


# ── câu hỏi tự nhập: không có nhãn nào để đối chiếu ──────────────────
#
# Nhãn gold và cờ impossible thuộc về CÂU HỎI GỐC của đoạn văn. Người dùng gõ câu
# khác thì hai thứ đó không còn áp dụng, và khẳng định "đúng, câu này impossible"
# cho một câu vừa được gõ ra là bịa dữ liệu.

@pytest.mark.parametrize("abstain", [True, False])
def test_an_unlabelled_question_never_claims_to_know_the_gold(abstain):
    explanation = verdict(abstain, None)[1]
    assert "impossible của ViQuAD" not in explanation
    assert "gold là rỗng" not in explanation


@pytest.mark.parametrize("abstain", [True, False])
def test_an_unlabelled_question_says_why_there_is_nothing_to_check(abstain):
    assert "tự nhập" in verdict(abstain, None)[1]


def test_an_unlabelled_question_still_reports_what_the_model_did():
    assert verdict(True, None)[0] == "Không có đáp án trong đoạn văn"
    assert verdict(False, None)[0] == "Đáp án trích xuất"


def test_unlabelled_verdicts_differ_from_every_labelled_one():
    labelled = {verdict(a, i)[1] for a in (True, False) for i in (True, False)}
    unlabelled = {verdict(a, None)[1] for a in (True, False)}
    assert not (labelled & unlabelled)


def test_refusing_an_impossible_question_is_reported_as_correct():
    assert "Đúng" in verdict(abstain=True, impossible=True)[1]


def test_refusing_an_answerable_question_suggests_lowering_the_threshold():
    assert "Hạ ngưỡng" in verdict(abstain=True, impossible=False)[1]


def test_answering_an_impossible_question_is_flagged_as_a_warning():
    assert "Cảnh báo" in verdict(abstain=False, impossible=True)[1]


def test_answering_an_answerable_question_restates_the_extractive_invariant():
    assert "không sinh chữ mới" in verdict(abstain=False, impossible=False)[1]


def test_the_kicker_matches_whether_the_model_answered():
    assert verdict(True, False)[0] == "Không có đáp án trong đoạn văn"
    assert verdict(False, False)[0] == "Đáp án trích xuất"


# ── điều hướng ───────────────────────────────────────────────────────
def test_every_screen_has_a_renderer():
    """NAV và bảng màn hình của app phải khớp nhau, không thừa không thiếu."""
    from app.streamlit_app import SCREENS
    from demo.catalog import NAV

    assert {key for key, _, _ in NAV} == set(SCREENS)


def test_screen_urls_are_unique():
    from demo.catalog import NAV

    urls = [url for _, _, url in NAV]
    assert len(urls) == len(set(urls))


def test_exactly_one_screen_is_served_at_the_root():
    """Streamlit chỉ phục vụ trang mặc định ở ``/`` và bỏ qua url_path của nó;
    đúng một màn hình được phép để URL rỗng, nếu không sẽ có màn hình 404."""
    from demo.catalog import NAV

    assert [url for _, _, url in NAV].count("") == 1



# ── nút chuyển màn hình ──────────────────────────────────────────────
def test_shell_can_reach_every_screen_it_registers():
    """``shell.goto`` dùng CHÍNH đối tượng trang mà st.navigation nhận; dựng một
    ``st.Page`` mới cho cùng màn hình thì ``st.switch_page`` không nhận ra."""
    from app import shell
    from demo.catalog import NAV

    sentinels = {key: object() for key, _, _ in NAV}
    shell.register_pages(sentinels)
    assert set(shell._PAGES) == {key for key, _, _ in NAV}


def test_goto_is_a_no_op_for_an_unknown_screen():
    """Sai khoá thì không được ném lỗi giữa lúc demo đang chạy."""
    from app import shell

    shell.register_pages({})
    assert shell.goto("khong-co") is None
