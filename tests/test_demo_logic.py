"""Logic của demo web, tách khỏi UI Streamlit.

ĐẶC TẢ: nhận context + question + predictor, trả về đáp án kèm vị trí span để tô
sáng và độ trễ đo được.

Tách khỏi UI để test được mà không chạy Streamlit. Phiên bản trước của dự án có
SyntaxError trong app khiến demo — tiêu chí chấm quan trọng nhất — không chạy được;
test import/compile ở đây chặn điều đó tái diễn.
"""
import py_compile
from pathlib import Path

from demo.logic import answer, highlight

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


class _Found:
    name = "stub"
    def predict(self, context, question):
        return "Hà Nội"


class _Empty:
    name = "stub-empty"
    def predict(self, context, question):
        return ""


# ── demo phải chạy được ──────────────────────────────────────────────
def test_app_file_compiles():
    py_compile.compile(str(APP), doraise=True)


def test_app_module_imports():
    import importlib
    importlib.import_module("app.streamlit_app")


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
