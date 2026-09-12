"""Phase 8 — demo. Phiên bản trước của dự án có SyntaxError trong app khiến demo
(tiêu chí chấm #1) không chạy được. Test ở đây chặn điều đó tái diễn."""
import py_compile
from pathlib import Path

from app.streamlit_app import answer, highlight

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


class StubFound:
    name = "stub"
    def predict(self, context, question):
        return "Hà Nội"


class StubEmpty:
    name = "stub-empty"
    def predict(self, context, question):
        return ""


def test_app_file_compiles():
    py_compile.compile(str(APP), doraise=True)


def test_app_module_imports():
    import importlib
    importlib.import_module("app.streamlit_app")


def test_answer_returns_prediction_and_latency():
    r = answer("Hà Nội là thủ đô.", "Thủ đô?", StubFound())
    assert r["answer"] == "Hà Nội"
    assert r["found"] is True
    assert r["latency_ms"] >= 0.0


def test_answer_locates_span_in_context():
    ctx = "Thủ đô của Việt Nam là Hà Nội."
    r = answer(ctx, "Thủ đô?", StubFound())
    s, e = r["span"]
    assert ctx[s:e] == "Hà Nội"


def test_answer_handles_no_answer_case():
    r = answer("Một đoạn văn.", "Câu hỏi?", StubEmpty())
    assert r["found"] is False and r["span"] is None


def test_answer_rejects_empty_input():
    assert answer("", "Câu hỏi?", StubFound())["found"] is False
    assert answer("Đoạn văn.", "", StubFound())["found"] is False


def test_highlight_wraps_the_span():
    out = highlight("abc Hà Nội def", (4, 10))
    assert "Hà Nội" in out and ":orange[" in out


def test_highlight_without_span_returns_context_unchanged():
    assert highlight("nguyên văn", None) == "nguyên văn"
