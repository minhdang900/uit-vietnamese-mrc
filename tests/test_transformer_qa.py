"""Phase 4b — transformer QA trên model thật. Đánh dấu ``slow`` (cần tải weights).

Chạy nhanh, bỏ qua phase này: ``pytest -m "not slow"``
"""
import pytest

pytestmark = pytest.mark.slow

CTX = (
    "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. "
    "Thành phố này có khoảng tám triệu dân và nằm bên bờ sông Hồng."
)


@pytest.fixture(scope="module")
def model():
    from mrc.transformer_qa import TransformerQA

    return TransformerQA()


def test_uses_fast_tokenizer(model):
    # Không có fast tokenizer thì không có offset_mapping -> không map được span
    assert model.tokenizer.is_fast


def test_runs_on_mps(model):
    assert model.device in {"mps", "cuda", "cpu"}


def test_answer_is_substring_of_context(model):
    # BẤT BIẾN CỐT LÕI của extractive QA
    assert model.predict(CTX, "Thủ đô của Việt Nam là gì?") in CTX


def test_answers_a_simple_factoid(model):
    assert "Hà Nội" in model.predict(CTX, "Thủ đô của Việt Nam là gì?")


def test_preserves_diacritics_in_answer(model):
    ctx = "Thành phố Hoà Bình nằm ở miền Bắc Việt Nam."
    out = model.predict(ctx, "Thành phố nào nằm ở miền Bắc?")
    assert out in ctx                       # cắt từ chuỗi gốc -> dấu nguyên vẹn


def test_handles_context_longer_than_max_length(model):
    long_ctx = ("Đây là một câu nhồi để làm context dài ra. " * 300) + \
               "Thủ đô của Việt Nam là Hà Nội."
    out = model.predict(long_ctx, "Thủ đô của Việt Nam là gì?")
    assert out in long_ctx                  # windowing không phá bất biến


def test_empty_context_returns_empty(model):
    assert model.predict("", "Câu hỏi?") == ""


def test_deterministic_across_calls(model):
    q = "Thủ đô của Việt Nam là gì?"
    assert model.predict(CTX, q) == model.predict(CTX, q)


def test_predict_timed_reports_measured_latency(model):
    out, ms = model.predict_timed(CTX, "Thủ đô của Việt Nam là gì?")
    assert isinstance(out, str) and ms > 0   # ĐO được, không ước lượng
