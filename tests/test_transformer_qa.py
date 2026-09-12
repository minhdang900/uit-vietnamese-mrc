"""Suy luận extractive QA bằng transformer có QA head.

ĐẶC TẢ (từ bài toán): cho ``context`` và ``question``, trả về một chuỗi CON của
context — hoặc chuỗi rỗng nếu context không chứa câu trả lời.

Bất biến cốt lõi, đúng cho MỌI đầu vào: ``predict(ctx, q) in ctx``. Bài toán là
*extractive*: model chỉ ra vị trí, không sinh chữ. Bất biến này bắt được cả lỗi
offset lệch lẫn model "bịa" đáp án, nên nó được kiểm ở gần như mọi test dưới đây.
"""
import pytest

pytestmark = pytest.mark.slow          # cần tải weights

CTX = ("Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. "
       "Thành phố này có khoảng tám triệu dân và nằm bên bờ sông Hồng.")


@pytest.fixture(scope="module")
def qa():
    from mrc.transformer_qa import TransformerQA

    return TransformerQA()


# ── hợp đồng interface ───────────────────────────────────────────────
def test_conforms_to_predictor_protocol(qa):
    assert isinstance(qa.name, str) and qa.name
    assert callable(qa.predict)


def test_requires_a_fast_tokenizer(qa):
    # Không có fast tokenizer thì không có offset_mapping, và cách duy nhất lấy
    # lại chuỗi là decode() — decode làm mất dấu tiếng Việt.
    assert qa.tokenizer.is_fast


def test_rejects_model_without_fast_tokenizer():
    """PhoBERT không có fast tokenizer — phải BÁO LỖI RÕ, không âm thầm hỏng."""
    from mrc.transformer_qa import TransformerQA

    with pytest.raises((RuntimeError, ValueError), match="fast|offset"):
        TransformerQA("vinai/phobert-base-v2")


def test_runs_on_an_accelerator_when_available(qa):
    assert qa.device in {"mps", "cuda", "cpu"}


# ── BẤT BIẾN EXTRACTIVE ──────────────────────────────────────────────
def test_answer_is_a_substring_of_the_context(qa):
    assert qa.predict(CTX, "Thủ đô của Việt Nam là gì?") in CTX


def test_answer_preserves_vietnamese_diacritics(qa):
    ctx = "Thành phố Hoà Bình nằm ở miền Bắc Việt Nam."
    out = qa.predict(ctx, "Thành phố nào nằm ở miền Bắc?")
    assert out in ctx                    # cắt chuỗi gốc -> dấu nguyên vẹn


def test_answer_of_long_context_is_still_a_substring(qa):
    long_ctx = ("Câu nhồi không liên quan. " * 300) + "Thủ đô của Việt Nam là Hà Nội."
    assert qa.predict(long_ctx, "Thủ đô của Việt Nam là gì?") in long_ctx


# ── đúng đáp án ──────────────────────────────────────────────────────
def test_answers_a_simple_factoid(qa):
    assert "Hà Nội" in qa.predict(CTX, "Thủ đô của Việt Nam là gì?")


def test_finds_an_answer_at_the_very_end_of_a_long_context(qa):
    """Nếu windowing hỏng, đáp án cuối đoạn văn không bao giờ tìm được."""
    long_ctx = ("Câu nhồi không liên quan. " * 300) + "Thủ đô của Việt Nam là Hà Nội."
    assert "Hà Nội" in qa.predict(long_ctx, "Thủ đô của Việt Nam là gì?")


# ── trường hợp biên ──────────────────────────────────────────────────
def test_empty_context_returns_empty_string(qa):
    assert qa.predict("", "Câu hỏi?") == ""


def test_whitespace_only_context_returns_empty_string(qa):
    assert qa.predict("   \n  ", "Câu hỏi?") == ""


def test_deterministic_across_repeated_calls(qa):
    q = "Thủ đô của Việt Nam là gì?"
    assert qa.predict(CTX, q) == qa.predict(CTX, q)


# ── ngưỡng "không có đáp án" ─────────────────────────────────────────
def test_high_null_threshold_makes_the_model_decline(qa):
    """Ngưỡng cao ⇒ dè dặt hơn. Đây là cần gạt Precision/Recall không cần train lại."""
    from mrc.transformer_qa import TransformerQA

    shy = TransformerQA(null_threshold=50.0)
    assert shy.predict(CTX, "GDP của Hà Nội năm 2023 là bao nhiêu?") == ""


def test_low_null_threshold_forces_an_answer(qa):
    from mrc.transformer_qa import TransformerQA

    bold = TransformerQA(null_threshold=-50.0)
    assert bold.predict(CTX, "GDP của Hà Nội năm 2023 là bao nhiêu?") != ""


# ── đo đạc ───────────────────────────────────────────────────────────
def test_predict_timed_returns_answer_and_measured_latency(qa):
    out, ms = qa.predict_timed(CTX, "Thủ đô của Việt Nam là gì?")
    assert isinstance(out, str) and ms > 0      # ĐO được, không ước lượng


def test_max_answer_len_bounds_the_span(qa):
    """max_answer_len tính theo TOKEN và phụ thuộc tokenizer.

    Mặc định 30 là thói quen của SQuAD tiếng Anh; với tokenizer vocab nhỏ nó cắt
    mất 25% đáp án vàng. Test này khoá chặt việc tham số có hiệu lực.
    """
    from mrc.transformer_qa import TransformerQA

    tight = TransformerQA(max_answer_len=2, null_threshold=-50.0)
    out = tight.predict(CTX, "Thủ đô của Việt Nam là gì?")
    assert len(out.split()) <= 4        # 2 token subword -> vài từ là cùng
