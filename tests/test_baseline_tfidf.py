"""Phase 3 — baseline TF-IDF: truy hồi câu (sentence retrieval) bằng cosine.

Baseline này định nghĩa SÀN. Không có nó, con số F1 của transformer là vô nghĩa —
không ai biết 56% là tốt hay tệ cho bài toán này.
"""
import pytest

from mrc.baseline_tfidf import TfidfRetriever, split_sentences


# ── tách câu ────────────────────────────────────────────────────────
def test_split_sentences_on_period():
    assert split_sentences("Câu một. Câu hai.") == ["Câu một.", "Câu hai."]


def test_split_sentences_handles_question_and_exclamation():
    assert len(split_sentences("Thật à? Đúng vậy! Xong.")) == 3


def test_split_sentences_does_not_split_on_decimal_numbers():
    # "8.5 triệu" không được tách thành hai câu
    assert len(split_sentences("Dân số là 8.5 triệu người.")) == 1


def test_split_sentences_empty_gives_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_single_sentence_without_period():
    assert split_sentences("Không có dấu chấm") == ["Không có dấu chấm"]


# ── bất biến extractive ─────────────────────────────────────────────
def test_prediction_is_always_a_substring_of_context():
    ctx = "Hà Nội là thủ đô Việt Nam. Paris là thủ đô Pháp."
    out = TfidfRetriever().predict(ctx, "Thủ đô Việt Nam là gì?")
    assert out in ctx


def test_picks_the_sentence_containing_the_answer():
    ctx = "Paris là thủ đô của Pháp. Hà Nội là thủ đô của Việt Nam."
    assert "Hà Nội" in TfidfRetriever().predict(ctx, "Thủ đô Việt Nam là gì?")


def test_conforms_to_predictor_protocol():
    m = TfidfRetriever()
    assert isinstance(m.name, str) and m.name
    assert callable(m.predict)


# ── trường hợp biên ─────────────────────────────────────────────────
def test_empty_context_returns_empty_string():
    assert TfidfRetriever().predict("", "câu hỏi?") == ""


def test_single_sentence_context_returns_that_sentence():
    assert TfidfRetriever().predict("Chỉ một câu.", "gì?") == "Chỉ một câu."


def test_no_lexical_overlap_still_returns_a_sentence_from_context():
    ctx = "Alpha beta. Gamma delta."
    out = TfidfRetriever().predict(ctx, "xyz không liên quan?")
    assert out in ctx and out != ""


def test_deterministic_across_calls():
    ctx, q = "Câu A ở đây. Câu B ở kia.", "Câu B là gì?"
    m = TfidfRetriever()
    assert m.predict(ctx, q) == m.predict(ctx, q)


def test_predict_timed_returns_positive_latency():
    out, ms = TfidfRetriever().predict_timed("Một câu.", "gì?")
    assert isinstance(out, str) and ms >= 0.0


# ── giả thuyết đăng ký trước ────────────────────────────────────────
def test_baseline_returns_whole_sentence_not_short_span():
    """Đây là LÝ DO baseline sẽ có F1 trung bình nhưng EM gần 0.

    Nó trả về cả câu; gold là cụm vài từ. Overlap token có, trùng khít không.
    Test này pin cơ chế đó để kết quả EM~0 sau này là DỰ ĐOÁN ĐƯỢC, không phải
    bất ngờ đáng nghi.
    """
    ctx = "Hà Nội là thủ đô của Việt Nam."
    out = TfidfRetriever().predict(ctx, "Thủ đô của Việt Nam là gì?")
    assert len(out.split()) > 2          # cả câu, không phải chỉ "Hà Nội"
