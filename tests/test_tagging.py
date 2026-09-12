"""Phase 2c — gán nhãn phân tích: loại câu hỏi và bucket độ dài context.

QUAN TRỌNG: ``question_type`` là HEURISTIC tự gán, KHÔNG phải nhãn có sẵn của
ViQuAD. Test ở đây chỉ pin HÀNH VI của heuristic — chúng không chứng minh
heuristic đúng. Báo cáo phải nói rõ điều này, nếu không người đọc sẽ hiểu sai
rằng dataset có nhãn reasoning-scope.
"""
import pytest

from mrc.tagging import length_bucket, question_type, tag_examples


# ── bucket độ dài ───────────────────────────────────────────────────
def test_bucket_boundaries_are_explicit():
    # Pin BIÊN: 200 thuộc bucket nào? Không để mơ hồ.
    assert length_bucket(0) == "<100"
    assert length_bucket(99) == "<100"
    assert length_bucket(100) == "100-200"
    assert length_bucket(200) == "100-200"      # biên trên ĐÓNG
    assert length_bucket(201) == "200-300"
    assert length_bucket(300) == "200-300"
    assert length_bucket(301) == "300+"


def test_bucket_counts_words_not_characters():
    # 150 từ -> bucket 100-200, dù chuỗi dài hàng trăm ký tự
    assert length_bucket(len(("từ " * 150).split())) == "100-200"


def test_bucket_returns_only_known_labels():
    labels = {length_bucket(n) for n in range(0, 500, 7)}
    assert labels <= {"<100", "100-200", "200-300", "300+"}


def test_bucket_rejects_negative_length():
    with pytest.raises(ValueError):
        length_bucket(-1)


# ── loại câu hỏi ────────────────────────────────────────────────────
def test_question_type_returns_only_two_labels():
    got = question_type("Huế ở đâu?", "Huế là thành phố ở miền Trung.")
    assert got in {"single-sentence", "multi-sentence"}


def test_high_overlap_with_one_sentence_is_single():
    ctx = "Hà Nội là thủ đô của Việt Nam. Paris là thủ đô của Pháp."
    assert question_type("Thủ đô của Việt Nam là gì?", ctx) == "single-sentence"


def test_evidence_spread_over_two_sentences_is_multi():
    # "thủ đô" ở câu 1, "dân" ở câu 2 -> không câu nào chứa đủ
    ctx = "Hà Nội là thủ đô. Thành phố này có tám triệu dân."
    assert question_type("Thủ đô có bao nhiêu dân?", ctx) == "multi-sentence"


def test_single_sentence_context_is_always_single():
    assert question_type("Bất kỳ câu hỏi nào?", "Chỉ có một câu duy nhất.") == "single-sentence"


def test_empty_context_does_not_crash():
    assert question_type("Câu hỏi?", "") in {"single-sentence", "multi-sentence"}


def test_question_type_is_deterministic():
    ctx, q = "Câu một ở đây. Câu hai ở kia.", "Câu một là gì?"
    assert question_type(q, ctx) == question_type(q, ctx)


# ── gắn nhãn cho cả tập ─────────────────────────────────────────────
def test_tag_examples_adds_both_tags(mini_squad):
    from mrc.data import parse_squad
    tags = tag_examples(parse_squad(mini_squad))
    for qid, t in tags.items():
        assert t["question_type"] in {"single-sentence", "multi-sentence"}
        assert t["length_bucket"] in {"<100", "100-200", "200-300", "300+"}
        assert isinstance(t["context_words"], int)


def test_tag_examples_covers_every_qid(mini_squad):
    from mrc.data import parse_squad
    ex = parse_squad(mini_squad)
    assert set(tag_examples(ex)) == {e.qid for e in ex}
