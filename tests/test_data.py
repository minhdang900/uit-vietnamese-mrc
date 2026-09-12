"""Phase 2 — tầng dữ liệu: parse SQuAD-2.0, dedup theo CONTEXT, split chống leakage."""
import pytest

from mrc.data import (
    Example,
    assert_no_leakage,
    compute_stats,
    deduplicate_contexts,
    parse_squad,
    references_from,
    split_by_context,
)


# ── parse ───────────────────────────────────────────────────────────
def test_parse_extracts_every_question(mini_squad):
    assert {e.qid for e in parse_squad(mini_squad)} == {"q1", "q2", "q3"}


def test_parse_returns_example_objects(mini_squad):
    assert all(isinstance(e, Example) for e in parse_squad(mini_squad))


def test_parse_attaches_context_and_title(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q1"].context.startswith("Hà Nội là thủ đô")
    assert ex["q1"].title == "Hà Nội"


def test_parse_reads_answers_as_list_of_strings(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q1"].answers == ["Hà Nội"]


def test_parse_keeps_all_alternative_answers(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q3"].answers == ["miền Trung Việt Nam", "miền Trung"]


def test_impossible_question_has_empty_answers(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q2"].is_impossible is True
    assert ex["q2"].answers == []


def test_plausible_answers_are_never_used_as_gold(mini_squad):
    # BẪY: dùng plausible_answers làm gold biến câu impossible thành answerable
    # và làm metric sai một cách âm thầm.
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert "8 triệu" not in ex["q2"].answers
    assert ex["q2"].answers == []


def test_answer_start_actually_points_at_the_answer(mini_squad):
    # Nếu offset lệch, fine-tuning sẽ học nhãn sai mà không báo lỗi.
    for e in parse_squad(mini_squad):
        if not e.answers:
            continue
        got = e.context[e.answer_start : e.answer_start + len(e.answers[0])]
        assert got == e.answers[0], f"{e.qid}: offset lệch -> {got!r}"


def test_parse_is_stable_in_order(mini_squad):
    assert [e.qid for e in parse_squad(mini_squad)] == [e.qid for e in parse_squad(mini_squad)]


# ── dedup: đơn vị là CONTEXT (paragraph), không phải ARTICLE ─────────
def test_dedup_collapses_identical_context_across_articles(duplicated_context_squad):
    ex = parse_squad(duplicated_context_squad)
    assert len({e.context for e in ex}) == 1          # 1 context duy nhất
    kept = deduplicate_contexts(ex)
    assert len({e.context for e in kept}) == 1


def test_dedup_keeps_all_questions_of_the_surviving_context(duplicated_context_squad):
    # Dedup context KHÔNG được làm mất câu hỏi — chỉ hợp nhất context.
    kept = deduplicate_contexts(parse_squad(duplicated_context_squad))
    assert {e.qid for e in kept} == {"a1", "b1"}


def test_dedup_is_noop_when_all_contexts_unique(mini_squad):
    ex = parse_squad(mini_squad)
    assert len(deduplicate_contexts(ex)) == len(ex)


# ── thống kê: sửa lỗi đếm article-thay-vì-context của dự án cũ ───────
def test_stats_counts_contexts_not_articles(mini_squad):
    s = compute_stats(parse_squad(mini_squad))
    assert s["num_contexts"] == 2
    assert s["num_articles"] == 2
    assert s["num_questions"] == 3


def test_stats_distinguishes_contexts_from_articles_when_they_differ():
    # MỘT article chứa BA context -> phải đếm 1 article, 3 context.
    # Mỗi context cần >=1 câu hỏi, vì Example được sinh theo câu hỏi.
    def qa(i):
        return {"id": f"x{i}", "question": f"Câu {i}?", "is_impossible": False,
                "answers": {"text": ["ctx"], "answer_start": [0]}}
    raw = {"data": [{"title": "Một article", "paragraphs": [
        {"context": f"ctx {w}", "qas": [qa(i)]} for i, w in enumerate("một hai ba".split())
    ]}]}
    s = compute_stats(parse_squad(raw))
    assert s["num_articles"] == 1 and s["num_contexts"] == 3


def test_stats_reports_impossible_count_and_pct(mini_squad):
    s = compute_stats(parse_squad(mini_squad))
    assert s["num_impossible"] == 1
    assert s["impossible_pct"] == pytest.approx(100 / 3, abs=0.01)


def test_stats_reports_questions_with_gold(mini_squad):
    assert compute_stats(parse_squad(mini_squad))["num_with_gold"] == 2


def test_stats_on_empty_input_does_not_divide_by_zero():
    s = compute_stats([])
    assert s["num_questions"] == 0 and s["impossible_pct"] == 0.0


# ── split theo context + leakage guard ──────────────────────────────
def test_split_never_puts_one_context_on_both_sides(mini_squad):
    tr, va = split_by_context(parse_squad(mini_squad), val_frac=0.5, seed=0)
    assert {e.context for e in tr}.isdisjoint({e.context for e in va})


def test_split_keeps_every_question_of_a_context_together(duplicated_context_squad):
    ex = parse_squad(duplicated_context_squad)          # 2 câu hỏi, 1 context
    tr, va = split_by_context(ex, val_frac=0.5, seed=0)
    # cùng context -> phải nằm CÙNG một phía, không bị xé đôi
    assert (len(tr), len(va)) in {(2, 0), (0, 2)}


def test_split_loses_no_questions(mini_squad):
    ex = parse_squad(mini_squad)
    tr, va = split_by_context(ex, val_frac=0.5, seed=0)
    assert len(tr) + len(va) == len(ex)


def test_split_is_deterministic_with_same_seed(mini_squad):
    ex = parse_squad(mini_squad)
    a1, b1 = split_by_context(ex, val_frac=0.5, seed=42)
    a2, b2 = split_by_context(ex, val_frac=0.5, seed=42)
    assert [e.qid for e in a1] == [e.qid for e in a2]
    assert [e.qid for e in b1] == [e.qid for e in b2]


def test_assert_no_leakage_passes_on_disjoint_splits(mini_squad):
    tr, va = split_by_context(parse_squad(mini_squad), val_frac=0.5, seed=0)
    assert_no_leakage(tr, va)              # không được raise


def test_assert_no_leakage_RAISES_on_shared_context(mini_squad):
    ex = parse_squad(mini_squad)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(ex, ex)


def test_assert_no_leakage_names_the_offending_context(mini_squad):
    ex = parse_squad(mini_squad)
    with pytest.raises(AssertionError) as err:
        assert_no_leakage(ex, ex)
    assert "1" in str(err.value) or "context" in str(err.value).lower()


# ── references cho harness ──────────────────────────────────────────
def test_references_maps_qid_to_gold_list(mini_squad):
    refs = references_from(parse_squad(mini_squad))
    assert refs["q1"] == ["Hà Nội"]
    assert refs["q2"] == []                # impossible -> danh sách rỗng
    assert refs["q3"] == ["miền Trung Việt Nam", "miền Trung"]


# ══════════════════════════════════════════════════════════════════════
# GRADEABILITY — phân biệt "không có đáp án" với "đáp án bị lược bỏ"
#
# Cả hai đều có gold RỖNG, nhưng ý nghĩa trái ngược:
#   - is_impossible=True  + gold rỗng -> câu KHÔNG có đáp án; dự đoán rỗng là ĐÚNG
#   - is_impossible=False + gold rỗng -> đáp án BỊ LƯỢC BỎ (blind split); KHÔNG
#     chấm được
# Gộp hai trường hợp này lại sẽ khiến một model luôn trả về "" đạt EM 100% trên
# blind test set — đúng loại lỗi âm thầm mà dự án này tồn tại để ngăn.
# ══════════════════════════════════════════════════════════════════════

def _blind_split():
    """Mô phỏng test split thật của ViQuAD: gold rỗng, is_impossible=False."""
    return {"data": [{"title": "T", "paragraphs": [{"context": "Một đoạn văn.", "qas": [
        {"id": "t1", "question": "Câu hỏi?", "is_impossible": False,
         "answers": {"text": [], "answer_start": []}}]}]}]}


def test_blind_split_question_is_not_marked_impossible():
    ex = parse_squad(_blind_split())[0]
    assert ex.is_impossible is False       # tôn trọng cờ của dataset
    assert ex.answers == []


def test_blind_split_question_is_not_gradeable():
    assert parse_squad(_blind_split())[0].is_gradeable is False


def test_genuine_impossible_question_IS_gradeable(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q2"].is_impossible is True
    assert ex["q2"].is_gradeable is True   # dự đoán rỗng là đáp án đúng


def test_answerable_question_with_gold_is_gradeable(mini_squad):
    assert {e.qid: e for e in parse_squad(mini_squad)}["q1"].is_gradeable is True


def test_assert_gradeable_raises_on_blind_split():
    from mrc.data import assert_gradeable
    with pytest.raises(ValueError, match="gradeable|chấm"):
        assert_gradeable(parse_squad(_blind_split()))


def test_assert_gradeable_passes_on_normal_split(mini_squad):
    from mrc.data import assert_gradeable
    assert_gradeable(parse_squad(mini_squad))      # không được raise


def test_stats_reports_gradeable_count(mini_squad):
    assert compute_stats(parse_squad(mini_squad))["num_gradeable"] == 3


def test_stats_reports_zero_gradeable_for_blind_split():
    assert compute_stats(parse_squad(_blind_split()))["num_gradeable"] == 0
