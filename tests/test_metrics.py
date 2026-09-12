"""Phase 1b — Exact Match + token-level F1 theo quy ước SQuAD-2.0.

Đây là THƯỚC ĐO của toàn dự án. Nó được test đầy đủ TRƯỚC khi có model nào được
viết, vì một model không có metric đã kiểm chứng là một model không có kết quả.
"""
import pytest

from mrc.metrics import (
    evaluate,
    exact_match,
    metric_max_over_ground_truths,
    token_f1,
)


# ── Exact Match ─────────────────────────────────────────────────────
def test_em_one_when_equal_after_normalization():
    assert exact_match("Hà Nội.", "hà nội") == 1.0


def test_em_zero_on_different_answer():
    assert exact_match("Hà Nội", "Sài Gòn") == 0.0


def test_em_zero_when_prediction_is_superset():
    # EM là toàn-hoặc-không: thừa từ vẫn là 0
    assert exact_match("thủ đô Hà Nội", "Hà Nội") == 0.0


def test_em_zero_when_prediction_is_subset():
    assert exact_match("Hà", "Hà Nội") == 0.0


def test_em_ignores_diacritic_free_lookalikes():
    # "hoa" KHÔNG khớp "hoà" — bảo vệ QUYẾT ĐỊNH C
    assert exact_match("Hoa Binh", "Hoà Bình") == 0.0


# ── token-level F1 ──────────────────────────────────────────────────
def test_f1_perfect_overlap_is_one():
    assert token_f1("Hà Nội", "hà nội.") == pytest.approx(1.0)


def test_f1_partial_overlap_has_exact_expected_value():
    # pred [thủ,đô,hà,nội]=4 · gold [hà,nội]=2 · common=2
    # P=2/4=0.5 · R=2/2=1.0 · F1=2(0.5)(1.0)/1.5=0.6667
    assert token_f1("thủ đô Hà Nội", "Hà Nội") == pytest.approx(2 / 3, abs=1e-4)


def test_f1_no_overlap_is_zero():
    assert token_f1("Sài Gòn", "Hà Nội") == 0.0


def test_f1_treats_multisyllable_name_as_two_tokens():
    # QUYẾT ĐỊNH B: "Hà Nội" là 2 token. pred [hà]=1 · gold=2 · common=1
    # P=1/1=1.0 · R=1/2=0.5 · F1=2(1)(0.5)/1.5=0.6667
    assert token_f1("Hà", "Hà Nội") == pytest.approx(2 / 3, abs=1e-4)


def test_f1_counts_token_multiplicity_not_set_membership():
    # pred [a,a,b]=3 · gold [a,b]=2 · common={a:1,b:1}=2
    # P=2/3 · R=1.0 · F1=2(2/3)(1)/(5/3)=0.8
    assert token_f1("a a b", "a b") == pytest.approx(0.8, abs=1e-4)


def test_f1_is_symmetric_in_precision_recall_swap():
    assert token_f1("a b c", "a") == pytest.approx(token_f1("a", "a b c"), abs=1e-9)


# ── quy ước câu impossible (SQuAD-2.0) ──────────────────────────────
def test_f1_both_empty_is_perfect():
    # gold rỗng = câu impossible; dự đoán rỗng là ĐÚNG
    assert token_f1("", "") == 1.0


def test_em_both_empty_is_perfect():
    assert exact_match("", "") == 1.0


def test_f1_zero_when_predicting_something_for_impossible():
    assert token_f1("Hà Nội", "") == 0.0


def test_f1_zero_when_predicting_nothing_for_answerable():
    assert token_f1("", "Hà Nội") == 0.0


# ── nhiều đáp án vàng ───────────────────────────────────────────────
def test_max_over_ground_truths_takes_the_best():
    assert metric_max_over_ground_truths(
        exact_match, "Hà Nội", ["Sài Gòn", "Hà Nội", "Đà Nẵng"]
    ) == 1.0


def test_max_over_ground_truths_uses_best_f1_not_first():
    # gold thứ hai khớp hoàn hảo -> phải lấy 1.0, không lấy điểm của gold đầu
    got = metric_max_over_ground_truths(token_f1, "miền Trung", ["miền Trung Việt Nam", "miền Trung"])
    assert got == pytest.approx(1.0)


def test_empty_ground_truth_list_means_impossible():
    # gold list rỗng -> chỉ dự đoán rỗng mới được điểm
    assert metric_max_over_ground_truths(exact_match, "", []) == 1.0
    assert metric_max_over_ground_truths(exact_match, "Hà Nội", []) == 0.0


# ── tổng hợp ────────────────────────────────────────────────────────
def test_evaluate_returns_percentages_not_fractions():
    r = evaluate({"q1": "Hà Nội", "q2": "sai rồi"}, {"q1": ["Hà Nội"], "q2": ["Sài Gòn"]})
    assert r["EM"] == pytest.approx(50.0)  # PHẦN TRĂM
    assert r["count"] == 2


def test_evaluate_f1_is_mean_of_per_item_f1():
    r = evaluate({"q1": "thủ đô Hà Nội", "q2": "Hà Nội"}, {"q1": ["Hà Nội"], "q2": ["Hà Nội"]})
    expected = 100.0 * (2 / 3 + 1.0) / 2
    assert r["F1"] == pytest.approx(expected, abs=1e-3)


def test_evaluate_on_empty_input_returns_zero_not_nan():
    r = evaluate({}, {})
    assert r["EM"] == 0.0 and r["F1"] == 0.0 and r["count"] == 0


def test_evaluate_raises_on_missing_prediction():
    # Thà FAIL TO ỒN hơn âm thầm tính EM trên tập con rồi báo cáo n đầy đủ.
    # Đây chính là căn bệnh đã giết phiên bản trước của dự án.
    with pytest.raises(KeyError):
        evaluate({"q1": "x"}, {"q1": ["x"], "q2": ["y"]})


def test_evaluate_ignores_extra_predictions_not_in_gold():
    r = evaluate({"q1": "x", "zz": "thừa"}, {"q1": ["x"]})
    assert r["count"] == 1


def test_evaluate_reports_impossible_subset_separately():
    preds = {"a": "", "b": "Hà Nội", "c": "sai"}
    golds = {"a": [], "b": ["Hà Nội"], "c": ["Sài Gòn"]}
    r = evaluate(preds, golds)
    assert r["n_impossible"] == 1
    assert r["EM_impossible"] == pytest.approx(100.0)   # "a" đúng
    assert r["EM_answerable"] == pytest.approx(50.0)    # "b" đúng, "c" sai


def test_evaluate_per_item_scores_cover_every_question():
    per = evaluate({"q1": "a", "q2": "b"}, {"q1": ["a"], "q2": ["x"]})["per_item"]
    assert set(per) == {"q1", "q2"}
    assert per["q1"]["em"] == 1.0 and per["q2"]["em"] == 0.0
