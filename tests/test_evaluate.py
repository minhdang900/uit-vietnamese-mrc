"""Phase 5 — harness đánh giá: provenance, breakdown, và các cổng an toàn.

Nhiều test ở đây là KHÁNG THỂ chống lại căn bệnh đã giết phiên bản trước của dự
án: báo cáo những con số chưa từng được sinh ra, bucket n=3 trình bày như kết quả,
và tổng subgroup không khớp n mà không giải thích.
"""
import pytest

from mrc.data import Example
from mrc.evaluate import breakdown, run_evaluation


class StubPerfect:
    """Luôn trả về đúng đáp án vàng đầu tiên."""
    name = "stub-perfect"

    def __init__(self, answer="Hà Nội"):
        self._a = answer

    def predict(self, context, question):
        return self._a


class StubEmpty:
    """Luôn trả về chuỗi rỗng — model 'luôn nói không có đáp án'."""
    name = "stub-empty"

    def predict(self, context, question):
        return ""


def _ex(qid, ctx, answers, impossible=False, question="Câu hỏi?"):
    return Example(qid=qid, question=question, context=ctx, title="T",
                   answers=list(answers), answer_start=0 if answers else -1,
                   is_impossible=impossible)


@pytest.fixture
def answerable_set():
    return [_ex("q1", "Hà Nội là thủ đô.", ["Hà Nội"]),
            _ex("q2", "Hà Nội là thủ đô.", ["Hà Nội"])]


@pytest.fixture
def all_impossible_set():
    return [_ex("i1", "Một đoạn văn.", [], impossible=True),
            _ex("i2", "Một đoạn văn.", [], impossible=True)]


# ── kiểm hai chiều bằng stub ─────────────────────────────────────────
def test_perfect_stub_scores_100(answerable_set):
    assert run_evaluation(StubPerfect(), answerable_set)["overall"]["EM"] == pytest.approx(100.0)


def test_empty_stub_scores_100_on_all_impossible(all_impossible_set):
    # Kiểm NGƯỢC: quy ước impossible đi đúng qua toàn bộ harness
    assert run_evaluation(StubEmpty(), all_impossible_set)["overall"]["EM"] == pytest.approx(100.0)


def test_empty_stub_scores_0_on_answerable(answerable_set):
    assert run_evaluation(StubEmpty(), answerable_set)["overall"]["EM"] == 0.0


# ── cổng an toàn: từ chối split không chấm được ──────────────────────
def test_refuses_to_evaluate_ungradeable_split():
    """Blind split (gold rỗng + is_impossible=False) phải bị TỪ CHỐI.

    Nếu không, StubEmpty sẽ đạt EM 100% trên test split của ViQuAD và con số đó
    trông hoàn toàn hợp lý trong báo cáo.
    """
    blind = [_ex("b1", "Đoạn văn.", [], impossible=False)]
    with pytest.raises(ValueError, match="gradeable|chấm"):
        run_evaluation(StubEmpty(), blind)


# ── provenance: mọi con số truy vết được ─────────────────────────────
def test_result_records_full_provenance(answerable_set):
    r = run_evaluation(StubPerfect(), answerable_set)
    for key in ("model", "n", "timestamp", "commit", "device", "split"):
        assert key in r, f"thiếu trường provenance: {key}"


def test_provenance_n_matches_actual_count(answerable_set):
    r = run_evaluation(StubPerfect(), answerable_set)
    assert r["n"] == r["overall"]["count"] == len(answerable_set)


def test_records_measured_latency(answerable_set):
    assert run_evaluation(StubPerfect(), answerable_set)["avg_latency_ms"] >= 0.0


# ── breakdown ────────────────────────────────────────────────────────
def test_breakdown_counts_sum_to_total():
    per_item = {"a": {"em": 1.0, "f1": 1.0}, "b": {"em": 0.0, "f1": 0.5},
                "c": {"em": 1.0, "f1": 1.0}}
    tags = {"a": "x", "b": "y", "c": "x"}
    b = breakdown(per_item, tags)
    assert sum(v["count"] for v in b.values() if isinstance(v, dict)) == 3


def test_breakdown_records_n_for_every_group():
    per_item = {"a": {"em": 1.0, "f1": 1.0}}
    for v in breakdown(per_item, {"a": "g"}).values():
        if isinstance(v, dict):
            assert "count" in v          # KHÔNG có số nào thiếu n


def test_breakdown_flags_small_groups_as_unreliable():
    # nhóm n<30 phải tự gắn cờ — chống việc trình bày bucket n=3 như kết quả
    per_item = {f"q{i}": {"em": 1.0, "f1": 1.0} for i in range(3)}
    b = breakdown(per_item, {f"q{i}": "tiny" for i in range(3)})
    assert b["tiny"]["unreliable"] is True
    assert b["tiny"]["count"] == 3


def test_breakdown_does_not_flag_large_groups():
    per_item = {f"q{i}": {"em": 1.0, "f1": 1.0} for i in range(40)}
    b = breakdown(per_item, {f"q{i}": "big" for i in range(40)})
    assert b["big"]["unreliable"] is False


def test_breakdown_averages_are_percentages():
    per_item = {"a": {"em": 1.0, "f1": 1.0}, "b": {"em": 0.0, "f1": 0.0}}
    b = breakdown(per_item, {"a": "g", "b": "g"})
    assert b["g"]["EM"] == pytest.approx(50.0)


def test_run_evaluation_includes_both_breakdowns(answerable_set):
    r = run_evaluation(StubPerfect(), answerable_set)
    assert "by_context_length" in r and "by_question_type" in r


def test_question_type_breakdown_notes_impossible_exclusion(all_impossible_set):
    """Tổng của by_question_type KHÁC n vì câu impossible bị loại.

    Báo cáo v1 để lại 149+136=285 trong khi ghi n=400 mà không giải thích. Harness
    giờ BẮT BUỘC phải phát ra ghi chú đó.
    """
    r = run_evaluation(StubEmpty(), all_impossible_set)
    qt = r["by_question_type"]
    assert "_note" in qt and str(qt["_note"]).strip()


def test_sample_predictions_are_substrings_of_their_context(answerable_set):
    r = run_evaluation(StubPerfect(), answerable_set)
    for s in r["sample_predictions"]:
        assert s["prediction"] in s["context"]


def test_result_is_json_serialisable(answerable_set, tmp_path):
    import json
    r = run_evaluation(StubPerfect(), answerable_set)
    (tmp_path / "r.json").write_text(json.dumps(r, ensure_ascii=False))
    assert json.loads((tmp_path / "r.json").read_text())["n"] == 2
