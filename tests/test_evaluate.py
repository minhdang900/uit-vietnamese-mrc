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


# ── bản ghi từng câu hỏi (preds JSONL) ───────────────────────────────
class StubDetailed:
    """Predictor có predict_detailed, như TransformerQA, không cần model."""
    name = "stub-detailed"
    null_threshold = 0.0

    def predict_detailed(self, context, question):
        return {"answer": "Hà Nội", "span": (0, 6), "null_delta": -1.5,
                "windows": [{"best_score": 2.0, "null_score": 0.5, "start_char": 0,
                             "end_char": 6, "text": "Hà Nội"}]}


def _mixed_set():
    return [_ex("q1", "Hà Nội là thủ đô.", ["Hà Nội"]),
            _ex("q2", "Huế là cố đô.", ["Huế"]),
            _ex("i1", "Một đoạn văn.", [], impossible=True)]


def test_preds_jsonl_has_one_line_per_question(tmp_path):
    from mrc.evaluate import read_jsonl

    path = tmp_path / "preds.jsonl"
    r = run_evaluation(StubDetailed(), _mixed_set(), preds_path=path, run_id="stub")
    recs = read_jsonl(path)
    assert len(recs) == r["n"] == 3
    assert [x["qid"] for x in recs] == ["q1", "q2", "i1"]


def test_preds_jsonl_schema_for_transformer_like_predictor(tmp_path):
    from mrc.evaluate import paragraph_id, read_jsonl

    path = tmp_path / "preds.jsonl"
    run_evaluation(StubDetailed(), _mixed_set(), preds_path=path)
    rec = read_jsonl(path)[0]
    assert set(rec) == {"qid", "paragraph_id", "title", "is_impossible", "gold", "pred",
                        "em", "f1", "null_delta", "windows"}
    assert rec["paragraph_id"] == paragraph_id("Hà Nội là thủ đô.")
    assert len(rec["paragraph_id"]) == 12
    assert rec["null_delta"] == -1.5 and rec["windows"][0]["text"] == "Hà Nội"


def test_preds_jsonl_for_baseline_has_same_schema_without_windows(tmp_path):
    from mrc.evaluate import read_jsonl

    path = tmp_path / "preds.jsonl"
    run_evaluation(StubEmpty(), _mixed_set(), preds_path=path)
    rec = read_jsonl(path)[0]
    assert "windows" not in rec and rec["null_delta"] is None
    assert rec["pred"] == ""


def test_record_scores_agree_with_aggregate(tmp_path):
    from mrc.evaluate import read_jsonl

    path = tmp_path / "preds.jsonl"
    r = run_evaluation(StubDetailed(), _mixed_set(), preds_path=path)
    recs = read_jsonl(path)
    assert 100.0 * sum(x["em"] for x in recs) / len(recs) == pytest.approx(r["overall"]["EM"], abs=1e-4)


def test_result_records_run_provenance_and_empty_rate():
    r = run_evaluation(StubEmpty(), _mixed_set(), run_id="empty", checkpoint=None,
                       inference_config={"tau": 0.0})
    assert r["run_id"] == "empty"
    assert r["empty_prediction_rate"] == pytest.approx(100.0)
    assert r["inference_config"] == {"tau": 0.0}
    assert "null_threshold" in r and "checkpoint" in r and "selected_on" in r


def test_empty_predictor_em_equals_impossible_share():
    from mrc.predictor import EmptyPredictor

    r = run_evaluation(EmptyPredictor(), _mixed_set())
    assert r["overall"]["EM"] == pytest.approx(100.0 / 3, abs=1e-4)
    assert r["overall"]["EM"] == pytest.approx(r["overall"]["F1"])


def test_no_preds_file_written_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run_evaluation(StubPerfect(), _mixed_set())
    assert r["preds_file"] is None and not list(tmp_path.iterdir())
