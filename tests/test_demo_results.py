"""Đọc số liệu từ ``results/`` và ``data/raw/``.

ĐẶC TẢ: đây là chỗ DUY NHẤT đọc file. Con số nào lên màn hình mà không đi qua
module này thì là số bịa — bất biến đầu tiên của dự án.

Test dựng cây thư mục giả trong ``tmp_path``, không đụng ``results/`` thật, nên
chúng chạy được cả khi chưa ai chạy đánh giá.
"""
import json

import pytest

from demo import results
from demo.catalog import MODELS


def _eval(model="mbert (fine-tuned)", em=50.8, f1=59.49, samples=None, **extra):
    payload = {
        "model": model, "dataset": "UIT-ViQuAD 2.0", "split": "validation",
        "n": 500, "timestamp": "2026-09-12T14:03:03+00:00", "commit": "590e775",
        "device": "mps", "env": {"torch": "2.14.0", "platform": "macOS"},
        "overall": {"EM": em, "F1": f1, "count": 500},
        "answerable_only": {"EM": 54.57, "F1": 66.6, "count": 361},
        "impossible_only": {"EM": 41.01, "count": 139},
        "avg_latency_ms": 12.264,
        "by_context_length": {}, "by_question_type": {},
        "sample_predictions": samples if samples is not None else [],
    }
    payload.update(extra)
    return payload


@pytest.fixture
def repo(tmp_path):
    """Gốc repo giả với đủ bốn file đánh giá."""
    (tmp_path / "results").mkdir()
    scores = {"mbert": 50.8, "xlmr": 40.6, "visobert": 27.8, "baseline": 0.8}
    for model in MODELS:
        (tmp_path / "results" / model.eval_file).write_text(
            json.dumps(_eval(model=model.id, em=scores[model.id])),
            encoding="utf-8",
        )
    results._cached_json.cache_clear()
    yield tmp_path
    results._cached_json.cache_clear()


# ── thiếu file thì phải ồn ào ────────────────────────────────────────
def test_missing_results_names_the_command_that_creates_them(tmp_path):
    """Màn hình toàn dấu gạch ngang trông như "model kém", không như "chưa chạy
    đánh giá" — đúng loại nhầm lẫn không được phép xảy ra lúc đang chấm."""
    results._cached_json.cache_clear()
    with pytest.raises(results.MissingResults) as error:
        results.load_eval("mbert", tmp_path)
    assert "run_eval.py" in str(error.value)


def test_unknown_model_is_rejected(repo):
    with pytest.raises(KeyError):
        results.load_eval("khong-co", repo)


# ── xuất xứ ──────────────────────────────────────────────────────────
def test_provenance_reads_the_file_rather_than_hard_coding(repo):
    prov = results.provenance("mbert", repo)
    assert (prov.device, prov.commit, prov.n) == ("mps", "590e775", 500)


def test_provenance_formats_the_date_the_vietnamese_way(repo):
    assert results.provenance("mbert", repo).date == "12/09/2026"


def test_provenance_survives_an_unparseable_timestamp(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "eval_mbert_validation.json").write_text(
        json.dumps(_eval(timestamp="chưa rõ")), encoding="utf-8")
    results._cached_json.cache_clear()
    assert results.provenance("mbert", tmp_path).date == "chưa rõ"


# ── xếp hạng tính từ dữ liệu ─────────────────────────────────────────
def test_best_by_ranks_from_the_data(repo):
    model, value = results.best_by("EM", results.load_all_evals(repo))
    assert (model.id, value) == ("mbert", 50.8)


def test_best_by_follows_the_data_when_another_model_wins(repo):
    """Huấn luyện lại và model khác dẫn đầu thì màn hình phải tự nói đúng."""
    path = repo / "results" / "eval_visobert_validation.json"
    path.write_text(json.dumps(_eval(model="visobert", em=99.0)), encoding="utf-8")
    results._cached_json.cache_clear()
    assert results.best_by("EM", results.load_all_evals(repo))[0].id == "visobert"


def test_best_by_raises_when_there_is_nothing_to_rank():
    with pytest.raises(results.MissingResults):
        results.best_by("EM", {})


# ── đoạn văn cho chip ────────────────────────────────────────────────
def _sample(qid, context, question, impossible=False):
    return {"qid": qid, "question": question, "context": context,
            "gold": [] if impossible else ["x"], "prediction": "",
            "em": 0.0, "f1": 0.0, "is_impossible": impossible}


def test_passages_always_start_with_the_opening_example(repo):
    assert results.passages("mbert", repo)[0].key == "default"


def test_passages_deduplicate_by_chip_label_not_by_context(repo):
    """Split validation có ba đoạn cùng article "Paris"; ba chip cùng tên thì
    người xem không chọn được cái nào."""
    samples = [_sample("q1", "ctx một", "câu một"),
               _sample("q2", "ctx hai", "câu hai")]
    (repo / "results" / "eval_mbert_validation.json").write_text(
        json.dumps(_eval(samples=samples)), encoding="utf-8")
    results._cached_json.cache_clear()
    chips = [p.chip for p in results.passages("mbert", repo)]
    assert len(chips) == len(set(chips))


def test_passages_keep_an_impossible_example(repo):
    """Chip nào cũng trả lời được thì thanh ngưỡng từ chối chẳng có gì để minh hoạ."""
    samples = [_sample(f"q{i}", f"ctx {i}", f"câu {i}") for i in range(6)]
    samples.append(_sample("qimp", "ctx imp", "câu imp", impossible=True))
    (repo / "results" / "eval_mbert_validation.json").write_text(
        json.dumps(_eval(samples=samples)), encoding="utf-8")
    results._cached_json.cache_clear()
    assert any(p.impossible for p in results.passages("mbert", repo))


def test_passages_respect_the_limit(repo):
    samples = [_sample(f"q{i}", f"ctx {i}", f"câu {i}") for i in range(20)]
    (repo / "results" / "eval_mbert_validation.json").write_text(
        json.dumps(_eval(samples=samples)), encoding="utf-8")
    results._cached_json.cache_clear()
    assert len(results.passages("mbert", repo, limit=4)) == 4


def test_impossible_passages_are_labelled_as_such(repo):
    passage = results.Passage("q", "Montréal", "ctx", "câu?", (), impossible=True)
    assert passage.chip == "Montréal · impossible"


def test_passage_reports_its_own_length_and_bucket():
    passage = results.Passage("q", "t", "một hai ba", "câu?")
    assert passage.syllables == 3 and passage.bucket == "<100"


# ── thống kê dataset ─────────────────────────────────────────────────
def test_dataset_stats_is_empty_without_the_raw_files(repo):
    """Thiếu data/raw/ thì trả rỗng để màn hình nói rõ cần chạy fetch_data.py."""
    assert results.dataset_stats(repo) == {}


def test_training_curves_are_empty_when_nothing_was_trained(repo):
    assert results.load_training_curves(repo) == {}


def test_training_curves_are_read_when_present(repo):
    (repo / "results" / "training_curve_mbert.json").write_text(
        json.dumps({"curve": [{"epoch": 1}], "config": {}}), encoding="utf-8")
    results._cached_json.cache_clear()
    assert "mbert" in results.load_training_curves(repo)
