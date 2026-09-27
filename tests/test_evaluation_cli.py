"""CLI đánh giá — chọn model, lấy mẫu, ghi kết quả.

ĐẶC TẢ: chạy một hoặc nhiều model trên một split, ghi ra ``results/eval_<model>.json``.

Hai quyết định ở đây từng gây lỗi thật và được khoá chặt bằng test:

1. **Lấy mẫu ngẫu nhiên, tái lập được** — không phải n câu đầu file. Các câu đầu
   thuộc vài article đầu, nên mẫu thiên lệch theo chủ đề; lỗi này từng làm sai 10
   điểm EM.
2. **``max_answer_len`` phụ thuộc tokenizer** — nó giới hạn span theo TOKEN, mà số
   token cho cùng một chuỗi phụ thuộc vocab. Một hằng số chung là sai: đo trên
   4.000 gold answer, p95 là 40 token với mBERT nhưng 64 với ViSoBERT (vocab 15k).
"""
import pytest

from mrc.data import Example


def _examples(n):
    return [Example(qid=f"q{i}", question=f"Câu hỏi {i}?", context=f"Ngữ cảnh {i}.",
                    title=f"Bài {i // 10}", answers=[f"đáp án {i}"], answer_start=0)
            for i in range(n)]


# ── lấy mẫu ──────────────────────────────────────────────────────────
def test_subset_is_reproducible_for_a_given_seed():
    from evaluation.cli import subset
    ex = _examples(100)
    assert [e.qid for e in subset(ex, 10, seed=7)] == [e.qid for e in subset(ex, 10, seed=7)]


def test_subset_does_not_take_the_first_n():
    from evaluation.cli import subset
    ex = _examples(200)
    assert [e.qid for e in subset(ex, 20, seed=42)] != [e.qid for e in ex[:20]]


def test_subset_matches_the_sampler_used_by_training_curves():
    """Đường cong huấn luyện và bảng cuối phải dùng CÙNG mẫu, nếu không hai con số
    trong báo cáo mâu thuẫn nhau."""
    from mrc.data import reproducible_subset
    from evaluation.cli import subset
    ex = _examples(100)
    assert [e.qid for e in subset(ex, 10, seed=42)] == \
           [e.qid for e in reproducible_subset(ex, 10, seed=42)]


def test_subset_none_returns_everything():
    from evaluation.cli import subset
    assert len(subset(_examples(40), None)) == 40


def test_subset_larger_than_population_returns_everything():
    from evaluation.cli import subset
    assert len(subset(_examples(5), 500)) == 5


def test_subset_does_not_mutate_input():
    from evaluation.cli import subset
    ex = _examples(50)
    before = [e.qid for e in ex]
    subset(ex, 10)
    assert [e.qid for e in ex] == before


def test_subset_samples_across_many_articles():
    from evaluation.cli import subset
    assert len({e.title for e in subset(_examples(200), 30, seed=42)}) > 5


# ── max_answer_len phụ thuộc tokenizer ───────────────────────────────
def test_visobert_gets_a_larger_max_answer_len():
    from evaluation.cli import max_answer_len_for
    assert max_answer_len_for("visobert") > max_answer_len_for("mbert")


def test_default_max_answer_len_applies_to_unknown_models():
    from evaluation.cli import DEFAULT_MAX_ANSWER_LEN, max_answer_len_for
    assert max_answer_len_for("some-new-model") == DEFAULT_MAX_ANSWER_LEN


def test_explicit_override_wins_over_per_model_default():
    from evaluation.cli import max_answer_len_for
    assert max_answer_len_for("visobert", override=7) == 7


# ── phân giải model ──────────────────────────────────────────────────
def test_baseline_builds_without_touching_disk():
    from evaluation.cli import build_predictor
    p = build_predictor("baseline")
    assert p.predict("Một câu. Hai câu.", "câu?") in "Một câu. Hai câu."


def test_unknown_finetuned_model_fails_loudly():
    from evaluation.cli import build_predictor
    with pytest.raises(SystemExit, match="finetune|models/"):
        build_predictor("khong-ton-tai-tren-dia")


# ── phân tích tham số ────────────────────────────────────────────────
def test_parse_args_defaults_to_validation_split():
    from evaluation.cli import parse_args
    assert parse_args(["--models", "baseline"]).split == "validation"


def test_parse_args_full_flag_disables_limit():
    from evaluation.cli import parse_args, resolve_limit
    assert resolve_limit(parse_args(["--models", "baseline", "--full"])) is None


def test_parse_args_limit_is_used_when_not_full():
    from evaluation.cli import parse_args, resolve_limit
    assert resolve_limit(parse_args(["--models", "baseline", "--limit", "42"])) == 42


def test_parse_args_accepts_several_models():
    from evaluation.cli import parse_args
    assert parse_args(["--models", "baseline", "xlmr"]).models == ["baseline", "xlmr"]


# ── cấu hình suy luận theo run (selection.json) ──────────────────────
class _SpyQA:
    """Thay TransformerQA: ghi lại tham số, không tải model."""

    def __init__(self, model_name, **kwargs):
        self.model_name, self.kwargs = model_name, kwargs
        self.null_threshold = kwargs.get("null_threshold")
        self.name = kwargs.get("name")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    import json

    import mrc.transformer_qa as tqa

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tqa, "TransformerQA", _SpyQA)

    def make(kind, selection=None):
        d = tmp_path / "models" / kind
        d.mkdir(parents=True)
        if selection is not None:
            (d / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return make


def _selection(**over):
    base = {"epoch": 2, "tau": -0.75, "selected_on": "dev", "max_length": 384,
            "doc_stride": 128, "max_answer_len": 64}
    base.update(over)
    return base


def test_selection_json_sets_window_config_for_len512(models_dir):
    from evaluation.cli import build_predictor
    models_dir("visobert-len512", _selection(max_length=512))
    kw = build_predictor("visobert-len512").kwargs
    assert (kw["max_length"], kw["doc_stride"], kw["max_answer_len"]) == (512, 128, 64)
    assert kw["null_threshold"] == -0.75


def test_selection_json_for_visobert_dev(models_dir):
    from evaluation.cli import build_predictor
    models_dir("visobert-dev", _selection())
    kw = build_predictor("visobert-dev").kwargs
    assert (kw["max_length"], kw["doc_stride"], kw["max_answer_len"]) == (384, 128, 64)


def test_legacy_checkpoint_without_selection_keeps_old_config(models_dir):
    from evaluation.cli import build_predictor, resolve_inference_config
    models_dir("mbert")
    kw = build_predictor("mbert").kwargs
    assert (kw["max_length"], kw["doc_stride"], kw["max_answer_len"]) == (384, 128, 30)
    assert kw["null_threshold"] == 0.0
    assert resolve_inference_config("mbert")["selected_on"] == "validation_subset_300"


def test_legacy_visobert_keeps_its_table_entry(models_dir):
    from evaluation.cli import build_predictor
    models_dir("visobert")
    assert build_predictor("visobert").kwargs["max_answer_len"] == 64


def test_cli_override_beats_selection_json(models_dir):
    from evaluation.cli import build_predictor, resolve_inference_config
    models_dir("visobert-dev", _selection())
    kw = build_predictor("visobert-dev", max_answer_len=40, null_threshold=1.0,
                         max_length=256, doc_stride=64).kwargs
    assert (kw["max_length"], kw["doc_stride"], kw["max_answer_len"],
            kw["null_threshold"]) == (256, 64, 40, 1.0)
    src = resolve_inference_config("visobert-dev", null_threshold=1.0)["source"]
    assert src["tau"] == "cli" and src["max_answer_len"] == "selection.json"


def test_empty_kind_always_answers_empty():
    from evaluation.cli import build_predictor
    p = build_predictor("empty")
    assert p.predict("Hà Nội là thủ đô.", "Thủ đô?") == ""


# ── chấm-một-lần (C5) ────────────────────────────────────────────────
def _write_split(tmp_path):
    import json
    data = {"version": 2.0, "data": [{"title": "T", "paragraphs": [{
        "context": "Hà Nội là thủ đô của Việt Nam.",
        "qas": [{"id": "q1", "question": "Thủ đô?", "is_impossible": False,
                 "answers": {"text": ["Hà Nội"], "answer_start": [0]}},
                {"id": "q2", "question": "GDP?", "is_impossible": True,
                 "answers": {"text": [], "answer_start": []}}]}]}]}
    d = tmp_path / "data"
    d.mkdir()
    (d / "viquad2_validation.json").write_text(json.dumps(data), encoding="utf-8")
    return d


def _run(tmp_path, *extra):
    from evaluation.cli import main
    data = _write_split(tmp_path) if not (tmp_path / "data").exists() else tmp_path / "data"
    main(["--models", "empty", "--full", "--data-dir", str(data),
          "--out-dir", str(tmp_path / "out"), *extra])


def test_eval_writes_json_preds_and_invocation_log(tmp_path):
    import json
    _run(tmp_path)
    out = tmp_path / "out"
    r = json.loads((out / "eval_empty_validation.json").read_text(encoding="utf-8"))
    assert r["run_id"] == "empty" and r["n"] == 2
    assert len((out / "preds_empty_validation.jsonl").read_text().splitlines()) == 2
    log = [json.loads(x) for x in (out / "eval_invocations.jsonl").read_text().splitlines()]
    assert log[0]["run_ids"] == ["empty"] and log[0]["forced"] is False
    assert {"utc", "argv", "commit", "reason"} <= set(log[0])


def test_second_eval_without_force_is_refused_and_file_untouched(tmp_path):
    import json
    _run(tmp_path)
    target = tmp_path / "out" / "eval_empty_validation.json"
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        _run(tmp_path)
    assert target.read_bytes() == before
    log = (tmp_path / "out" / "eval_invocations.jsonl").read_text().splitlines()
    assert len(log) == 2 and json.loads(log[1])["refused"]


def test_force_requires_a_reason(tmp_path):
    _run(tmp_path)
    with pytest.raises(SystemExit, match="reason"):
        _run(tmp_path, "--force")


def test_force_with_reason_overwrites_and_is_logged(tmp_path):
    import json
    _run(tmp_path)
    _run(tmp_path, "--force", "--reason", "bug X")
    log = [json.loads(x) for x in
           (tmp_path / "out" / "eval_invocations.jsonl").read_text().splitlines()]
    assert log[-1]["forced"] is True and log[-1]["reason"] == "bug X"


def test_no_preds_flag_skips_jsonl(tmp_path):
    _run(tmp_path, "--no-preds")
    assert not (tmp_path / "out" / "preds_empty_validation.jsonl").exists()
