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
