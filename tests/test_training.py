"""Logic huấn luyện — tách khỏi CLI để test được mà không cần GPU hay model.

Phiên bản trước của `scripts/finetune.py` gộp tất cả vào một hàm `main()`: đọc dữ
liệu, tính lịch học, vòng lặp train, chọn epoch tốt nhất, ghi file. Không có phần
nào test được, và đúng chỗ đó đã chứa lỗi lấy mẫu thiên lệch làm sai kết quả 10
điểm EM.

Các test dưới đây mô tả API MONG MUỐN: mọi quyết định (lịch học, chọn epoch,
chẩn đoán overfitting) là hàm THUẦN, tách khỏi phần chạy torch.
"""
import pytest

from mrc.training import (
    EpochRecord,
    compute_schedule,
    detect_overfitting,
    select_best_epoch,
    summarise_curve,
)


# ── lịch học: số bước, warmup ────────────────────────────────────────
def test_schedule_steps_per_epoch_accounts_for_grad_accumulation():
    # 100 feature, batch 10 -> 10 batch; grad_accum 2 -> 5 bước optimizer
    s = compute_schedule(n_features=100, batch_size=10, grad_accum=2, epochs=1)
    assert s.steps_per_epoch == 5


def test_schedule_total_steps_multiplies_by_epochs():
    s = compute_schedule(n_features=100, batch_size=10, grad_accum=2, epochs=3)
    assert s.total_steps == 15


def test_schedule_warmup_is_fraction_of_total():
    s = compute_schedule(n_features=1000, batch_size=10, grad_accum=1, epochs=1,
                         warmup_ratio=0.1)
    assert s.warmup_steps == 10          # 10% của 100 bước


def test_schedule_never_returns_zero_steps():
    # dataset nhỏ hơn một batch vẫn phải chạy được ít nhất 1 bước
    s = compute_schedule(n_features=3, batch_size=12, grad_accum=2, epochs=1)
    assert s.steps_per_epoch >= 1 and s.total_steps >= 1


def test_schedule_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError):
        compute_schedule(n_features=100, batch_size=0, grad_accum=1, epochs=1)


def test_schedule_rejects_nonpositive_epochs():
    with pytest.raises(ValueError):
        compute_schedule(n_features=100, batch_size=10, grad_accum=1, epochs=0)


def test_schedule_warmup_ratio_zero_gives_no_warmup():
    s = compute_schedule(n_features=100, batch_size=10, grad_accum=1, epochs=1,
                         warmup_ratio=0.0)
    assert s.warmup_steps == 0


# ── chọn epoch tốt nhất (early stopping thực chất) ───────────────────
def _curve(*triples):
    return [EpochRecord(epoch=i, train_loss=l, val_em=em, val_f1=f1)
            for i, (l, em, f1) in enumerate(triples, start=1)]


def test_select_best_epoch_uses_f1_not_loss():
    # loss thấp nhất ở epoch 3 nhưng F1 cao nhất ở epoch 2 -> phải chọn 2
    c = _curve((3.0, 40, 50), (2.0, 45, 60), (1.0, 38, 48))
    assert select_best_epoch(c) == 2


def test_select_best_epoch_returns_last_when_still_improving():
    c = _curve((3.0, 40, 50), (2.0, 45, 58), (1.0, 52, 60))
    assert select_best_epoch(c) == 3


def test_select_best_epoch_breaks_ties_toward_earlier_epoch():
    # cùng F1 -> chọn epoch SỚM hơn (model đơn giản hơn, ít train hơn)
    c = _curve((3.0, 40, 55.0), (2.0, 44, 55.0))
    assert select_best_epoch(c) == 1


def test_select_best_epoch_on_single_epoch():
    assert select_best_epoch(_curve((2.0, 40, 50))) == 1


def test_select_best_epoch_rejects_empty_curve():
    with pytest.raises(ValueError):
        select_best_epoch([])


# ── chẩn đoán overfitting (đặc tả trong docs/PLAN_TDD.md Phase 6) ─────
def test_detects_overfitting_when_loss_falls_but_val_falls():
    # đây là đường cong THẬT của mBERT-kiểu: loss giảm, val quay đầu
    d = detect_overfitting(_curve((4.64, 28, 28.00), (3.57, 28, 28.44), (2.78, 23, 24.47)))
    assert d["overfitting"] is True
    assert d["best_epoch"] == 2


def test_no_overfitting_when_both_improve():
    d = detect_overfitting(_curve((2.13, 46, 58.39), (1.30, 52, 59.75)))
    assert d["overfitting"] is False
    assert d["best_epoch"] == 2


def test_reports_underfitting_when_val_still_rising_at_end():
    # val vẫn tăng ở epoch cuối -> còn thiếu epoch, không phải overfit
    d = detect_overfitting(_curve((2.13, 46, 58.39), (1.30, 52, 59.75)))
    assert d["still_improving"] is True


def test_still_improving_false_when_val_declined():
    d = detect_overfitting(_curve((3.0, 40, 60.0), (2.0, 38, 55.0)))
    assert d["still_improving"] is False


def test_detect_overfitting_needs_at_least_two_epochs():
    d = detect_overfitting(_curve((2.0, 40, 50)))
    assert d["overfitting"] is False and d["reason"]


# ── phát hiện suy sụp "luôn trả rỗng" ────────────────────────────────
def test_flags_degenerate_collapse_when_em_equals_f1():
    # EM == F1 -> model chỉ sinh đúng-hoàn-toàn hoặc sai-hoàn-toàn = luôn trả rỗng
    d = detect_overfitting(_curve((3.05, 25.67, 25.67), (2.57, 25.33, 25.61)))
    assert d["degenerate_collapse"] is True


def test_no_collapse_flag_when_f1_exceeds_em():
    d = detect_overfitting(_curve((2.13, 46.0, 58.39), (1.30, 52.0, 59.75)))
    assert d["degenerate_collapse"] is False


# ── tóm tắt đường cong để ghi ra file ────────────────────────────────
def test_summarise_curve_is_json_serialisable():
    import json
    s = summarise_curve(_curve((2.13, 46, 58.39), (1.30, 52, 59.75)))
    json.dumps(s)                                   # không được ném lỗi


def test_summarise_curve_includes_every_epoch():
    s = summarise_curve(_curve((3.0, 40, 50), (2.0, 45, 55)))
    assert [r["epoch"] for r in s["curve"]] == [1, 2]


def test_summarise_curve_embeds_diagnosis():
    s = summarise_curve(_curve((4.64, 28, 28.0), (3.57, 28, 28.44), (2.78, 23, 24.47)))
    assert s["diagnosis"]["overfitting"] is True
    assert s["best_epoch"] == 2


def test_epoch_record_rejects_negative_epoch():
    with pytest.raises(ValueError):
        EpochRecord(epoch=0, train_loss=1.0, val_em=1.0, val_f1=1.0)


# ══════════════════════════════════════════════════════════════════════
# QADataset — bọc feature dict thành torch Dataset
# ══════════════════════════════════════════════════════════════════════

def _feats(n=3, seq=8):
    return {
        "input_ids": [[1] * seq for _ in range(n)],
        "attention_mask": [[1] * seq for _ in range(n)],
        "start_positions": [0] * n,
        "end_positions": [1] * n,
    }


def test_dataset_length_matches_feature_count():
    from mrc.training import QADataset
    assert len(QADataset(_feats(5))) == 5


def test_dataset_item_has_exactly_the_keys_the_model_consumes():
    from mrc.training import QADataset
    item = QADataset(_feats())[0]
    assert set(item) == {"input_ids", "attention_mask", "start_positions", "end_positions"}


def test_dataset_returns_long_tensors():
    import torch
    from mrc.training import QADataset
    for v in QADataset(_feats())[0].values():
        assert v.dtype == torch.long      # nhãn vị trí phải là long, không phải float


def test_dataset_rejects_ragged_features():
    from mrc.training import QADataset
    bad = _feats(3)
    bad["start_positions"] = [0, 1]       # thiếu một nhãn
    with pytest.raises(ValueError, match="khớp|length"):
        QADataset(bad)


def test_dataset_rejects_empty_features():
    from mrc.training import QADataset
    with pytest.raises(ValueError):
        QADataset(_feats(0))


# ══════════════════════════════════════════════════════════════════════
# evaluate_checkpoint — ĐÂY là nơi lỗi lấy mẫu thiên lệch đã từng sống.
#
# Bản cũ gọi val_examples[:limit] tức n câu ĐẦU file, làm sai kết quả 10 điểm EM.
# Các test sau khoá chặt hành vi đúng và tiêm predictor vào để test không cần model.
# ══════════════════════════════════════════════════════════════════════

class _StubPredictor:
    """Trả về đúng đáp án vàng đầu tiên — điểm phải là 100."""
    name = "stub"

    def __init__(self, refs):
        self._refs = refs

    def predict(self, context, question):
        return self._refs.get((context, question), "")


def _mk_examples(n):
    from mrc.data import Example
    return [Example(qid=f"q{i}", question=f"Câu hỏi {i}?", context=f"Ngữ cảnh số {i}.",
                    title=f"Bài {i // 10}", answers=[f"đáp án {i}"], answer_start=0)
            for i in range(n)]


def test_evaluate_checkpoint_samples_randomly_not_first_n():
    """Khoá chặt lỗi đã từng xảy ra: KHÔNG được lấy n câu đầu."""
    from mrc.training import evaluate_checkpoint
    ex = _mk_examples(200)
    seen = evaluate_checkpoint(ex, _StubPredictor({}), limit=20, seed=42)["qids"]
    assert seen != [e.qid for e in ex[:20]]


def test_evaluate_checkpoint_is_reproducible_for_same_seed():
    from mrc.training import evaluate_checkpoint
    ex = _mk_examples(100)
    a = evaluate_checkpoint(ex, _StubPredictor({}), limit=10, seed=7)["qids"]
    b = evaluate_checkpoint(ex, _StubPredictor({}), limit=10, seed=7)["qids"]
    assert a == b


def test_evaluate_checkpoint_matches_run_eval_sampling():
    """Đường cong huấn luyện và bảng kết quả cuối PHẢI dùng cùng một mẫu.

    Nếu khác nhau, hai con số trong báo cáo mâu thuẫn và người chấm sẽ hỏi tại sao.
    """
    from mrc.data import reproducible_subset
    from mrc.training import evaluate_checkpoint
    ex = _mk_examples(100)
    got = evaluate_checkpoint(ex, _StubPredictor({}), limit=10, seed=42)["qids"]
    want = [e.qid for e in reproducible_subset(ex, 10, seed=42)]
    assert got == want


def test_evaluate_checkpoint_perfect_predictor_scores_100():
    from mrc.training import evaluate_checkpoint
    ex = _mk_examples(30)
    refs = {(e.context, e.question): e.answers[0] for e in ex}
    r = evaluate_checkpoint(ex, _StubPredictor(refs), limit=10, seed=1)
    assert r["em"] == pytest.approx(100.0) and r["f1"] == pytest.approx(100.0)


def test_evaluate_checkpoint_empty_predictor_scores_0_on_answerable():
    from mrc.training import evaluate_checkpoint
    r = evaluate_checkpoint(_mk_examples(20), _StubPredictor({}), limit=10, seed=1)
    assert r["em"] == 0.0


def test_evaluate_checkpoint_limit_none_uses_everything():
    from mrc.training import evaluate_checkpoint
    assert evaluate_checkpoint(_mk_examples(15), _StubPredictor({}), limit=None)["n"] == 15
