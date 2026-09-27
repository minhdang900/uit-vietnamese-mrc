"""Chọn τ offline: replay từ bản ghi cửa sổ phải TRÙNG KHÍT quyết định online.

Nếu replay lệch online dù chỉ một câu, τ chọn trên dev không còn là τ mà model
thật sự dùng lúc chấm validation — và headline mất nghĩa. Vì vậy ngoài phép so
replay == online trên stub, mỗi điều kiện chính xác (D5 a/b/c) có test riêng.
"""
import json
import random

import pytest

from mrc.threshold import (
    DEFAULT_TAU_GRID,
    replay,
    select_epoch_and_tau,
    select_tau,
    sweep,
    tau_grid,
)
from mrc.windowing import Window


def _w(best, null, start, end, text="x"):
    return {"best_score": best, "null_score": null, "start_char": start,
            "end_char": end, "text": text}


# ── D5 điều kiện chính xác ───────────────────────────────────────────
def test_equal_best_score_in_two_windows_first_window_wins():
    windows = [_w(3.0, 0.0, 0, 5), _w(3.0, 0.0, 10, 15)]
    assert replay(windows, 0.0) == (0, 5)


def test_later_window_wins_only_when_strictly_better():
    windows = [_w(3.0, 0.0, 0, 5), _w(3.0000001, 0.0, 10, 15)]
    assert replay(windows, 0.0) == (10, 15)


@pytest.mark.parametrize("tau", [-100.0, 0.0, 100.0])
def test_window_without_null_score_passes_at_any_tau(tau):
    assert replay([_w(-50.0, None, 2, 4)], tau) == (2, 4)


def test_skipped_window_is_ignored():
    windows = [{"skipped": True}, _w(1.0, 0.0, 7, 9)]
    assert replay(windows, 0.0) == (7, 9)
    assert replay([{"skipped": True}], -100.0) is None


def test_threshold_comparison_is_less_or_equal_like_online():
    # online: best <= null + tau ⇒ từ chối. Đúng bằng biên cũng là từ chối.
    assert replay([_w(2.0, 1.0, 0, 1)], 1.0) is None
    assert replay([_w(2.0, 1.0, 0, 1)], 0.999) == (0, 1)


def test_all_windows_prefer_null_gives_none():
    assert replay([_w(1.0, 5.0, 0, 1), _w(2.0, 5.0, 3, 4)], 0.0) is None


def test_json_round_trip_is_lossless():
    windows = [_w(0.1 + 0.2, 0.30000000000000004, 0, 3)]
    back = json.loads(json.dumps(windows))
    assert back[0]["best_score"] == windows[0]["best_score"]
    assert replay(back, 0.0) == replay(windows, 0.0)


# ── replay == online trên TransformerQA với model stub ───────────────
CONTEXT = "abcdefghijklmnopqrstuvwxyz" * 3


def _make_stub_qa(window_logits, tau):
    """TransformerQA không tải model: cửa sổ và logit do test chỉ định."""
    from mrc.transformer_qa import TransformerQA

    qa = TransformerQA.__new__(TransformerQA)
    qa.max_length, qa.doc_stride, qa.max_answer_len = 16, 8, 4
    qa.null_threshold, qa.tokenizer, qa.name = tau, None, "stub"
    qa._logits = window_logits
    qa._score_window = lambda window: qa._logits[window.input_ids[0]]
    return qa


def _random_case(rng, n_windows=4, n_tokens=10):
    """Logit ngẫu nhiên, có cố ý tạo hoà điểm và cửa sổ không có token context."""
    windows, logits = [], []
    for w in range(n_windows):
        base = w * 5
        offsets = [None] + [(base + i, base + i + 1) for i in range(n_tokens - 2)] + [None]
        if rng.random() < 0.15:
            offsets = [None] * n_tokens  # score_spans -> None ⇒ online bỏ qua
        windows.append(Window(input_ids=[w], attention_mask=[1] * n_tokens,
                              offset_mapping=offsets, context_char_start=base,
                              context_char_end=base + n_tokens))
        # Bội số của 0,5 ⇒ hoà điểm xảy ra thường xuyên, giữa cửa sổ lẫn với null.
        start = [rng.randint(-8, 8) / 2 for _ in range(n_tokens)]
        end = [rng.randint(-8, 8) / 2 for _ in range(n_tokens)]
        # [CLS] được đẩy lên để null thắng khá thường xuyên — cả hai nhánh đều chạy.
        start[0], end[0] = rng.randint(0, 12) / 2, rng.randint(0, 12) / 2
        logits.append((start, end))
    return windows, logits


@pytest.mark.parametrize("tau", [-2.0, 0.0, 2.0])
def test_replay_equals_online_predict_on_stub(monkeypatch, tau):
    import mrc.transformer_qa as tqa

    rng = random.Random(1234)
    checked = answered = 0
    for _ in range(300):
        windows, logits = _random_case(rng)
        monkeypatch.setattr(tqa, "make_windows", lambda *a, _w=windows, **k: _w)
        qa = _make_stub_qa(logits, tau)
        d = qa.predict_detailed(CONTEXT, "q?")
        stored = json.loads(json.dumps(d["windows"]))
        assert len(stored) == len(windows)
        assert replay(stored, tau) == d["span"]
        checked += 1
        answered += d["span"] is not None
    assert checked == 300 and 0 < answered < 300  # cả hai nhánh đều được kiểm


def test_replay_at_zero_equals_predict_with_default_threshold(monkeypatch):
    import mrc.transformer_qa as tqa

    rng = random.Random(7)
    for _ in range(100):
        windows, logits = _random_case(rng, n_windows=3)
        monkeypatch.setattr(tqa, "make_windows", lambda *a, _w=windows, **k: _w)
        qa = _make_stub_qa(logits, 0.0)
        span = replay(qa.predict_detailed(CONTEXT, "q?")["windows"], 0.0)
        expected = "" if span is None else CONTEXT[span[0]:span[1]]
        assert qa.predict(CONTEXT, "q?") == expected


def test_windows_record_text_is_the_decoded_span(monkeypatch):
    import mrc.transformer_qa as tqa

    windows, logits = _random_case(random.Random(3), n_windows=2)
    monkeypatch.setattr(tqa, "make_windows", lambda *a, **k: windows)
    for w in _make_stub_qa(logits, 0.0).predict_detailed(CONTEXT, "q?")["windows"]:
        if not w.get("skipped"):
            assert w["text"] == CONTEXT[w["start_char"]:w["end_char"]]


# ── sweep / select ───────────────────────────────────────────────────
def _rec(qid, gold, windows):
    return {"qid": qid, "gold": gold, "windows": windows}


def _records():
    return [
        _rec("a", ["Hà Nội"], [_w(3.0, 1.0, 0, 6, "Hà Nội")]),          # delta -2
        _rec("b", [], [_w(2.0, 1.5, 0, 3, "sai")]),                       # delta -0.5
        _rec("c", ["Huế"], [_w(1.0, 2.0, 0, 3, "Huế")]),                  # delta +1
        _rec("d", [], [{"skipped": True}]),
    ]


def test_sweep_empty_rate_is_monotone_in_tau():
    rows = sweep(_records(), DEFAULT_TAU_GRID)
    rates = [r["empty_rate"] for r in rows]
    assert rates == sorted(rates)


def test_sweep_scores_match_hand_computation():
    by_tau = {r["tau"]: r for r in sweep(_records(), [-2.0, 0.0, 3.0])}
    # τ=0: a trả lời đúng, b trả lời sai (impossible), c rỗng (sai), d rỗng (đúng)
    assert by_tau[0.0]["EM"] == pytest.approx(50.0)
    assert by_tau[0.0]["empty_rate"] == pytest.approx(50.0)
    # τ=−2: c cũng trả lời (1 > 2 − 2) ⇒ đúng thêm 1
    assert by_tau[-2.0]["EM"] == pytest.approx(75.0)
    # τ=3: mọi câu rỗng ⇒ đúng đúng 2 câu impossible
    assert by_tau[3.0]["EM"] == pytest.approx(50.0)
    assert by_tau[3.0]["empty_rate"] == pytest.approx(100.0)


def test_sweep_at_zero_matches_metrics_evaluate():
    from mrc.metrics import evaluate

    recs = _records()
    preds = {}
    for r in recs:
        span = replay(r["windows"], 0.0)
        preds[r["qid"]] = "" if span is None else next(
            w["text"] for w in r["windows"] if not w.get("skipped")
            and (w["start_char"], w["end_char"]) == span)
    ref = evaluate(preds, {r["qid"]: r["gold"] for r in recs})
    row = sweep(recs, [0.0])[0]
    assert row["EM"] == ref["EM"] and row["F1"] == ref["F1"]


def test_sweep_rejects_records_without_windows():
    with pytest.raises(ValueError, match="windows"):
        sweep([{"qid": "x", "gold": []}], [0.0])


def test_tau_grid_is_exact_and_includes_edges():
    g = tau_grid(-5, 5, 0.25)
    assert len(g) == 41 and g[0] == -5.0 and g[-1] == 5.0 and 0.0 in g


def test_select_tau_picks_max_f1_ties_closest_to_zero():
    rows = [{"tau": -1.0, "F1": 60.0}, {"tau": 0.5, "F1": 60.0},
            {"tau": 2.0, "F1": 59.0}]
    assert select_tau(rows)["tau"] == 0.5


def test_select_tau_symmetric_tie_prefers_negative_deterministically():
    rows = [{"tau": 0.5, "F1": 60.0}, {"tau": -0.5, "F1": 60.0}]
    assert select_tau(rows)["tau"] == -0.5
    assert select_tau(list(reversed(rows)))["tau"] == -0.5


def test_select_epoch_and_tau_is_joint():
    sweeps = {
        1: [{"tau": 0.0, "F1": 50.0}, {"tau": -1.0, "F1": 58.0}],
        2: [{"tau": 0.0, "F1": 55.0}, {"tau": -1.0, "F1": 56.0}],
    }
    # Chọn theo τ=0 sẽ ra epoch 2; chọn CẶP ra epoch 1, τ=−1.
    got = select_epoch_and_tau(sweeps)
    assert (got["epoch"], got["tau"]) == (1, -1.0)


def test_select_epoch_and_tau_ties_prefer_earlier_epoch_then_small_tau():
    sweeps = {
        2: [{"tau": 0.0, "F1": 60.0}],
        1: [{"tau": 1.0, "F1": 60.0}, {"tau": -0.25, "F1": 60.0}],
    }
    got = select_epoch_and_tau(sweeps)
    assert (got["epoch"], got["tau"]) == (1, -0.25)


def test_select_epoch_and_tau_rejects_empty():
    with pytest.raises(ValueError):
        select_epoch_and_tau({})
