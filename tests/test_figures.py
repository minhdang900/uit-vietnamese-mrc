"""Phase 7 — hình vẽ. Bất biến: mọi hình tái tạo được TỪ results/*.json.

Không có hình "mồ côi" nào được vẽ tay rồi dán vào báo cáo.
"""
import json

import pytest

from scripts.make_figures import load_evals, fig_model_comparison


def _fake_eval(model, em, f1, n=100):
    return {
        "model": model, "n": n, "split": "validation", "commit": "abc1234",
        "overall": {"EM": em, "F1": f1, "count": n},
        "answerable_only": {"EM": em, "F1": f1, "count": 70},
        "impossible_only": {"EM": 20.0, "count": 30},
        "by_context_length": {"100-200": {"EM": em, "F1": f1, "count": 80, "unreliable": False},
                              "300+": {"EM": em, "F1": f1, "count": 3, "unreliable": True}},
        "by_question_type": {"single-sentence": {"EM": em, "F1": f1, "count": 40, "unreliable": False},
                             "_note": "ghi chú"},
    }


def test_load_evals_fails_loudly_when_results_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval_"):
        load_evals(tmp_path)


def test_figure_is_generated_from_results(tmp_path):
    (tmp_path / "eval_a.json").write_text(json.dumps(_fake_eval("A", 10.0, 20.0)))
    out = tmp_path / "cmp.png"
    fig_model_comparison(load_evals(tmp_path), out)
    assert out.exists() and out.stat().st_size > 5_000


def test_figure_regenerates_after_deletion(tmp_path):
    (tmp_path / "eval_a.json").write_text(json.dumps(_fake_eval("A", 10.0, 20.0)))
    out = tmp_path / "cmp.png"
    evals = load_evals(tmp_path)
    fig_model_comparison(evals, out)
    out.unlink()
    fig_model_comparison(evals, out)
    assert out.exists()
