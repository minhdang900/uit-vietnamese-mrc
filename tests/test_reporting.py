"""Sinh hình cho báo cáo TỪ results/*.json.

ĐẶC TẢ: mọi hình phải tái tạo được từ artifact do code sinh. Không có hình "mồ côi"
vẽ tay rồi dán vào báo cáo — đó chính là cách phiên bản trước của dự án trình bày
những con số chưa từng được sinh ra.

Hệ quả: thiếu results thì phải FAIL TO ỒN, không được vẽ hình rỗng.
"""
import json

import pytest

from reporting.figures import (
    FIGURE_RUNS,
    collect_results,
    figure_results,
    format_results_table,
    order_length_buckets,
    unreliable_groups,
)


def _result(model, em, f1, n=100, buckets=None):
    return {
        "model": model, "n": n, "split": "validation", "commit": "abc1234",
        "overall": {"EM": em, "F1": f1, "count": n},
        "answerable_only": {"EM": em, "F1": f1, "count": 70},
        "impossible_only": {"EM": 20.0, "count": 30},
        "avg_latency_ms": 1.5,
        "by_context_length": buckets or {
            "100-200": {"EM": em, "F1": f1, "count": 80, "unreliable": False},
            "300+": {"EM": em, "F1": f1, "count": 3, "unreliable": True},
        },
        "by_question_type": {
            "single-sentence": {"EM": em, "F1": f1, "count": 40, "unreliable": False},
            "_note": "ghi chú",
        },
    }


# ── đọc kết quả ──────────────────────────────────────────────────────
def test_collect_results_fails_loudly_when_directory_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval_"):
        collect_results(tmp_path)


def test_collect_results_reads_every_eval_file(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"eval_{name}.json").write_text(json.dumps(_result(name, 10, 20)))
    assert len(collect_results(tmp_path)) == 2


def test_collect_results_ignores_non_eval_files(tmp_path):
    (tmp_path / "eval_a.json").write_text(json.dumps(_result("a", 10, 20)))
    (tmp_path / "training_curve_x.json").write_text(json.dumps({"curve": []}))
    (tmp_path / "hypotheses.json").write_text(json.dumps({}))
    assert [r["model"] for r in collect_results(tmp_path)] == ["a"]


def test_collect_results_sorted_by_score_ascending(tmp_path):
    (tmp_path / "eval_hi.json").write_text(json.dumps(_result("hi", 50, 60)))
    (tmp_path / "eval_lo.json").write_text(json.dumps(_result("lo", 5, 10)))
    assert [r["model"] for r in collect_results(tmp_path)] == ["lo", "hi"]


# ── thứ tự bucket ────────────────────────────────────────────────────
def test_length_buckets_are_ordered_by_size_not_alphabetically():
    # sắp xếp chữ cái sẽ cho "<100" sau "300+" và biểu đồ trở nên vô nghĩa
    assert order_length_buckets(["300+", "<100", "200-300", "100-200"]) == \
           ["<100", "100-200", "200-300", "300+"]


def test_order_length_buckets_ignores_unknown_labels():
    assert order_length_buckets(["300+", "linh tinh", "<100"]) == ["<100", "300+"]


# ── đánh dấu nhóm không đáng tin ─────────────────────────────────────
def test_unreliable_groups_are_identified():
    r = _result("m", 10, 20)
    assert unreliable_groups(r["by_context_length"]) == ["300+"]


def test_no_unreliable_groups_when_all_large():
    groups = {"100-200": {"EM": 1, "F1": 2, "count": 80, "unreliable": False}}
    assert unreliable_groups(groups) == []


def test_unreliable_ignores_the_note_key():
    groups = {"a": {"count": 3, "unreliable": True}, "_note": "văn bản"}
    assert unreliable_groups(groups) == ["a"]


# ── bảng kết quả cho báo cáo ─────────────────────────────────────────
def test_results_table_has_one_row_per_model(tmp_path):
    rows = format_results_table([_result("a", 10, 20), _result("b", 30, 40)])
    assert len(rows) == 2


def test_results_table_includes_n_with_every_row(tmp_path):
    for row in format_results_table([_result("a", 10, 20)]):
        assert "n" in row and row["n"] == 100


def test_results_table_separates_answerable_from_impossible(tmp_path):
    row = format_results_table([_result("a", 10, 20)])[0]
    assert "answerable_EM" in row and "impossible_EM" in row


# ── run nào được vẽ (FIGURE_RUNS) ────────────────────────────────────
def _write(results, name, em, **extra):
    data = {**_result(name, em, em + 5), **extra}
    (results / f"eval_{name}_validation.json").write_text(json.dumps(data), encoding="utf-8")


def test_figures_plot_only_whitelisted_runs(tmp_path):
    """Cộng thêm eval_*.json (run seed, run lạ) không được lặng lẽ chui vào hình."""
    for name, em in (("baseline", 1.0), ("xlmr", 40.0), ("mbert", 50.0),
                     ("visobert", 27.0), ("mbert-dev-s43", 49.0), ("scratch", 99.0)):
        _write(tmp_path, name, em)

    runs = [r["_run_id"] for r in figure_results(tmp_path)]

    assert runs == ["baseline", "visobert", "xlmr", "mbert"], "chỉ run trong whitelist, EM tăng dần"


def test_run_id_field_wins_over_the_filename(tmp_path):
    _write(tmp_path, "renamed", 30.0, run_id="xlmr")

    assert [r["_run_id"] for r in figure_results(tmp_path)] == ["xlmr"]


def test_dev_selected_runs_replace_their_v1_counterparts(tmp_path):
    """Có ``mbert-dev`` thì hình tiêu đề vẽ nó, không vẽ thêm ``mbert`` v1 (D2)."""
    for name, em in (("mbert", 50.0), ("mbert-dev", 55.0), ("visobert", 27.0),
                     ("empty", 28.0)):
        _write(tmp_path, name, em)

    runs = [r["_run_id"] for r in figure_results(tmp_path)]

    assert runs == ["visobert", "empty", "mbert-dev"]
    assert "empty" in FIGURE_RUNS, "N7: baseline luôn-trả-rỗng thuộc bảng/hình tiêu đề"


def test_figure_results_fails_loudly_when_no_whitelisted_run(tmp_path):
    _write(tmp_path, "scratch", 10.0)
    with pytest.raises(FileNotFoundError, match="FIGURE_RUNS"):
        figure_results(tmp_path)
