"""Phase 2.2 — scripts/compute_stats.py: ``stats_{split}.json`` từ preds JSONL.

Chưa có preds JSONL thật (P2.3 sẽ tạo, chạy inference thật) — mọi test ở đây
dựng MỘT bộ fixture nhỏ, đúng schema ``mrc.evaluate.build_records`` (qid,
paragraph_id, title, is_impossible, gold, pred, em, f1[, windows]), rồi ghi ra
``preds_{run_id}_{split}.jsonl`` bằng ``mrc.evaluate.write_jsonl`` — không phụ
thuộc model hay tokenizer thật.
"""
from __future__ import annotations

import json

import pytest

from mrc.evaluate import write_jsonl
from scripts.compute_stats import build_stats, compute_pair_stats, compute_run_stats, discover_pred_files


def _record(qid, paragraph_id, em, f1, is_impossible=False, pred="x"):
    return {
        "qid": qid,
        "paragraph_id": paragraph_id,
        "title": "T",
        "is_impossible": is_impossible,
        "gold": [] if is_impossible else ["vàng"],
        "pred": "" if (is_impossible and em == 1.0) else pred,
        "em": em,
        "f1": f1,
        "null_delta": None,
    }


def _fixture_records(correct_qids, n=20, n_paragraphs=5):
    """``n`` câu hỏi rải đều trên ``n_paragraphs`` đoạn văn; câu trong
    ``correct_qids`` được điểm EM=F1=1.0, còn lại 0.0. 2 câu cuối là impossible."""
    out = []
    for i in range(n):
        qid = f"q{i}"
        para = f"p{i % n_paragraphs}"
        impossible = i >= n - 2
        correct = qid in correct_qids
        em = 1.0 if correct else 0.0
        f1 = 1.0 if correct else (0.0 if impossible else 0.3)
        out.append(_record(qid, para, em, f1, is_impossible=impossible))
    return out


def _write_run(results_dir, run_id, split, records):
    write_jsonl(results_dir / f"preds_{run_id}_{split}.jsonl", records)


# ── discover_pred_files ──────────────────────────────────────────────────


def test_discover_pred_files_parses_run_id(tmp_path):
    (tmp_path / "preds_mbert-dev_validation.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "preds_xlmr_validation.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "preds_mbert_dev.jsonl").write_text("", encoding="utf-8")  # split khác, bỏ qua

    found = discover_pred_files(tmp_path, "validation")

    assert set(found) == {"mbert-dev", "xlmr"}


# ── compute_run_stats ────────────────────────────────────────────────────


def test_compute_run_stats_schema_and_values():
    correct = {f"q{i}" for i in range(0, 20, 2)}  # 10/20 đúng
    records = _fixture_records(correct)

    stats = compute_run_stats(records, n_boot=200, seed=0)

    assert stats["n"] == 20
    assert stats["EM"] == pytest.approx(50.0)
    assert 0.0 <= stats["empty_rate"] <= 100.0
    for key in ("EM_wilson", "EM_cluster_ci", "F1_cluster_ci"):
        lo, hi = stats[key]
        assert lo <= hi
    assert "has_ans" in stats and "no_ans" in stats
    assert stats["has_ans"]["n"] == 18
    assert stats["no_ans"]["n"] == 2
    assert "F1" in stats["has_ans"]
    assert "F1" not in stats["no_ans"]  # hợp đồng: F1 không có ý nghĩa riêng cho no_ans


def test_compute_run_stats_no_impossible_questions_omits_no_ans():
    records = [_record(f"q{i}", f"p{i % 3}", 1.0, 1.0) for i in range(6)]
    stats = compute_run_stats(records, n_boot=100, seed=0)
    assert "no_ans" not in stats
    assert "has_ans" in stats


def test_compute_run_stats_empty_records_raises():
    with pytest.raises(ValueError):
        compute_run_stats([], n_boot=100, seed=0)


# ── compute_pair_stats ───────────────────────────────────────────────────


def test_compute_pair_stats_mcnemar_counts_and_ci():
    # a đúng {q0..q9}, b đúng {q5..q14} trên 20 câu ⇒ b10 (a đúng,b sai) = |{0..4}| = 5;
    # b01 (a sai,b đúng) = |{10..14}| = 5.
    a_correct = {f"q{i}" for i in range(0, 10)}
    b_correct = {f"q{i}" for i in range(5, 15)}
    records_a = _fixture_records(a_correct)
    records_b = _fixture_records(b_correct)

    pair = compute_pair_stats(records_a, records_b, n_boot=200, seed=0)

    assert pair["b10"] == 5
    assert pair["b01"] == 5
    assert pair["p_mcnemar"] == pytest.approx(1.0)  # b01 == b10 ⇒ p=1
    assert pair["dEM"] == pytest.approx(0.0)
    for key in ("dEM_cluster_ci", "dF1_cluster_ci"):
        lo, hi = pair[key]
        assert lo <= hi


def test_compute_pair_stats_rejects_mismatched_question_sets():
    records_a = _fixture_records({"q0"}, n=10)
    records_b = [r for r in _fixture_records({"q0"}, n=10) if r["qid"] != "q9"]
    with pytest.raises(ValueError, match="qid"):
        compute_pair_stats(records_a, records_b)


def test_compute_pair_stats_tolerates_different_file_order():
    """Hai file JSONL có thể ghi câu hỏi theo thứ tự khác nhau — compute_pair_stats
    tự ghép lại theo qid, không cần thứ tự dòng khớp nhau."""
    a_correct = {f"q{i}" for i in range(0, 10)}
    records_a = _fixture_records(a_correct)
    records_b = list(reversed(_fixture_records(a_correct)))  # cùng nội dung, đảo thứ tự

    pair = compute_pair_stats(records_a, records_b, n_boot=100, seed=0)

    assert pair["b01"] == 0 and pair["b10"] == 0
    assert pair["dEM"] == pytest.approx(0.0)


# ── build_stats / CLI end-to-end ─────────────────────────────────────────


def test_build_stats_matches_reporting_contract_shape(tmp_path):
    a_correct = {f"q{i}" for i in range(0, 10)}
    b_correct = {f"q{i}" for i in range(5, 15)}
    _write_run(tmp_path, "mbert", "validation", _fixture_records(a_correct))
    _write_run(tmp_path, "xlmr", "validation", _fixture_records(b_correct))

    payload = build_stats(tmp_path, split="validation", n_boot=150, seed=0)

    for key in ("split", "commit", "timestamp", "B", "seed", "cluster", "n_clusters",
                "runs", "pairs"):
        assert key in payload
    assert payload["split"] == "validation"
    assert payload["cluster"] == "paragraph_id"
    assert payload["B"] == 150
    assert set(payload["runs"]) == {"mbert", "xlmr"}
    for run in payload["runs"].values():
        for key in ("n", "EM", "F1", "empty_rate", "EM_wilson", "EM_cluster_ci", "F1_cluster_ci"):
            assert key in run
    assert payload["pairs"], "cặp mbert/xlmr phải xuất hiện — cả hai run đều có mặt"
    pair = payload["pairs"][0]
    assert pair["a"] == "mbert" and pair["b"] == "xlmr"
    for key in ("b01", "b10", "p_mcnemar", "dEM", "dEM_cluster_ci", "dF1_cluster_ci"):
        assert key in pair


def test_build_stats_skips_pairs_not_yet_available(tmp_path):
    """Chỉ mbert có preds (P3 chưa chạy) ⇒ không cặp nào, không lỗi (D2)."""
    _write_run(tmp_path, "mbert", "validation", _fixture_records({"q0"}, n=6, n_paragraphs=2))

    payload = build_stats(tmp_path, split="validation", n_boot=50, seed=0)

    assert set(payload["runs"]) == {"mbert"}
    assert payload["pairs"] == []


def test_build_stats_raises_when_no_preds_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_stats(tmp_path, split="validation")


def test_build_stats_is_reproducible_given_fixed_seed(tmp_path):
    _write_run(tmp_path, "mbert", "validation", _fixture_records({f"q{i}" for i in range(10)}))

    p1 = build_stats(tmp_path, split="validation", n_boot=100, seed=42)
    p2 = build_stats(tmp_path, split="validation", n_boot=100, seed=42)

    assert p1["runs"]["mbert"]["EM_cluster_ci"] == p2["runs"]["mbert"]["EM_cluster_ci"]


def test_main_writes_stats_json(tmp_path):
    import scripts.compute_stats as compute_stats_mod

    _write_run(tmp_path, "mbert", "validation", _fixture_records({f"q{i}" for i in range(10)}))

    rc = compute_stats_mod.main(["--results-dir", str(tmp_path), "--split", "validation",
                                "--n-boot", "50"])

    assert rc == 0
    out = json.loads((tmp_path / "stats_validation.json").read_text(encoding="utf-8"))
    assert out["runs"]["mbert"]["n"] == 20


def test_main_returns_1_and_message_when_no_preds(tmp_path, capsys):
    import scripts.compute_stats as compute_stats_mod

    rc = compute_stats_mod.main(["--results-dir", str(tmp_path)])

    assert rc == 1
    assert "preds_" in capsys.readouterr().out
