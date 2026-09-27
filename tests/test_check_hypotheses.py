"""Đối chiếu giả thuyết: khớp run_id chính xác, cổng dự đoán, báo động v2,
và kiểm lại thứ tự đăng ký < huấn luyện < chấm từ git."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ch():
    spec = importlib.util.spec_from_file_location("check_hypotheses_script",
                                                  ROOT / "scripts" / "check_hypotheses.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval(run_id, em, f1, has_em=None, empty=10.0, model=None, ts="2026-09-28T12:00:00+00:00",
          n_imp=30, n=100):
    r = {"model": model or f"{run_id} (fine-tuned)", "n": n, "timestamp": ts,
         "overall": {"EM": em, "F1": f1, "count": n},
         "answerable_only": {"EM": has_em if has_em is not None else em, "F1": f1,
                             "count": n - n_imp},
         "impossible_only": {"EM": 90.0, "count": n_imp},
         "empty_prediction_rate": empty}
    if run_id is not None:
        r["run_id"] = run_id
    return r


SPEC = {
    "bands": {"mbert": {"EM": [50, 60], "F1": [68, 78]},
              "visobert": {"EM": [60, 70], "F1": [78, 85]},
              "TF-IDF Baseline": {"EM": [0, 3], "F1": [20, 30]}},
    "alarms": {"degenerate_all_empty": {"kind": "em_equals_f1", "tolerance": 0.5,
                                        "meaning": "suy sụp"}},
    "registrations": {"P3": {
        "run_ids": ["mbert-dev", "visobert-dev"],
        "bands": {"mbert-dev": {"EM": [45, 60], "F1": [55, 70]}},
        "predictions": [{
            "id": "visobert-dev-collapse", "run_id": "visobert-dev",
            "claim": "visobert-dev vẫn suy sụp", "expected": True,
            "gate": {"any": [{"metric": "empty_rate", "op": ">=", "value": 90},
                             {"metric": "HasAns_EM", "op": "<", "value": 15}]}}],
        "alarms": {
            "collapse": {"kind": "empty_rate_above", "max": 95, "meaning": "suy sụp"},
            "dev_gap": {"kind": "dev_val_gap", "max": 10, "meaning": "dev không đại diện"},
            "edge": {"kind": "tau_at_grid_edge", "meaning": "lưới hẹp"}}}},
}


def _setup(tmp_path, evals, spec=SPEC, git=False):
    res = tmp_path / "results"
    res.mkdir()
    (res / "hypotheses.json").write_text(json.dumps(spec), encoding="utf-8")
    (res / "hypotheses.md").write_text("#\n", encoding="utf-8")
    if git:
        for args in (["init", "-q"], ["config", "user.email", "t@e.com"],
                     ["config", "user.name", "t"], ["add", "results"],
                     ["commit", "-q", "-m", "prereg"]):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    for r in evals:
        (res / f"eval_{r.get('run_id') or r['model'][:5]}_validation.json").write_text(
            json.dumps(r), encoding="utf-8")
    return res


def _head(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def _curve(res, run_id, commit, started):
    (res / f"training_curve_{run_id}.json").write_text(json.dumps(
        {"config": {"prereg_commit": commit, "started_utc": started}}), encoding="utf-8")


# ── khớp dải ─────────────────────────────────────────────────────────
def test_len512_is_not_graded_against_visobert_band(ch):
    assert ch.find_band(_eval("visobert-len512", 1, 2), SPEC) is None


def test_v2_run_uses_its_registered_band(ch):
    assert ch.find_band(_eval("mbert-dev", 1, 2), SPEC) == {"EM": [45, 60], "F1": [55, 70]}


def test_v1_run_with_run_id_matches_exact_and_alias(ch):
    assert ch.find_band(_eval("mbert", 1, 2), SPEC) == SPEC["bands"]["mbert"]
    assert ch.find_band(_eval("baseline", 1, 2), SPEC) == SPEC["bands"]["TF-IDF Baseline"]


def test_old_eval_without_run_id_falls_back_to_substring(ch):
    assert ch.find_band(_eval(None, 1, 2, model="visobert (fine-tuned)"), SPEC) \
        == SPEC["bands"]["visobert"]


# ── cổng dự đoán ─────────────────────────────────────────────────────
def test_collapse_prediction_confirmed_by_hasans_em(ch, tmp_path):
    res = _setup(tmp_path, [_eval("visobert-dev", 28, 31, has_em=6, empty=70)])
    _, report = ch.check(res, repo=tmp_path)
    assert report["predictions"][0]["verdict"] == "CONFIRMED"


def test_collapse_prediction_refuted_when_model_answers(ch, tmp_path):
    res = _setup(tmp_path, [_eval("visobert-dev", 55, 65, has_em=50, empty=30)])
    _, report = ch.check(res, repo=tmp_path)
    assert report["predictions"][0]["verdict"] == "REFUTED"
    assert json.loads((res / "hypotheses_report.json").read_text())["predictions"]


def test_prediction_pending_without_eval(ch, tmp_path):
    res = _setup(tmp_path, [_eval("mbert", 50, 60)])
    _, report = ch.check(res, repo=tmp_path)
    assert report["predictions"][0]["verdict"] == "PENDING"


# ── báo động v2 ──────────────────────────────────────────────────────
def _alarms(report):
    return " ".join(a["message"] for a in report["alarms"])


def test_empty_rate_above_fires(ch, tmp_path):
    res = _setup(tmp_path, [_eval("visobert-dev", 30, 35, empty=97)])
    code, report = ch.check(res, repo=tmp_path)
    assert code == 1 and "[collapse]" in _alarms(report)


def test_dev_val_gap_and_tau_edge_fire_from_threshold_file(ch, tmp_path):
    res = _setup(tmp_path, [_eval("mbert-dev", 50, 60)])
    (res / "threshold_mbert-dev.json").write_text(json.dumps({"selection": {
        "dev_metrics": {"EM": 65.0}, "tau": -5.0, "tau_at_grid_edge": True}}))
    _, report = ch.check(res, repo=tmp_path)
    assert "[dev_gap]" in _alarms(report) and "[edge]" in _alarms(report)


def test_empty_predictor_is_exempt_from_collapse_alarms(ch, tmp_path):
    res = _setup(tmp_path, [_eval("empty", 30, 30, empty=100, model="Luôn trả rỗng (empty)")])
    code, report = ch.check(res, repo=tmp_path)
    assert code == 0 and not report["alarms"]


# ── thứ tự đăng ký < huấn luyện < chấm ───────────────────────────────
def test_prereg_order_ok(ch, tmp_path):
    res = _setup(tmp_path, [], git=True)
    head = _head(tmp_path)
    _curve(res, "mbert-dev", head, "2099-01-01T00:00:00+00:00")
    (res / "eval_mbert-dev_validation.json").write_text(json.dumps(
        _eval("mbert-dev", 50, 60, ts="2099-01-02T00:00:00+00:00")))
    code, report = ch.check(res, repo=tmp_path)
    assert "prereg_order_violated" not in _alarms(report)


def test_prereg_order_fails_when_timestamps_swapped(ch, tmp_path):
    res = _setup(tmp_path, [], git=True)
    head = _head(tmp_path)
    _curve(res, "mbert-dev", head, "2099-01-02T00:00:00+00:00")
    (res / "eval_mbert-dev_validation.json").write_text(json.dumps(
        _eval("mbert-dev", 50, 60, ts="2099-01-01T00:00:00+00:00")))
    code, report = ch.check(res, repo=tmp_path)
    assert code == 1 and "prereg_order_violated" in _alarms(report)


def test_prereg_order_fails_when_training_started_before_registration(ch, tmp_path):
    res = _setup(tmp_path, [], git=True)
    _curve(res, "mbert-dev", _head(tmp_path), "2000-01-01T00:00:00+00:00")
    (res / "eval_mbert-dev_validation.json").write_text(json.dumps(
        _eval("mbert-dev", 50, 60, ts="2099-01-01T00:00:00+00:00")))
    code, report = ch.check(res, repo=tmp_path)
    assert code == 1 and "prereg_order_violated" in _alarms(report)


def test_prereg_order_fails_without_training_curve(ch, tmp_path):
    res = _setup(tmp_path, [_eval("mbert-dev", 50, 60)], git=True)
    code, report = ch.check(res, repo=tmp_path)
    assert code == 1 and "prereg_order_violated" in _alarms(report)


def test_relative_gate_against_reference_run(ch):
    gate = {"metric": "EM", "op": "abs_le", "ref_run": "mbert", "value": 3}
    runs = {"mbert": _eval("mbert", 50, 60)}
    assert ch.evaluate_gate(gate, _eval("mbert-dev", 52.5, 61), runs)
    assert not ch.evaluate_gate(gate, _eval("mbert-dev", 46.0, 61), runs)
    with pytest.raises(ch.MissingReference):
        ch.evaluate_gate(gate, _eval("mbert-dev", 50, 60), {})
