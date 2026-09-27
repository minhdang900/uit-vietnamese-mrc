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


# ── cổng P4/P5: delta, abs (đường dẫn chấm), if/then/else, McNemar, seed_std ──
def test_dotted_metric_paths(ch):
    r = _eval("x", 40, 50, has_em=20, empty=33)
    assert ch.metric(r, "answerable_only.EM") == 20
    assert ch.metric(r, "empty_prediction_rate") == 33
    with pytest.raises(KeyError):
        ch.metric(r, "answerable_only.nope")


def test_delta_and_abs_kinds(ch):
    runs = {"visobert-dev": _eval("visobert-dev", 30, 33, has_em=8, empty=85)}
    r = _eval("visobert-len512", 40, 48, has_em=20, empty=50)
    assert ch.evaluate_gate({"kind": "delta", "metric": "answerable_only.EM",
                             "vs": "visobert-dev", "op": ">=", "value": 10.0}, r, runs)
    assert ch.evaluate_gate({"kind": "delta", "metric": "empty_prediction_rate",
                             "vs": "visobert-dev", "op": "<=", "value": -30.0}, r, runs)
    assert ch.evaluate_gate({"kind": "abs", "metric": "empty_prediction_rate",
                             "op": "<", "value": 65.0}, r, runs)
    assert not ch.evaluate_gate({"kind": "abs", "metric": "answerable_only.EM",
                                 "op": ">", "value": 20.0}, r, runs)


def test_if_then_else_branches_on_another_run(ch):
    collapsed = {"run": "visobert-dev", "any": [
        {"metric": "empty_rate", "op": ">=", "value": 90},
        {"metric": "HasAns_EM", "op": "<", "value": 15}]}
    gate = {"if": collapsed,
            "then": {"kind": "abs", "metric": "answerable_only.EM", "op": ">", "value": 16.93},
            "else": {"kind": "delta", "metric": "answerable_only.EM", "vs": "visobert-dev",
                     "op": ">=", "value": 3.0}}
    r = _eval("visobert-len512", 40, 48, has_em=14)
    still = {"visobert-dev": _eval("visobert-dev", 30, 33, has_em=8, empty=85)}
    fixed = {"visobert-dev": _eval("visobert-dev", 45, 55, has_em=10.5, empty=30)}
    fixed["visobert-dev"]["answerable_only"]["EM"] = 10.5
    assert ch.gate_branch(gate, r, still, None) == "then"
    assert not ch.evaluate_gate(gate, r, still)            # 14 ≤ 16.93
    # HasAns 10.5 < 15 vẫn là suy sụp theo định nghĩa ⇒ vẫn nhánh "then".
    assert ch.gate_branch(gate, r, fixed, None) == "then"
    fixed["visobert-dev"]["answerable_only"]["EM"] = 20.0
    r["answerable_only"]["EM"] = 24.0
    assert ch.gate_branch(gate, r, fixed, None) == "else"
    assert ch.evaluate_gate(gate, r, fixed)                # 24 − 20 ≥ 3


def _preds(res, run, correct, impossible):
    with (res / f"preds_{run}_validation.jsonl").open("w") as f:
        for i, (c, imp) in enumerate(zip(correct, impossible)):
            f.write(json.dumps({"qid": f"q{i}", "is_impossible": imp, "em": float(c)}) + "\n")


def test_mcnemar_gate_one_sided_on_has_ans_subset(ch, tmp_path):
    n = 40
    imp = [i >= 30 for i in range(n)]
    # HasAns (30 câu): len512 đúng 15 câu mà control sai, không có chiều ngược lại.
    _preds(tmp_path, "visobert-len512", [i < 15 or i >= 30 for i in range(n)], imp)
    _preds(tmp_path, "visobert-dev", [i >= 30 for i in range(n)], imp)
    assert ch.mcnemar_counts(tmp_path, "visobert-len512", "visobert-dev", "has_ans") == (0, 15)
    gate = {"kind": "mcnemar", "subset": "has_ans", "vs": "visobert-dev",
            "alternative": "greater", "direction": "b10>b01", "alpha": 0.01}
    r = _eval("visobert-len512", 1, 2)
    assert ch.evaluate_gate(gate, r, {}, tmp_path)
    # Đảo chiều: H1 "greater" không thể đạt.
    reverse = {**gate, "vs": "visobert-len512"}
    assert not ch.evaluate_gate(reverse, _eval("visobert-dev", 1, 2), {}, tmp_path)


def test_mcnemar_missing_preds_is_pending_not_error(ch, tmp_path):
    gate = {"kind": "mcnemar", "subset": "has_ans", "vs": "visobert-dev",
            "alternative": "greater", "alpha": 0.01}
    with pytest.raises(ch.MissingReference):
        ch.evaluate_gate(gate, _eval("visobert-len512", 1, 2), {}, tmp_path)


def test_mcnemar_refuses_mismatched_qids(ch, tmp_path):
    _preds(tmp_path, "a", [1, 0], [False, False])
    _preds(tmp_path, "b", [1], [False])
    with pytest.raises(ValueError, match="qid"):
        ch.mcnemar_counts(tmp_path, "a", "b")


def test_seed_std_gate(ch):
    runs = {"mbert-dev": _eval("mbert-dev", 50.0, 1), "mbert-dev-s43": _eval("s", 51.0, 1),
            "mbert-dev-s44": _eval("s", 49.0, 1)}
    gate = {"kind": "seed_std", "runs": ["mbert-dev", "mbert-dev-s43", "mbert-dev-s44"],
            "metric": "overall.EM", "op": "<=", "value": 1.5}
    assert ch.seed_std(runs, gate["runs"], "overall.EM") == pytest.approx(1.0)
    assert ch.evaluate_gate(gate, runs["mbert-dev-s44"], runs)
    runs["mbert-dev-s44"] = _eval("s", 45.0, 1)
    assert not ch.evaluate_gate(gate, runs["mbert-dev-s44"], runs)


def test_gate_alarm_and_config_alarm(ch, tmp_path):
    spec = {"bands": {}, "alarms": {}, "registrations": {"P4": {
        "run_ids": ["visobert-len512"],
        "alarms": {
            "too_good": {"kind": "gate", "meaning": "EM > mbert-dev + 10",
                         "gate": {"kind": "delta", "metric": "overall.EM", "vs": "mbert-dev",
                                  "op": ">", "value": 10}},
            "oom": {"kind": "config_differs", "key": "batch_size", "expected": 12,
                    "meaning": "dùng OOM fallback"}}}}}
    res = _setup(tmp_path, [_eval("visobert-len512", 65, 70), _eval("mbert-dev", 50, 60)],
                 spec=spec)
    (res / "training_curve_visobert-len512.json").write_text(json.dumps(
        {"config": {"batch_size": 6}}))
    _, report = ch.check(res, repo=tmp_path)
    msgs = _alarms(report)
    assert "[too_good]" in msgs and "[oom]" in msgs


def test_seed_prediction_pending_until_all_seeds_exist(ch, tmp_path):
    spec = {"bands": {}, "alarms": {}, "registrations": {"P5": {
        "run_ids": ["mbert-dev-s43", "mbert-dev-s44"],
        "predictions": [{"id": "seed-std", "run_id": "mbert-dev-s44", "expected": True,
                         "gate": {"kind": "seed_std", "metric": "overall.EM", "op": "<=",
                                  "value": 1.5, "runs": ["mbert-dev", "mbert-dev-s43",
                                                         "mbert-dev-s44"]}}]}}}
    res = _setup(tmp_path, [_eval("mbert-dev", 50, 60), _eval("mbert-dev-s44", 50.5, 60)],
                 spec=spec)
    _, report = ch.check(res, repo=tmp_path)
    assert report["predictions"][0]["verdict"] == "PENDING"
