"""Cổng hồi quy C6 — code mới không được đổi con số n=500 nào đã công bố."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reg():
    spec = importlib.util.spec_from_file_location("regression_n500",
                                                  ROOT / "scripts" / "regression_n500.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval(em=50.8, pred="Hà Nội"):
    return {"n": 500,
            "overall": {"EM": em, "F1": 59.4886, "count": 500},
            "answerable_only": {"EM": 54.5706, "F1": 66.6047, "count": 361},
            "impossible_only": {"EM": 41.0072, "count": 139},
            "by_context_length": {"<100": {"EM": 1.0}}, "by_question_type": {},
            "sample_predictions": [{"qid": "q1", "prediction": pred}]}


def _write(d, run, r):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"eval_{run}_validation.json").write_text(json.dumps(r), encoding="utf-8")


def test_identical_files_pass(reg, tmp_path):
    _write(tmp_path / "a", "mbert", _eval())
    _write(tmp_path / "b", "mbert", _eval())
    assert reg.compare(tmp_path / "a", tmp_path / "b", runs=["mbert"]) == []


def test_last_bit_difference_fails(reg, tmp_path):
    _write(tmp_path / "a", "mbert", _eval(em=50.8))
    _write(tmp_path / "b", "mbert", _eval(em=50.800000000000004))
    assert reg.compare(tmp_path / "a", tmp_path / "b", runs=["mbert"])


def test_changed_sample_prediction_fails(reg, tmp_path):
    _write(tmp_path / "a", "mbert", _eval())
    _write(tmp_path / "b", "mbert", _eval(pred="Huế"))
    assert reg.compare(tmp_path / "a", tmp_path / "b", runs=["mbert"])


def test_missing_file_fails(reg, tmp_path):
    _write(tmp_path / "a", "mbert", _eval())
    (tmp_path / "b").mkdir()
    assert reg.compare(tmp_path / "a", tmp_path / "b", runs=["mbert"])


def test_reference_prefers_history_dir(reg, tmp_path):
    assert reg.reference_dir(tmp_path) == tmp_path
    (tmp_path / "history" / "n500").mkdir(parents=True)
    assert reg.reference_dir(tmp_path) == tmp_path / "history" / "n500"


@pytest.mark.slow
def test_v1_checkpoints_reproduce_published_n500_numbers(tmp_path):
    """Chạy thật bốn checkpoint v1 (MPS ~3 phút) và so từng bit."""
    if not (ROOT / "models" / "mbert").exists():
        pytest.skip("cần models/mbert và models/visobert")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, "scripts/run_eval.py", "--models", "baseline", "xlmr",
                    "mbert", "visobert", "--limit", "500", "--out-dir", str(tmp_path)],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    done = subprocess.run([sys.executable, "scripts/regression_n500.py", str(tmp_path)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
