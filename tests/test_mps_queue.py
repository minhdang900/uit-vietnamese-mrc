"""Hàng đợi MPS (scripts/mps_queue.sh + queue_deadline.py): số học hạn chót và
luồng điều khiển — chạy script thật trong sandbox với một ``python`` giả."""
import importlib.util
import os
import shutil
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qd():
    spec = importlib.util.spec_from_file_location("queue_deadline",
                                                  ROOT / "scripts" / "queue_deadline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── số học hạn chót ──────────────────────────────────────────────────
def test_deadline_is_20h_plus7_on_oct_1(qd):
    assert qd.DEADLINE == datetime(2026, 10, 1, 20, 0,
                                   tzinfo=timezone(timedelta(hours=7))).astimezone(timezone.utc)


def test_fits_boundary_is_inclusive(qd):
    at = qd.DEADLINE - timedelta(hours=2.5)
    assert qd.fits(at, 2.5)
    assert not qd.fits(at + timedelta(minutes=1), 2.5)


def test_parse_utc_converts_offsets_and_rejects_naive(qd):
    assert qd.parse_utc("2026-10-01T20:00:00+07:00") == qd.DEADLINE
    assert qd.parse_utc("2026-10-01T13:00:00Z") == qd.DEADLINE
    with pytest.raises(ValueError):
        qd.parse_utc("2026-10-01T13:00:00")


def test_simulation_skips_late_training_and_its_eval(qd):
    # Bắt đầu 8 giờ trước hạn: P3 eval + P4 (2,55) + s43 + s44 (3,5) kịp, P6 (2) thì không.
    rows = {r["step"]: r["status"] for r in qd.simulate(qd.DEADLINE - timedelta(hours=8))}
    assert rows["p4_smoke_train"] == "run" and rows["p5_s44"] == "run"
    assert rows["p6_train"] == "skipped: deadline"
    assert rows["eval_p6"] == "skipped (no run)" and rows["post_p6"] == "skipped (no run)"


def test_skipped_training_consumes_no_time(qd):
    start = qd.DEADLINE - timedelta(hours=1)
    rows = qd.simulate(start)
    assert all(r["status"] != "run" or r["step"].endswith("p3") for r in rows)
    assert rows[-1]["end"] == start + timedelta(hours=0.2)   # chỉ eval+post P3


def test_check_cli_exit_codes():
    py, script = str(ROOT / ".venv" / "bin" / "python"), "scripts/queue_deadline.py"
    ok = subprocess.run([py, script, "check", "--hours", "2.5", "--now", "2026-10-01T10:30:00Z"],
                        cwd=ROOT, capture_output=True)
    late = subprocess.run([py, script, "check", "--hours", "2.5", "--now", "2026-10-01T10:31:00Z"],
                          cwd=ROOT, capture_output=True)
    assert (ok.returncode, late.returncode) == (0, 1)


# ── luồng điều khiển của mps_queue.sh (sandbox, python giả) ──────────
FAKE_PY = """#!/usr/bin/env bash
# Ghi lại lời gọi; queue_deadline.py chạy THẬT (bằng python thật).
if [[ "$1" == scripts/queue_deadline.py ]]; then exec "$REAL_PY" "$@"; fi
echo "$*" >> "$CALLS"
if [[ -n "${FAIL_ON:-}" && "$*" == *"$FAIL_ON"* ]]; then
  echo "${FAIL_MSG:-boom}"; exit 3
fi
if [[ "$1" == scripts/check_hypotheses.py ]]; then exit 1; fi   # báo động: không dừng
exit 0
"""


@pytest.fixture
def sandbox(tmp_path):
    for rel in ("scripts/mps_queue.sh", "scripts/queue_deadline.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    logs = tmp_path / "results" / "logs"
    logs.mkdir(parents=True)
    (logs / "p3_queue.log").write_text("visobert-dev exit=0\n")
    return tmp_path


def _run(sandbox, **env):
    full = {**os.environ, "CALLS": str(sandbox / "calls.txt"),
            "REAL_PY": str(ROOT / ".venv" / "bin" / "python"),
            "MPS_QUEUE_WAIT_PATTERN": "khong-co-tien-trinh-nao-ten-nay-xyz",
            "MPS_QUEUE_CAFFEINATE": "env", "MPS_QUEUE_POLL": "0",
            "MPS_QUEUE_NOW": "2026-09-28T00:00:00Z",
            "TMPDIR": str(sandbox / "tmp"), **env}
    done = subprocess.run(["bash", "scripts/mps_queue.sh"], cwd=sandbox, env=full,
                          capture_output=True, text=True, timeout=120)
    calls = (sandbox / "calls.txt").read_text().splitlines()
    qlog = (sandbox / "results" / "logs" / "mps_queue.log").read_text()
    return done.returncode, calls, qlog


def test_full_queue_order_and_no_git(sandbox):
    code, calls, qlog = _run(sandbox)
    assert code == 0, qlog
    order = [c.split()[0] + (" " + c.split()[c.split().index("--out") + 1]
                             if "--out" in c else "") for c in calls]
    trains = [o.split()[1] for o in order if o.startswith("scripts/finetune.py")]
    assert trains[1:] == ["models/visobert-len512", "models/mbert-dev-s43",
                          "models/mbert-dev-s44", "models/phobert-dev"]
    assert trains[0].endswith("mps_queue_smoke512/model")
    evals = [c for c in calls if c.startswith("scripts/run_eval.py")]
    assert evals == [
        "scripts/run_eval.py --models mbert-dev visobert-dev --full --preds",
        "scripts/run_eval.py --models visobert-len512 --full --preds",
        "scripts/run_eval.py --models mbert-dev-s43 mbert-dev-s44 --full --preds",
        "scripts/run_eval.py --models phobert-dev --full --preds"]
    assert sum(c.startswith("scripts/compute_stats.py") for c in calls) == 4
    assert "check_p3: có BÁO ĐỘNG" in qlog and "queue: xong" in qlog   # alarm không dừng
    assert not any("git" in c.split()[0] for c in calls)


def test_registered_configs_are_passed(sandbox):
    _, calls, _ = _run(sandbox)
    by_out = {c.split("--out ")[1].split()[0]: c for c in calls if "--out " in c}
    assert "--max-length 512 --doc-stride 128 --max-answer-len 64" in by_out["models/visobert-len512"]
    assert "--seed 43" in by_out["models/mbert-dev-s43"]
    assert "--max-length 256 --doc-stride 64 --max-answer-len 30" in by_out["models/phobert-dev"]
    for c in by_out.values():
        if c.split("--out ")[1].startswith("models/"):
            assert "--dev-frac 0.1 --dev-group title --dev-seed 42" in c and "--lr 3e-5" in c


def test_oom_in_smoke_stops_queue_without_fallback(sandbox):
    code, calls, qlog = _run(sandbox, FAIL_ON="mps_queue_smoke512",
                             FAIL_MSG="RuntimeError: MPS backend out of memory")
    assert code == 1 and "OOM" in qlog
    assert not any("visobert-len512" in c or "--batch-size" in c for c in calls)


def test_training_failure_stops_everything_after(sandbox):
    code, calls, qlog = _run(sandbox, FAIL_ON="models/mbert-dev-s43")
    assert code == 1 and "STOP: p5_s43" in qlog
    assert not any("mbert-dev-s44" in c or "phobert" in c for c in calls)


def test_eval_refusal_stops_queue(sandbox):
    code, calls, qlog = _run(sandbox, FAIL_ON="--models mbert-dev visobert-dev")
    assert code == 1 and "STOP: eval_p3" in qlog
    assert not any(c.startswith("scripts/finetune.py") for c in calls)


def test_dry_run_prints_plan(sandbox):
    done = subprocess.run(["bash", "scripts/mps_queue.sh", "--dry-run", "2026-09-27T19:40:00Z"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "p6_train" in done.stdout and "Hạn chót" in done.stdout


def test_queue_skips_training_after_deadline(sandbox):
    code, calls, qlog = _run(sandbox, MPS_QUEUE_NOW="2026-10-01T12:00:00Z")
    assert code == 0
    assert not any(c.startswith("scripts/finetune.py") for c in calls)
    assert "p4 skipped: deadline" in qlog and "p6 skipped: deadline" in qlog
    assert [c for c in calls if c.startswith("scripts/run_eval.py")] == [
        "scripts/run_eval.py --models mbert-dev visobert-dev --full --preds"]
