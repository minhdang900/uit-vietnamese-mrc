"""Đăng ký trước được CƯỠNG CHẾ bằng git, không chỉ bằng lời hứa.

Một giả thuyết "đăng ký trước" chỉ có giá trị nếu chứng minh được nó có mặt
TRƯỚC khi run bắt đầu. Git làm chứng: ``finetune.py`` từ chối chạy nếu
``HEAD:results/hypotheses.json`` chưa đăng ký ``run_id``, hoặc nếu file giả thuyết
còn sửa dở chưa commit (sửa dở = có thể đổi sau khi thấy kết quả). Nó ghi lại
commit đăng ký, và ``check_hypotheses.py`` kiểm lại thứ tự thời gian:
commit đăng ký < lúc bắt đầu huấn luyện < lúc chấm.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

__all__ = [
    "PreregError",
    "HYPOTHESES_FILES",
    "registered_run_ids",
    "registrations_at",
    "assert_preregistered",
    "commit_time",
    "check_order",
]

HYPOTHESES_JSON = "results/hypotheses.json"
HYPOTHESES_FILES = ("results/hypotheses.md", HYPOTHESES_JSON)


class PreregError(RuntimeError):
    """Run chưa được đăng ký trước đúng quy trình."""


def _git(repo: str | Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=30, check=False)


def registered_run_ids(spec: dict) -> set[str]:
    """Mọi ``run_id`` trong ``registrations`` (dict theo phase, hoặc list)."""
    regs = spec.get("registrations") or {}
    entries = regs.values() if isinstance(regs, dict) else regs
    return {rid for reg in entries for rid in (reg.get("run_ids") or [])}


def registrations_at(repo: str | Path, rev: str = "HEAD") -> dict:
    """Nội dung ``results/hypotheses.json`` tại ``rev``; raise nếu không có."""
    done = _git(repo, "show", f"{rev}:{HYPOTHESES_JSON}")
    if done.returncode != 0:
        raise PreregError(f"Không đọc được {HYPOTHESES_JSON} tại {rev}: {done.stderr.strip()}")
    return json.loads(done.stdout)


def commit_time(repo: str | Path, rev: str) -> str:
    """Thời điểm commit (ISO 8601, có múi giờ) của ``rev``."""
    done = _git(repo, "log", "-1", "--format=%cI", rev)
    if done.returncode != 0 or not done.stdout.strip():
        raise PreregError(f"Không tìm thấy commit {rev}: {done.stderr.strip()}")
    return done.stdout.strip()


def assert_preregistered(run_id: str, repo: str | Path) -> dict:
    """Raise :class:`PreregError` trừ khi ``run_id`` đã đăng ký và đã commit sạch.

    Returns:
        ``{"prereg_commit", "prereg_commit_time"}`` để ghi vào cấu hình run.
    """
    dirty = _git(repo, "status", "--porcelain", "--", *HYPOTHESES_FILES)
    if dirty.returncode != 0:
        raise PreregError(f"git status thất bại: {dirty.stderr.strip()}")
    if dirty.stdout.strip():
        raise PreregError(
            "File giả thuyết còn thay đổi chưa commit:\n" + dirty.stdout
            + "Commit đăng ký trước (một commit riêng) rồi mới chạy."
        )
    ids = registered_run_ids(registrations_at(repo, "HEAD"))
    if run_id not in ids:
        raise PreregError(
            f"run_id {run_id!r} chưa được đăng ký trong HEAD:{HYPOTHESES_JSON} "
            f"(đã đăng ký: {sorted(ids)}). Đăng ký trước, commit, rồi mới chạy."
        )
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return {"prereg_commit": head, "prereg_commit_time": commit_time(repo, head)}


def check_order(prereg_commit_time: str, started_utc: str, eval_timestamp: str) -> bool:
    """``commit đăng ký < bắt đầu huấn luyện < chấm`` — so theo thời điểm tuyệt đối."""
    t = [datetime.fromisoformat(x) for x in (prereg_commit_time, started_utc, eval_timestamp)]
    return t[0] < t[1] < t[2]
