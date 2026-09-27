"""Đăng ký trước được cưỡng chế bằng git (C4) — test trên một repo git tạm.

Ba cách vi phạm, mỗi cách một test: run chưa đăng ký, file giả thuyết sửa dở,
và thứ tự thời gian đảo (chấm trước khi đăng ký).
"""
import json
import subprocess

import pytest

from mrc.prereg import (
    PreregError,
    assert_preregistered,
    check_order,
    commit_time,
    registered_run_ids,
    registrations_at,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit_spec(repo, spec, msg="prereg"):
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results" / "hypotheses.json").write_text(json.dumps(spec), encoding="utf-8")
    (repo / "results" / "hypotheses.md").write_text("# giả thuyết\n", encoding="utf-8")
    _git(repo, "add", "results")
    _git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    _commit_spec(tmp_path, {"bands": {}, "registrations": {
        "P3": {"run_ids": ["mbert-dev", "visobert-dev"]}}})
    return tmp_path


def test_registered_run_ids_reads_dict_and_list_forms():
    assert registered_run_ids({"registrations": {"P3": {"run_ids": ["a", "b"]}}}) == {"a", "b"}
    assert registered_run_ids({"registrations": [{"run_ids": ["c"]}]}) == {"c"}
    assert registered_run_ids({"bands": {}}) == set()


def test_registered_clean_run_is_allowed_and_commit_recorded(repo):
    info = assert_preregistered("mbert-dev", repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    assert info["prereg_commit"] == head
    assert info["prereg_commit_time"] == commit_time(repo, head)


def test_unregistered_run_id_is_refused(repo):
    with pytest.raises(PreregError, match="chưa được đăng ký"):
        assert_preregistered("visobert-len512", repo)


def test_run_id_matching_is_exact_not_substring(repo):
    with pytest.raises(PreregError):
        assert_preregistered("mbert", repo)


def test_dirty_hypotheses_file_is_refused(repo):
    (repo / "results" / "hypotheses.md").write_text("# sửa dở\n", encoding="utf-8")
    with pytest.raises(PreregError, match="chưa commit"):
        assert_preregistered("mbert-dev", repo)


def test_registration_only_in_working_tree_is_refused(repo):
    """Thêm run_id vào file mà chưa commit: vừa bẩn, vừa không có ở HEAD."""
    spec = json.loads((repo / "results" / "hypotheses.json").read_text())
    spec["registrations"]["P4"] = {"run_ids": ["visobert-len512"]}
    (repo / "results" / "hypotheses.json").write_text(json.dumps(spec))
    with pytest.raises(PreregError):
        assert_preregistered("visobert-len512", repo)


def test_registrations_at_old_commit(repo):
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True).stdout.strip()
    _commit_spec(repo, {"registrations": {"P4": {"run_ids": ["x"]}}}, "p4")
    assert registered_run_ids(registrations_at(repo, first)) == {"mbert-dev", "visobert-dev"}


def test_order_check_passes_in_order_and_fails_when_swapped():
    commit, start, ev = ("2026-09-27T10:00:00+07:00", "2026-09-27T03:30:00+00:00",
                         "2026-09-27T06:00:00+00:00")
    assert check_order(commit, start, ev)
    assert not check_order(start, commit, ev)       # huấn luyện trước khi đăng ký
    assert not check_order(commit, ev, start)       # chấm trước khi huấn luyện xong
