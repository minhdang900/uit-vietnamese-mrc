"""Nối dây của ``scripts/finetune.py`` — không cần GPU, không tải model.

Kiểm những gì mà một lỗi nối dây sẽ làm hỏng âm thầm: validation lọt vào khâu
chọn, một trong ba phép kiểm leakage bị bỏ, schema ``selection.json`` thiếu khoá
mà ``evaluation.cli`` cần, và chạy thử ghi đè ``results/`` thật.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ft():
    spec = importlib.util.spec_from_file_location("finetune_script",
                                                  ROOT / "scripts" / "finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _squad(prefix, n_titles, ctx_per_title=2):
    return {"version": 2.0, "data": [
        {"title": f"{prefix} bài {t}", "paragraphs": [
            {"context": f"{prefix} đoạn {t}-{c}: Hà Nội là thủ đô.",
             "qas": [{"id": f"{prefix}{t}-{c}-{q}", "question": "Thủ đô?",
                      "is_impossible": q == 1,
                      "answers": ({"text": [], "answer_start": []} if q == 1 else
                                  {"text": ["Hà Nội"], "answer_start": [len(f"{prefix} đoạn {t}-{c}: ")]})}
                     for q in range(2)]}
            for c in range(ctx_per_title)]}
        for t in range(n_titles)]}


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "viquad2_train.json").write_text(json.dumps(_squad("tr", 20)), encoding="utf-8")
    (d / "viquad2_validation.json").write_text(json.dumps(_squad("va", 3)), encoding="utf-8")
    return d


def _args(ft, tmp_path, data_dir, *extra):
    return ft.parse_args(["--model", "stub", "--out", str(tmp_path / "models" / "mbert-dev"),
                          "--results-dir", str(tmp_path / "res"), "--data-dir", str(data_dir),
                          *extra])


def test_run_id_defaults_to_out_dir_name(ft, tmp_path, data_dir):
    assert _args(ft, tmp_path, data_dir).run_id == "mbert-dev"


def test_defaults_are_the_preregistered_dev_split(ft, tmp_path, data_dir):
    a = _args(ft, tmp_path, data_dir)
    assert (a.dev_frac, a.dev_group, a.dev_seed, a.eval_limit) == (0.1, "title", 42, None)


def test_all_three_leakage_asserts_run(ft, tmp_path, data_dir, monkeypatch):
    calls = []
    real = ft.assert_no_leakage
    monkeypatch.setattr(ft, "assert_no_leakage", lambda a, b: (calls.append((a, b)), real(a, b)))
    train, dev = ft.load_and_split(_args(ft, tmp_path, data_dir), tmp_path / "res")
    kinds = [({e.qid[:2] for e in a}, {e.qid[:2] for e in b}) for a, b in calls]
    assert len(calls) == 3
    assert kinds[0] == ({"tr"}, {"tr"})          # train / dev
    assert kinds[1] == ({"tr"}, {"va"})          # train / val
    assert kinds[2] == ({"tr"}, {"va"})          # dev / val (dev cũng đến từ train)
    assert calls[0][1] == dev


def test_validation_never_leaves_load_and_split(ft, tmp_path, data_dir):
    train, dev = ft.load_and_split(_args(ft, tmp_path, data_dir), tmp_path / "res")
    assert not any(e.qid.startswith("va") for e in train + dev)


def test_dev_is_title_disjoint_and_split_file_written(ft, tmp_path, data_dir):
    train, dev = ft.load_and_split(_args(ft, tmp_path, data_dir), tmp_path / "res")
    assert len({e.title for e in dev}) == 2               # 10 % của 20 article
    assert not {e.title for e in train} & {e.title for e in dev}
    info = json.loads((tmp_path / "res" / "split_dev_mbert-dev.json").read_text())
    assert info["dev_qids"] == [e.qid for e in dev]
    assert (info["group"], info["dev_seed"], info["dev_frac"]) == ("title", 42, 0.1)
    assert len(info["train_file_sha256"]) == 64


def test_train_size_is_applied_after_the_split(ft, tmp_path, data_dir):
    full_train, full_dev = ft.load_and_split(_args(ft, tmp_path, data_dir), tmp_path / "r1")
    train, dev = ft.load_and_split(_args(ft, tmp_path, data_dir, "--train-size", "5"),
                                   tmp_path / "r2")
    assert len(train) == 5 and dev == full_dev


def _records(good):
    """Bản ghi dev: ``good`` câu có span đúng, còn lại span sai và null tự tin hơn."""
    recs = []
    for i in range(4):
        ok = i < good
        w = {"best_score": 2.0, "null_score": 1.0 if ok else 3.0,
             "start_char": 0, "end_char": 6 if ok else 3, "text": "Hà Nội" if ok else "sai"}
        recs.append({"qid": f"q{i}", "gold": ["Hà Nội"], "windows": [w]})
    return recs


def test_finalize_selection_writes_the_schema_cli_reads(ft, tmp_path, data_dir):
    from mrc.training import EpochRecord

    args = _args(ft, tmp_path, data_dir, "--max-length", "512", "--max-answer-len", "64")
    out, res = tmp_path / "models" / "mbert-dev", tmp_path / "res"
    curve = [EpochRecord(1, 2.0, 50.0, 60.0), EpochRecord(2, 1.5, 50.0, 55.0)]
    sel = ft.finalize_selection(args, curve, {1: _records(1), 2: _records(3)}, out, res, dev_n=4)
    on_disk = json.loads((out / "selection.json").read_text())
    assert on_disk == sel
    for key in ("epoch", "tau", "selected_on", "dev_metrics", "max_length", "doc_stride",
                "max_answer_len", "model_name", "seed", "dev_n", "dev_seed", "group"):
        assert key in sel, key
    assert (sel["max_length"], sel["doc_stride"], sel["max_answer_len"]) == (512, 128, 64)
    assert sel["selected_on"] == "dev" and sel["epoch"] == 2
    thr = json.loads((res / "threshold_mbert-dev.json").read_text())
    assert set(thr["sweeps"]) == {"1", "2"} and len(thr["tau_grid"]) == 41


def test_selection_json_drives_the_eval_cli(ft, tmp_path, data_dir, monkeypatch):
    """Vòng kín: thứ finetune ghi ra là thứ evaluation.cli đọc vào."""
    from evaluation.cli import resolve_inference_config
    from mrc.training import EpochRecord

    args = _args(ft, tmp_path, data_dir, "--max-length", "512", "--max-answer-len", "64")
    ft.finalize_selection(args, [EpochRecord(1, 1.0, 1.0, 1.0)], {1: _records(2)},
                          tmp_path / "models" / "mbert-dev", tmp_path / "res", dev_n=4)
    monkeypatch.chdir(tmp_path)
    cfg = resolve_inference_config("mbert-dev")
    assert (cfg["max_length"], cfg["doc_stride"], cfg["max_answer_len"]) == (512, 128, 64)
    assert cfg["selected_on"] == "dev" and cfg["source"]["tau"] == "selection.json"


def test_smoke_run_skips_prereg_and_leaves_repo_results_untouched(ft, tmp_path, data_dir):
    before = {p.name: p.stat().st_mtime_ns for p in (ROOT / "results").iterdir()}
    args = _args(ft, tmp_path, data_dir)
    assert ft.preregistration(args)["smoke"] is True
    ft.load_and_split(args, Path(args.results_dir))
    from mrc.training import EpochRecord
    ft.finalize_selection(args, [EpochRecord(1, 1.0, 1.0, 1.0)], {1: _records(2)},
                          Path(args.out), Path(args.results_dir), dev_n=4)
    after = {p.name: p.stat().st_mtime_ns for p in (ROOT / "results").iterdir()}
    assert after == before


def test_real_results_dir_requires_registration(ft, tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(ft, "REPO_RESULTS", tmp_path / "res")
    args = _args(ft, tmp_path, data_dir, "--run-id", "chua-dang-ky-bao-gio")
    with pytest.raises(SystemExit, match="TỪ CHỐI"):
        ft.preregistration(args)
