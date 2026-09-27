"""``numbers.tex`` và bộ bắt số gõ tay — bất biến #1 áp cho báo cáo LaTeX.

Mọi số KẾT QUẢ trong ``report/latex/chapters/*.tex`` phải đi qua một macro sinh
từ ``results/`` (``scripts/make_numbers.py``). Test chia hai nhóm:

* matcher + generator: chạy trên cây ``results/`` giả trong ``tmp_path``;
* kiểm tra trên repo thật (1–5 của kế hoạch 1.3). Những kiểm tra NỘI DUNG còn đỏ
  vì văn bản chưa được viết lại (làn P1.4/1.5) được đánh ``xfail(strict=True)``:
  bộ test nhanh vẫn xanh, và ngay khi làn sửa xong thì chúng XPASS → đỏ, buộc
  người sửa gỡ dấu xfail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from reporting.assets import result_literals
from reporting.literals import (
    LiteralCollision,
    check_collisions,
    find_literals,
    load_allowlist,
    normalise,
    significant_digits,
    unallowed,
    variants,
)
from reporting.numbers import (
    FLAGS,
    MACRO_SPEC,
    collect_all,
    collect_values,
    fmt_vn,
    get_path,
    known_non_results,
    render_numbers_tex,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LATEX = ROOT / "report" / "latex"
CHAPTERS = sorted((LATEX / "chapters").glob("*.tex"))
ALLOWLIST = ROOT / "tests" / "report_literal_allowlist.txt"
WORDING_PENDING = "P1.4/1.5 wording lanes pending"


# ── matcher (C7) ────────────────────────────────────────────────────────────

def test_significant_digits_ignores_leading_zeros_and_the_separator():
    assert significant_digits("50,80") == 4
    assert significant_digits("0,80") == 2
    assert significant_digits("6,93") == 3
    assert significant_digits("12,3") == 3
    assert significant_digits("0,5") == 1


def test_variants_cover_comma_dot_dropped_zero_and_one_decimal_rounding():
    v = variants(50.8)
    assert {"50,80", "50.80", "50,8", "50.8"} <= v

    w = variants(12.264)
    assert {"12,26", "12.26", "12,3", "12.3", "12,264"} <= w
    assert "2,1316" in variants(2.1316), "giá trị đúng như trong JSON cũng là một cách viết"


def test_values_with_fewer_than_three_significant_digits_never_enter():
    assert variants(0.5) == set()
    assert variants(1.3) == set()
    assert variants(0.8) == set(), "0,80 chỉ có 2 chữ số có nghĩa"
    assert "6,93" in variants(6.93)


def test_one_decimal_rounding_is_dropped_when_it_leaves_two_digits():
    """1,108 → 1,11 (3 chữ số) được giữ; 1,1 (2 chữ số) thì không."""
    v = variants(1.108)
    assert "1,11" in v
    assert "1,1" not in v


def test_integers_are_excluded_entirely():
    assert variants(95) == set()
    assert variants(3814) == set()


def test_integer_valued_floats_are_integers_too():
    """Luật C7: chỉ giá trị KHÔNG nguyên. ``52.0`` (EM của val 300 câu) không vào."""
    assert variants(52.0) == set()


def test_latex_decimal_comma_is_caught_after_normalisation():
    assert normalise(r"EM đạt 50{,}80\% và 3\,814") == "EM đạt 50,80\\% và 3814"
    hits = find_literals(r"EM đạt 50{,}80\%", {50.8: "x"})
    assert [h.text for h in hits] == ["50,80"]


@pytest.mark.parametrize("text", ["đạt 50,80%", "(50,80)", "EM 50,80.", "50.8 điểm",
                                  "50,8~điểm", "=50{,}8$"])
def test_each_variant_is_caught_as_a_whole_token(text):
    assert find_literals(text, {50.8: "x"}), text


@pytest.mark.parametrize("text", ["150,80", "50,801", "50,80,5", "1.050,80", "50,80.5",
                                  "50", "5080"])
def test_longer_numbers_containing_the_value_do_not_trigger(text):
    assert not find_literals(text, {50.8: "x"}), text


def test_hits_carry_line_numbers_for_the_error_message():
    hits = find_literals("dòng 1\nEM 50,80\n", {50.8: "eval_mbert:overall.EM"})
    assert hits[0].line == 2
    assert hits[0].source == "eval_mbert:overall.EM"


def test_a_result_equal_to_a_known_non_result_fails_loudly():
    with pytest.raises(LiteralCollision) as err:
        check_collisions({32.39: "eval_empty:overall.EM"},
                         {32.39: "data_stats.json:train.impossible_pct"})
    assert "eval_empty" in str(err.value) and "data_stats" in str(err.value)


def test_no_collision_when_values_differ():
    check_collisions({50.8: "a"}, {3e-5: "lr", 0.01: "wd", 30.44: "data"})


# ── allowlist ───────────────────────────────────────────────────────────────

def test_allowlist_parses_file_literal_and_justification(tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("# comment\n\nreport/latex/chapters/07_phu_luc.tex:68,5  # claim v1 trích nguyên\n",
                 encoding="utf-8")
    [entry] = load_allowlist(f)
    assert entry.file == "report/latex/chapters/07_phu_luc.tex"
    assert entry.literal == "68,5"
    assert entry.justification


def test_allowlist_entry_without_justification_is_rejected(tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("a.tex:50,80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="justification"):
        load_allowlist(f)


def test_allowlist_with_more_than_ten_entries_is_rejected(tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("".join(f"a.tex:{i},55  # lý do\n" for i in range(11)), encoding="utf-8")
    with pytest.raises(ValueError, match="10"):
        load_allowlist(f)


def test_unallowed_filters_hits_and_reports_stale_entries(tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("a.tex:50,80  # trích v1\nb.tex:59,49  # không còn trong b.tex\n",
                 encoding="utf-8")
    entries = load_allowlist(f)
    hits = {"a.tex": find_literals("EM 50,80 và 59,49", {50.8: "x", 59.49: "y"})}

    offenders, stale = unallowed(hits, entries)

    assert [(p, h.text) for p, h in offenders] == [("a.tex", "59,49")]
    assert [e.file for e in stale] == ["b.tex"]


# ── generator ───────────────────────────────────────────────────────────────

def test_vn_formatting():
    assert fmt_vn(50.8, "f2") == "50{,}80"
    assert fmt_vn(12.264, "f1") == "12{,}3"
    assert fmt_vn(1.296, "f4") == "1{,}2960"
    assert fmt_vn(3814, "int") == "3.814"
    assert fmt_vn(15002, "int") == "15.002"
    assert fmt_vn(500, "int") == "500"
    assert fmt_vn(-2.5, "f2") == r"\ensuremath{-}2{,}50"
    assert fmt_vn(0.00012, "p") == r"\ensuremath{<}0{,}001"
    assert fmt_vn(0.0342, "p") == "0{,}034"


def test_get_path_supports_keys_indices_and_selectors():
    obj = {"curve": [{"epoch": 1, "em": 1.0}, {"epoch": 2, "em": 2.0}],
           "pairs": [{"a": "m", "b": "x", "p": 0.5}], "ci": [3.0, 4.0]}
    assert get_path(obj, "curve[-1].em") == 2.0
    assert get_path(obj, "curve[epoch=1].em") == 1.0
    assert get_path(obj, "pairs[a=m,b=x].p") == 0.5
    assert get_path(obj, "ci[1]") == 4.0
    with pytest.raises(KeyError):
        get_path(obj, "pairs[a=z].p")


def test_macro_and_flag_names_are_letters_only_and_unique():
    names = [s.name for s in MACRO_SPEC]
    assert len(names) == len(set(names)), "tên macro trùng"
    for name in names + list(FLAGS):
        assert re.fullmatch(r"[A-Za-z]+", name), name
    for spec in MACRO_SPEC:
        assert spec.flag is None or spec.flag in FLAGS, spec


def make_eval(model, em=50.8, n=500, latency=12.264, **extra):
    return {
        "model": model, "split": "validation", "n": n, "commit": "590e775",
        "timestamp": "2026-09-12T14:03:03+00:00", "device": "mps",
        "overall": {"EM": em, "F1": 59.4886, "count": n},
        "answerable_only": {"EM": 54.5706, "F1": 66.6047, "count": 361},
        "impossible_only": {"EM": 41.0072, "count": 139},
        "avg_latency_ms": latency, **extra,
    }


def make_curve(model, lr=3e-05):
    return {"model": model,
            "config": {"lr": lr, "weight_decay": 0.01, "warmup_ratio": 0.1, "epochs": 2},
            "curve": [{"epoch": 1, "train_loss": 2.1316, "val_em": 46.0, "val_f1": 58.39,
                       "val_em_biased_first300": 36.67, "val_f1_biased_first300": 45.79},
                      {"epoch": 2, "train_loss": 1.296, "val_em": 52.0, "val_f1": 59.75,
                       "val_em_biased_first300": 42.0, "val_f1_biased_first300": 49.13}]}


def write_tree(root: Path, history_moved=False) -> Path:
    """``results/`` tối thiểu: 4 eval n=500 + 2 đường cong (mọi file BẮT BUỘC)."""
    results = root / "results"
    results.mkdir()
    evals = results / "history" / "n500" if history_moved else results
    evals.mkdir(parents=True, exist_ok=True)
    for run, em in (("baseline", 0.8), ("xlmr", 40.6), ("mbert", 50.8), ("visobert", 27.8)):
        (evals / f"eval_{run}_validation.json").write_text(
            json.dumps(make_eval(run, em=em)), encoding="utf-8")
    (results / "training_curve_mbert.json").write_text(
        json.dumps(make_curve("mbert")), encoding="utf-8")
    viso = make_curve("visobert", lr=5e-05)
    viso["curve"].append({"epoch": 3, "train_loss": 2.1908, "val_em": 27.0, "val_f1": 30.48})
    (results / "training_curve_visobert.json").write_text(json.dumps(viso), encoding="utf-8")
    return results


def test_collect_reads_history_from_top_level_n500_files_until_they_are_moved(tmp_path):
    macros, flags = collect_all(write_tree(tmp_path))

    assert macros["histMbertEM"] == "50{,}80"
    assert macros["histMbertFone"] == "59{,}49"
    assert macros["histMbertLatency"] == "12{,}3"
    assert macros["histN"] == "500"


def test_collect_prefers_the_history_directory(tmp_path):
    results = write_tree(tmp_path, history_moved=True)
    (results / "eval_mbert_validation.json").write_text(
        json.dumps(make_eval("mbert", em=55.55, n=3814, empty_prediction_rate=20.0)),
        encoding="utf-8")

    macros, flags = collect_all(results)

    assert macros["histMbertEM"] == "50{,}80"
    assert macros["mbertVoneEM"] == "55{,}55"
    assert flags["hasFullEval"] is True


def test_an_n500_top_level_file_is_never_read_as_the_full_evaluation(tmp_path):
    macros, flags = collect_all(write_tree(tmp_path))

    assert flags["hasFullEval"] is False
    assert "mbertVoneEM" not in macros


def test_missing_optional_files_turn_flags_off_and_omit_macros(tmp_path):
    macros, flags = collect_all(write_tree(tmp_path))

    assert flags["hasVisoLong"] is False and flags["hasSeeds"] is False
    assert flags["hasPho"] is False
    assert not [m for m in macros if m.startswith(("visoLong", "pho", "mbertSeed"))]


def test_present_optional_file_turns_its_flag_on(tmp_path):
    results = write_tree(tmp_path)
    (results / "eval_visobert-len512_validation.json").write_text(
        json.dumps(make_eval("visobert-len512", em=33.33, n=3814,
                             empty_prediction_rate=41.5, null_threshold=0.25)),
        encoding="utf-8")

    macros, flags = collect_all(results)

    assert flags["hasVisoLong"] is True
    assert macros["visoLongEM"] == "33{,}33"
    assert macros["visoLongEmptyRate"] == "41{,}50"


def test_missing_required_file_raises(tmp_path):
    results = write_tree(tmp_path)
    (results / "training_curve_mbert.json").unlink()

    with pytest.raises(FileNotFoundError, match="training_curve_mbert"):
        collect_all(results)


def test_a_present_file_missing_a_spec_key_fails_loudly(tmp_path):
    """Lệch schema phải ồn, không lặng lẽ bỏ macro."""
    results = write_tree(tmp_path)
    (results / "eval_visobert-len512_validation.json").write_text(
        json.dumps({"n": 3814, "overall": {}}), encoding="utf-8")

    with pytest.raises(KeyError, match="visoLong"):
        collect_all(results)


def test_render_has_a_generated_header_flags_and_one_newcommand_per_macro(tmp_path):
    macros, flags = collect_all(write_tree(tmp_path))

    tex = render_numbers_tex(macros, flags)

    assert tex.splitlines()[0].startswith("% GENERATED")
    assert "do not edit" in tex.splitlines()[0]
    assert r"\newif\ifhasVisoLong" in tex and r"\hasVisoLongfalse" in tex
    assert r"\newcommand*{\histMbertEM}{50{,}80}" in tex
    assert tex.count(r"\newcommand") == len(macros)


def test_result_literals_come_from_eval_files_and_result_macros(tmp_path):
    lits = result_literals(write_tree(tmp_path))

    assert 50.8 in lits and 59.4886 in lits and 12.264 in lits
    assert 42.0 in lits, "val_em_biased_first300 (N8) là kết quả"
    assert all(not isinstance(v, int) for v in lits), "số nguyên không bao giờ vào"


def test_known_non_results_hold_hyperparameters(tmp_path):
    known = known_non_results(write_tree(tmp_path))

    assert 3e-05 in known and 0.01 in known


# ── kiểm tra trên repo thật ─────────────────────────────────────────────────

def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _hits(paths, literals) -> dict[str, list]:
    return {_rel(p): find_literals(p.read_text(encoding="utf-8"), literals) for p in paths}


def _assert_clean(paths, literals):
    offenders, _ = unallowed(_hits(paths, literals), load_allowlist(ALLOWLIST))
    assert not offenders, "Số kết quả gõ tay (dùng macro trong numbers.tex):\n" + "\n".join(
        f"  {p}:{h.line}: {h.text}  ← {h.source}" for p, h in offenders)


def test_1_numbers_tex_is_fresh():
    """Sinh lại trong bộ nhớ phải BẰNG file đã commit."""
    committed = (LATEX / "numbers.tex").read_text(encoding="utf-8")

    assert render_numbers_tex(*collect_all(RESULTS)) == committed, (
        "numbers.tex cũ so với results/ — chạy: .venv/bin/python scripts/make_numbers.py")


def test_main_tex_inputs_numbers_before_the_document():
    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    assert r"\input{numbers}" in main
    assert main.index(r"\input{numbers}") < main.index(r"\begin{document}")


@pytest.mark.xfail(strict=True, reason=WORDING_PENDING)
def test_2_chapters_contain_no_result_literal():
    _assert_clean(CHAPTERS, result_literals(RESULTS))


def test_3_every_numbers_macro_used_in_chapters_is_defined():
    """Tên sai chính tả (``\\mbertEm``) hay macro chưa có file nguồn → gãy ở đây,
    không phải thành ``Undefined control sequence`` lúc biên dịch.

    Macro mà cờ của nó đang tắt được phép vắng: chương phải bọc nó trong
    ``\\if<cờ> … \\fi``, và nhánh bị bỏ qua thì TeX không mở rộng.
    """
    macros, flags = collect_all(RESULTS)
    spec = {s.name: s for s in MACRO_SPEC}
    prefixes = tuple(sorted({re.match(r"[a-z]+", n).group() for n in spec}))
    problems = []
    for chapter in CHAPTERS:
        for cs in set(re.findall(r"\\([A-Za-z]+)", chapter.read_text(encoding="utf-8"))):
            if cs.startswith("if") and cs[2:] in flags:
                continue
            ours = re.match(rf"(?:{'|'.join(prefixes)})[A-Z]", cs)
            if cs in spec:
                s = spec[cs]
                if cs not in macros and not (s.flag and not flags[s.flag]):
                    problems.append(f"{chapter.name}: \\{cs} chưa được định nghĩa")
            elif ours:
                problems.append(f"{chapter.name}: \\{cs} không có trong MACRO_SPEC")
    assert not problems, "\n".join(problems)


def test_3b_flags_used_in_chapters_are_known():
    _, flags = collect_all(RESULTS)
    text = "".join(c.read_text(encoding="utf-8") for c in CHAPTERS)
    unknown = {f for f in re.findall(r"\\ifhas([A-Za-z]+)", text) if "has" + f not in flags}
    assert not unknown, unknown


@pytest.mark.xfail(strict=True, reason=WORDING_PENDING)
def test_4a_readme_results_table_is_generated():
    from reporting.assets import render_readme_table

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- BEGIN:results -->\n(.*?)<!-- END:results -->", readme, re.S)
    assert m, "README thiếu vùng <!-- BEGIN:results --> … <!-- END:results -->"
    assert m.group(1) == render_readme_table(RESULTS)


@pytest.mark.xfail(strict=True, reason=WORDING_PENDING)
def test_4b_readme_has_no_result_literal_outside_the_markers():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "<!-- BEGIN:results -->" in text
    outside = re.sub(r"<!-- BEGIN:results -->.*?<!-- END:results -->", "", text, flags=re.S)
    hits = {"README.md": find_literals(outside, result_literals(RESULTS))}
    offenders, _ = unallowed(hits, load_allowlist(ALLOWLIST))
    assert not offenders, [(h.line, h.text) for _, h in offenders]


def test_5_negative_control_a_planted_literal_is_caught(tmp_path):
    chapter = tmp_path / "99_planted.tex"
    chapter.write_text("mBERT đạt EM $50{,}80$ trên n=500.\n", encoding="utf-8")

    hits = find_literals(chapter.read_text(encoding="utf-8"), result_literals(RESULTS))

    assert [h.text for h in hits] == ["50,80"]


def test_allowlist_is_valid_and_has_no_stale_entries():
    entries = load_allowlist(ALLOWLIST)
    literals = result_literals(RESULTS)
    files = sorted({ROOT / e.file for e in entries})
    missing = [f for f in files if not f.is_file()]
    assert not missing, missing
    _, stale = unallowed(_hits(files, literals), entries)
    assert not stale, f"mục allowlist không còn khớp: {stale}"


def test_real_results_have_no_literal_collision():
    check_collisions(result_literals(RESULTS), known_non_results(RESULTS))
