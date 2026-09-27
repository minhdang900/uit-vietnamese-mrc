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
import shutil
from pathlib import Path

import pytest

from reporting.assets import result_literals
from reporting.literals import (
    LiteralCollision,
    check_collisions,
    find_literals,
    load_allowlist,
    normalise,
    scannable_text,
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
    wilson,
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


def test_integer_valued_floats_match_only_in_fixed_decimal_form():
    """``52.0`` là float trong JSON (EM trên 300 câu) → ``52,00``/``52,0`` là gõ tay
    kết quả; ``52`` trần thì không (đếm, epoch không được báo động giả)."""
    assert {"52,00", "52{,}00", "52.00", "52,0"} <= variants(52.0)
    assert "52" not in variants(52.0)
    assert variants(0.0) == set(), "0,00 không có chữ số có nghĩa"


def test_integer_valued_float_is_caught_as_decimal_but_not_as_bare_integer():
    lits = {52.0: "training_curve_mbert.json:curve[-1].val_em"}
    assert [h.text for h in find_literals(r"EM $52{,}00$ trên 300 câu", lits)] == ["52,00"]
    assert [h.text for h in find_literals("EM 52,0.", lits)] == ["52,0"]
    assert not find_literals("sau 52 bước, 52 câu", lits)
    assert not find_literals("152,00 và 52,001", lits)


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
    assert fmt_vn(0.1234, "f3") == "0{,}123"
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
        "avg_latency_ms": latency,
        "by_context_length": {b: {"EM": em, "F1": 60.12, "count": c, "unreliable": c < 30}
                              for b, c in (("<100", 1), ("100-200", 405), ("200-300", 85),
                                           ("300+", 9))},
        "by_question_type": {"single-sentence": {"EM": 60.8392, "F1": 73.5126, "count": 143},
                             "multi-sentence": {"EM": 50.4587, "F1": 62.0734, "count": 218},
                             "_note": "361 câu answerable"},
        **extra,
    }


def make_curve(model, lr=3e-05):
    return {"model": model,
            "config": {"lr": lr, "weight_decay": 0.01, "warmup_ratio": 0.1, "epochs": 2},
            "curve": [{"epoch": 1, "train_loss": 2.1316, "val_em": 46.0, "val_f1": 58.39,
                       "val_em_biased_first300": 36.67, "val_f1_biased_first300": 45.79},
                      {"epoch": 2, "train_loss": 1.296, "val_em": 52.0, "val_f1": 59.75,
                       "val_em_biased_first300": 42.0, "val_f1_biased_first300": 49.13}]}


EVIDENCE = ("null_labels_train.json", "tokenizer_stats.json", "ci_n500.json",
            "sample_overlap.json", "data_stats.json")


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
    for name in EVIDENCE:   # bằng chứng P1.2 là nguồn BẮT BUỘC; schema lấy từ tệp thật
        shutil.copy(RESULTS / name, results / name)
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
    assert flags["hasFullEval"] is False, "mới 1/5 run v1 ở n đầy đủ"


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


@pytest.mark.parametrize("name", EVIDENCE)
def test_missing_evidence_file_raises(tmp_path, name):
    results = write_tree(tmp_path)
    (results / name).unlink()

    with pytest.raises(FileNotFoundError, match=name.removesuffix(".json")):
        collect_all(results)


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
        "report/latex/numbers.tex is STALE vs results/ (a result JSON changed or landed). "
        "Fix: run `.venv/bin/python scripts/make_numbers.py`, then commit "
        "report/latex/numbers.tex together with the new results/*.json.")


def test_main_tex_inputs_numbers_before_the_document():
    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    assert r"\input{numbers}" in main
    assert main.index(r"\input{numbers}") < main.index(r"\begin{document}")


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


def test_4a_readme_results_table_is_generated():
    from reporting.assets import render_readme_table

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- BEGIN:results -->\n(.*?)<!-- END:results -->", readme, re.S)
    assert m, "README thiếu vùng <!-- BEGIN:results --> … <!-- END:results -->"
    assert m.group(1) == render_readme_table(RESULTS)


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


def test_evidence_files_map_to_their_macros(tmp_path):
    """Schema của scripts/{null_labels,tokenizer_stats,ci,overlap_audit}.py (P1.2)."""
    results = write_tree(tmp_path)
    (results / "null_labels_train.json").write_text(json.dumps([
        {"model": "mbert", "features": 30540, "null_frac": 36.16, "q_multiwindow": 900},
        {"model": "visobert", "features": 40984, "null_frac": 49.3, "q_multiwindow": 5000},
    ]), encoding="utf-8")
    tok = {"vocab_size": 15002, "embedding_rows": 15004, "total_params": 97_000_000,
           "embedding_params": 11_523_072, "encoder_body_params": 85_054_464,
           "sample_sentence_tokens": 23, "val_ctx_mean_tokens": 250.1,
           "val_ctx_tokens_per_word": 1.61, "val_ctx_over_357": 154, "val_ctx_total": 557,
           "questions_gt2_windows": 2, "val_ctx_gt2_windows": 2,
           "train_answers_over_30_pct": 1.2}
    (results / "tokenizer_stats.json").write_text(
        json.dumps({"mbert": tok, "visobert": tok}), encoding="utf-8")
    (results / "sample_overlap.json").write_text(json.dumps({
        "report_subset_contexts": 331, "report_subset_articles": 19,
        "shared_titles_train_validation": 0,
        "overlaps": {"300": {"n": 300, "overlap_with_report_n": 284},
                     "200": {"n": 200, "overlap_with_report_n": 195}}}), encoding="utf-8")

    macros, _ = collect_all(results)

    assert macros["nullRateViso"] == "49{,}30"
    assert macros["tokVocabViso"] == "15.002" and macros["tokEmbRowsViso"] == "15.004"
    assert macros["tokEmbParamsViso"] == "11{,}5"
    assert macros["tokBodyParamsMbert"] == "85{,}1"
    assert macros["overlapSelEval"] == "284" and macros["overlapTauEval"] == "195"
    assert macros["tokCtxGtTwoWinMbert"] == "2", "G5: đoạn văn (không phải câu) cần >2 cửa sổ"
    assert 49.3 in result_literals(results), "tỉ lệ nhãn null là kết quả"


def make_stats(runs=("mbert", "xlmr"), pairs=(("mbert", "xlmr"),)):
    def run(em):
        return {"n": 3814, "EM": em, "F1": em + 8.0, "empty_rate": 30.0,
                "EM_wilson": [em - 1.55, em + 1.61], "EM_cluster_ci": [em - 2.25, em + 2.35],
                "F1_cluster_ci": [em + 5.75, em + 10.25], "icc_paragraph": -0.0194,
                "has_ans": {"n": 2640, "EM": 54.12, "F1": 66.3, "EM_wilson": [52.21, 56.02]},
                "no_ans": {"n": 1174, "EM": 41.33, "EM_wilson": [38.55, 44.17]}}
    return {"split": "validation", "B": 10000, "seed": 0, "cluster": "paragraph_id",
            "runs": {r: run(50.5 + i) for i, r in enumerate(runs)},
            "pairs": [{"a": a, "b": b, "b01": 150, "b10": 310, "p_mcnemar": 0.00012,
                       "dEM": 9.87, "dEM_cluster_ci": [7.65, 12.05],
                       "dF1_cluster_ci": [5.15, 9.35]} for a, b in pairs]}


def test_stats_file_maps_to_ci_and_mcnemar_macros(tmp_path):
    results = write_tree(tmp_path)
    (results / "stats_validation.json").write_text(json.dumps(make_stats()), encoding="utf-8")

    macros, flags = collect_all(results)

    assert flags["hasStats"] is True
    assert macros["mbertVoneEMlo"] == "48{,}95" and macros["mbertVoneEMhi"] == "52{,}11"
    assert macros["mbertVoneHasAnsEMlo"] == "52{,}21"
    assert macros["mbertVoneNoAnsEMhi"] == "44{,}17"
    assert macros["mbertVoneEMclo"] == "48{,}25" and macros["mbertVoneEMchi"] == "52{,}85"
    assert macros["mbertVoneFoneclo"] == "56{,}25" and macros["mbertVoneFonechi"] == "60{,}75"
    assert macros["mbertVoneICC"] == r"\ensuremath{-}0{,}019", "ICC: hệ số, 3 chữ số lẻ"
    assert macros["mbertVoneXlmrP"] == r"\ensuremath{<}0{,}001"
    assert macros["mbertVoneXlmrDiffEM"] == "9{,}87"
    assert macros["mbertVoneXlmrDiffEMclo"] == "7{,}65"
    assert macros["mbertVoneXlmrDiffEMchi"] == "12{,}05"
    assert 48.95 in result_literals(results), "CI là kết quả"


def test_runs_and_pairs_not_yet_in_stats_are_omitted(tmp_path):
    """P3 chưa chạy → stats chưa có mbert-dev; không gãy, macro vắng."""
    results = write_tree(tmp_path)
    (results / "stats_validation.json").write_text(json.dumps(make_stats()), encoding="utf-8")

    macros, _ = collect_all(results)

    assert "mbertEMlo" not in macros and "mbertXlmrP" not in macros


def test_stats_entry_missing_a_key_fails_loudly(tmp_path):
    results = write_tree(tmp_path)
    stats = make_stats()
    del stats["runs"]["mbert"]["EM_cluster_ci"]
    (results / "stats_validation.json").write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(KeyError, match="mbertVoneEMclo"):
        collect_all(results)


# ── bổ sung cho làn A (57b756a) ─────────────────────────────────────────────

FULL_RUNS = ("empty", "baseline", "xlmr", "mbert", "visobert")


def _full(results, run, em=44.44):
    (results / f"eval_{run}_validation.json").write_text(
        json.dumps(make_eval(run, em=em, n=3814, empty_prediction_rate=12.5,
                             null_threshold=0.25)), encoding="utf-8")


def test_full_eval_flag_needs_every_v1_run_at_full_n(tmp_path):
    """Giữa P2 chỉ mới có vài run 3.814 câu → cờ vẫn tắt, không macro nào bị thiếu
    trong nhánh ``\\ifhasFullEval``."""
    results = write_tree(tmp_path, history_moved=True)
    for run in FULL_RUNS[:-1]:
        _full(results, run)

    assert collect_all(results)[1]["hasFullEval"] is False

    _full(results, "visobert")
    macros, flags = collect_all(results)
    assert flags["hasFullEval"] is True
    assert macros["visoVoneEM"] == "44{,}44" and macros["alwaysEmptyEmptyRate"] == "12{,}50"


def test_wilson_matches_the_reference_values():
    assert wilson(50, 100) == pytest.approx((40.38, 59.62), abs=0.01)
    assert wilson(0, 10)[1] == pytest.approx(27.75, abs=0.01)


def test_history_wilson_bounds_for_answerable_and_impossible(tmp_path):
    macros, _ = collect_all(write_tree(tmp_path))

    lo, hi = wilson(round(54.5706 * 361 / 100), 361)
    assert macros["histMbertHasAnsEMlo"] == fmt_vn(lo, "f2")
    assert macros["histMbertHasAnsEMhi"] == fmt_vn(hi, "f2")
    lo, hi = wilson(round(41.0072 * 139 / 100), 139)
    assert macros["histViso" + "NoAnsEMlo"] == fmt_vn(lo, "f2")
    assert macros["histBaselineNoAnsEMhi"] == fmt_vn(hi, "f2")


def test_history_f1_difference_and_breakdowns(tmp_path):
    macros, _ = collect_all(write_tree(tmp_path))
    ci = json.loads((RESULTS / "ci_n500.json").read_text(encoding="utf-8"))["per_model"]

    assert macros["histMbertXlmrDiffFone"] == fmt_vn(
        ci["mbert"]["F1"] - ci["xlmr"]["F1"], "f2")
    assert macros["histMbertSingleEM"] == "60{,}84"
    assert macros["histMbertMultiFone"] == "62{,}07"
    assert macros["histMbertSingleN"] == "143"
    assert macros["histMbertLenOneTwoN"] == "405"
    assert macros["histXlmrLenOverThreeFone"] == "60{,}12"
    assert macros["histViso" + "LenUnderHundredN"] == "1"


def test_p4_verdict_and_one_sided_has_ans_mcnemar(tmp_path):
    results = write_tree(tmp_path)
    _full(results, "visobert-len512")
    report = {"predictions": [{"phase": "P3", "id": "x", "verdict": "REFUTED"}]}
    (results / "hypotheses_report.json").write_text(json.dumps(report), encoding="utf-8")

    assert "visoLongVerdict" not in collect_all(results)[0], "chưa chấm P4 → vắng"

    report["predictions"] += [{"phase": "P4", "id": "a", "verdict": "CONFIRMED"},
                              {"phase": "P4", "id": "b", "verdict": "CONFIRMED"}]
    (results / "hypotheses_report.json").write_text(json.dumps(report), encoding="utf-8")
    stats = make_stats(runs=("visobert-len512",), pairs=())
    stats["pairs"] = [{"a": "visobert-len512", "b": "visobert-dev", "b01": 20, "b10": 90,
                       "p_mcnemar": 0.0001, "p_mcnemar_has_ans_greater": 0.00731,
                       "dEM": 5.5, "dEM_cluster_ci": [3.1, 7.9], "dF1_cluster_ci": [2.2, 8.8]}]
    (results / "stats_validation.json").write_text(json.dumps(stats), encoding="utf-8")

    macros, _ = collect_all(results)
    assert macros["visoLongVerdict"] == "CONFIRMED"
    assert macros["visoLongMcnemarP"] == "0{,}007"

    report["predictions"][-1]["verdict"] = "REFUTED"
    (results / "hypotheses_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert collect_all(results)[0]["visoLongVerdict"] == "REFUTED"


@pytest.mark.parametrize("text", [r"\renewcommand{\arraystretch}{1.15}",
                                  "minimum width=1.15cm", "xshift=1.15em", r"1.15\linewidth",
                                  r"\setstretch{1.15}"])
def test_tex_layout_lengths_are_not_results(text):
    assert not find_literals(text, {1.1523: "x"}), text


def test_a_value_next_to_a_word_is_still_caught():
    assert find_literals("chênh 1.15 điểm", {1.1523: "x"})


# ── deck: chỉ chữ người xem thấy, không phải toạ độ dàn trang ───────────────

def test_js_numbers_in_code_are_layout_but_strings_are_text():
    src = ('card(s, M, 1.78, 6.1, 3.05, { size: 11.5 });\n'
           'body(s, "EM đạt 50,80% — chênh 3,05", 1, 2);  // 50,80 trong chú thích\n'
           "const t = `ViSoBERT ${fmt(3.05)} và 11,5M`; /* 11,5 */\n"
           "const re = i.replace(/\\B(?=(\\d{3})+(?!\\d))/g, '.');\n")
    text = scannable_text("deck.js", src)

    assert text.count("\n") == src.count("\n"), "giữ nguyên số dòng cho thông báo lỗi"
    hits = find_literals(text, {3.0512: "loss", 11.523: "emb", 50.8: "em"})
    assert sorted((h.line, h.text) for h in hits) == [(2, "3,05"), (2, "50,80"), (3, "11,5")]


def test_html_scans_visible_text_and_script_strings_not_css():
    src = ('<style>\n.x{line-height:1.15;max-width:22ch}\n</style>\n'
           '<p class="a" style="margin:1.15em">EM 1,15 điểm</p>\n'
           '<script>\nconst w = 1.15; const s = "chênh 1,15";\n</script>\n')
    text = scannable_text("index.html", src)

    hits = find_literals(text, {1.1523: "x"})
    assert [h.line for h in hits] == [4, 6]


def test_other_files_are_scanned_verbatim():
    assert scannable_text("README.md", "a 1.15 b") == "a 1.15 b"
