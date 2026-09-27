"""``report/latex/numbers.tex`` — mọi số kết quả của báo cáo LaTeX, SINH từ ``results/``.

Chương không gõ ``50,80``; nó gõ ``\\histMbertEM``. Chấm lại model rồi chạy
``scripts/make_numbers.py`` là báo cáo tự đúng theo (bất biến #1/#3).

``MACRO_SPEC`` là bảng duy nhất: tên macro → (nguồn, đường dẫn JSON, định dạng).
Nguồn thiếu:

* ``required`` (kết quả đã có từ v1: eval n=500, đường cong huấn luyện) → gãy
  ``FileNotFoundError``;
* còn lại → macro bị bỏ, và cờ ``\\ifhas…`` tương ứng là false. Chương bọc phần
  dùng macro đó trong ``\\ifhas… … \\else <dự định đã đăng ký> \\fi``, nên đến
  hạn đóng băng không cần viết lại gì.

Tên macro chỉ gồm chữ cái (giới hạn của TeX): ``F1`` → ``Fone``, epoch 1–3 →
``I``/``II``/``III``.

Hợp đồng ``results/stats_validation.json`` (``scripts/compute_stats.py``, P2.2)
— mọi metric là phần trăm 0–100, mọi CI là ``[lo, hi]``::

    {"split": "validation", "commit", "timestamp", "B": 10000, "seed": 0,
     "cluster": "paragraph_id", "n_clusters": 557,
     "runs": {"<run_id>": {"n", "EM", "F1", "empty_rate",
                           "EM_wilson": [lo, hi],          # → <p>EMlo / <p>EMhi
                           "EM_cluster_ci": [lo, hi],      # → <p>EMclo / <p>EMchi
                           "F1_cluster_ci": [lo, hi],      # → <p>Foneclo / <p>Fonechi
                           "has_ans": {"n", "EM", "F1", "EM_wilson": [lo, hi]},
                           "no_ans": {"n", "EM", "EM_wilson": [lo, hi]}}},
     "pairs": [{"a", "b", "b01", "b10", "p_mcnemar",   # → <pair>P
                "dEM",                                  # a − b → <pair>DiffEM
                "dEM_cluster_ci": [lo, hi],             # → <pair>DiffEMclo / chi
                "dF1_cluster_ci": [lo, hi]}],
     "seed_summary": {"mean", "std", "n_seeds", "per_seed"}}   # chỉ sau P5

Run/cặp CHƯA có trong tệp (phase chưa chạy) → macro vắng, không lỗi; mục đã có
mà thiếu khoá → ``KeyError`` (lệch hợp đồng phải ồn).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = ["Spec", "MACRO_SPEC", "FLAGS", "HISTORY_N", "fmt_vn", "get_path",
           "collect_values", "collect_all", "render_numbers_tex", "known_non_results"]

#: Cỡ mẫu của bảng v1. Một ``eval_*.json`` cấp trên cùng với n này là LỊCH SỬ,
#: không bao giờ được đọc như kết quả đầy đủ 3.814 câu.
HISTORY_N = 500


@dataclass(frozen=True)
class Spec:
    name: str
    source: str   # "hist:<run>" | "eval:<run>" | "file:<tên>.json" | "stats:<run>" | "pair:<a>:<b>" | "seeds:"
    path: str | Callable[[Any], Any]
    fmt: str = "f2"                 # f2 | f1 | f4 | int | p
    scale: float = 1.0
    required: bool = False
    flag: str | None = None         # cờ bọc chỗ dùng macro trong chương
    result: bool = True             # False: số liệu dữ liệu, được phép gõ tay


# ── bảng macro ──────────────────────────────────────────────────────────────

def _eval_metrics(prefix: str, source: str, *, required=False, flag=None,
                  extra: tuple[str, ...] = ()) -> list[Spec]:
    rows = [("EM", "overall.EM", "f2"), ("Fone", "overall.F1", "f2"),
            ("HasAnsEM", "answerable_only.EM", "f2"),
            ("HasAnsFone", "answerable_only.F1", "f2"),
            ("NoAnsEM", "impossible_only.EM", "f2"),
            ("Latency", "avg_latency_ms", "f1")]
    rows += [r for r in (("EmptyRate", "empty_prediction_rate", "f2"),
                         ("Tau", "null_threshold", "f2")) if r[0] in extra]
    return [Spec(prefix + suffix, source, path, fmt, required=required, flag=flag)
            for suffix, path, fmt in rows]


def _curve(prefix: str, run: str, epochs: int, biased: bool) -> list[Spec]:
    source = f"file:training_curve_{run}.json"
    out = []
    for k, roman in zip(range(epochs), ("I", "II", "III")):
        out += [Spec(f"{prefix}Loss{roman}", source, f"curve[epoch={k + 1}].train_loss",
                     "f4", required=True),
                Spec(f"{prefix}ValEM{roman}", source, f"curve[epoch={k + 1}].val_em",
                     "f2", required=True),
                Spec(f"{prefix}ValFone{roman}", source, f"curve[epoch={k + 1}].val_f1",
                     "f2", required=True)]
        if biased:
            out += [Spec(f"{prefix}FirstEM{roman}", source,
                         f"curve[epoch={k + 1}].val_em_biased_first300", "f2", required=True),
                    Spec(f"{prefix}FirstFone{roman}", source,
                         f"curve[epoch={k + 1}].val_f1_biased_first300", "f2", required=True)]
    return out


_HIST = {"Baseline": "baseline", "Xlmr": "xlmr", "Mbert": "mbert", "Viso": "visobert"}

#: (tiền tố macro, run_id, cờ). v1 đầy đủ mang hậu tố ``Vone``; bản chọn trên dev
#: (tiêu đề, D2) giữ tên ngắn ``mbert…``/``viso…``.
_FULL = (("alwaysEmpty", "empty", "hasFullEval"), ("baseline", "baseline", "hasFullEval"),
         ("xlmr", "xlmr", "hasFullEval"), ("mbertVone", "mbert", "hasFullEval"),
         ("visoVone", "visobert", "hasFullEval"))
_DEV = (("mbert", "mbert-dev", "hasDev"), ("viso", "visobert-dev", "hasDev"),
        ("visoLong", "visobert-len512", "hasVisoLong"), ("pho", "phobert-dev", "hasPho"))


def _gap(curve: dict) -> float:
    last = curve["curve"][-1]
    return last["val_em"] - last["val_em_biased_first300"]


# Bằng chứng P1.2 (scripts/{null_labels,tokenizer_stats,ci,overlap_audit,data_stats}.py).
_TOK = {"Mbert": "mbert", "Viso": "visobert"}


def _evidence() -> list[Spec]:
    nl, tok = "file:null_labels_train.json", "file:tokenizer_stats.json"
    ci, ov, ds = "file:ci_n500.json", "file:sample_overlap.json", "file:data_stats.json"
    out: list[Spec] = []
    for name, run in _TOK.items():
        out += [Spec(f"nullRate{name}", nl, f"[model={run}].null_frac"),
                Spec(f"nullFeatures{name}", nl, f"[model={run}].features", "int"),
                Spec(f"nullMultiWindow{name}", nl, f"[model={run}].q_multiwindow", "int"),
                Spec(f"tokVocab{name}", tok, f"{run}.vocab_size", "int"),
                Spec(f"tokEmbRows{name}", tok, f"{run}.embedding_rows", "int"),
                Spec(f"tokParams{name}", tok, f"{run}.total_params", "f1", scale=1e-6),
                Spec(f"tokEmbParams{name}", tok, f"{run}.embedding_params", "f1", scale=1e-6),
                Spec(f"tokBodyParams{name}", tok, f"{run}.encoder_body_params", "f1",
                     scale=1e-6),
                Spec(f"tokSampleTokens{name}", tok, f"{run}.sample_sentence_tokens", "int"),
                Spec(f"tokCtxMean{name}", tok, f"{run}.val_ctx_mean_tokens", "f1"),
                Spec(f"tokPerWord{name}", tok, f"{run}.val_ctx_tokens_per_word", "f2"),
                Spec(f"tokCtxOver{name}", tok, f"{run}.val_ctx_over_357", "int"),
                Spec(f"tokGtTwoWin{name}", tok, f"{run}.questions_gt2_windows", "int"),
                Spec(f"tokAnsOverThirty{name}", tok, f"{run}.train_answers_over_30_pct", "f1")]
    out.append(Spec("tokCtxTotal", tok, "mbert.val_ctx_total", "int"))
    # Wilson/Wald trên n=500 (lịch sử).
    for name, run in _HIST.items():
        out += [Spec(f"hist{name}EMlo", ci, f"per_model.{run}.EM_wilson[0]"),
                Spec(f"hist{name}EMhi", ci, f"per_model.{run}.EM_wilson[1]"),
                Spec(f"hist{name}EMwald", ci, f"per_model.{run}.EM_wald_half"),
                Spec(f"hist{name}HasAnsEMwald", ci, f"per_model.{run}.ans_EM_wald_half"),
                Spec(f"hist{name}NoAnsEMwald", ci, f"per_model.{run}.imp_EM_wald_half")]
    mx = "comparisons.mbert_vs_xlmr"
    for suffix, key in (("", "overall_EM"), ("HasAns", "answerable_EM"),
                        ("NoAns", "impossible_EM")):
        out += [Spec(f"histMbertXlmr{suffix}DiffEM", ci, f"{mx}.{key}.diff"),
                Spec(f"histMbertXlmr{suffix}ZEM", ci, f"{mx}.{key}.z")]
    out += [Spec("histMbertXlmrContribHasAnsFone", ci, f"{mx}.F1_contrib_answerable"),
            Spec("histMbertXlmrContribNoAnsFone", ci, f"{mx}.F1_contrib_impossible"),
            Spec("histMbertXlmrContribHasAnsEM", ci, f"{mx}.EM_contrib_answerable"),
            Spec("histMbertXlmrContribNoAnsEM", ci, f"{mx}.EM_contrib_impossible"),
            Spec("histMbertVisoHasAnsDiffEM", ci,
                 "comparisons.mbert_vs_visobert_answerable_EM.diff"),
            Spec("histMbertVisoHasAnsZEM", ci, "comparisons.mbert_vs_visobert_answerable_EM.z")]
    # G1/G2/N6: chồng lấp mẫu chọn-model với mẫu báo cáo (đếm, không phải kết quả).
    out += [Spec("overlapSelEval", ov, "overlaps.300.overlap_with_report_n", "int"),
            Spec("overlapSelN", ov, "overlaps.300.n", "int"),
            Spec("overlapTauEval", ov, "overlaps.200.overlap_with_report_n", "int"),
            Spec("overlapTauN", ov, "overlaps.200.n", "int"),
            Spec("overlapHistContexts", ov, "report_subset_contexts", "int"),
            Spec("overlapHistArticles", ov, "report_subset_articles", "int"),
            Spec("overlapSharedTitles", ov, "shared_titles_train_validation", "int")]
    # Số liệu dữ liệu: ĐƯỢC gõ tay (result=False) — có macro để giải va chạm C7
    # (vd. 27,8 % câu impossible của mẫu n=500 == EM của ViSoBERT v1).
    for name, split in (("Train", "train"), ("Val", "validation"), ("Test", "test")):
        out += [Spec(f"data{name}N", ds, f"{split}.num_questions", "int", result=False),
                Spec(f"data{name}Contexts", ds, f"{split}.num_contexts", "int", result=False),
                Spec(f"data{name}Articles", ds, f"{split}.num_articles", "int", result=False),
                Spec(f"data{name}ImpPct", ds, f"{split}.impossible_pct", "f2", result=False)]
    out += [Spec("dataHistImpPct", ds, "subset.impossible_pct", "f2", result=False),
            Spec("dataHistImp", ds, "subset.impossible", "int", result=False)]
    return out


#: Cặp so sánh trong stats_validation.json: (tiền tố macro, a, b, cờ).
_PAIRS = (("mbertVoneXlmr", "mbert", "xlmr", "hasStats"),
          ("mbertVoneVisoVone", "mbert", "visobert", "hasStats"),
          ("xlmrBaseline", "xlmr", "baseline", "hasStats"),
          ("mbertXlmr", "mbert-dev", "xlmr", "hasDev"),
          ("mbertViso", "mbert-dev", "visobert-dev", "hasDev"),
          ("mbertMbertVone", "mbert-dev", "mbert", "hasDev"))


def _stats() -> list[Spec]:
    """CI Wilson + bootstrap cụm đoạn văn + McNemar (P2.2). Cờ: ``hasStats`` cho
    run v1, cờ của chính run cho run v2 (stats sinh ngay sau mỗi lần eval)."""
    out: list[Spec] = []
    for prefix, run, flag in _FULL + _DEV:
        src, flag = f"stats:{run}", (flag if flag != "hasFullEval" else "hasStats")
        out += [Spec(prefix + name, src, path, flag=flag) for name, path in (
            ("EMlo", "EM_wilson[0]"), ("EMhi", "EM_wilson[1]"),
            ("HasAnsEMlo", "has_ans.EM_wilson[0]"), ("HasAnsEMhi", "has_ans.EM_wilson[1]"),
            ("NoAnsEMlo", "no_ans.EM_wilson[0]"), ("NoAnsEMhi", "no_ans.EM_wilson[1]"),
            ("EMclo", "EM_cluster_ci[0]"), ("EMchi", "EM_cluster_ci[1]"),
            ("Foneclo", "F1_cluster_ci[0]"), ("Fonechi", "F1_cluster_ci[1]"))]
    for prefix, a, b, flag in _PAIRS:
        src = f"pair:{a}:{b}"
        out += [Spec(prefix + "P", src, "p_mcnemar", "p", flag=flag),
                Spec(prefix + "DiffEM", src, "dEM", flag=flag),
                Spec(prefix + "DiffEMclo", src, "dEM_cluster_ci[0]", flag=flag),
                Spec(prefix + "DiffEMchi", src, "dEM_cluster_ci[1]", flag=flag),
                Spec(prefix + "DiffFoneclo", src, "dF1_cluster_ci[0]", flag=flag),
                Spec(prefix + "DiffFonechi", src, "dF1_cluster_ci[1]", flag=flag),
                Spec(prefix + "Bzo", src, "b01", "int", flag=flag),
                Spec(prefix + "Boz", src, "b10", "int", flag=flag)]
    out += [Spec("mbertSeedMean", "seeds:", "mean", flag="hasSeeds"),
            Spec("mbertSeedStd", "seeds:", "std", flag="hasSeeds")]
    return out


MACRO_SPEC: tuple[Spec, ...] = (
    # Bảng v1, n=500 (phụ lục lịch sử). Bắt buộc: đây là kết quả đã nộp.
    *(s for name, run in _HIST.items()
      for s in _eval_metrics("hist" + name, f"hist:{run}", required=True)),
    Spec("histN", "hist:mbert", "overall.count", "int", required=True),
    Spec("histHasAnsN", "hist:mbert", "answerable_only.count", "int", required=True),
    Spec("histNoAnsN", "hist:mbert", "impossible_only.count", "int", required=True),
    # Đường cong huấn luyện v1 (N8: 42,00 trên 300 câu đầu vs 52,00 ngẫu nhiên).
    *_curve("curveMbert", "mbert", 2, biased=True),
    *_curve("curveViso", "visobert", 3, biased=False),
    Spec("curveMbertFirstGap", "file:training_curve_mbert.json", _gap, "f2", required=True),
    # Toàn bộ validation 3.814 câu (P2) và các run v2 (P3–P6).
    *(s for prefix, run, flag in _FULL
      for s in _eval_metrics(prefix, f"eval:{run}", flag=flag, extra=("EmptyRate",))),
    Spec("valN", "eval:mbert", "overall.count", "int", flag="hasFullEval"),
    Spec("valHasAnsN", "eval:mbert", "answerable_only.count", "int", flag="hasFullEval"),
    Spec("valNoAnsN", "eval:mbert", "impossible_only.count", "int", flag="hasFullEval"),
    *(s for prefix, run, flag in _DEV
      for s in _eval_metrics(prefix, f"eval:{run}", flag=flag, extra=("EmptyRate", "Tau"))),
    *_evidence(),
    *_stats(),
)

#: Cờ → điều kiện (tên tệp phải tồn tại, tính tương đối ``results/``). Cờ bật khi
#: MỌI tệp có mặt; ``eval:`` áp luật n > HISTORY_N.
FLAGS: dict[str, tuple[str, ...]] = {
    "hasFullEval": ("eval:mbert",),
    "hasDev": ("eval:mbert-dev", "eval:visobert-dev"),
    "hasStats": ("file:stats_validation.json",),
    "hasVisoLong": ("eval:visobert-len512",),
    "hasSeeds": ("eval:mbert-dev-s43", "eval:mbert-dev-s44"),
    "hasPho": ("eval:phobert-dev",),
}


# ── định dạng & truy cập JSON ───────────────────────────────────────────────

def fmt_vn(value: float, fmt: str) -> str:
    """Số kiểu Việt Nam cho LaTeX: thập phân ``{,}``, nghìn ``.``."""
    if fmt == "p":
        if value < 0.001:
            return r"\ensuremath{<}0{,}001"
        return f"{value:.3f}".replace(".", "{,}")
    sign = r"\ensuremath{-}" if value < 0 else ""
    value = abs(value)
    if fmt == "int":
        return sign + f"{round(value):,}".replace(",", ".")
    decimals = {"f4": 4, "f2": 2, "f1": 1}[fmt]
    return sign + f"{value:.{decimals}f}".replace(".", "{,}")


_TOKEN = re.compile(r"([^.\[\]]+)|\[([^\]]+)\]")


def get_path(obj: Any, path: str) -> Any:
    """``"curve[-1].val_em"``, ``"pairs[a=mbert,b=xlmr].p"``, ``"ci[0]"``."""
    for key, bracket in _TOKEN.findall(path):
        if key:
            if not isinstance(obj, dict) or key not in obj:
                raise KeyError(f"{path}: thiếu khoá {key!r}")
            obj = obj[key]
        elif re.fullmatch(r"-?\d+", bracket):
            obj = obj[int(bracket)]
        else:
            want = dict(kv.split("=", 1) for kv in bracket.split(","))
            found = [o for o in obj
                     if all(str(o.get(k)) == v for k, v in want.items())]
            if len(found) != 1:
                raise KeyError(f"{path}: [{bracket}] khớp {len(found)} phần tử, cần đúng 1")
            obj = found[0]
    return obj


# ── đọc results/ ────────────────────────────────────────────────────────────

class _Sources:
    def __init__(self, results_dir: Path, history_dir: Path | None):
        self.results = Path(results_dir)
        self.history = Path(history_dir) if history_dir else self.results / "history" / "n500"
        self._cache: dict[str, tuple[Path, Any] | None] = {}

    def _read(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def describe(self, source: str) -> str:
        kind, _, name = source.partition(":")
        if kind == "file":
            return str(self.results / name)
        if kind == "hist":
            return str(self.history / f"eval_{name}_validation.json")
        return str(self.results / f"eval_{name}_validation.json")

    def get(self, source: str) -> tuple[Path, Any] | None:
        if source not in self._cache:
            self._cache[source] = self._resolve(source)
        return self._cache[source]

    def _resolve(self, source: str) -> tuple[Path, Any] | None:
        kind, _, name = source.partition(":")
        if kind == "file":
            path = self.results / name
            return (path, self._read(path)) if path.is_file() else None
        if kind in ("stats", "pair", "seeds"):
            return self._stats_entry(kind, name)
        top = self.results / f"eval_{name}_validation.json"
        top_data = self._read(top) if top.is_file() else None
        if kind == "hist":
            moved = self.history / f"eval_{name}_validation.json"
            if moved.is_file():
                return moved, self._read(moved)
            # Chưa dời sang history/: tệp cấp trên cùng CHÍNH LÀ lịch sử nếu n=500.
            if top_data is not None and top_data.get("n") == HISTORY_N:
                return top, top_data
            return None
        if kind == "eval":
            if top_data is not None and top_data.get("n", 0) > HISTORY_N:
                return top, top_data
            return None
        raise ValueError(f"nguồn lạ: {source!r}")


    def _stats_entry(self, kind: str, name: str):
        got = self.get("file:stats_validation.json")
        if got is None:
            return None
        path, data = got
        if kind == "seeds":
            entry = data.get("seed_summary")
        elif kind == "stats":
            entry = data.get("runs", {}).get(name)
        else:
            a, b = name.split(":")
            found = [p for p in data.get("pairs", []) if p.get("a") == a and p.get("b") == b]
            entry = found[0] if found else None
        return None if entry is None else (path, entry)


def _flags(sources: _Sources) -> dict[str, bool]:
    return {flag: all(sources.get(s) is not None for s in needs)
            for flag, needs in FLAGS.items()}


def collect_values(results_dir: str | Path, history_dir: str | Path | None = None
                   ) -> tuple[dict[str, tuple[Any, Spec, Path]], dict[str, bool]]:
    """Giá trị THÔ của mọi macro có nguồn: tên → (giá trị, spec, tệp nguồn)."""
    sources = _Sources(Path(results_dir), history_dir)
    values: dict[str, tuple[Any, Spec, Path]] = {}
    for spec in MACRO_SPEC:
        got = sources.get(spec.source)
        if got is None:
            if spec.required:
                raise FileNotFoundError(
                    f"\\{spec.name} cần {sources.describe(spec.source)} — tệp kết quả "
                    "bắt buộc không có. Không sinh numbers.tex từ kết quả thiếu.")
            continue
        path, data = got
        try:
            raw = spec.path(data) if callable(spec.path) else get_path(data, spec.path)
        except (KeyError, IndexError, TypeError) as err:
            raise KeyError(f"\\{spec.name}: {path} không có {spec.path!r} ({err}) — "
                           "schema lệch MACRO_SPEC") from err
        if raw is None:
            raise KeyError(f"\\{spec.name}: {path}:{spec.path} là null")
        values[spec.name] = (raw * spec.scale if spec.scale != 1.0 else raw, spec, path)
    return values, _flags(sources)


def collect_all(results_dir: str | Path, history_dir: str | Path | None = None
                ) -> tuple[dict[str, str], dict[str, bool]]:
    """(macro → chuỗi VN đã định dạng, cờ → bool)."""
    values, flags = collect_values(results_dir, history_dir)
    return {name: fmt_vn(v, spec.fmt) for name, (v, spec, _) in values.items()}, flags


HEADER = "% GENERATED by scripts/make_numbers.py from results/ — do not edit."


def render_numbers_tex(macros: dict[str, str], flags: dict[str, bool]) -> str:
    lines = [HEADER,
             "% Mọi số kết quả của báo cáo. Sửa kết quả rồi chạy lại script; đừng sửa tay.",
             "", "% Cờ: true khi tệp kết quả tương ứng tồn tại."]
    for flag in sorted(flags):
        lines.append(f"\\newif\\if{flag} \\{flag}{'true' if flags[flag] else 'false'}")
    lines.append("")
    for name in sorted(macros):
        lines.append(f"\\newcommand*{{\\{name}}}{{{macros[name]}}}")
    return "\n".join(lines) + "\n"


# ── số KHÔNG phải kết quả (cho kiểm tra va chạm C7) ─────────────────────────

def _floats(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _floats(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _floats(v, f"{prefix}[{i}]")
    elif isinstance(obj, float):
        yield prefix, obj


def known_non_results(results_dir: str | Path) -> dict[float, str]:
    """Thống kê dữ liệu (``data_stats.json``) + siêu tham số (config đường cong).

    Một số liệu đã có macro trong ``MACRO_SPEC`` thì không còn là "số gõ tay hợp
    lệ" — nó rời tập này (đó là cách giải một va chạm).
    """
    results_dir = Path(results_dir)
    covered = {(s.source.partition(":")[2], s.path) for s in MACRO_SPEC
               if s.source.startswith("file:") and isinstance(s.path, str)}
    known: dict[float, str] = {}
    data_stats = results_dir / "data_stats.json"
    if data_stats.is_file():
        for path, v in _floats(json.loads(data_stats.read_text(encoding="utf-8"))):
            if ("data_stats.json", path) not in covered:
                known.setdefault(v, f"data_stats.json:{path}")
    for curve in sorted(results_dir.glob("training_curve_*.json")):
        config = json.loads(curve.read_text(encoding="utf-8")).get("config", {})
        for path, v in _floats(config):
            known.setdefault(v, f"{curve.name}:config.{path}")
    return known
