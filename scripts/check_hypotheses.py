"""So kết quả thật với giả thuyết đã đăng ký TRƯỚC khi chạy.

Đăng ký trước chỉ có giá trị nếu việc đối chiếu là bắt buộc và tự động. Script này
biến nó từ lời hứa thành một cổng kiểm tra chạy được.

Exit code 1 nếu có tín hiệu BÁO ĐỘNG (nghi leakage / bug / vi phạm thứ tự đăng ký),
0 nếu không — kể cả khi dự đoán sai. Dự đoán sai là kết quả khoa học hợp lệ;
leakage thì không.

Hai thế hệ giả thuyết cùng tồn tại trong ``hypotheses.json``:

* ``bands``/``alarms`` ở cấp gốc (v1, 2026-09-12): khớp theo ``run_id`` CHÍNH XÁC
  (qua bảng bí danh), chỉ rơi về khớp chuỗi con với eval JSON cũ không có
  ``run_id``. Khớp chuỗi con cho run mới là sai: ``visobert-len512`` sẽ bị chấm
  theo dải của ``visobert``.
* ``registrations`` (v2, theo phase): mỗi đăng ký có ``run_ids``, ``bands``,
  ``predictions`` (cổng ``{metric, op, value}`` lồng ``any``/``all``, ra
  CONFIRMED/REFUTED) và ``alarms`` riêng. Với mọi run đã đăng ký, thứ tự
  commit đăng ký < bắt đầu huấn luyện < chấm được kiểm lại từ git.

Kết quả chi tiết ghi ra ``hypotheses_report.json`` cạnh các eval JSON.
"""

from __future__ import annotations

import json
import operator
import sys
from datetime import datetime, timezone
from pathlib import Path

# Đặt sys.path TRƯỚC mọi import của dự án (xem scripts/finetune.py).
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.prereg import PreregError, check_order, commit_time, registered_run_ids, registrations_at

#: run_id của CLI -> khoá dải v1 viết theo tên hiển thị.
LEGACY_ALIASES = {"baseline": "TF-IDF Baseline", "xlmr": "XLM-R (squad2, zero-shot)"}

#: Run mà "luôn trả rỗng" là THIẾT KẾ, không phải triệu chứng: bỏ các báo động suy sụp.
DEGENERATE_BY_DESIGN = {"empty"}
COLLAPSE_ALARM_KINDS = {"em_equals_f1", "em_equals_impossible_rate", "empty_rate_above"}

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "==": operator.eq}


def metric(r: dict, name: str) -> float:
    """Tên metric trong giả thuyết -> giá trị trong eval JSON (đều theo %)."""
    table = {
        "EM": lambda: r["overall"]["EM"],
        "F1": lambda: r["overall"]["F1"],
        "HasAns_EM": lambda: r["answerable_only"]["EM"],
        "HasAns_F1": lambda: r["answerable_only"]["F1"],
        "NoAns_EM": lambda: r["impossible_only"]["EM"],
        "empty_rate": lambda: r["empty_prediction_rate"],
    }
    if name not in table:
        raise KeyError(f"metric không hỗ trợ: {name!r} (có: {sorted(table)})")
    return table[name]()


class MissingReference(LookupError):
    """Cổng so với một run khác mà run đó chưa được chấm."""


def evaluate_gate(gate: dict, r: dict, runs: dict | None = None) -> bool:
    """``{metric, op, value}``, hoặc ``{"any": [...]}`` / ``{"all": [...]}`` lồng nhau.

    Có ``ref_run`` thì so HIỆU ``metric(r) − metric(ref_run)``; op ``abs_le`` là
    ``|hiệu| <= value`` (dự đoán "nằm trong ±value của run tham chiếu").
    """
    if "any" in gate:
        return any(evaluate_gate(g, r, runs) for g in gate["any"])
    if "all" in gate:
        return all(evaluate_gate(g, r, runs) for g in gate["all"])
    value = metric(r, gate["metric"])
    if "ref_run" in gate:
        ref = (runs or {}).get(gate["ref_run"])
        if ref is None:
            raise MissingReference(gate["ref_run"])
        value -= metric(ref, gate["metric"])
    if gate["op"] == "abs_le":
        return abs(value) <= gate["value"]
    return OPS[gate["op"]](value, gate["value"])


def _registrations(spec: dict) -> list[tuple[str, dict]]:
    regs = spec.get("registrations") or {}
    if isinstance(regs, dict):
        return list(regs.items())
    return [(reg.get("phase", str(i)), reg) for i, reg in enumerate(regs)]


def find_band(r: dict, spec: dict) -> dict | None:
    run_id = r.get("run_id")
    if run_id:
        for _, reg in _registrations(spec):
            if run_id in (reg.get("bands") or {}):
                return reg["bands"][run_id]
        legacy = spec.get("bands", {})
        return legacy.get(run_id) or legacy.get(LEGACY_ALIASES.get(run_id, ""))
    # Eval JSON cũ (trước khi có run_id): khớp chuỗi con như bản v1.
    model = r["model"].lower()
    return next((b for k, b in spec.get("bands", {}).items() if k.lower() in model), None)


def band_verdict(r: dict, band: dict | None) -> tuple[str, dict]:
    if band is None:
        return "chưa đăng ký giả thuyết", {}
    hits = {m: lo <= metric(r, m) <= hi for m, (lo, hi) in band.items()}
    if all(hits.values()):
        return "KHỚP mọi dải", hits
    if not any(hits.values()):
        return "LỆCH mọi dải", hits
    return "lệch " + ", ".join(m for m, ok in hits.items() if not ok), hits


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def legacy_alarms(r: dict, alarms: dict) -> list[str]:
    """Báo động v1 (giữ nguyên ngữ nghĩa cũ)."""
    out = []
    model, em, f1 = r["model"], r["overall"]["EM"], r["overall"]["F1"]
    for name, a in alarms.items():
        kind = a.get("kind")
        if kind in COLLAPSE_ALARM_KINDS and r.get("run_id") in DEGENERATE_BY_DESIGN:
            continue
        if kind == "em_equals_f1":
            # F1 cho điểm BÁN PHẦN nên bình thường phải cao hơn EM vài điểm.
            if abs(em - f1) <= a["tolerance"]:
                out.append(f"[{name}]: EM={em:.2f} ≈ F1={f1:.2f} — {a['meaning']}")
            continue
        if kind == "em_equals_impossible_rate":
            n_imp = r.get("impossible_only", {}).get("count", 0)
            total = r["overall"]["count"] or 1
            rate = 100.0 * n_imp / total
            if n_imp and abs(em - rate) <= a["tolerance"]:
                out.append(f"[{name}]: EM={em:.2f} ≈ tỉ lệ impossible ={rate:.2f}% "
                           f"— {a['meaning']}")
            continue
        if "model" in a and a["model"].lower() not in model.lower():
            continue
        val = r["overall"][a["metric"]]
        if val > a["max"]:
            out.append(f"[{name}]: {a['metric']}={val:.2f} > {a['max']} — {a['meaning']}")
    return out


def registration_alarms(r: dict, reg: dict, results_dir: Path) -> list[str]:
    """Báo động v2: chỉ áp cho run_id thuộc đăng ký đó (hoặc ``run_ids`` của báo động)."""
    out = []
    run_id = r.get("run_id")
    threshold = _read_json(results_dir / f"threshold_{run_id}.json") or {}
    selection = threshold.get("selection") or {}
    for name, a in (reg.get("alarms") or {}).items():
        if run_id not in a.get("run_ids", reg.get("run_ids", [])):
            continue
        kind = a.get("kind")
        if kind in COLLAPSE_ALARM_KINDS and run_id in DEGENERATE_BY_DESIGN:
            continue
        if kind == "empty_rate_above":
            rate = r.get("empty_prediction_rate")
            if rate is not None and rate > a["max"]:
                out.append(f"[{name}]: tỉ lệ rỗng {rate:.2f}% > {a['max']} — {a['meaning']}")
        elif kind == "dev_val_gap":
            dev_em = (selection.get("dev_metrics") or {}).get("EM")
            if dev_em is None:
                out.append(f"[{name}]: thiếu threshold_{run_id}.json — không kiểm được "
                           f"khoảng cách dev/val")
            elif dev_em - r["overall"]["EM"] > a["max"]:
                out.append(f"[{name}]: dev EM {dev_em:.2f} − val EM {r['overall']['EM']:.2f}"
                           f" > {a['max']} — {a['meaning']}")
        elif kind == "tau_at_grid_edge":
            if selection.get("tau_at_grid_edge"):
                out.append(f"[{name}]: τ={selection.get('tau')} nằm ở mép lưới — {a['meaning']}")
        elif kind in ("em_equals_f1", "em_equals_impossible_rate") or "metric" in a:
            out.extend(legacy_alarms(r, {name: a}))
    return out


def prereg_order_alarm(r: dict, results_dir: Path, repo: Path) -> str | None:
    """Commit đăng ký < bắt đầu huấn luyện < chấm, và đăng ký có mặt ở commit đó."""
    run_id = r["run_id"]
    curve = _read_json(results_dir / f"training_curve_{run_id}.json")
    config = (curve or {}).get("config") or {}
    commit, started = config.get("prereg_commit"), config.get("started_utc")
    if not commit or not started:
        return (f"[prereg_order_violated]: {run_id} không có prereg_commit/started_utc "
                f"trong training_curve_{run_id}.json")
    try:
        if run_id not in registered_run_ids(registrations_at(repo, commit)):
            return f"[prereg_order_violated]: {run_id} chưa đăng ký ở commit {commit[:10]}"
        committed = commit_time(repo, commit)
    except PreregError as e:
        return f"[prereg_order_violated]: {e}"
    if not check_order(committed, started, r["timestamp"]):
        return (f"[prereg_order_violated]: thứ tự sai — đăng ký {committed}, "
                f"bắt đầu {started}, chấm {r['timestamp']}")
    return None


def check(results_dir: Path, repo: Path | None = None) -> tuple[int, dict]:
    repo = repo or results_dir.resolve().parent
    spec = json.loads((results_dir / "hypotheses.json").read_text(encoding="utf-8"))
    evals = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(results_dir.glob("eval_*.json"))]
    report: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "runs": [], "predictions": [], "alarms": []}
    if not evals:
        print("Chưa có eval_*.json — chạy scripts/run_eval.py trước.")
        return 0, report

    registered = registered_run_ids(spec)
    by_run = {r["run_id"]: r for r in evals if r.get("run_id")}

    print(f"{'model':34s} {'EM':>7s} {'F1':>7s}  kết luận")
    print("-" * 90)
    for r in evals:
        band = find_band(r, spec)
        verdict, hits = band_verdict(r, band)
        em, f1 = r["overall"]["EM"], r["overall"]["F1"]
        print(f"{r['model'][:34]:34s} {em:7.2f} {f1:7.2f}  {verdict}"
              + (f"  dải {band}" if band else ""))
        report["runs"].append({"run_id": r.get("run_id"), "model": r["model"], "EM": em,
                               "F1": f1, "band": band, "in_band": hits, "verdict": verdict})

        found = legacy_alarms(r, spec.get("alarms", {}))
        run_id = r.get("run_id")
        for _, reg in _registrations(spec):
            if run_id and run_id in (reg.get("run_ids") or []):
                found += registration_alarms(r, reg, results_dir)
        if run_id in registered:
            order = prereg_order_alarm(r, results_dir, repo)
            if order:
                found.append(order)
        for msg in found:
            print(f"    *** BÁO ĐỘNG {msg}")
            report["alarms"].append({"run_id": run_id, "model": r["model"], "message": msg})

    for phase, reg in _registrations(spec):
        for p in reg.get("predictions") or []:
            r = by_run.get(p["run_id"])
            expected = p.get("expected", True)
            try:
                if r is None:
                    raise MissingReference(p["run_id"])
                verdict = ("CONFIRMED" if evaluate_gate(p["gate"], r, by_run) == expected
                           else "REFUTED")
            except MissingReference:
                verdict = "PENDING"
            print(f"  [{phase}] {p['id']}: {verdict} — {p.get('claim', '')}")
            report["predictions"].append({"phase": phase, **p, "verdict": verdict})

    (results_dir / "hypotheses_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if report["alarms"]:
        print("Có tín hiệu báo động: ĐIỀU TRA trước khi báo cáo con số này.")
        return 1, report
    print("Không có tín hiệu báo động. Dự đoán lệch (nếu có) là kết quả hợp lệ, "
          "phải báo cáo nguyên trạng — không sửa giả thuyết cho khớp.")
    return 0, report


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    results_dir = Path(argv[0] if argv else "results")
    return check(results_dir)[0]


if __name__ == "__main__":
    raise SystemExit(main())
