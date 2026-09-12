"""So kết quả thật với giả thuyết đã đăng ký TRƯỚC khi chạy.

Đăng ký trước chỉ có giá trị nếu việc đối chiếu là bắt buộc và tự động. Script này
biến nó từ lời hứa thành một cổng kiểm tra chạy được.

Exit code 1 nếu có tín hiệu BÁO ĐỘNG (nghi leakage / bug), 0 nếu không — kể cả khi
dự đoán sai. Dự đoán sai là kết quả khoa học hợp lệ; leakage thì không.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    spec = json.loads((results_dir / "hypotheses.json").read_text(encoding="utf-8"))
    bands, alarms = spec["bands"], spec["alarms"]

    evals = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(results_dir.glob("eval_*.json"))]
    if not evals:
        print("Chưa có eval_*.json — chạy scripts/run_eval.py trước.")
        return 0

    print(f"{'model':34s} {'EM':>7s} {'dự đoán':>12s}  {'F1':>7s} {'dự đoán':>12s}  kết luận")
    print("-" * 100)

    alarm_hit = False
    for r in evals:
        model, em, f1 = r["model"], r["overall"]["EM"], r["overall"]["F1"]
        band = next((b for k, b in bands.items() if k.lower() in model.lower()), None)

        if band is None:
            verdict = "chưa đăng ký giả thuyết"
            em_b = f1_b = "—"
        else:
            in_em = band["EM"][0] <= em <= band["EM"][1]
            in_f1 = band["F1"][0] <= f1 <= band["F1"][1]
            em_b = f"[{band['EM'][0]},{band['EM'][1]}]"
            f1_b = f"[{band['F1'][0]},{band['F1'][1]}]"
            verdict = ("KHỚP cả hai" if in_em and in_f1 else
                       "khớp EM, lệch F1" if in_em else
                       "lệch EM, khớp F1" if in_f1 else "LỆCH cả hai")

        print(f"{model[:34]:34s} {em:7.2f} {em_b:>12s}  {f1:7.2f} {f1_b:>12s}  {verdict}")

        # Tín hiệu báo động
        for name, a in alarms.items():
            if "model" in a and a["model"].lower() not in model.lower():
                continue
            val = r["overall"][a["metric"]]
            if val > a["max"]:
                print(f"    *** BÁO ĐỘNG [{name}]: {a['metric']}={val:.2f} > {a['max']} "
                      f"— {a['meaning']}")
                alarm_hit = True

    print()
    if alarm_hit:
        print("Có tín hiệu báo động: ĐIỀU TRA trước khi báo cáo con số này.")
        return 1
    print("Không có tín hiệu báo động. Dự đoán lệch (nếu có) là kết quả hợp lệ, "
          "phải báo cáo nguyên trạng — không sửa giả thuyết cho khớp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
