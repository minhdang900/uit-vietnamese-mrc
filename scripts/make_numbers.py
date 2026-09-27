"""Vỏ mỏng cho ``reporting.numbers`` — sinh ``report/latex/numbers.tex``.

    python scripts/make_numbers.py            # ghi file
    python scripts/make_numbers.py --check    # exit 1 nếu file đã commit bị cũ

Trước khi ghi, kiểm tra va chạm (luật C7): một kết quả trùng giá trị với thống
kê dữ liệu hay siêu tham số thì test chữ không phân biệt được hai thứ — gãy ngay
ở đây, liệt kê cả hai nguồn.
"""

import argparse

# Đặt sys.path TRƯỚC mọi import của dự án (xem scripts/make_report.py).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reporting.assets import result_literals
from reporting.literals import check_collisions
from reporting.numbers import collect_all, known_non_results, render_numbers_tex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_ROOT / "results"))
    ap.add_argument("--history-dir", default=None,
                    help="mặc định <results-dir>/history/n500, lùi về eval_*.json n=500")
    ap.add_argument("--out", default=str(_ROOT / "report" / "latex" / "numbers.tex"))
    ap.add_argument("--check", action="store_true", help="chỉ so sánh, không ghi")
    args = ap.parse_args()

    check_collisions(result_literals(args.results_dir), known_non_results(args.results_dir))
    macros, flags = collect_all(args.results_dir, args.history_dir)
    tex = render_numbers_tex(macros, flags)
    out = _Path(args.out)

    if args.check:
        fresh = out.is_file() and out.read_text(encoding="utf-8") == tex
        print(f"{out}: {'fresh' if fresh else 'STALE — chạy lại không có --check'}")
        return 0 if fresh else 1

    out.write_text(tex, encoding="utf-8")
    on = ", ".join(f for f, v in sorted(flags.items()) if v) or "—"
    print(f"  {out}  ({len(macros)} macro; cờ bật: {on})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
