"""Vỏ mỏng cho ``reporting.assets`` — logic nằm trong module đã có test.

    python scripts/make_report.py

Sinh bảng, tờ xuất xứ và bản sao hình vào ``report/assets/``. Chạy lại bất cứ
lúc nào: mọi thứ trong đó là sản phẩm của ``results/``, không có gì viết tay.
"""

import argparse

# Đặt sys.path TRƯỚC mọi import của dự án. Chạy ``python scripts/x.py`` chỉ đưa
# ``scripts/`` vào path, không bối cảnh nào tự thấy ``src/``. Vài dòng ở đây rẻ
# hơn việc bắt người chấm phải ``pip install -e .`` trước khi README chạy được.
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reporting.assets import build_report_assets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="report/assets")
    args = ap.parse_args()

    for path in build_report_assets(args.results_dir, args.out_dir):
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
