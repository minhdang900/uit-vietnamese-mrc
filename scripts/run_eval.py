"""Vỏ mỏng cho ``evaluation.cli`` — mọi quyết định nằm trong module đã có test.

    python scripts/run_eval.py --models baseline xlmr mbert visobert --full
"""

# Đặt sys.path TRƯỚC mọi import của dự án. Chạy ``python scripts/x.py`` chỉ đưa
# ``scripts/`` vào path, không bối cảnh nào tự thấy ``src/``. Vài dòng ở đây rẻ
# hơn việc bắt người chấm phải ``pip install -e .`` trước khi README chạy được.
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation.cli import main

if __name__ == "__main__":
    main()
