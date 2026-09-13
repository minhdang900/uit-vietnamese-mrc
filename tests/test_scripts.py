"""Mọi script trong ``scripts/`` phải chạy được ngay sau khi clone.

README hứa ``python scripts/make_figures.py``. Lời hứa đó SAI cho tới khi có
test này: ``scripts/`` không có gì đưa ``src/`` vào ``sys.path``, nên lệnh trong
README gãy với ``ModuleNotFoundError: No module named 'reporting'``. pytest thì
không bắt được vì ``pyproject.toml`` tự đặt ``pythonpath = ["src", "."]`` —
nghĩa là bộ test xanh trong khi hướng dẫn chạy thì hỏng.

``app/streamlit_app.py`` đã tự bootstrap đúng vì lý do này ("rẻ hơn việc bắt
người chấm phải ``pip install -e .``"). Script cũng phải vậy.

Chạy bằng subprocess với môi trường sạch: gọi hàm trong tiến trình hiện tại sẽ
thừa hưởng ``sys.path`` của pytest và test luôn xanh một cách vô nghĩa.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    """Chặn trường hợp glob hỏng khiến test dưới lặng lẽ không kiểm gì."""
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_imports_project_modules_without_pythonpath(script: Path):
    """Chạy từ gốc repo không được gãy ở khâu import.

    Chỉ khẳng định đúng một điều: không có ``ModuleNotFoundError``. Bản đầu của
    test này đòi ``returncode == 0``, và nó đỏ ở ``check_hypotheses.py`` vì
    script đó nhận đường dẫn ở vị trí thứ nhất chứ không dùng argparse — ``--help``
    bị hiểu thành tên thư mục. Đó là một hành vi KHÁC, không phải lỗi sys.path;
    gộp hai thứ vào một test thì thông điệp khi đỏ không còn chỉ đúng chỗ.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    done = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )

    assert "ModuleNotFoundError" not in done.stderr, (
        f"{script.name} không import được module dự án nếu thiếu PYTHONPATH — "
        f"lệnh trong README sẽ gãy:\n{done.stderr}"
    )
