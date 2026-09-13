"""Vật liệu cho báo cáo, SINH từ ``results/*.json``.

``figures.py`` lo phần hình; file này lo phần chữ và số — bảng Markdown/CSV, tờ
xuất xứ, danh sách kiểm. Tách ra vì phần này không cần matplotlib, nên nó chạy
trong bộ test nhanh.

Lý do module này tồn tại nằm ở bất biến #3: *mọi số trong báo cáo ⟶ một file
``results/*.json`` ⟶ một commit hash*. Qua 30–50 trang thì không ai giữ nổi điều
đó bằng kỷ luật chép tay — bản v1 của dự án chết đúng vì vậy. Sinh ra thì bảng
trong báo cáo và bảng trong ``results/`` không thể lệch nhau.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

__all__ = ["markdown_table", "csv_table", "provenance_rows", "impossible_share",
           "build_report_assets"]

#: Trường bắt buộc để một kết quả được phép xuất hiện trong báo cáo. Thiếu bất
#: kỳ trường nào thì con số ấy không dựng lại được, và bất biến #3 đứt.
REQUIRED_PROVENANCE = ("commit", "timestamp", "device", "split")


def _columns(rows: list[dict]) -> list[str]:
    """Tên cột theo thứ tự xuất hiện, gộp mọi dòng.

    Không dùng ``rows[0].keys()``: một model thiếu ``latency_ms`` thì cột đó
    biến mất khỏi cả bảng, và báo cáo im lặng mất một chiều so sánh.
    """
    seen: dict[str, None] = {}
    for row in rows:
        seen.update(dict.fromkeys(row))
    return list(seen)


def _require(rows: list[dict]) -> list[str]:
    if not rows:
        raise ValueError(
            "Không có dòng nào để dựng bảng. Chạy scripts/run_eval.py trước — "
            "một bảng rỗng trong báo cáo trông y hệt bảng thật."
        )
    return _columns(rows)


def markdown_table(rows: list[dict]) -> str:
    """Bảng Markdown: tiêu đề, gạch phân cách, rồi mỗi bản ghi một dòng."""
    columns = _require(rows)

    def line(cells) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    return "\n".join([
        line(columns),
        line("---" for _ in columns),
        *(line(row.get(c, "") for c in columns) for row in rows),
    ]) + "\n"


def csv_table(rows: list[dict]) -> str:
    """Cùng dữ liệu, dạng CSV — để dán vào Word/Excel/Google Docs."""
    columns = _require(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({c: row.get(c, "") for c in columns} for row in rows)
    return buffer.getvalue()


def _vn_date(timestamp: str) -> str:
    """``2026-09-12T14:03:03+00:00`` → ``"12/09/2026"``."""
    try:
        year, month, day = timestamp[:10].split("-")
    except ValueError:
        return timestamp
    return f"{day}/{month}/{year}"


def provenance_rows(results: list[dict]) -> list[dict]:
    """Xuất xứ từng kết quả: commit, thiết bị, phiên bản torch, ngày, n.

    Gãy ồn ào nếu thiếu trường bắt buộc. Một con số không gắn được commit thì
    người chấm không dựng lại được nó — và "số không dựng lại được" chính là
    thứ đã giết bản v1 của dự án.
    """
    rows = []
    for result in results:
        missing = [f for f in REQUIRED_PROVENANCE if not result.get(f)]
        if missing:
            raise ValueError(
                f"Kết quả {result.get('model', '?')!r} thiếu {', '.join(missing)} "
                f"— không đủ xuất xứ để đưa vào báo cáo (bất biến #3). "
                f"Sinh lại bằng scripts/run_eval.py."
            )
        env = result.get("env") or {}
        rows.append({
            "model": result["model"],
            "split": result["split"],
            "n": result["overall"]["count"],
            "device": result["device"],
            "torch": env.get("torch", "?"),
            "platform": env.get("platform", "?"),
            "commit": result["commit"],
            "date": _vn_date(result["timestamp"]),
        })
    return rows


def impossible_share(results: list[dict]) -> float:
    """Phần trăm câu impossible trong mẫu đã chấm, TÍNH từ chính kết quả.

    Giới hạn bắt buộc #2 của báo cáo nói metric tổng trộn hai kỹ năng, và tỉ lệ
    này là cái quyết định nó trộn mạnh tới đâu. Viết tay "≈30%" thì đổi ``n``
    một cái là con số lặng lẽ sai.
    """
    if not results:
        raise ValueError("Không có kết quả nào để tính tỉ lệ impossible.")
    first = results[0]
    total = first["overall"]["count"]
    return round(100.0 * first["impossible_only"]["count"] / total, 2)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def build_report_assets(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Sinh toàn bộ vật liệu số cho báo cáo. Trả về các đường dẫn đã ghi.

    Ghi cả hai định dạng: ``.md`` cho báo cáo viết bằng Markdown, ``.csv`` để
    dán vào Word/Excel/Google Docs. Rẻ hơn nhiều so với việc đoán sai định dạng
    rồi bắt người viết gõ lại số bằng tay.
    """
    from .figures import collect_results, format_results_table

    results_dir, out_dir = Path(results_dir), Path(out_dir)
    results = collect_results(results_dir)          # gãy ồn ào nếu rỗng
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    table = format_results_table(results)
    written.append(_write(out_dir / "table_results.md", markdown_table(table)))
    written.append(_write(out_dir / "table_results.csv", csv_table(table)))

    provenance = provenance_rows(results)
    written.append(_write(
        out_dir / "provenance.md",
        "# Xuất xứ của mọi con số trong báo cáo\n\n"
        "Bất biến #3: mỗi số ⟶ một file `results/*.json` ⟶ một commit hash.\n"
        "Bảng này sinh tự động; đừng sửa tay.\n\n"
        + markdown_table(provenance)
        + f"\nCâu impossible chiếm **{impossible_share(results):.2f}%** mẫu đã "
        "chấm — đây là lý do báo cáo tách `answerable_only` và "
        "`impossible_only`: metric tổng trộn hai kỹ năng khác nhau.\n",
    ))
    written.append(_write(out_dir / "provenance.csv", csv_table(provenance)))

    figures = sorted((results_dir / "figures").glob("*.png"))
    if figures:
        figures_out = out_dir / "figures"
        figures_out.mkdir(exist_ok=True)
        for figure in figures:
            written.append(Path(shutil.copy2(figure, figures_out / figure.name)))

    return written
