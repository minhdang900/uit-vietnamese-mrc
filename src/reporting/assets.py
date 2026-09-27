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
import re
import shutil
from pathlib import Path

__all__ = ["markdown_table", "csv_table", "provenance_rows", "impossible_share",
           "build_report_assets", "render_report", "metric_literals",
           "result_literals", "render_readme_table", "sync_readme_results"]

#: Chỗ đánh dấu nhúng bảng trong template báo cáo.
INCLUDE = re.compile(
    r"^[ \t]*<!--[ \t]*include:[ \t]*(?P<name>[\w.\-/]+)[ \t]*-->[ \t]*$",
    re.MULTILINE,
)

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


def render_report(template: str, assets_dir: str | Path) -> str:
    """Thay mọi ``<!-- include: tên.md -->`` bằng nội dung ``assets_dir/tên.md``.

    Báo cáo do đó KHÔNG chứa con số nào của riêng nó: mỗi bảng là bảng đã sinh
    từ ``results/``. Sửa kết quả rồi chạy lại là báo cáo tự đúng theo.
    """
    assets_dir = Path(assets_dir)

    def swap(match: re.Match) -> str:
        path = assets_dir / match.group("name")
        if not path.is_file():
            raise FileNotFoundError(
                f"Báo cáo nhúng {match.group('name')} nhưng {path} không có. "
                f"Chạy scripts/make_report.py trước."
            )
        return path.read_text(encoding="utf-8").rstrip("\n")

    return INCLUDE.sub(swap, template)


def _vn(value: float) -> str:
    """``50.8`` → ``"50,80"`` — dấu phẩy thập phân như phần còn lại của dự án."""
    return f"{value:.2f}".replace(".", ",")


def metric_literals(results: list[dict]) -> list[str]:
    """Những chuỗi số KHÔNG được phép gõ tay vào văn xuôi báo cáo.

    Chỉ lấy EM/F1 tổng thể: đó là các con số dễ bị chép nhất, và cũng là các con
    số mà bản v1 đã chép sai. Không quét mọi số trong file, vì báo cáo còn nhiều
    số hợp lệ khác (năm, mã môn, số trang) mà chặn thì chỉ gây phiền.
    """
    out: list[str] = []
    for result in results:
        overall = result.get("overall") or {}
        out.extend(_vn(overall[key]) for key in ("EM", "F1") if key in overall)
    return out


#: Trường của một ``eval_*.json`` là KẾT QUẢ (không bao giờ gõ tay), ngoài các
#: macro kết quả trong ``numbers.MACRO_SPEC``.
EVAL_RESULT_PATHS = ("overall.EM", "overall.F1", "answerable_only.EM",
                     "answerable_only.F1", "impossible_only.EM", "avg_latency_ms",
                     "empty_prediction_rate", "null_threshold")


def result_literals(results_dir: str | Path) -> dict[float, str]:
    """Mọi kết quả KHÔNG được gõ tay vào văn bản: giá trị → nguồn.

    Thay ``metric_literals`` (chỉ EM/F1 tổng): gồm EM/F1 tách answerable/
    impossible, latency, tỉ lệ rỗng, τ của MỌI ``eval_*.json`` (cả
    ``history/n500/``), cộng mọi macro kết quả của ``numbers.tex`` — CI, p-value,
    tỉ lệ nhãn null, độ lệch chuẩn seed… Số nguyên bị loại ở bước so khớp
    (``literals.variants``).
    """
    import json

    from .numbers import collect_values, get_path

    results_dir = Path(results_dir)
    out: dict[float, str] = {}
    files = sorted(results_dir.glob("eval_*.json"))
    files += sorted((results_dir / "history").glob("**/eval_*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for path in EVAL_RESULT_PATHS:
            try:
                value = get_path(data, path)
            except KeyError:
                continue
            if isinstance(value, float):
                out.setdefault(value, f"{f.relative_to(results_dir)}:{path}")
    values, _ = collect_values(results_dir)
    for name, (value, spec, _src) in values.items():
        if spec.result and isinstance(value, float):
            out.setdefault(value, f"\\{name}")
    return out


#: Nhãn đẹp cho README — khớp với ``model`` thô ghi trong mỗi ``eval_*.json``.
#: Model chưa có trong bảng này (chạy mới, vd. ``mbert-dev``) thì dùng luôn
#: chuỗi thô: README vẫn đúng, chỉ chưa được đánh bóng tên.
README_MODEL_NAMES: dict[str, str] = {
    "TF-IDF Baseline": "TF-IDF Baseline",
    "Luôn trả rỗng (empty)": "*Luôn trả rỗng*",
    "mbert (fine-tuned)": "mBERT + QA (fine-tuned)",
    "visobert (fine-tuned)": "ViSoBERT + QA (fine-tuned)",
    "XLM-R (squad2, zero-shot)": "XLM-R squad2 (zero-shot)",
}

#: Cột nào được in đậm ở GIÁ TRỊ LỚN NHẤT trong cột (không phải "câu chuyện" —
#: đó là chú thích viết tay mà bảng SINH này thay thế, bất biến #3).
README_BOLD_COLUMNS = ("F1", "answerable_EM", "answerable_F1", "impossible_EM")


#: Chuỗi con để nhận ra model trong ``result["model"]`` thô — không phụ thuộc
#: định dạng chính xác của chuỗi đó (đổi cách viết trong eval_*.json thì đây
#: vẫn khớp).
README_MODEL_NEEDLES: dict[str, str] = {
    "baseline": "tf-idf", "xlmr": "xlm-r", "mbert": "mbert",
    "visobert": "visobert", "empty": "rỗng",
}


def _by_model(results: list[dict], key: str) -> dict | None:
    needle = README_MODEL_NEEDLES[key]
    for r in results:
        if needle in r["model"].lower():
            return r
    return None


def _read_json(results_dir: Path, name: str) -> dict | None:
    import json
    path = Path(results_dir) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _render_findings(results: list[dict], n: int) -> list[str]:
    """"Ba điều bảng này nói ra" — CÙNG số với bảng, không phải bản chép lại.

    Trước phần này từng là văn xuôi viết tay trích số từ bảng (bất biến #3
    bị vi phạm ngay trong README của chính dự án chống việc đó). Sinh ra thì
    không thể lệch nhau nữa, và số tự cập nhật khi có model mới/kết quả mới.
    """
    baseline, xlmr, mbert = _by_model(results, "baseline"), _by_model(results, "xlmr"), _by_model(results, "mbert")
    viso, empty = _by_model(results, "visobert"), _by_model(results, "empty")
    lines = ["### Ba điều bảng này nói ra, mà con số tổng thì không", ""]
    if baseline:
        lines += [
            f"**1. Baseline: F1 {baseline['overall']['F1']:.0f}% nhưng EM ~"
            f"{round(baseline['overall']['EM'])}%.** Nó trả về **cả một câu**, còn gold là **cụm "
            "vài từ** — overlap token có, trùng khít thì không. Khoảng cách EM-F1 đó là bằng chứng "
            "trực quan rằng hai metric đo hai thứ khác nhau.", "",
        ]
    if xlmr and mbert:
        lines += [
            "**2. mBERT thắng XLM-R KHÔNG phải vì tìm span giỏi hơn.** Trên câu answerable, "
            f"XLM-R zero-shot thực ra **tốt hơn** (F1 {xlmr['answerable_only']['F1']:.2f} so với "
            f"{mbert['answerable_only']['F1']:.2f}). mBERT thắng tổng thể vì **biết khi nào KHÔNG "
            f"nên trả lời** tốt hơn hẳn (impossible EM {mbert['impossible_only']['EM']:.2f} so với "
            f"{xlmr['impossible_only']['EM']:.2f}). Với ~30% câu là unanswerable, kỹ năng thứ hai "
            "quyết định bảng xếp hạng. Đây là lý do báo cáo tách `answerable_only` và "
            "`impossible_only` — con số tổng trộn hai kỹ năng và che mất điều này.", "",
        ]
    if viso and empty:
        gap = viso["overall"]["EM"] - empty["overall"]["EM"]
        lines += [
            "**3. ViSoBERT gần suy sụp về \"luôn trả rỗng\".** EM tổng "
            f"{viso['overall']['EM']:.2f} chỉ nhỉnh hơn **{empty['overall']['EM']:.2f}** — EM của "
            f"chính mốc *luôn trả rỗng* (tỉ lệ câu không có đáp án của tập) — đúng **{gap:.2f} "
            f"điểm** (quan sát ở n={n}, không phải một đẳng thức luôn đúng). Bóc tách ra: "
            f"impossible EM **{viso['impossible_only']['EM']:.2f}** nhưng answerable EM chỉ "
            f"**{viso['answerable_only']['EM']:.2f}** — nó gần như chỉ ăn điểm từ việc từ chối trả "
            "lời. `scripts/check_hypotheses.py` phát hiện tự động điều này.", "",
        ]
    return lines


#: (nhãn hiển thị, tên tệp ``training_curve_*.json``).
TRAINING_CURVES = (("mBERT", "training_curve_mbert.json"), ("ViSoBERT", "training_curve_visobert.json"))


def _render_training_curves(results_dir: str | Path) -> list[str]:
    """Đường cong huấn luyện — ĐỌC ``training_curve_*.json``, không chép tay.

    Model chưa có ``training_curve_*.json`` (chưa fine-tune) thì hàng của nó
    vắng mặt thay vì lỗi — script chạy được ở bất kỳ giai đoạn nào của lộ trình.
    """
    rows: list[str] = []
    for label, fname in TRAINING_CURVES:
        curve = _read_json(Path(results_dir), fname)
        if not curve:
            continue
        lr = f"{curve['config']['lr']:.0e}".replace("e-0", "e-")
        for i, point in enumerate(curve["curve"]):
            model_cell = f"**{label}** (lr {lr})" if i == 0 else ""
            rows.append(
                f"| {model_cell} | {point['epoch']} | {point['train_loss']:.4f} | "
                f"{point['val_em']:.2f} | {point['val_f1']:.2f} |"
            )
    if not rows:
        return []
    return ["### Đường cong huấn luyện", "",
            "| | epoch | train loss | val EM | val F1 |",
            "|---|---:|---:|---:|---:|", *rows]


def render_readme_table(results_dir: str | Path) -> str:
    """Toàn bộ vùng SINH của README giữa ``<!-- BEGIN:results -->``/``<!-- END:results -->``.

    Không chỉ bảng: bất biến #3 (mọi số kết quả ⟶ ``results/*.json``) từng bị
    chính README vi phạm — phần văn xuôi "Ba điều bảng này nói ra" và bảng
    "Đường cong huấn luyện" chép tay lại đúng những số đã có trong bảng/JSON
    phía trên, và hai bản chép đó lệch nhau là chuyện chỉ còn là thời gian.
    Gộp cả ba vào một hàm SINH duy nhất thì không còn chỗ nào để lệch.

    Cùng nguồn với ``report/assets/table_results.md``
    (``figures.collect_results`` + ``format_results_table``): mọi
    ``results/eval_*.json`` ở cấp cao nhất, kể cả dòng "luôn trả rỗng" (N7) một
    khi ``results/eval_empty_validation.json`` tồn tại — không cần đặc cách,
    glob đã bắt được nó.

    Định dạng README dùng dấu CHẤM thập phân (quy ước đã có từ trước của
    README, khác dấu phẩy của LaTeX/slide) — xem quy ước hiện tại trong phần
    "Kết quả" phía trên vùng sinh này.
    """
    from .figures import collect_results, format_results_table

    results = collect_results(results_dir)
    rows = format_results_table(results)
    n = rows[0]["n"]

    best_em_model = max(rows, key=lambda r: r["EM"])["model"]
    col_best = {col: max(r[col] for r in rows) for col in README_BOLD_COLUMNS}

    def cell(row: dict, col: str) -> str:
        text = f"{row[col]:.2f}"
        return f"**{text}**" if row[col] == col_best[col] else text

    lines = [
        "| Model | EM | F1 | answerable EM / F1 | impossible EM | Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = README_MODEL_NAMES.get(row["model"], row["model"])
        em_text = f"{row['EM']:.2f}"
        if row["model"] == best_em_model:
            name, em_text = f"**{name}**", f"**{em_text}**"
        lines.append(
            f"| {name} | {em_text} | {cell(row, 'F1')} | "
            f"{cell(row, 'answerable_EM')} / {cell(row, 'answerable_F1')} | "
            f"{cell(row, 'impossible_EM')} | {row['latency_ms']:.1f} ms |"
        )
    lines += ["", *_render_findings(results, n), *_render_training_curves(results_dir)]
    return "\n".join(lines) + "\n"


#: Cặp đánh dấu vùng sinh trong README — ``BEGIN`` luôn đứng đầu dòng riêng.
README_MARKERS = ("<!-- BEGIN:results -->\n", "<!-- END:results -->")


def sync_readme_results(results_dir: str | Path, readme_path: str | Path = "README.md") -> Path:
    """Ghi lại vùng đánh dấu của README bằng ``render_readme_table`` hiện tại.

    Đây là hành động thật đứng sau lời hứa ở đầu README ("chạy lại
    ``scripts/make_report.py`` để cập nhật"). Không tự thêm marker nếu README
    chưa có — gãy ồn ào thay vì âm thầm bỏ qua, cùng triết lý với phần còn lại
    của module này (thiếu bằng chứng thì dừng, không đoán).
    """
    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    begin, end = README_MARKERS
    start = text.find(begin)
    # tìm END SAU START: README có thể nhắc marker trong văn xuôi (vd. hướng dẫn
    # ở đầu mục "Kết quả"), và END của câu văn đó đứng trước BEGIN thật.
    stop = text.find(end, start + len(begin)) if start != -1 else -1
    if start == -1 or stop == -1:
        raise ValueError(
            f"{readme_path} thiếu cặp {begin.strip()!r} / {end!r} — không đồng bộ được."
        )
    start += len(begin)
    new_text = text[:start] + render_readme_table(results_dir) + text[stop:]
    if new_text != text:
        readme_path.write_text(new_text, encoding="utf-8")
    return readme_path


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

    # Báo cáo ghép sau cùng: nó nhúng chính các bảng vừa ghi ở trên.
    template = out_dir.parent / "BAO_CAO.template.md"
    if template.is_file():
        written.append(_write(
            out_dir.parent / "BAO_CAO.md",
            render_report(template.read_text(encoding="utf-8"), out_dir),
        ))

    figures = sorted((results_dir / "figures").glob("*.png"))
    if figures:
        figures_out = out_dir / "figures"
        figures_out.mkdir(exist_ok=True)
        for figure in figures:
            written.append(Path(shutil.copy2(figure, figures_out / figure.name)))

    return written
