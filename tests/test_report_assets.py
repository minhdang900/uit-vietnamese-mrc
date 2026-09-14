"""Gom vật liệu cho báo cáo TỪ ``results/*.json``.

Bất biến #3 của dự án: mọi số trong báo cáo ⟶ một file ``results/*.json`` ⟶ một
commit hash. Cách duy nhất giữ được điều đó qua 30–50 trang là **sinh** bảng chứ
không chép tay — chính việc chép tay đã giết phiên bản v1 (xem docs/PLAN_TDD.md
§1: 68,5% EM là số chưa từng được sinh ra).

Test dựng cây results giả trong ``tmp_path``, không đụng ``results/`` thật.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from reporting.assets import (
    build_report_assets,
    csv_table,
    impossible_share,
    markdown_table,
    metric_literals,
    provenance_rows,
    render_report,
)


def make_eval(model="mbert (fine-tuned)", em=50.8, f1=59.49, n=500,
              commit="590e775", timestamp="2026-09-12T14:03:03+00:00") -> dict:
    """Một file đánh giá tối thiểu, đủ trường mà báo cáo cần."""
    return {
        "model": model, "dataset": "UIT-ViQuAD 2.0", "split": "validation",
        "n": n, "timestamp": timestamp, "commit": commit, "device": "mps",
        "env": {"torch": "2.14.0", "platform": "macOS-26.4.1-arm64-arm-64bit"},
        "overall": {"EM": em, "F1": f1, "count": n},
        "answerable_only": {"EM": 54.57, "F1": 66.60, "count": 361},
        "impossible_only": {"EM": 41.01, "count": 139},
        "avg_latency_ms": 12.264,
    }


def test_markdown_table_renders_a_header_rule_and_one_row_per_record():
    rows = [{"model": "A", "EM": 1.0}, {"model": "B", "EM": 2.0}]

    lines = markdown_table(rows).strip().splitlines()

    assert len(lines) == 4, "phải là: tiêu đề, gạch phân cách, rồi 2 dòng dữ liệu"
    assert lines[0].startswith("|") and lines[0].endswith("|")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= set("-:")
    assert "A" in lines[2] and "B" in lines[3]


def test_markdown_table_keeps_n_beside_every_number():
    """Bất biến #4: không bảng nào có số mà thiếu ``count``."""
    rows = [{"model": "A", "EM": 1.0, "n": 500}]

    body = markdown_table(rows).strip().splitlines()[2]

    assert "500" in body


def test_csv_table_round_trips_through_a_csv_reader():
    """CSV để dán thẳng vào Word/Excel — phải đọc lại được, không chỉ nhìn giống."""
    rows = [{"model": "A", "EM": 1.0}, {"model": "B", "EM": 2.0}]

    parsed = list(csv.DictReader(io.StringIO(csv_table(rows))))

    assert [r["model"] for r in parsed] == ["A", "B"]
    assert parsed[0]["EM"] == "1.0"


def test_rendering_an_empty_table_fails_loudly():
    """Bảng rỗng trong báo cáo trông y hệt bảng thật cho tới khi ai đó nhìn kỹ."""
    with pytest.raises(ValueError):
        markdown_table([])


def test_provenance_carries_commit_and_device_for_every_model():
    """Bất biến #3: mỗi con số phải truy được về commit nào, máy nào, lúc nào.

    Không có tờ này thì bảng kết quả trong báo cáo chỉ là "một con số nào đó" —
    người chấm không có cách nào dựng lại nó.
    """
    results = [make_eval(model="A", commit="590e775"),
               make_eval(model="B", commit="abc1234")]

    rows = provenance_rows(results)

    assert [r["model"] for r in rows] == ["A", "B"]
    assert [r["commit"] for r in rows] == ["590e775", "abc1234"]
    for row in rows:
        assert row["device"] == "mps"
        assert row["torch"] == "2.14.0"
        assert row["split"] == "validation"
        assert row["n"] == 500
        assert row["date"] == "12/09/2026", "ngày kiểu Việt Nam, không phải ISO"


def test_provenance_refuses_a_result_with_no_commit():
    """Kết quả không gắn commit thì KHÔNG được vào báo cáo.

    Đây chính xác là căn bệnh của v1: số tồn tại nhưng không dựng lại được.
    Thà gãy lúc sinh vật liệu còn hơn để người chấm phát hiện.
    """
    orphan = make_eval(model="A")
    del orphan["commit"]

    with pytest.raises(ValueError, match="commit"):
        provenance_rows([orphan])


def test_impossible_share_is_computed_not_transcribed():
    """Giới hạn bắt buộc #2 nêu tỉ lệ câu impossible — phải TÍNH từ dữ liệu.

    139/500 = 27,8%. Viết tay "≈30%" thì khi đổi n con số đó lặng lẽ sai.
    """
    assert impossible_share([make_eval()]) == pytest.approx(27.8)


def write_results(root, models=("mbert", "xlmr")):
    """Cây ``results/`` giả: một eval mỗi model, cộng một hình."""
    results = root / "results"
    (results / "figures").mkdir(parents=True)
    for i, name in enumerate(models):
        (results / f"eval_{name}_validation.json").write_text(
            json.dumps(make_eval(model=name, em=10.0 * (i + 1))), encoding="utf-8"
        )
    (results / "figures" / "model_comparison.png").write_bytes(b"\x89PNG stub")
    return results


def test_build_writes_tables_in_both_markdown_and_csv(tmp_path):
    """Markdown cho báo cáo viết bằng Markdown, CSV để dán vào Word/Excel."""
    results = write_results(tmp_path)
    out = tmp_path / "report" / "assets"

    build_report_assets(results, out)

    assert (out / "table_results.md").is_file()
    assert (out / "table_results.csv").is_file()
    assert "mbert" in (out / "table_results.md").read_text(encoding="utf-8")


def test_build_copies_every_figure(tmp_path):
    """Hình phải nằm cạnh bảng, không bắt người viết đi lục ``results/``."""
    results = write_results(tmp_path)
    out = tmp_path / "report" / "assets"

    build_report_assets(results, out)

    assert (out / "figures" / "model_comparison.png").read_bytes() == b"\x89PNG stub"


def test_build_writes_a_provenance_sheet(tmp_path):
    results = write_results(tmp_path)
    out = tmp_path / "report" / "assets"

    build_report_assets(results, out)

    text = (out / "provenance.md").read_text(encoding="utf-8")
    assert "590e775" in text and "12/09/2026" in text


def test_build_returns_every_path_it_wrote(tmp_path):
    """Người gọi in ra được đúng những gì vừa sinh, không phải đoán."""
    results = write_results(tmp_path)
    out = tmp_path / "report" / "assets"

    written = build_report_assets(results, out)

    assert written, "không trả về đường dẫn nào"
    for path in written:
        assert path.exists(), f"{path} được báo là đã ghi nhưng không tồn tại"


def test_build_fails_loudly_when_no_evaluation_exists(tmp_path):
    """Thiếu ``results/`` thì gãy, không sinh ra một thư mục rỗng trông như thật."""
    empty = tmp_path / "results"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        build_report_assets(empty, tmp_path / "out")


# ── Báo cáo: ghép từ template + bảng đã sinh ────────────────────────────────

def test_render_report_replaces_an_include_marker_with_the_asset(tmp_path):
    """Báo cáo KHÔNG chép số; nó nhúng bảng đã sinh tại chỗ đánh dấu."""
    (tmp_path / "table_results.md").write_text("| model |\n| --- |\n| A |\n",
                                               encoding="utf-8")

    out = render_report("Kết quả:\n\n<!-- include: table_results.md -->\n", tmp_path)

    assert "| model |" in out
    assert "include:" not in out, "còn sót chỗ đánh dấu chưa thay"


def test_render_report_fails_when_an_included_asset_is_missing(tmp_path):
    """Thiếu bảng thì gãy, không để lại một lỗ trống trong báo cáo đã nộp."""
    with pytest.raises(FileNotFoundError, match="table_results.md"):
        render_report("<!-- include: table_results.md -->", tmp_path)


def test_metric_literals_lists_the_numbers_that_must_never_be_typed_by_hand():
    """Rút ra chính những con số chỉ được phép đến từ results/."""
    literals = metric_literals([make_eval(model="A", em=50.8, f1=59.49)])

    assert "50,80" in literals and "59,49" in literals, (
        "phải bắt được EM/F1 ở định dạng số Việt Nam"
    )


def test_the_real_report_template_contains_no_hand_written_metrics():
    """Bất biến #1, áp cho chính bản báo cáo — cơ chế, không phải kỷ luật.

    Đây là test duy nhất trong file đọc repo thật thay vì tmp_path, và có lý do:
    thứ cần bảo vệ là BẢN BÁO CÁO SẼ NỘP. Nếu một ngày ai đó gõ "50,80" thẳng
    vào văn xuôi rồi sau đó chấm lại model, con số ấy lặng lẽ sai — đúng kịch
    bản đã giết bản v1.
    """
    from pathlib import Path

    from reporting.figures import collect_results

    root = Path(__file__).resolve().parents[1]
    template = root / "report" / "BAO_CAO.template.md"
    if not template.is_file():
        pytest.skip("chưa có template báo cáo")

    text = template.read_text(encoding="utf-8")
    offenders = [n for n in metric_literals(collect_results(root / "results"))
                 if n in text]

    assert not offenders, (
        f"Số đo bị gõ tay vào template: {offenders}. Nhúng bảng bằng "
        f"<!-- include: ... --> thay vì chép."
    )


def test_the_slide_deck_contains_no_hand_written_metrics():
    """Bất biến #1, áp cho cả slide — CẢ deck HTML lẫn mã dựng deck .pptx.

    Slide dễ bị chép số hơn báo cáo — người ta gõ nhanh một con số cho đẹp ô
    rồi quên. Deck chỉ được hiển thị số qua HÌNH sinh từ ``results/`` hoặc đọc
    thẳng ``results/*.json`` lúc dựng, nên chấm lại model là slide tự đúng theo.

    Vì sao soi cả ``build_deck.js``: bản nộp bài là ``.pptx`` dựng từ tệp đó,
    nhưng test cũ chỉ soi ``index.html`` — và đúng ở tệp quan trọng nhất thì bất
    biến âm thầm ngừng bảo vệ (số test trong deck đã trôi 398 → 423 mà không ai
    phát hiện). Đổi định dạng deliverable thì phải dời test theo.
    """
    from pathlib import Path

    from reporting.figures import collect_results

    root = Path(__file__).resolve().parents[1]
    decks = [root / "slides" / "index.html", root / "slides" / "build_deck.js"]
    decks = [d for d in decks if d.is_file()]
    if not decks:
        pytest.skip("chưa có slide")

    literals = metric_literals(collect_results(root / "results"))
    for deck in decks:
        text = deck.read_text(encoding="utf-8")
        offenders = [n for n in literals if n in text]

        assert not offenders, (
            f"Số đo bị gõ tay vào {deck.name}: {offenders}. Đọc từ results/ "
            f"lúc dựng, hoặc dùng hình trong report/assets/figures/, thay vì chép số."
        )
