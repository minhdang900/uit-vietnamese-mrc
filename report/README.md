# Vật liệu cho báo cáo

Thư mục này gom mọi thứ cần để **viết** báo cáo CS116 T11. Bản thân báo cáo
(30–50 trang) không nằm ở đây — đây là nguyên liệu.

## Quy tắc một dòng

> **Không chép số bằng tay.** Mọi con số trong báo cáo phải đến từ `assets/`,
> và `assets/` sinh ra từ `results/`.

Đây không phải sự cầu toàn. Phiên bản v1 của đúng dự án này **chết vì chép tay**:
báo cáo ghi 68,5% EM / 84,5% F1 — những con số chưa từng được sinh ra, mâu thuẫn
giữa README, báo cáo và app, trong khi artifact thật duy nhất ghi 0,6% EM
(`docs/PLAN_TDD.md` §1). Sinh tự động thì bảng trong báo cáo và bảng trong
`results/` **không thể** lệch nhau.

## Sinh lại

```bash
python scripts/run_eval.py --models baseline xlmr mbert visobert --full  # nếu cần chấm lại
python scripts/make_figures.py     # hình  → results/figures/
python scripts/make_report.py      # bảng + xuất xứ + bản sao hình → report/assets/
```

Chạy lại `make_report.py` bất cứ lúc nào; nó ghi đè `assets/`.

## Có gì trong đây

| Đường dẫn | Sinh tự động? | Dùng để |
|---|:--:|---|
| `assets/table_results.md` | ✅ | Bảng kết quả chính, dán vào báo cáo Markdown |
| `assets/table_results.csv` | ✅ | Cùng bảng, dán vào Word / Excel / Google Docs |
| `assets/provenance.md` | ✅ | Phụ lục truy vết: commit, thiết bị, torch, ngày, `n` |
| `assets/provenance.csv` | ✅ | Cùng nội dung, dạng bảng tính |
| `assets/figures/*.png` | ✅ | 5 hình, bản sao từ `results/figures/` |
| `BAO_CAO.md` | ✅ | **Bản báo cáo hoàn chỉnh** — ghép từ template + bảng |
| `BAO_CAO.template.md` | ❌ | Văn xuôi báo cáo; nhúng bảng bằng `<!-- include: -->` |
| `outline.md` | ❌ | Dàn ý + danh sách kiểm bắt buộc (viết tay) |
| `README.md` | ❌ | Chính file này |
| `../slides/index.html` | ❌ | Slide thuyết trình (mở bằng trình duyệt) |

Hai file viết tay có nội dung số — `BAO_CAO.template.md` và `slides/index.html` —
đều bị một test **cấm gõ tay giá trị EM/F1**: số chỉ được vào qua
`<!-- include: -->` (báo cáo) hoặc qua hình sinh từ `results/` (slide). Chấm lại
model là cả hai tự đúng theo.

Xuất báo cáo sang PDF hoặc Word:

```bash
pandoc report/BAO_CAO.md -o BAO_CAO.pdf --resource-path=report
pandoc report/BAO_CAO.md -o BAO_CAO.docx --resource-path=report
```

Slide in ra PDF bằng phím `P` ngay trong trình duyệt.

Hai file không sinh tự động là **văn xuôi và cấu trúc**, không chứa số liệu —
cố ý như vậy, để không có chỗ nào cho một con số lạc đường.

## Nguồn khác, không chép vào đây

Những thứ sau đã nằm sẵn trong repo và nên **trích dẫn tại chỗ** thay vì nhân bản:

| Nguồn | Nội dung |
|---|---|
| `results/hypotheses.md` | Giả thuyết ghi **trước** khi chạy — đối chiếu bằng `python scripts/check_hypotheses.py results` |
| `results/eval_*.json` | Nguồn gốc của mọi con số, kèm `sample_predictions` để trích ví dụ lỗi |
| `results/training_curve_*.json` | Loss và val EM/F1 từng epoch |
| `results/finetune_*.log` | Nhật ký huấn luyện thật |
| `docs/SOLUTION.md` | Lập luận thiết kế và các quyết định kỹ thuật |
| `docs/PLAN_TDD.md` | Kế hoạch TDD, bốn bất biến, bốn giới hạn bắt buộc |
| `README.md` (gốc repo) | Kết quả tóm tắt, kiến trúc, hướng dẫn chạy |
