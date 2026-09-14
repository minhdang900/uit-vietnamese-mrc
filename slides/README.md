# Slide thuyết trình

| Tệp | Nội dung |
|---|---|
| `CS116_T11_Slide_BaoCao.pptx` | **Deck ~18 phút** — 21 slide theo cấu trúc sáu mục, có speaker notes kèm mốc thời gian từng slide |
| `CS116_T11_Slide_BaoCao.pdf` | Bản PDF của deck — dùng khi máy trình chiếu không có font |
| `build_deck.js` | Mã sinh deck (`pptxgenjs`). Sửa số liệu ở đây rồi chạy lại, đừng sửa tay trong PowerPoint |
| `fonts/` | Sáu font TTF cần cài trước khi mở `.pptx` |
| `index.html` | Deck HTML cũ — mở thẳng bằng trình duyệt, không cần server |

## Cài font trước khi mở .pptx

Deck dùng đúng hệ thiết kế "Organic" của demo: **Baloo 2** cho tiêu đề, **Nunito**
cho thân chữ. Đây là font của Google Fonts, **không có sẵn trên macOS/Windows** —
chưa cài thì PowerPoint tự thay bằng font khác và bố cục lệch đi.

```bash
# macOS
cp slides/fonts/*.ttf ~/Library/Fonts/
```

Hoặc chọn cả sáu tệp trong `fonts/` → mở bằng Font Book → "Install Font".

Sáu tệp này đã được gộp từ ba subset `latin` + `latin-ext` + `vietnamese`, nên
phủ đủ dấu tiếng Việt — bản tải thẳng từ Google Fonts theo từng subset thì không.

Không muốn cài font thì dùng `CS116_T11_Slide_BaoCao.pdf`; bản PDF đã nhúng sẵn font.

## Dựng lại deck

```bash
cd slides
node build_deck.js     # cần pptxgenjs: npm install pptxgenjs
```

## Cấu trúc 21 slide — sáu mục

Deck đi theo cấu trúc báo cáo sáu mục (không phải thứ tự kể chuyện của bản cũ):

| # | Slide | Mục | Mốc |
|---:|---|---|---|
| 1 | Bìa + nội dung sáu mục | — | 0:00 |
| 2 | Bối cảnh & động lực | 1 Tổng quan | 0:30 |
| 3 | Phát biểu bài toán — Input → Output | 1 | 1:15 |
| 4 | Đóng góp của nhóm | 1 | 2:05 |
| 5 | Bốn hướng tiếp cận + các bộ dữ liệu đã có | 2 Công trình liên quan | 2:45 |
| 6 | GAP — chỗ nhóm đứng vào | 2 | 3:35 |
| 7 | Nguồn & thống kê split · test là tập ẩn nhãn | 3 Xây dựng dữ liệu | 4:20 |
| 8 | Phân bố nhãn + phân bố độ dài | 3 | 5:20 |
| 9 | Phân bố chủ đề + cái bẫy lấy mẫu | 3 | 6:10 |
| 10 | Bốn hệ thống, bốn câu hỏi | 4 Phương pháp | 7:10 |
| 11 | Kiến trúc bốn tầng | 4 | 7:55 |
| 12 | Bốn bất biến + cơ chế thực thi | 4 | 8:45 |
| 13 | Thang đo cho tiếng Việt | 4 | 9:35 |
| 14 | Hai quyết định kỹ thuật | 4 | 10:20 |
| 15 | Pipeline huấn luyện: siêu tham số + đường cong | 4 | 11:15 |
| 16 | Bảng kết quả chính | 5 Kết quả | 12:10 |
| 17 | Phát hiện 1 — điểm tổng trộn hai kỹ năng | 5 | 13:05 |
| 18 | Phát hiện 2 — ViSoBERT suy sụp, ca lỗi được chẩn đoán | 5 | 14:05 |
| 19 | Ứng dụng: sáu màn hình + Docker/TDD | 6 Kết luận | 15:20 |
| 20 | Hạn chế & hướng phát triển | 6 | 16:05 |
| 21 | Kết luận | 6 | 17:00 |

Tổng khoảng **18 phút**. Muốn về đúng 15 phút thì cắt **slide 13** (thang đo) và
**slide 19** (ứng dụng) — hai slide này được viết để cắt được. Slide 17 và 18 là
phần không nên cắt: đó là hai phát hiện chính.

## Nguồn số liệu

`build_deck.js` **đọc `../results/*.json` lúc dựng**: mọi EM / F1 / độ trễ / đường
cong / siêu tham số đến từ `eval_*_validation.json` và `training_curve_*.json`.
Huấn luyện lại rồi chạy lại `node build_deck.js` là deck tự đúng theo — không ai
phải sửa số bằng tay.

Bất biến này được test:
`tests/test_report_assets.py::test_the_slide_deck_contains_no_hand_written_metrics`
soi **cả `index.html` và `build_deck.js`** (bản cũ chỉ soi `index.html`, nên số
test trong deck từng trôi 398 → 423 mà không ai phát hiện).

Thống kê mô tả dữ liệu (số câu hỏi, phân bố độ dài, phân bố chủ đề, phân mảnh
token, khoảng tin cậy) nằm trong hằng số `DATA` ở đầu `build_deck.js`, kèm tên
script đã sinh ra chúng (`06_BaoCao_T11/05_BANG_CHUNG/`). Đổi dữ liệu thì chạy
lại script rồi cập nhật đúng một khối đó.
