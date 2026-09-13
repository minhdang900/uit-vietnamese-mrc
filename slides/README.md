# Slide thuyết trình

| Tệp | Nội dung |
|---|---|
| `CS116_T11_Slide_BaoCao.pptx` | **Deck 15 phút** — 15 slide, có speaker notes kèm mốc thời gian từng slide |
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

## Cấu trúc 15 slide

| # | Slide | Mốc |
|---:|---|---|
| 1 | Bìa + lộ trình | 0:00 |
| 2 | Bài toán — trích xuất, không sinh | 0:30 |
| 3 | Dữ liệu · Phát hiện 1 — test split là blind set | 1:30 |
| 4 | Bốn hệ thống, bốn câu hỏi | 2:45 |
| 5 | Kiến trúc bốn tầng | 3:45 |
| 6 | Bốn bất biến + cơ chế thực thi | 5:15 |
| 7 | Hai quyết định kỹ thuật (windowing, fast tokenizer) | 6:30 |
| 8 | Kết quả — bảng chính | 7:45 |
| 9 | Phát hiện 2 — mBERT thắng vì biết từ chối | 8:45 |
| 10 | Phát hiện 3 — ViSoBERT suy sụp về trả rỗng | 10:00 |
| 11 | Vì sao ViSoBERT thất bại | 11:00 |
| 12 | Kỹ thuật phần mềm — TDD + Docker | 12:00 |
| 13 | Demo sáu màn hình | 12:45 |
| 14 | Giới hạn + hướng phát triển | 13:45 |
| 15 | Kết luận | 14:30 |

Slide 12 và 13 là hai slide cắt được nếu thiếu thời gian; slide 9–11 là phần
không nên cắt, vì đó là ba phát hiện chính.

## Nguồn số liệu

Mọi con số trong deck đã được đối chiếu với `results/*.json`. Deck **không** đọc
file lúc chạy như `index.html` — nên khi huấn luyện lại, sửa số trong
`build_deck.js` rồi dựng lại, và đối chiếu với `report/assets/table_results.md`.
