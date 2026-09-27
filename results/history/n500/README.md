# Lịch sử: đánh giá v1 trên mẫu n = 500

Bốn tệp ở đây là bảng kết quả v1 (2026-09-12), chấm trên **500 câu ngẫu nhiên**
(`reproducible_subset`, seed 42) của validation, không phải toàn bộ 3.814 câu.
Chúng được dời khỏi `results/` ở P2 để `collect_results`/`check_hypotheses`
(chỉ đọc `results/eval_*.json` cấp trên cùng) không bao giờ nhầm chúng với đánh
giá đầy đủ. Không sửa các tệp này.

| Tệp | commit sinh ra | timestamp (UTC) |
|---|---|---|
| `eval_baseline_validation.json` | 590e775 | 2026-09-12T14:06:43 |
| `eval_mbert_validation.json` | 590e775 | 2026-09-12T14:03:03 |
| `eval_xlmr_validation.json` | c63b28c | 2026-09-12T11:27:54 |
| `eval_visobert_validation.json` | c63b28c | 2026-09-12T11:28:13 |

**Tái lập được từng bit.** Trước khi dời, bốn checkpoint được chấm lại với code
P2 (`--limit 500`) và `scripts/regression_n500.py` xác nhận mọi metric tổng,
breakdown và dự đoán mẫu trùng khít (cổng C6, commit c17351a).

Những con số này được trích trong phụ lục (macro `\hist…` của `numbers.tex`).
Checkpoint v1 chọn epoch trên 300 câu của CHÍNH validation — xem
`results/hypotheses.md`, phần P3.
