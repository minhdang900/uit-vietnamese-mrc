# Giả thuyết đăng ký trước (pre-registration)

**Ghi ngày 2026-09-12, TRƯỚC khi chạy đánh giá nào.** Commit tại thời điểm ghi:
xem `git log` cho commit ngay trước commit chứa file này.

Mục đích: nếu không biết trước mình mong đợi gì, **mọi con số đều trông hợp lý** —
kể cả con số sai. Đây là cơ chế phòng vệ chống tự lừa mình, và là phản ứng trực
tiếp với thất bại của phiên bản trước dự án (báo cáo số liệu chưa từng được sinh ra).

## Dự đoán định lượng

| Model | EM kỳ vọng | F1 kỳ vọng | Cơ sở của dự đoán |
|---|---|---|---|
| TF-IDF Baseline | **0–3** | 20–30 | Trả về **cả câu**; gold là **cụm vài từ**. Overlap token có, trùng khít gần như không |
| XLM-R squad2 (zero-shot) | 35–45 | 50–60 | Đã fine-tune QA nhưng trên SQuAD-2.0 tiếng Anh; chưa thấy tiếng Việt in-domain |
| mBERT + QA (fine-tuned) | 50–60 | 68–78 | Fine-tune in-domain, nhưng encoder đa ngữ không tối ưu cho tiếng Việt |
| PhoBERT-v2 + QA (fine-tuned) | **60–70** | **78–85** | Pretraining chuyên tiếng Việt + fine-tune in-domain |

## Dự đoán định tính

1. **Khoảng cách EM–F1 của TF-IDF sẽ rất lớn** (F1 cao hơn EM khoảng 20+ điểm).
   Đây là bằng chứng trực quan rằng hai metric đo hai thứ khác nhau.
2. **Mọi model sẽ kém hơn trên context dài.** Context vượt `max_length=384` bị cắt
   thành nhiều window; span có thể rơi vào window thiếu ngữ cảnh.
3. **Mọi model sẽ kém hơn trên câu multi-sentence.** Suy luận qua nhiều câu khó hơn
   so khớp trong một câu.
4. **PhoBERT > mBERT.** Nếu KHÔNG đúng, đó là phát hiện đáng báo cáo: nó gợi ý
   pretraining tiếng Việt không bù được việc PhoBERT có vocab nhỏ hơn, hoặc cấu hình
   fine-tune chưa tối ưu.

## Tín hiệu BÁO ĐỘNG — điều tra thay vì ăn mừng

| Quan sát | Nghi vấn |
|---|---|
| TF-IDF EM > 10% | Bug trong metric, hoặc leakage |
| Bất kỳ model EM > 85% | Leakage — SOTA trên ViQuAD quanh mức đó |
| Model fine-tuned **kém hơn** zero-shot | Lỗi cấu hình training: label alignment sai, lr quá cao, hoặc `fp16=True` trên MPS |
| Điểm trên câu impossible = 100% mà answerable ≈ 0% | Model học "luôn trả về rỗng" — kiểm `null_threshold` |
| EM == F1 chính xác trên mọi model | Metric bị nối sai — F1 phải cho điểm bán phần |

## Cam kết

Kết quả thật sẽ được **so với bảng này** trong báo cáo, kể cả (đặc biệt là) khi
không khớp. Không sửa giả thuyết sau khi thấy kết quả.

---

## Đăng ký v2 — P3 (dev split, huấn luyện lại mBERT & ViSoBERT)

**Ghi ngày 2026-09-27, TRƯỚC khi chạy `mbert-dev` và `visobert-dev`.** Commit chứa
phần này phải là một commit RIÊNG (`prereg(P3): …`); `scripts/finetune.py` từ chối
chạy nếu `HEAD:results/hypotheses.json` chưa liệt kê `run_id` hoặc file giả thuyết
còn sửa dở, và ghi `prereg_commit` vào `training_curve_{run_id}.json`.
`scripts/check_hypotheses.py` kiểm lại: commit đăng ký < `started_utc` < thời điểm chấm.

### Vì sao có P3

Checkpoint v1 (`models/mbert`, `models/visobert`) chọn epoch trên 300 câu của CHÍNH
validation rồi báo cáo trên validation. P3 chọn **epoch và ngưỡng null τ** trên một
tập dev tách từ train; validation chỉ được tải để kiểm leakage và được chấm **đúng
một lần** sau khi huấn luyện xong (`run_eval.py` từ chối ghi đè nếu không có
`--force --reason`, mọi lần gọi ghi vào `results/eval_invocations.jsonl`).

### Tập dev (cố định trước)

- `split_by_context(train, val_frac=0.1, seed=42, group="title")`: 10 % **article**
  của train. Theo code hiện tại: **13 article, 515 context, 3.420 câu** (1.109
  impossible, 32,43 %); train còn 125 article / 3.586 context / 25.034 câu.
- `val_frac` đếm theo article, mà cỡ article rất lệch, nên số câu phụ thuộc seed.
  Seed 42 được đăng ký ở đây; con số trên được CHẤP NHẬN như nó là — **không đổi
  seed để có cỡ đẹp hơn**.
- Ba phép kiểm `assert_no_leakage` (train/dev, train/val, dev/val) chạy trước khi
  tải model. Danh sách qid của dev ghi vào `results/split_dev_{run_id}.json` kèm
  sha256 của file train.
- **Sai khác so với đặc tả (D3).** Đặc tả AC-4 viết "dev split qua
  `split_by_context()`" (mặc định chia theo context). Validation của ViQuAD không
  chung article nào với train, nên dev chia theo context sẽ dễ hơn validation
  (cùng article, khác đoạn). Nhóm chọn chia theo **title/article** — vẫn là hàm
  đó, thêm tham số `group`. Chia theo article mạnh hơn: không chung article thì
  cũng không chung context.

### Cấu hình huấn luyện (cố định)

| run_id | model | lr | epoch | max_length / doc_stride | max_answer_len | seed |
|---|---|---|---|---|---|---|
| `mbert-dev` | bert-base-multilingual-cased | 3e-5 | 3 | 384 / 128 | 30 | 42 |
| `visobert-dev` | uitnlp/visobert | **3e-5** | 3 | 384 / 128 | 64 | 42 |

batch 12 × grad-accum 2, warmup 0,1, weight decay 0,01. ViSoBERT v1 dùng lr 5e-5;
`visobert-dev` dùng 3e-5 như mBERT để bỏ lr khỏi danh sách khác biệt (D4). Run
này là **đối chứng** cho P4 (chỉ đổi `max_length` 384 → 512).

### Quy tắc chọn (cố định)

- Mỗi epoch k: chấm TOÀN BỘ dev ở τ = 0, lưu cửa sổ vào
  `preds_{run}_dev_epoch{k}.jsonl`; quét τ ∈ [−5; 5] bước 0,25 (41 giá trị) offline.
- Chọn **cặp (epoch, τ)** có **dev F1** cao nhất; hoà ⇒ epoch sớm hơn, rồi |τ| nhỏ
  hơn, rồi τ nhỏ hơn. Ghi `models/{run}/selection.json` (epoch, τ, max_length,
  doc_stride, max_answer_len, dev_n, dev_seed, group); `run_eval.py` đọc file này.
- Headline = checkpoint + τ đã chọn, chấm MỘT lần trên validation đầy đủ (3.814 câu).
  Báo động bên dưới chỉ dẫn tới điều tra/viết lại, **không bao giờ** chọn lại
  epoch/τ/checkpoint sau khi đã thấy validation.

### Dự đoán định lượng (validation đầy đủ, n = 3.814)

| run_id | EM | F1 | Khác | Cơ sở |
|---|---|---|---|---|
| `mbert-dev` | 47–58 | 56–68 | \|EM − EM(`mbert` v1, full val)\| ≤ 3 | Ít hơn ~12 % dữ liệu train, thêm 1 epoch, τ chọn trên dev: các hiệu ứng nhỏ và ngược chiều nhau. v1 trên n = 500: EM 50,80 / F1 59,49 |
| `visobert-dev` | 25–38 | 27–42 | HasAns EM 0–15; tỉ lệ rỗng 50–100 % | Xem dự đoán suy sụp |

### Dự đoán nhị phân: `visobert-dev` ở lr 3e-5 **VẪN suy sụp — CÓ**

- **Định nghĩa suy sụp (vận hành):** trên validation đầy đủ, ở τ đã chọn,
  tỉ lệ dự đoán rỗng ≥ 90 % **hoặc** HasAns EM < 15.
- **Cơ sở.** v1 (lr 5e-5) không phân kỳ: train loss giảm đều 3,05 → 2,57 → 2,19
  qua 3 epoch, và epoch 3 bắt đầu nhích (val F1 25,6 → 30,5 trên 300 câu). Tức là
  model học CHẬM chứ không mất ổn định vì lr cao. Hạ lr xuống 3e-5 và bớt ~12 % dữ
  liệu làm việc học chậm thêm, nên trong 3 epoch nhóm dự đoán HasAns EM vẫn < 15.
  Bằng chứng trực tiếp nhất (nhưng bị gây nhiễu): lần chạy ViSoBERT đầu tiên ở
  **lr 3e-5** (cùng lúc với `max_answer_len` 30, dừng sau epoch 1) chỉ đạt val F1
  12,25 — thấp hơn cả mốc "luôn trả rỗng". Lần đó đổi hai biến so với v1 nên không
  tách được phần của lr; dự đoán này vì vậy là suy luận, không phải kết quả đã biết.
  Giả thuyết cạnh tranh mà P4 kiểm: tokenizer vocab 15k sinh nhiều token hơn ⇒
  nhiều cửa sổ hơn ⇒ nhiều nhãn null hơn khi huấn luyện.
- **Hệ quả cho P4.** Nếu dự đoán ĐÚNG (vẫn suy sụp), P4 kiểm xem riêng
  `max_length 512` có gỡ suy sụp không. Nếu SAI (hết suy sụp ở 3e-5), lr 5e-5 là
  nguyên nhân chính của v1 và P4 đo tác động của 512 trên một model không suy sụp.
- Không sửa dự đoán này sau khi thấy kết quả dev hay validation.

### Tín hiệu BÁO ĐỘNG (điều tra, không chọn lại)

| Quan sát | Nghi vấn | Kiểm tự động |
|---|---|---|
| dev EM − val EM > 10 | Dev không đại diện cho validation | `dev_val_gap` |
| τ chọn được ở mép lưới (−5 hoặc 5) | Lưới quá hẹp | `tau_at_grid_edge` |
| Tỉ lệ rỗng > 95 % ở bất kỳ run nào | Suy sụp | `empty_rate_above` |
| `assert_no_leakage` thất bại | Leakage — run dừng trước khi tốn GPU | trong `finetune.py` |
| Thứ tự commit đăng ký / bắt đầu / chấm sai | Đăng ký trước không hợp lệ | `prereg_order_violated` |
