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

---

## Đăng ký v2 — P4 (ViSoBERT: một biến duy nhất `max_length` 384 → 512)

**Ghi TRƯỚC khi có kết quả `visobert-dev` trên validation**, commit riêng
(`prereg(P4): …`). Run: `visobert-len512`.

### Biến thay đổi — và chỉ biến đó

`max_length` 384 → **512**. Mọi thứ khác giữ nguyên như đối chứng `visobert-dev`
(P3): lr 3e-5, 3 epoch, batch 12 × grad-accum 2, warmup 0,1, weight decay 0,01,
`doc_stride` 128, `max_answer_len` 64, seed 42, cùng tập dev (title, 10 %, seed
42), cùng quy tắc chọn (epoch, τ) trên dev F1, lưới τ [−5; 5] bước 0,25.
Nếu 512 hết bộ nhớ MPS: dùng batch 6 × grad-accum 4 (cùng batch hiệu dụng 24) và
ghi sai khác đó vào file này bằng một commit MỚI **trước** khi chạy thật.

### Cơ chế dự đoán — số đo đầu vào (không phải kết quả)

Đo bằng `scripts/null_labels.py` (chỉ tokenizer, trên toàn bộ 28.454 câu train,
commit 896bc5e), trước khi huấn luyện:

| ViSoBERT, doc_stride 128 | feature | nhãn null | trong đó "đáp án ngoài cửa sổ" | câu nhiều cửa sổ |
|---|---|---|---|---|
| `max_length` 384 | 40.984 | **49,30 %** | 6.895 | 10.386 |
| `max_length` 512 | 32.292 | **39,33 %** | 2.241 | 3.573 |
| (tham chiếu) mBERT, 384 | 30.540 | 36,16 % | 1.135 | 1.884 |

Giả thuyết cửa sổ: vocab 15k khiến ViSoBERT cắt context thành nhiều cửa sổ hơn,
một nửa số feature huấn luyện mang nhãn null, và QA head học "luôn trả rỗng".
512 đưa tỉ lệ null về gần mức của mBERT (model không suy sụp).

### Mốc tham chiếu

ViSoBERT v1 trên validation ĐẦY ĐỦ (n = 3.814): HasAns EM **8,29**, tỉ lệ rỗng
82,88 %. Con số 6,93 trong đặc tả là trên mẫu n = 500; ngưỡng tuyệt đối 16,93
(= 6,93 + 10, "rõ ràng cao hơn") được GIỮ như đặc tả viết, nhưng cổng quyết định
chính là so với đối chứng `visobert-dev`, không phải với v1.

### Cổng — rẽ nhánh CƠ HỌC theo kết quả P3 (đều trên validation đầy đủ)

Nhánh được chọn tự động bởi `check_hypotheses.py` từ eval của `visobert-dev`,
theo đúng định nghĩa suy sụp của P3 (tỉ lệ rỗng ≥ 90 % hoặc HasAns EM < 15):

**A. Đối chứng vẫn suy sụp ⇒ P4 là phép thử quyết định.** CONFIRMED khi và chỉ
khi CẢ BỐN điều sau đúng; ngược lại REFUTED và chẩn đoán cửa sổ/nhãn null được
báo cáo là bị bác bỏ:
1. ΔHasAns EM (len512 − đối chứng) **≥ +10,0**;
2. tỉ lệ rỗng giảm **≥ 30 điểm** so với đối chứng **và** tỉ lệ rỗng tuyệt đối **< 65 %**;
3. HasAns EM tuyệt đối **> 16,93**;
4. McNemar chính xác trên tập **HasAns**, **một phía** H1: b10 > b01
   (b10 = len512 đúng & đối chứng sai), **p < 0,01**.

**B. Đối chứng KHÔNG còn suy sụp (lr 3e-5 đã đủ) ⇒ phát hiện chính là nhiễu lr:**
suy sụp của v1 được quy cho lr 5e-5 (cộng tỉ lệ nhãn null), và báo cáo nói đúng
như vậy. P4 vẫn chạy như đăng ký, cổng trở thành: ΔHasAns EM **≥ +3** **và**
cùng McNemar HasAns một phía p < 0,01 — một hiệu ứng phụ của ngân sách cửa sổ.

Dù nhánh nào, không chọn lại epoch/τ/checkpoint sau khi thấy validation.

### Tín hiệu BÁO ĐỘNG

| Quan sát | Nghi vấn | Kiểm tự động |
|---|---|---|
| EM(len512) > EM(`mbert-dev`) + 10 | Đáng ngờ — nghi leakage/bug | `gate` |
| batch/grad-accum khác 12/2 | Đã dùng phương án OOM — phải có commit ghi sai khác | `config_differs` |
| dev EM − val EM > 10; τ ở mép lưới; tỉ lệ rỗng > 95 % | như P3 | như P3 |

---

## Đăng ký v2 — P5 (mBERT nhiều seed)

**Ghi TRƯỚC khi chạy**, commit riêng (`prereg(P5): …`). Run: `mbert-dev-s43`,
`mbert-dev-s44`.

- Cấu hình giống HỆT `mbert-dev` (P3); chỉ `--seed` đổi (43, 44). Tập dev cố
  định ở `--dev-seed 42` cho cả ba run.
- **Headline vẫn là `mbert-dev` (seed 42) bất kể kết quả seed** (D2): hai seed mới
  chỉ dùng để ước lượng độ dao động, KHÔNG BAO GIỜ để chọn "seed tốt nhất".
- MPS không tất định: ngay cả cùng seed cũng có thể khác — đó là một phần của thứ
  được đo, không phải lỗi.
- **Dự đoán:** độ lệch chuẩn mẫu (ddof = 1) của EM validation đầy đủ qua
  {`mbert-dev`, `mbert-dev-s43`, `mbert-dev-s44`} **≤ 1,5**.
- **Báo động:** bất kỳ seed nào lệch EM so với `mbert-dev` **> 5** ⇒ nghi bug hoặc
  huấn luyện bất ổn — điều tra, không bỏ run. Cộng các báo động của P3.
- Khoảng cách giữa các model trong báo cáo được so với độ lệch chuẩn seed này.

---

## Đăng ký v2 — P6 (PhoBERT, `phobert-dev`)

**Ghi TRƯỚC khi chạy `phobert-dev`**, commit riêng (`prereg(P6): …`).

### Điều kiện vào (cổng khứ hồi — quyết định của người dùng, đã đo)

`scripts/roundtrip_audit.py` → `results/roundtrip_train.json` (19.238 câu
answerable của TRAIN, commit 3c1af71). Cổng: PhoBERT khứ hồi **sau chuẩn hoá EM
≥ 99 %** VÀ cách mBERT **≤ 0,5 điểm**. Kết quả: **ĐẠT** (99,303 % vs mBERT
99,470 %, cách 0,167).

| Tokenizer | cửa sổ | khứ hồi chính xác | sau chuẩn hoá EM | lỗi chính |
|---|---|---|---|---|
| mBERT | 384/128 | 99,366 % | 99,470 % | gold cắt giữa từ (106) |
| ViSoBERT | 384/128 | 92,936 % | 99,574 % | dấu câu dính cuối (1.234) |
| PhoBERT | 256/64 | 75,200 % | 99,303 % | dấu câu dính cuối (4.611) |

Tỉ lệ "chính xác" thấp của PhoBERT là do tách theo khoảng trắng: dấu câu dính vào
từ đứng trước, nên nhãn span có thêm "." — EM bỏ dấu câu nên không đổi. Cổng 100 %
chính xác của kế hoạch ban đầu không đạt được với BẤT KỲ tokenizer nào (kể cả hai
model đã huấn luyện), nên được thay bằng cổng trên.

### Cấu hình (cố định) và các bất lợi đã biết

| run_id | model | lr | epoch | max_length / doc_stride | max_answer_len | seed |
|---|---|---|---|---|---|---|
| `phobert-dev` | vinai/phobert-base-v2 | 3e-5 | 3 | **256 / 64** | 30 | 42 |

batch 12 × 2, cùng tập dev (title, 10 %, seed 42) và quy tắc chọn (epoch, τ) như P3.
`max_answer_len` 30: p95 độ dài đáp án là 37 token (mBERT 38, ViSoBERT 60).

Bất lợi — ghi trước để không thành lời bào chữa sau:
1. **256 token/cửa sổ** (`max_position_embeddings` = 258). `doc_stride` 128 không
   còn chỗ, phải hạ về 64 — sai khác bắt buộc.
2. **Không tách từ.** PhoBERT được pretrain trên văn bản đã tách từ
   (VnCoreNLP/RDRSegmenter, Java, không có offline); ở đây đầu vào là âm tiết thô.
3. **Tokenizer nhanh ≠ tokenizer pretrain ở mẩu hiếm.** Bản bọc `tokenizer.json`
   cho id khác bản chậm ở 69/4.101 context và 80/3.000 cặp câu hỏi + context (mẩu
   BPE hiếm bản chậm đổi thành `<unk>`). Offset thì khớp tuyệt đối.

### Đầu vào cơ chế (đo trước, chỉ tokenizer, toàn bộ train)

Nhãn null của PhoBERT ở 256/64: **47,70 %** (38.498 feature; 5.891 "đáp án ngoài
cửa sổ"; 8.817 câu nhiều cửa sổ) — gần bằng ViSoBERT ở 384 (49,30 %, đã suy sụp)
và xa mBERT (36,16 %, không suy sụp).

### Dự đoán

- **Suy sụp: CÓ** (cùng định nghĩa P3: tỉ lệ rỗng ≥ 90 % hoặc HasAns EM < 15).
  Đây là HỆ QUẢ của chẩn đoán cửa sổ đã đăng ký ở P4: nếu tỉ lệ nhãn null ~50 %
  gây suy sụp cho ViSoBERT thì cũng gây cho PhoBERT. Nếu P4 bác bỏ chẩn đoán đó,
  dự đoán này cũng được kỳ vọng sai — cả hai được báo cáo nguyên trạng.
- **Dải mới** (validation đầy đủ): EM 25–40, F1 27–45, HasAns EM 0–15.
- **Dự đoán gốc giữ nguyên trong hồ sơ và vẫn được chấm** (2026-09-12): EM 60–70,
  F1 78–85, và "PhoBERT > mBERT" (so với `mbert-dev`). Không xoá, không sửa.

### Tín hiệu BÁO ĐỘNG

Như P3 (khoảng cách dev/val > 10, τ ở mép lưới, tỉ lệ rỗng > 95 %, leakage, thứ tự
đăng ký), cộng: EM(`phobert-dev`) > EM(`mbert-dev`) + 10 (đáng ngờ); cấu hình
huấn luyện khác 256/64/batch 12 (sai khác chưa ghi).
