# ĐỀ XUẤT GIẢI PHÁP & TECH STACK
## T11 — Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt
### Vietnamese Extractive Machine Reading Comprehension

**Repo:** `uit-vietnamese-mrc` · **Môn:** CS116 — UIT · **Ngày:** 2026-09-12

---

# 1. Bài toán

| | |
|---|---|
| **Task** | Extractive Machine Reading Comprehension (MRC) |
| **Input** | `context` (đoạn văn Wikipedia tiếng Việt) + `question` |
| **Output** | `answer span` — chuỗi **con** của context, xác định bởi `(start_char, end_char)` |
| **Không phải** | Generative QA. Model **không sinh** chữ mới; nó **chỉ ra vị trí** |
| **Dataset** | UIT-ViQuAD 2.0 (`taidng/UIT-ViQuAD2.0`), định dạng SQuAD-2.0 |
| **Metric** | Exact Match (EM) + token-level F1 |

**Vì sao "extractive" quyết định mọi thứ phía sau:** không gian đầu ra bị chặn trong context ⇒ (a) metric so khớp chuỗi là hợp lý, (b) có một **bất biến kiểm được**: `predict(ctx, q) in ctx` luôn đúng. Bất biến này được test cho mọi model — nó bắt được cả lỗi tokenizer lệch offset lẫn lỗi model "bịa" đáp án.

---

# 2. Ràng buộc thật của máy này — và vì sao nó đổi kiến trúc

Khảo sát phần cứng:

```
Apple M5 Pro · 18 cores · 48 GB unified memory · arm64
```

PyTorch hỗ trợ **MPS (Metal Performance Shaders)** trên Apple Silicon.

| | Dự án cũ (CPU-only) | Dự án này (MPS) |
|---|---|---|
| Fine-tune transformer | ❌ bất khả thi | ✅ **khả thi** |
| PhoBERT + QA head | ❌ không làm được (PhoBERT không có QA head sẵn) | ✅ **làm được** |
| Chiến lược buộc phải chọn | Inference-only trên checkpoint có sẵn | **Fine-tune thật trên train split đầy đủ** |
| EM kỳ vọng | ~40 | **~55–70** |

> **Đây là điểm khác biệt căn bản.** Giới hạn lớn nhất của dự án cũ — *"không fine-tune được nên phải dùng inference-only"* — **không còn tồn tại**. 48 GB unified memory đủ để fine-tune model `base` (~135M params) với batch size thực tế.

**Hệ quả:** đề tài T11 nêu *"PhoBERT encoder + QA head"*. Dự án cũ không thực hiện được phần này. Dự án này **thực hiện đúng yêu cầu đó**.

---

# 3. Kiến trúc giải pháp

## 3.1 Sơ đồ khối

```
┌──────────────────────────────────────────────────────────────────┐
│  TẦNG DỮ LIỆU                                                    │
│                                                                  │
│  HuggingFace: taidng/UIT-ViQuAD2.0                               │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐   parse SQuAD-2.0 JSON → List[Example]      │
│  │  data.py        │   dedup theo CONTEXT (không phải article)    │
│  │                 │   split theo CONTEXT  → chống group leakage  │
│  │                 │   assert_no_leakage() → FAIL RUN nếu vi phạm │
│  └────────┬────────┘                                             │
│           │                                                      │
│  ┌────────▼────────┐   question_type: single- vs multi-sentence   │
│  │  tagging.py     │   length_bucket: <100/100-200/200-300/300+   │
│  └────────┬────────┘   (heuristic — ghi rõ trong báo cáo)         │
└───────────┼──────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────┐
│  TẦNG MODEL — chung một interface: Predictor.predict(ctx,q)->str │
│           │                                                      │
│  ┌────────▼──────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ baseline_tfidf.py │  │ transformer_qa.py│  │ finetune.py   │ │
│  │                   │  │                  │  │               │ │
│  │ TF-IDF + cosine   │  │ HF QA head       │  │ MPS training  │ │
│  │ sentence retrieval│  │ + windowing.py   │  │ AdamW + warmup│ │
│  │ (SÀN so sánh)     │  │   doc-stride     │  │ early stopping│ │
│  │                   │  │   offset mapping │  │               │ │
│  └───────────────────┘  └──────────────────┘  └───────┬───────┘ │
│                                                        │         │
│                            ┌───────────────────────────▼───────┐ │
│                            │ 4 MODEL ĐƯỢC SO SÁNH:             │ │
│                            │  1. TF-IDF          (baseline)    │ │
│                            │  2. XLM-R squad2    (zero-shot)   │ │
│                            │  3. mBERT      + QA (fine-tuned)  │ │
│                            │  4. PhoBERT-v2 + QA (fine-tuned)  │ │
│                            └───────────────────────────────────┘ │
└───────────┬──────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────┐
│  TẦNG ĐÁNH GIÁ                                                   │
│  metrics.py   EM + token-F1, max-over-ground-truths              │
│  evaluate.py  harness → results/*.json  (+ provenance: commit,   │
│               timestamp, n, split — MỌI số truy vết được)        │
│               breakdown: by_context_length · by_question_type    │
│               bucket n<30 tự gắn cờ unreliable                   │
└───────────┬──────────────────────────────────────────────────────┘
            │
   ┌────────┴────────┬──────────────────┐
   ▼                 ▼                  ▼
figures/*.png   Streamlit demo    Báo cáo 30–50 trang
```

## 3.2 Vì sao 4 model — mỗi cái trả lời một câu hỏi khác nhau

| Model | Câu hỏi khoa học nó trả lời | Fine-tune? |
|---|---|---|
| **TF-IDF** | *Bài toán này giải được bằng so khớp từ khoá thuần không?* | không |
| **XLM-R squad2** | *Transfer từ SQuAD-2.0 tiếng Anh sang tiếng Việt được bao nhiêu?* | không (zero-shot) |
| **mBERT + QA** | *Fine-tune trên tiếng Việt thêm được bao nhiêu, với encoder đa ngữ không tối ưu cho tiếng Việt?* | ✅ MPS |
| **PhoBERT-v2 + QA** | *Pretraining chuyên tiếng Việt đóng góp bao nhiêu?* | ✅ MPS |

**Thiết kế này tạo ra một ablation tự nhiên:**
```
TF-IDF  →  XLM-R zero-shot  →  mBERT fine-tuned  →  PhoBERT fine-tuned
  │              │                    │                     │
  └─ sàn         └─ giá trị của       └─ giá trị của         └─ giá trị của
                    transfer            fine-tune              pretraining
                    đa ngữ              in-domain              tiếng Việt
```
Mỗi bước cô lập **một** biến. Đây là điều đề bài gọi là *"thiết kế thí nghiệm"* — và là thứ phân biệt mức Trung bình với mức Khó.

---

# 4. Tech stack

## 4.1 Lựa chọn và lý do

| Lớp | Công nghệ | Vì sao chọn cái này |
|---|---|---|
| **Ngôn ngữ** | Python **3.12** | Python 3.9.6 hệ thống không chạy được torch mới. 3.12 có wheel đầy đủ cho arm64 |
| **Quản lý package** | **uv** 0.11 | Nhanh hơn pip nhiều lần; `uv pip freeze` pin phiên bản chính xác ⇒ **tái lập** |
| **Deep learning** | **PyTorch** + **MPS** backend | MPS = GPU của Apple Silicon. Không dùng CUDA vì không có NVIDIA |
| **Model hub** | **HuggingFace transformers** | `AutoModelForQuestionAnswering` + `offset_mapping` là hạ tầng chuẩn cho extractive QA |
| **Tokenizer** | **HF tokenizers** (fast, Rust) | **Bắt buộc dùng fast tokenizer** — chỉ nó cung cấp `return_offsets_mapping`, thứ cần để map token span về ký tự gốc |
| **Dataset** | **HF datasets** | Tải `taidng/UIT-ViQuAD2.0` trực tiếp, có cache |
| **Baseline** | **scikit-learn** `TfidfVectorizer` | Đề bài yêu cầu TF-IDF; sklearn là chuẩn của môn học (L02–L05) |
| **Test** | **pytest** + **pytest-cov** | TDD. `-m "not slow"` cho vòng lặp nhanh, không cần tải model |
| **Hình vẽ** | **matplotlib** | Chuẩn của môn học; đủ cho training curve + bar chart |
| **Demo** | **Streamlit** | Tiêu chí chấm #1 của thầy: web demo trực quan nhập context + question |
| **Bảng biểu** | **pandas** | Xuất bảng kết quả sang markdown/latex cho báo cáo |

## 4.2 Hai quyết định kỹ thuật dễ bị làm sai âm thầm

### A. Bắt buộc fast tokenizer + offset mapping

```python
tok = AutoTokenizer.from_pretrained(name, use_fast=True)   # KHÔNG được dùng slow
enc = tok(question, context,
          return_offsets_mapping=True,     # ← thứ cho phép map token → ký tự
          return_overflowing_tokens=True,  # ← tạo nhiều window cho context dài
          stride=128, max_length=384, truncation="only_second")
```

Không có `offset_mapping`, cách duy nhất để lấy lại chuỗi đáp án là `tokenizer.decode()` — và **decode làm mất dấu, mất khoảng trắng gốc, thêm/bớt ký tự**. Với tiếng Việt điều này đặc biệt tai hại: `"hoà"` có thể ra `"hoa"`. Lỗi này **không crash**, chỉ làm EM thấp một cách khó hiểu.

→ Có test: `decode_span()` phải trả về **đúng substring** của context gốc.

### B. `truncation="only_second"` — không được cắt câu hỏi

`only_second` cắt **context** (đối số thứ hai), giữ nguyên **question**. Nếu dùng `truncation=True`, câu hỏi dài có thể bị cắt ⇒ model mất thông tin cần thiết.

→ Có test: token của context luôn nằm **sau** token của question; span không bao giờ lấy từ vùng question.

## 4.3 Cấu hình fine-tuning cho MPS

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"

TrainingArguments(
    per_device_train_batch_size = 12,      # 48GB unified memory cho phép
    gradient_accumulation_steps = 2,       # batch hiệu dụng 24
    learning_rate               = 3e-5,    # chuẩn cho BERT-family fine-tune
    num_train_epochs            = 2,       # ViQuAD train 28k câu; 2 epoch thường đủ
    warmup_ratio                = 0.1,     # tránh phá biểu diễn pretrained
    weight_decay                = 0.01,    # regularization
    fp16                        = False,   # ⚠️ MPS KHÔNG hỗ trợ fp16 của CUDA
    eval_strategy               = "epoch",
    load_best_model_at_end      = True,    # ← early stopping thực chất
    metric_for_best_model       = "f1",
)
```

> **⚠️ `fp16=False` là bắt buộc trên MPS.** Đặt `fp16=True` sẽ crash hoặc cho NaN loss. Đây là lỗi số 1 khi port code viết cho CUDA sang Apple Silicon.

## 4.4 Ba quyết định về metric đặc thù tiếng Việt

Đây là chỗ ViQuAD **khác** SQuAD tiếng Anh. Cả ba đều được **pin bằng test** để người sau không "sửa cho giống SQuAD".

| # | Quyết định | Lý do |
|---|---|---|
| **A** | **Không loại bỏ mạo từ** | SQuAD tiếng Anh loại `a/an/the`. Tiếng Việt **không có mạo từ**; `các`, `con`, `những` là **loại từ/lượng từ** và **có thể là phần của đáp án đúng** |
| **B** | **Token hoá theo khoảng trắng** (âm tiết), không word-segment | Khớp với eval chính thức của ViQuAD ⇒ so sánh được với công trình khác. Tránh phụ thuộc `underthesea`/`VnCoreNLP` và tránh segmenter sai làm nhiễu metric |
| **C** | **Giữ dấu tiếng Việt** | `"hoà"` ≠ `"hoa"`. Chuẩn hoá **không được** strip diacritics |

---

# 5. Sự thật về dữ liệu — đã kiểm trực tiếp

| Split (sau dedup theo context) | Questions | Contexts | Impossible | Gold answers |
|---|---|---|---|---|
| train | 28.454 | 4.101 | 9.216 (32,4%) | ✅ có |
| validation | 2.854 | 424 | 851 (29,8%) | ✅ có |
| test | 4.407 | 659 | 0 (0,0%) | ❌ **rỗng toàn bộ** |

## 5.1 ⚠️ Test split là blind set — không thể đánh giá trên đó

Kiểm trực tiếp: **cả 4.407 câu test có `answers.text` rỗng**, đồng thời `is_impossible=False` cho toàn bộ. Hai điều này **mâu thuẫn nhau** nếu coi là nhãn thật (is_impossible=False thì phải có đáp án) ⇒ đây là **placeholder**: đáp án bị lược bỏ để dùng cho leaderboard.

**Hệ quả bắt buộc:**
1. **Mọi** số báo cáo là trên **validation split**. Ghi rõ ở mọi nơi.
2. Đây là **tính chất của dataset**, không phải thiếu sót của nhóm.
3. Dùng train → fit, validation → report. Test chỉ dùng để kiểm pipeline chạy end-to-end.

## 5.2 32,4% câu là impossible — metric đo hai kỹ năng

ViQuAD 2.0 theo chuẩn SQuAD-2.0 nên có câu **không có đáp án trong context**. Model phải vừa (a) tìm đúng span, vừa (b) **biết khi nào nên trả lời rỗng**.

⇒ EM tổng thể **không phải** "độ chính xác khi tìm span". Báo cáo phải tách hai chỉ số này.

---

# 6. Kế hoạch TDD — 9 phase

Chi tiết đầy đủ (test specs từng phase): `docs/PLAN_TDD.md`.

| Phase | Nội dung | Cổng kiểm soát |
|---|---|---|
| 0 | Môi trường + verify MPS | `torch.backends.mps.is_available()` → True |
| **1** | **normalize + metrics** ⭐ | **100% coverage; 3 quyết định A/B/C có test pin** |
| 2 | Tầng dữ liệu + leakage guard | `assert_no_leakage` **raise** khi vi phạm |
| 3 | TF-IDF baseline | output ⊆ context (property test) |
| 4 | windowing + transformer QA | windowing test pass **không cần tải model** |
| 5 | Harness đánh giá | provenance đủ 6 trường; bucket nhỏ tự gắn cờ |
| 6 | Fine-tune MPS (mBERT, PhoBERT) | training curve thật; `detect_overfitting` có test |
| 7 | Hình vẽ | sinh lại được từ `results/*.json` |
| 8 | Streamlit demo | module **import được** (v1 cũ chết vì SyntaxError) |

## 6.1 Vì sao METRIC trước MODEL

Dự án tiền nhiệm (`AUDIT.md`) thất bại vì **báo cáo số liệu chưa từng được sinh ra**: 68,5% EM / 84,5% F1 là số bịa, mâu thuẫn giữa README / báo cáo / app, và trái với artifact thật duy nhất (0,6% EM).

Đó không phải lỗi cẩu thả — đó là **lỗi kiến trúc**: không có thước đo đáng tin nào được xác lập *trước* khi có số để báo cáo.

> **Nguyên tắc:** thước đo phải được kiểm chứng **trước** vật được đo.

Vì vậy Phase 1 là metrics — hàm thuần, dễ test nhất, và mọi thứ khác phụ thuộc vào nó.

## 6.2 Bốn bất biến — vi phạm là chặn merge

1. **Extractive:** mọi `predict()` trả về substring của context.
2. **Không leakage:** `assert_no_leakage` chạy trong mọi đường split và **làm fail run**.
3. **Truy vết được:** mọi số trong báo cáo ⟶ một file `results/*.json` ⟶ một commit hash.
4. **Có n kèm số:** không bảng nào có số mà thiếu `count`; `count < 30` tự gắn `unreliable`.

---

# 7. Kết quả kỳ vọng — ghi TRƯỚC khi chạy

Đăng ký giả thuyết trước (pre-registration) để phòng vệ chống tự lừa mình: nếu chưa biết mình mong đợi gì, **mọi con số đều trông hợp lý**.

| Model | EM kỳ vọng | F1 kỳ vọng | Lý do |
|---|---|---|---|
| TF-IDF | **~0–3** | ~20–30 | Trả về **cả câu**; gold là **cụm vài từ** ⇒ overlap có, trùng khít không |
| XLM-R zero-shot | ~35–45 | ~50–60 | Transfer từ SQuAD-2.0 tiếng Anh, không thấy tiếng Việt in-domain |
| mBERT fine-tuned | ~50–60 | ~68–78 | Fine-tune in-domain, nhưng encoder không tối ưu tiếng Việt |
| PhoBERT fine-tuned | **~60–70** | **~78–85** | Pretraining chuyên tiếng Việt + fine-tune in-domain |

**Tín hiệu báo động cần điều tra thay vì ăn mừng:**
- TF-IDF EM > 10% ⇒ nghi **bug** hoặc **leakage**
- Bất kỳ model nào EM > 85% ⇒ nghi **leakage** (SOTA ViQuAD quanh mức đó)
- Fine-tuned **kém hơn** zero-shot ⇒ nghi lỗi cấu hình training (lr, fp16 trên MPS, label alignment)

---

# 8. Giới hạn sẽ nêu trong báo cáo

- [ ] Đánh giá trên **validation**, không phải test — vì test split là **blind, toàn bộ 4.407 gold rỗng** (§5.1)
- [ ] **29,8% câu validation là impossible** — metric tổng trộn hai kỹ năng (§5.2)
- [ ] `question_type` (single/multi-sentence) là **heuristic tự gán**, **không** phải nhãn có sẵn của ViQuAD
- [ ] Mọi bảng breakdown ghi `n`; bucket `n < 30` không dùng để kết luận
