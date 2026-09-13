# Kiến trúc hệ thống — `uit-vietnamese-mrc`

**Dự án:** Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt (Vietnamese Extractive MRC)
**Môn:** CS116 — Lập trình Python cho Máy học · Đề tài T11 · UIT, ĐHQG-HCM
**Trạng thái:** Accepted · **Ngày:** 13/09/2026 · **Commit tham chiếu:** `dc85f58`

> Tài liệu này mô tả *hệ thống đã được xây*, không phải hệ thống dự định xây.
> Mọi con số trích ở đây đến từ `results/*.json`, `report/assets/provenance.md`
> hoặc từ script đo trong `06_BaoCao_T11/05_BANG_CHUNG/`. Xem thêm
> [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) cho phần thiết kế chi tiết từng thành
> phần, và [`SOLUTION.md`](SOLUTION.md) cho đề xuất giải pháp ban đầu.

---

## 1. Bối cảnh

### 1.1 Bài toán

Cho một đoạn văn `context` (Wikipedia tiếng Việt) và một `question`, hệ thống
**trích xuất** đoạn văn bản chứa câu trả lời — một chuỗi **con** của chính
context đó, xác định bởi cặp `(start_char, end_char)`.

| | |
|---|---|
| Task | Extractive Machine Reading Comprehension |
| Dataset | UIT-ViQuAD 2.0 (`taidng/UIT-ViQuAD2.0`), định dạng SQuAD-2.0 |
| Metric | Exact Match + token-level F1 (quy ước SQuAD-2.0) |
| **Không phải** | Generative QA — model **không sinh** chữ mới, nó **chỉ ra vị trí** |

Chữ *extractive* quyết định toàn bộ kiến trúc phía sau. Vì không gian đầu ra bị
chặn trong context, hệ thống có một **bất biến kiểm được bằng máy**:

```
predict(context, question) là substring của context   — luôn đúng, cho mọi model
```

Bất biến này không phải một lời hứa trong báo cáo; nó là một assertion trong test
chạy cho **mọi** implementation của `Predictor`. Nó bắt được hai lớp lỗi hoàn
toàn khác nhau bằng cùng một phép kiểm: tokenizer lệch offset, và model "bịa"
đáp án.

### 1.2 Lực tác động lên thiết kế

Bốn ràng buộc, xếp theo mức độ định hình kiến trúc:

| # | Ràng buộc | Hệ quả kiến trúc |
|---|---|---|
| 1 | **Phiên bản trước của dự án đã báo cáo những con số chưa từng được sinh ra** | Provenance trở thành cấu trúc dữ liệu bắt buộc, không phải tuỳ chọn (§5.3) |
| 2 | **Test split của ViQuAD 2.0 là blind set** — 7.301 câu gold rỗng nhưng `is_impossible = False` | Cổng `assert_gradeable()` chặn trước mọi lần chấm (§5.4) |
| 3 | **PhoBERT không có fast tokenizer** ⇒ không có `offset_mapping` | Đổi encoder tiếng Việt sang ViSoBERT (ADR-004) |
| 4 | **Máy chạy là Apple Silicon có MPS**, 48 GB unified memory | Fine-tune thật khả thi ⇒ có tầng huấn luyện, không chỉ inference |

Ràng buộc #1 là ràng buộc *quan trọng nhất* và cũng là ràng buộc dễ bị xem nhẹ
nhất. Nó không phải ràng buộc kỹ thuật mà là ràng buộc về **tính trung thực**, và
nó được giải quyết bằng kiến trúc chứ không bằng kỷ luật cá nhân: khi mọi con số
bắt buộc phải mang theo commit hash và thiết bị sinh ra nó, việc gõ tay một con
số trở thành *khó hơn* việc chạy lại thí nghiệm.

---

## 2. Sơ đồ kiến trúc

### 2.1 Bốn tầng

```
┌─ TẦNG DỮ LIỆU ───────────────────────────────────────────────────────────┐
│                                                                          │
│  HuggingFace Hub ──► scripts/fetch_data.py ──► data/raw/viquad2_*.json    │
│                                                       │                  │
│                                    ┌──────────────────▼───────────────┐  │
│                                    │ mrc/data.py                      │  │
│                                    │   parse_squad → list[Example]    │  │
│                                    │   deduplicate_contexts           │  │
│                                    │   split_by_context  (group=ctx)  │  │
│                                    │   assert_no_leakage  ── FAIL RUN │  │
│                                    │   assert_gradeable   ── FAIL RUN │  │
│                                    └──────────────────┬───────────────┘  │
│                                    ┌──────────────────▼───────────────┐  │
│                                    │ mrc/tagging.py                   │  │
│                                    │   question_type  (heuristic)     │  │
│                                    │   length_bucket                  │  │
│                                    └──────────────────┬───────────────┘  │
└───────────────────────────────────────────────────────┼──────────────────┘
                                                        │
┌─ TẦNG MODEL ── một interface duy nhất: Predictor.predict(ctx, q) -> str ─┐
│                                                       │                  │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌──▼───────────────┐  │
│  │ baseline_tfidf.py  │  │ transformer_qa.py    │  │ features.py      │  │
│  │  TF-IDF + cosine   │  │  QA head + inference │  │  char→token map  │  │
│  │  sentence retrieval│  │  predict_detailed()  │  │ training.py      │  │
│  └────────────────────┘  └──────────┬───────────┘  │  lịch học, chọn  │  │
│                                     │              │  epoch (hàm thuần)│  │
│                          ┌──────────▼───────────┐  └──┬───────────────┘  │
│                          │ windowing.py         │     │                  │
│                          │  make_windows (tự cài)│    │ scripts/finetune │
│                          │  score_spans (thuần) │     └─► models/{mbert, │
│                          │  select_best_span    │           visobert}/   │
│                          │  decode_span         │                        │
│                          └──────────────────────┘                        │
└───────────────────────────────────────────────────┬──────────────────────┘
                                                    │
┌─ TẦNG ĐÁNH GIÁ ────────────────────────────────────▼──────────────────────┐
│  mrc/normalize.py   chuẩn hoá chuỗi — 3 quyết định đặc thù tiếng Việt      │
│  mrc/metrics.py     EM + token-F1, max-over-ground-truths                  │
│  mrc/evaluate.py    harness + PROVENANCE + breakdown + cờ unreliable       │
│                            │                                              │
│                            └──► results/eval_<model>_validation.json       │
└────────────────────────────────────────────────────┬──────────────────────┘
                                                     │
┌─ TẦNG TRÌNH BÀY ─── chỉ ĐỌC results/, không bao giờ sinh số ───────────────┐
│                                                    │                       │
│   scripts/make_figures.py ──► results/figures/*.png │                      │
│   scripts/make_report.py  ──► report/assets/ (bảng + provenance + hình)    │
│                                                                            │
│   ┌─────────────────────┐        ┌──────────────────────────────────────┐  │
│   │ src/demo/           │◄───────┤ app/  (CHỈ gọi widget Streamlit)     │  │
│   │  results.py  ← đọc  │        │   streamlit_app.py  entry + routing   │  │
│   │  logic.py    thuần  │        │   shell.py          theme, state      │  │
│   │  render.py   thuần  │        │   screens/*.py      6 màn hình        │  │
│   │  theme.py    token  │        └──────────────────────────────────────┘  │
│   └─────────────────────┘                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Quy tắc phụ thuộc

Một mũi tên duy nhất, một chiều, không có ngoại lệ:

```
app/  ──►  src/demo/  ──►  src/mrc/  ──►  results/, data/raw/
```

* `src/mrc/` **không biết** demo tồn tại.
* `src/demo/` **không import** Streamlit.
* `app/` **không chứa** logic — mọi hàm quyết định *nội dung* nằm dưới `src/demo/`.
* `results/` và `data/raw/` là **nguồn đọc**, không phải nguồn ghi của tầng trình bày.

Quy tắc cuối cùng được kernel bắt buộc chứ không chỉ là quy ước: trong
`compose.yaml`, cả ba thư mục `models/`, `data/`, `results/` được mount vào
service `app` với cờ `:ro`.

---

## 3. Ranh giới module

### 3.1 `src/mrc/` — lõi nghiệp vụ (1.503 dòng)

| Module | Dòng | Trách nhiệm | Phụ thuộc nặng |
|---|---:|---|---|
| `normalize.py` | 46 | Chuẩn hoá chuỗi trước khi so khớp | không |
| `metrics.py` | 135 | EM + token-F1, max-over-ground-truths | không |
| `data.py` | 264 | Parse, dedup, split, hai assertion cổng | không |
| `tagging.py` | 84 | `question_type` (heuristic) + `length_bucket` | không |
| `predictor.py` | 37 | `Protocol` chung + `TimedPredictorMixin` | không |
| `baseline_tfidf.py` | 78 | TF-IDF + cosine sentence retrieval | scikit-learn |
| `windowing.py` | 402 | Chọn span, map token→ký tự, doc-stride | không |
| `features.py` | 125 | Map vị trí đáp án ký tự → token cho fine-tune | không |
| `transformer_qa.py` | 182 | Inference với QA head | torch, transformers |
| `training.py` | 247 | Lịch học, chọn epoch, chẩn đoán (hàm thuần) | không |
| `evaluate.py` | 173 | Harness + provenance + breakdown | không |
| `device.py` | 37 | Chọn MPS / CUDA / CPU | torch |

**Điểm cần chú ý:** chín trên mười hai module **không phụ thuộc torch**. Đó là
lựa chọn có chủ đích, không phải may mắn. `windowing.py` — module 402 dòng chứa
phần logic dễ sai âm thầm nhất của toàn hệ thống — gồm toàn hàm thuần nhận
`list[float]` và trả về `tuple[int, int]`. Hệ quả đo được: **43 test cho
`windowing.py`** chạy trong vài millisecond, không tải model, không cần GPU, và
bộ test nhanh của cả dự án (423 test) hoàn thành trong **~1 giây**.

Cùng nguyên tắc áp cho `training.py`: mọi quyết định trong huấn luyện — tính lịch
học, chọn epoch tốt nhất, phát hiện overfitting, phát hiện suy sụp về "luôn trả
rỗng" — là hàm thuần nhận số trả về số. Nếu chúng nằm trong `main()` của script
huấn luyện thì cách duy nhất để kiểm là chạy một epoch **35 phút**.

### 3.2 `src/demo/` — logic trình bày (1.593 dòng)

| Module | Dòng | Trách nhiệm |
|---|---:|---|
| `results.py` | 282 | **Chỗ duy nhất** đọc `results/*.json` và `data/raw/` |
| `render.py` | 671 | Hàm thuần: dữ liệu → chuỗi HTML |
| `theme.py` | 379 | Token hệ thiết kế + CSS + hàm escape |
| `logic.py` | 153 | Gọi predictor, quyết định trả lời/từ chối theo ngưỡng |
| `catalog.py` | 148 | Danh mục model, màn hình, thành viên — **không có metric** |
| `vi.py` | 79 | Định dạng số theo quy ước Việt |
| `text.py` | 60 | Đếm âm tiết, cắt trích đoạn |

`results.py` là điểm nghẽn có chủ ý (*intentional bottleneck*): nếu một con số
xuất hiện trên màn hình mà không đi qua module này thì đó là số bịa. Cả thứ hạng
"tốt nhất" lẫn ô in đậm trong bảng so sánh đều được **tính** từ dữ liệu qua
`best_by()`, không gõ tay — nên huấn luyện lại là màn hình tự nói đúng.

`catalog.py` cố tình chỉ chứa thứ **không đo được**: tên hiển thị, đường dẫn
checkpoint, thứ tự màn hình, tên thành viên. Ranh giới này khiến câu hỏi "số này
từ đâu ra?" luôn có đúng một câu trả lời.

### 3.3 `app/` — vỏ Streamlit (643 dòng)

Sáu màn hình, mỗi màn hình một URL deep-link được: `/` (Hỏi đáp), `/ket-qua`,
`/phan-tich-loi`, `/du-lieu`, `/so-sanh`, `/huan-luyen`.

Ranh giới `app/` ↔ `src/demo/` được giữ nghiêm vì một lý do cụ thể: **một
`SyntaxError` trong file app từng làm hỏng demo mà không test nào bắt được**.
Phản ứng kiến trúc là hai phần — (a) đẩy mọi thứ quyết định nội dung xuống
`src/demo/` dưới dạng hàm thuần, và (b) cho test **compile và import mọi file**
trong `app/`, không riêng file entry.

---

## 4. Luồng dữ liệu

### 4.1 Luồng huấn luyện

```
data/raw/viquad2_train.json
  │  load_squad_file        28.454 câu · 4.101 context · 138 article
  ▼
list[Example]
  │  prepare_train_features(tokenizer, max_length=384, doc_stride=128)
  │    └─ make_windows()  ← TỰ CÀI, không dùng return_overflowing_tokens
  │    └─ _locate_answer_tokens()  char offset → token index
  │       · câu impossible          → nhãn về [CLS]
  │       · window không chứa đáp án → nhãn về [CLS]
  ▼
QADataset ── AdamW + linear warmup (warmup_ratio 0.1, weight_decay 0.01)
  │  compute_schedule() đếm theo optimizer.step(), KHÔNG theo batch
  ▼
models/{mbert,visobert}/  +  results/training_curve_*.json
```

Nhãn `[CLS]` không phải trường hợp biên: **32,4% câu train của ViQuAD 2.0 là
impossible**, nên nhánh này chiếm một phần ba dữ liệu.

Chỗ dễ sai nhất trong toàn luồng là `_locate_answer_tokens`. Nếu map lệch, model
học nhãn sai mà **loss vẫn giảm bình thường** — không exception, không cảnh báo.
Vì vậy bất biến được test không phải "chạy không lỗi" mà là *giải mã nhãn token
phải ra lại đúng đáp án vàng*.

### 4.2 Luồng đánh giá

```
examples (validation)
  │  ① assert_gradeable()   ◄── CỔNG: raise nếu split là blind set
  ▼
for ex in examples:  predictor.predict_timed(ctx, q)   ── latency ĐO, không ước lượng
  ▼
metrics.evaluate(predictions, references)
  │  ② KeyError nếu thiếu bất kỳ qid nào  ◄── không cho phép chấm trên tập con
  ▼
{EM, F1, EM_answerable, F1_answerable, EM_impossible, per_item}
  │  ③ breakdown() theo length_bucket và question_type
  │     count < 30  →  unreliable: true
  ▼
results/eval_<model>_validation.json
   + commit · timestamp · device · torch · platform · split · n   ◄── ④ PROVENANCE
```

Bốn cơ chế phòng vệ được đánh số ở trên đều nằm **trong cấu trúc dữ liệu kết
quả**, không nằm trong quy trình làm việc. Một quy trình có thể bị bỏ qua lúc
gấp; một `raise` thì không.

### 4.3 Luồng demo

```
người dùng chọn model + gõ câu hỏi
  ▼
demo.logic.answer(context, question, predictor)
  │  ưu tiên predict_detailed() → giữ lại BẰNG CHỨNG thay vì vứt đi:
  │     null_delta · start_prob · end_prob · top_k span xếp sau
  ▼
demo.logic.margin(null_delta, threshold) = null_delta + threshold
  │  < 0  → trả lời      ≥ 0  → từ chối
  │  (cùng phép so sánh với select_best_span, viết lại thành hàm thuần)
  ▼
demo.render.*  →  HTML  →  app/screens/ask.py  →  màn hình
```

Điểm thiết kế đáng nói: `TransformerQA.predict()` chỉ trả về chuỗi đáp án và
**vứt bỏ** mọi bằng chứng đã được tính trong `score_spans`. Demo cần đúng những
số đó để giải thích *vì sao* model trả lời hay từ chối. Thay vì tính lại bằng số
minh hoạ viết tay, `predict_detailed()` giữ chúng lại — nên thanh "độ tin cậy"
và biên độ từ chối trên màn hình là **số đo được**.

---

## 5. Bốn bất biến và cơ chế thực thi

Điều làm dự án này khác một bài tập thông thường không phải là bốn bất biến, mà
là việc mỗi bất biến có một **cơ chế thực thi bằng máy** thay vì một đoạn văn
trong báo cáo.

### 5.1 Extractive — mọi `predict()` trả về substring của context

*Thực thi:* test cho mọi implementation của `Predictor`; `decode_span()` cắt
trực tiếp chuỗi gốc và `raise ValueError` nếu khoảng không hợp lệ.

*Vì sao fail-loud:* một span sai là **bug cần sửa**, không phải "không tìm thấy
đáp án". Trả về chuỗi rỗng khi offset hỏng sẽ biến một lỗi thành một con số.

### 5.2 Không leakage — một context chỉ thuộc một split

*Thực thi:* `assert_no_leakage()` chạy trong **mọi** đường split và làm fail cả
run khi vi phạm.

*Chi tiết quyết định thành bại:* đơn vị split là **context**, không phải
article. Dự án tiền nhiệm ghi `train.num_contexts = 138` trong khi dữ liệu thật
có 138 *article* chứa **4.101 context** — trường đó đang đếm title. Vì cơ chế
chống leakage dựa trên việc một context chỉ thuộc một split, nhầm hai đơn vị này
**làm vô hiệu hoá chính cơ chế bảo vệ**. `compute_stats()` báo cáo tách bạch cả
hai con số.

### 5.3 Truy vết được — mọi kết quả mang commit, timestamp, device, split, n

*Thực thi:* `run_evaluation()` gọi `git rev-parse --short HEAD` và `device_info()`
rồi nhét vào chính dict kết quả. Không có đường nào sinh ra một dict kết quả
thiếu các trường này.

*Kết quả thực tế:* bảng provenance của báo cáo sinh tự động từ bốn file JSON —
bốn dòng, mỗi dòng một commit (`590e775`, `c63b28c`), thiết bị `mps`, torch
`2.14.0`, `macOS-26.4.1-arm64`.

### 5.4 Có n kèm số — nhóm `count < 30` tự gắn `unreliable: true`

*Thực thi:* `breakdown()` gắn cờ; không bảng nào trong báo cáo có số thiếu `n`.

*Vì sao ngưỡng 30:* với `n = 3`, mỗi câu đúng/sai làm điểm nhảy 33 điểm. Bucket
`<100` trong kết quả validation có **đúng 1 câu** — F1 của nó (66,67 với TF-IDF,
33,33 với mBERT, 88,89 với XLM-R) là nhiễu thuần tuý, và cờ `unreliable` tồn tại
để bảng không thể trình bày nó như một kết quả mà không bị chú ý.

**Bất biến thứ năm, ở tầng trình bày:** demo không sinh ra con số nào. Mọi số
trên màn hình đọc từ `results/*.json` hoặc đo trực tiếp từ `data/raw/` qua
`demo.results`. Cơ chế thực thi: `:ro` trên ba bind-mount trong `compose.yaml`.

---

## 6. Các quyết định kiến trúc (ADR)

### ADR-001 — Một `Protocol` chung cho mọi model

**Status:** Accepted · **Ngày:** 12/09/2026

**Context.** Hệ thống so sánh bốn thứ rất khác nhau: một bộ truy hồi TF-IDF
không có tham số học, một transformer zero-shot, và hai transformer fine-tuned.
Harness đánh giá phải chạy được cả bốn.

**Decision.** Chốt `Predictor` — một `typing.Protocol` với đúng `name: str` và
`predict(context, question) -> str` — **trước khi viết model nào**.

**Options Considered.**

| | Option A: `Protocol` chốt trước | Option B: lớp cơ sở trừu tượng | Option C: `if isinstance(...)` trong harness |
|---|---|---|---|
| Độ phức tạp | Thấp | Trung bình | Thấp lúc đầu, cao dần |
| Ràng buộc kế thừa | Không (structural typing) | Có — TF-IDF phải kế thừa lớp có "model" | Không |
| Chỗ cho nhánh đặc biệt | Không có | Có | Có, và sẽ được dùng |
| Test bằng stub | Tự nhiên | Phải dựng lớp con | Phải giả lập cả kiểu |

**Trade-off.** `Protocol` cho structural typing: `TfidfRetriever` không cần biết
gì về transformer, và test có thể tiêm một stub năm dòng. Đổi lại, không có chỗ
nào ép buộc tuân thủ lúc biên dịch — điều này được bù bằng
`@runtime_checkable` và bằng test bất biến substring chạy cho mọi model.

**Consequences.**
- ✅ Không có một nhánh `if isinstance(...)` nào trong code đánh giá.
- ✅ Thêm model thứ năm = viết một lớp, không sửa harness.
- ⚠️ `TimedPredictorMixin` tách riêng vì không phải mọi predictor cần đo latency.

---

### ADR-002 — Dedup và split theo CONTEXT, không theo article hay question

**Status:** Accepted · **Ngày:** 12/09/2026

**Context.** ViQuAD cung cấp sẵn train/validation, nhưng hai split có thể chia sẻ
context, và dự án cần chia thêm cho đường cong huấn luyện.

**Decision.** Đơn vị nhóm là **chuỗi context**. `split_by_context()` là
`GroupShuffleSplit` với group = context; `remove_contexts_present_in()` dedup
chéo split; `assert_no_leakage()` canh cửa.

**Options Considered.**

| | Option A: group = context | Option B: group = article (title) | Option C: chia theo câu hỏi |
|---|---|---|---|
| Chống group leakage | ✅ Đúng đơn vị | ✅ Chặt hơn cần thiết | ❌ Hỏng hoàn toàn |
| Số nhóm (train) | 4.101 | 138 | 28.454 |
| Rủi ro | Không | Split mất cân bằng vì nhóm quá lớn | Kết quả cao giả tạo |

**Trade-off.** Option C là cái bẫy: chia theo câu hỏi khiến model được "đọc" đoạn
văn lúc train rồi bị hỏi về chính đoạn đó lúc test. Option B an toàn nhưng 138
nhóm cho 28.454 câu khiến tỉ lệ split dao động mạnh. Option A đúng đơn vị của
bất biến cần bảo vệ.

**Consequences.**
- ✅ `compute_stats()` báo cáo tách bạch `num_contexts` (4.101) và `num_articles` (138).
- ⚠️ Context giống hệt nhau xuất hiện dưới nhiều article là bình thường trong
  ViQuAD; `deduplicate_contexts()` chỉ bỏ cặp `(context, question)` trùng, không
  bỏ câu hỏi của context đó.

---

### ADR-003 — Tự cài doc-stride windowing

**Status:** Accepted · **Ngày:** 12/09/2026

**Context.** Context dài hơn `max_length=384` token phải được cắt thành các cửa
sổ chồng lấp. `transformers` có sẵn `return_overflowing_tokens` cho việc này.

**Decision.** Tự cài `make_windows()` trong `windowing.py`.

**Lý do — một phép đo, không phải một linh cảm.** Trên `transformers 5.17.0`:

| Độ dài context | Số window sinh ra | Đáng lẽ phải là |
|---:|---:|---:|
| 210 token | 2 | 2 |
| 420 token | **2** | 5 |
| 700 token | **2** | 8 |
| 1400 token | **2** | 16 |

Giới hạn ở 2 window bất kể `stride` bằng bao nhiêu. Phần đuôi context bị cắt
**âm thầm**: không exception, không cảnh báo — chỉ là đáp án nằm cuối đoạn văn
thì không bao giờ tìm được.

**Trade-off.** Ảnh hưởng thực tế ở `max_length=384` là nhỏ: chỉ **2/557 context
validation** vượt ngưỡng với tokenizer của mBERT. Nhưng đây là lỗi **đúng-sai âm
thầm**, không phải lỗi hiệu năng, nên chi phí 402 dòng tự cài là đáng — và với
ViSoBERT con số không còn nhỏ: **154/557 context** vượt 357 token (xem §7.2).

**Consequences.**
- ✅ Offset trả về là **tuyệt đối** trong context gốc, không phải trong chunk.
- ✅ Test `test_at_least_one_window_of_long_context_finds_the_answer` là thứ bắt
  được lỗi này, và là thứ ngăn nó quay lại.
- ⚠️ Thêm một đường code phải tự bảo trì khi `transformers` nâng cấp.

---

### ADR-004 — ViSoBERT thay cho PhoBERT

**Status:** Accepted · **Ngày:** 12/09/2026 · **Supersedes:** kế hoạch ban đầu trong `SOLUTION.md`

**Context.** Đề tài T11 nêu đích danh *"PhoBERT encoder + QA head"*. Giả thuyết
đăng ký trước dự đoán PhoBERT-v2 đạt EM 60–70, cao nhất trong bốn model.

**Decision.** Dùng `uitnlp/visobert` — encoder tiếng Việt của chính UIT NLP.

**Options Considered.**

| Model | `is_fast` | `offset_mapping` | Kết luận |
|---|---|---|---|
| `vinai/phobert-base` | ❌ False | ❌ không có | không dùng được |
| `vinai/phobert-base-v2` | ❌ False | ❌ không có | không dùng được |
| `nguyenvulebinh/vi-mrc-base` | — | — | **gated** sau HF auth |
| `FPTAI/videberta-base` | — | — | model id không tồn tại |
| `uitnlp/visobert` | ✅ True | ✅ có | **chọn** |

**Trade-off — vì sao thiếu fast tokenizer là chặn đứng chứ không phải bất tiện.**
Extractive QA cần map *token span* do model dự đoán về *vị trí ký tự* trong
context gốc. Chỉ fast tokenizer (Rust) cung cấp `return_offsets_mapping`. Không
có nó, cách duy nhất lấy lại chuỗi là `tokenizer.decode()` — và decode **làm mất
dấu tiếng Việt** (`"hoà"` → `"hoa"`), phá vỡ cả EM lẫn bất biến substring.
PhoBERT còn đòi input đã word-segment sẵn, khiến việc map ngược về context thô
càng phức tạp.

Đây là ràng buộc **kỹ thuật**, độc lập với việc có GPU hay không.

**Consequences.**
- ✅ `TransformerQA.__init__` **từ chối khởi tạo** model không có fast tokenizer,
  kèm thông điệp giải thích — thay vì chạy rồi cho kết quả sai.
- ❌ Giả thuyết "encoder tiếng Việt chuyên biệt sẽ thắng" **không được xác nhận**:
  ViSoBERT đạt EM 27,80 so với mBERT 50,80. Xem §7.2 — đây là kết quả hợp lệ và
  là phát hiện chính của dự án, không phải một thất bại.

---

### ADR-005 — Cắt chuỗi theo offset ký tự, không dùng `tokenizer.decode()`

**Status:** Accepted · **Ngày:** 12/09/2026

**Context.** Sau khi QA head chọn cặp `(start_token, end_token)`, cần lấy lại
chuỗi đáp án.

**Decision.** `decode_span(context, start_char, end_char)` cắt trực tiếp chuỗi
gốc theo offset ký tự.

**Trade-off.** `tokenizer.decode()` đơn giản hơn một dòng, nhưng nó tái dựng văn
bản từ token — và quá trình đó **làm mất dấu tiếng Việt** và thay đổi khoảng
trắng. Với tiếng Việt, `"hoà"` ≠ `"hoa"`; sai một dấu là sai cả EM lẫn bất biến
substring. Cắt chuỗi gốc là cách **duy nhất** bảo toàn cả hai.

**Consequences.**
- ✅ Bất biến "đáp án luôn là substring của context" trở thành hệ quả cấu trúc,
  không phải điều phải kiểm.
- ✅ Quyết định này lan ngược lên ADR-004: nó chính là lý do fast tokenizer là
  ràng buộc cứng.

---

### ADR-006 — Tách `app/` khỏi `src/demo/`

**Status:** Accepted · **Ngày:** 13/09/2026

**Context.** Demo web là tiêu chí chấm quan trọng của môn. Phiên bản trước có
`SyntaxError` trong file app khiến demo không chạy được, và **không test nào
phát hiện** — vì test không import file app.

**Decision.** Mọi hàm quyết định *nội dung* nằm trong `src/demo/` dưới dạng hàm
thuần; `app/` chỉ gọi widget Streamlit và ghép chúng lại. Test **compile và
import mọi file** trong `app/`, không riêng file entry.

**Options Considered.**

| | Option A: tách thuần/vỏ | Option B: viết test Streamlit end-to-end | Option C: giữ nguyên, cẩn thận hơn |
|---|---|---|---|
| Thời gian chạy test | ~1 giây | Hàng chục giây, cần server | — |
| Bắt được `SyntaxError` | ✅ | ✅ | ❌ |
| Bắt được lỗi logic hiển thị | ✅ (hàm thuần) | ✅ | ❌ |
| Độ giòn | Thấp | Cao — phụ thuộc DOM của Streamlit | — |

**Consequences.**
- ✅ 141 test cho tầng demo (`test_demo_*.py`) chạy không cần Streamlit.
- ✅ `render.py` trả về **chuỗi HTML**, nên test so khớp được nội dung thật.
- ⚠️ Mọi chuỗi từ dữ liệu phải đi qua `theme.esc()`: context ViQuAD là văn bản
  Wikipedia tuỳ ý và câu hỏi do người dùng gõ — một dấu `<` lọt vào sẽ nuốt mất
  phần còn lại của thẻ.

---

### ADR-007 — Một ảnh Docker cho cả test lẫn demo

**Status:** Accepted · **Ngày:** 13/09/2026

**Context.** Người chấm cần chạy được test và demo mà không cài Python, không cần
mạng.

**Decision.** Bốn service (`app`, `tests`, `tests-full`, `fetch-data`) dùng
**chung một ảnh**, khác nhau ở `command` và ở chỗ thư mục nào được mount cho ghi.

**Trade-off.** Ba ảnh riêng sẽ cần đúng một bộ thư viện, nên tách ra chỉ tạo chỗ
cho chúng lệch nhau âm thầm — và *"test xanh trên một bộ thư viện còn demo chạy
trên bộ khác"* là đúng loại lỗi không được phép xảy ra lúc chấm.

**Quyết định đóng gói đi kèm, mỗi cái được ghim bằng test trong `tests/test_docker.py`:**

| Quyết định | Lý do | Ảnh hưởng đo được |
|---|---|---|
| PyTorch từ **index CPU trên mọi kiến trúc** | Wheel aarch64 của torch 2.x **cũng** khai báo các gói `nvidia-*` | `site-packages` 6,1 GB → **1,7 GB**; ảnh 10,2 GB → **2,48 GB**; nén 3,52 GB → **527 MB** |
| Mount `models/`, `data/`, `results/` **read-only** | Biến bất biến "demo không sinh số" thành thứ kernel bắt buộc | — |
| Chạy dưới user `app` (uid 1000) | Bind-mount không rơi vào tay root | — |
| Tạo sẵn `/home/app/.cache/huggingface` **trong ảnh** | Docker chỉ sao chép quyền sở hữu sang volume rỗng khi đường dẫn đã có | Thiếu dòng này → `PermissionError` lúc tải XLM-R, demo mất một model |
| COPY cả `Dockerfile`, `compose.yaml`, `.dockerignore` vào ảnh | `test_docker.py` chạy **bên trong** container và đọc chúng | Bộ test tự kiểm chính môi trường đang chạy nó |

**Consequences.**
- ✅ `docker compose run --rm tests` — 423 test, ~1 giây, `HF_HUB_OFFLINE=1`.
- ⚠️ Container Linux không thấy Metal: demo trong Docker chạy trên CPU và **chậm
  hơn ~4 lần** (57,0 ms so với 13,5 ms cho cùng câu hỏi mở màn). Chỉ là độ trễ —
  span trả về y hệt, và mỗi màn hình in kèm thiết bị sinh ra nó.

---

### ADR-008 — Provenance là cấu trúc dữ liệu, không phải quy trình

**Status:** Accepted · **Ngày:** 12/09/2026

**Context.** Thất bại của phiên bản trước: báo cáo trình bày những con số **chưa
từng được sinh ra**. Không có cách nào, khi đọc báo cáo, phân biệt số đo được với
số gõ tay.

**Decision.** Bốn cơ chế cài thẳng vào `run_evaluation()`:

1. **Provenance bắt buộc** — `commit`, `timestamp`, `device`, `env`, `split`, `n`.
2. **Cổng `assert_gradeable()`** chạy trước khi chấm.
3. **`KeyError` nếu thiếu dự đoán** cho bất kỳ qid nào.
4. **Cờ `unreliable`** cho nhóm `count < 30`, và `_note` giải thích khi tổng của
   bảng breakdown nhỏ hơn `n`.

**Trade-off.** Một quy trình ("nhớ ghi lại commit") rẻ hơn nhưng có thể bị bỏ qua
lúc gấp — đúng lúc nó cần nhất. Một `raise` thì không bị bỏ qua.

**Consequences.**
- ✅ `report/assets/provenance.md` sinh tự động từ bốn file JSON; sửa tay là sai.
- ✅ `results/hypotheses.md` ghi giả thuyết **trước** khi chạy;
  `scripts/check_hypotheses.py` đối chiếu tự động — kể cả khi kết quả không khớp.
- ⚠️ Mọi lần chạy đánh giá gọi `git rev-parse`; nếu thư mục không phải git repo,
  trường `commit` nhận `"unknown"` thay vì làm fail run.

---

## 7. Kiến trúc đã trả lời được gì

### 7.1 Kết quả

UIT-ViQuAD 2.0, **validation split**, n = 500 (cùng mẫu ngẫu nhiên seed=42 cho
mọi model), thiết bị **MPS**:

| Model | EM | F1 | answerable EM / F1 | impossible EM | Latency |
|---|---:|---:|---:|---:|---:|
| TF-IDF Baseline | 0,80 | 23,09 | 1,11 / 31,99 | 0,00 | 0,54 ms |
| ViSoBERT + QA (fine-tuned) | 27,80 | 31,31 | 6,93 / 11,78 | **82,01** | 22,73 ms |
| XLM-R squad2 (zero-shot) | 40,60 | 56,84 | 45,71 / **68,20** | 27,34 | 14,35 ms |
| **mBERT + QA (fine-tuned)** | **50,80** | **59,49** | **54,57** / 66,60 | 41,01 | 12,26 ms |

### 7.2 Ba điều chỉ thấy được nhờ kiến trúc tách `answerable_only` / `impossible_only`

**(1) mBERT thắng XLM-R không phải vì tìm span giỏi hơn.** Trên câu answerable,
XLM-R zero-shot thực ra **tốt hơn** (F1 68,20 so với 66,60). mBERT thắng tổng thể
vì **biết khi nào không nên trả lời** tốt hơn hẳn (impossible EM 41,01 so với
27,34). Với ~30% câu là unanswerable, kỹ năng thứ hai quyết định bảng xếp hạng.
Con số tổng trộn hai kỹ năng và che mất điều này.

**(2) ViSoBERT suy sụp về "luôn trả rỗng".** EM tổng 27,80 **bằng đúng tỉ lệ
impossible của mẫu (27,80%)**. Bóc tách: impossible EM 82,01 nhưng answerable EM
chỉ 6,93. `scripts/check_hypotheses.py` phát hiện tự động điều này, và
`training.py` có hằng số `COLLAPSE_TOLERANCE = 0.5` để nhận ra dấu hiệu ngay trên
đường cong huấn luyện — epoch 1–2 của ViSoBERT có `val_EM == val_F1`.

**(3) Vì sao ViSoBERT thất bại — và vì sao đó là kết quả hợp lệ.** Ba confound đã
được loại trừ trước khi kết luận:

| Nghi vấn | Kiểm tra | Kết quả |
|---|---|---|
| `max_position_embeddings` < 384? | đọc config | 514 — không phải nguyên nhân |
| `max_answer_len=30` quá ngắn? | đo p95 gold answer | **đúng là confound** — 27,8% gold vượt 30 token. Sửa thành 64 |
| Learning rate quá thấp? | 3e-5 → 5e-5 | có cải thiện (F1 12,25 → 30,48) nhưng vẫn kém xa |
| Ngưỡng null lệch? | quét `null_threshold` | tốt nhất F1 34,77; ép trả lời cho EM 11,50 |

Sau khi rà lại để viết báo cáo: **một trục vẫn CHƯA được bù trừ** — `max_length`
giữ nguyên 384 (và `doc_stride` 128) cho cả hai model, dù tokenizer ViSoBERT sinh
ra chuỗi dài hơn ~60%. Đó **không** phải giới hạn thật của model mà là một nguyên
nhân cấu hình. Bằng chứng đo được từ `evidence/tokenizer_stats.py`:

| | vocab | tham số | câu mẫu | context validation trung bình | token/từ | vượt 357 token |
|---|---:|---:|---:|---:|---:|---:|
| mBERT | 119.547 | 177,3 M | 17 token | 204,6 token | 1,22 | 16/557 |
| ViSoBERT | 15.002 | 97,0 M | **23 token** | **326,5 token** | **1,95** | **154/557** |

⇒ 27,6% context của ViSoBERT vượt ngân sách cửa sổ, so với 2,9% của mBERT — gấp
gần mười lần. Cùng với loss chưa hội tụ (2,19 sau 3 epoch so với 1,30 sau 2), sức
chứa nhỏ hơn 45%, và hố cực tiểu "luôn trả rỗng" do 32,39% câu impossible trong
train, **bốn yếu tố này đã đủ** để giải thích sự suy sụp.

⇒ Vì vậy **chưa kết luận được** rằng tiền huấn luyện tiếng Việt không giúp ích;
đây là **một lần huấn luyện thất bại đã được chẩn đoán**, không phải một phép đo
năng lực encoder. PhoBERT — mô hình mà giả thuyết gốc nói tới — chưa từng được
chạy. Phép kiểm trực tiếp: huấn luyện lại với `max_length` 768 và lr 3e-5.

### 7.3 Kiến trúc nào đã cho phép phát hiện này

Không có quyết định kiến trúc nào trong danh sách dưới đây là "để đẹp":

| Phát hiện | Nhờ quyết định kiến trúc |
|---|---|
| Test split không chấm được | ADR-008 — cổng `assert_gradeable()` |
| ViSoBERT suy sụp về trả rỗng | ADR-008 — tách `answerable_only` / `impossible_only` |
| Vocab nhỏ là nguyên nhân | ADR-004 — `TransformerQA` từ chối tokenizer không fast, buộc phải đo tokenizer |
| `return_overflowing_tokens` giới hạn 2 window | ADR-003 — test span trên context dài |
| Lấy mẫu thiên lệch làm lệch 8,8 điểm EM | `reproducible_subset()` dùng chung cho cả đường cong lẫn bảng cuối |

Mục cuối đáng nhắc riêng: lấy `examples[:n]` thay vì mẫu ngẫu nhiên có seed cho
**EM 42,00 so với 50,80** trên cùng model — chênh 8,8 điểm chỉ do cách lấy mẫu.
Cả đường cong huấn luyện lẫn bảng kết quả cuối dùng chung một hàm, nên hai nơi đó
so sánh được với nhau.

---

## 8. Hệ quả và giới hạn của kiến trúc hiện tại

**Dễ hơn:**

- Thêm model thứ năm: viết một lớp theo `Predictor`, thêm một dòng trong
  `catalog.MODELS`. Harness, demo, bảng so sánh, thứ hạng "tốt nhất" tự cập nhật.
- Huấn luyện lại: chạy `run_eval.py` rồi `make_figures.py` — báo cáo, slide và
  demo tự nói đúng theo, vì không nơi nào gõ tay con số.
- Đổi hệ thiết kế: mọi màu/khoảng cách là biến CSS trong `theme.TOKENS`.

**Khó hơn:**

- Thêm một nguồn số mới vào demo: bắt buộc phải đi qua `demo.results`, kể cả khi
  chỉ cần một con số. Đây là chi phí có chủ đích của bất biến "không gõ tay số".
- Nâng cấp `transformers`: `make_windows()` tự cài phải được kiểm lại (ADR-003).

**Cần xem lại:**

1. `question_type` là **heuristic tự gán** với ngưỡng `_SINGLE_SENTENCE_COVERAGE
   = 0.6` chọn theo quan sát, **không** phải nhãn có sẵn của ViQuAD. Mọi bảng
   breakdown theo trường này phải nói rõ điều đó.
2. Đánh giá trên **validation**, không phải test — vì test là blind split. Đây là
   giới hạn của dataset, không phải của kiến trúc.
3. Kết quả `n = 500` là mẫu con: ở mức ~40%, khoảng tin cậy 95% là khoảng **±4,3
   điểm** (đo trong `evidence/ci.py`). Chênh lệch mBERT − XLM-R trên EM tổng là
   **10,2 điểm ± 6,14** — vẫn có ý nghĩa, nhưng chênh lệch answerable F1 giữa
   hai model thì **không**. Dùng `--full` cho số cuối cùng.
4. `_git_commit()` trả `"unknown"` thay vì fail khi không ở trong git repo. Đây
   là nhượng bộ với trường hợp chạy từ gói `delivery/` (`git archive` không mang
   theo `.git`), và là lỗ hổng duy nhất còn lại của bất biến #3.

---

## 9. Action items

1. [ ] Chạy `run_eval.py --full` trên toàn bộ validation (3.814 câu) để thay số
       `n = 500` trong bảng cuối, loại bỏ khoảng tin cậy ±4,3 điểm.
2. [ ] Thêm epoch 3–4 cho mBERT: đường cong cho thấy loss vẫn giảm và val vẫn
       tăng ⇒ **chưa overfit, thậm chí còn thiếu epoch**.
3. [ ] Ghim phiên bản `transformers` trong `requirements.txt` kèm chú thích trỏ
       về ADR-003, để lần nâng cấp sau biết phải kiểm lại gì.
4. [ ] Bổ sung test cho nhánh `_git_commit()` trả `"unknown"`, để lỗ hổng ở §8.4
       được ghi nhận thay vì bị quên.

---

## Phụ lục — Bản đồ tệp

```
uit-vietnamese-mrc/
├── src/mrc/          lõi nghiệp vụ — 1.503 dòng, 9/12 module không cần torch
├── src/demo/         logic trình bày — 1.593 dòng, hàm thuần, không import Streamlit
├── app/              vỏ Streamlit — 643 dòng, chỉ gọi widget
├── scripts/          fetch_data · finetune · run_eval · make_figures ·
│                     make_report · check_hypotheses · make_delivery.sh
├── tests/            21 file · 432 hàm test · 398 nhanh (~1 giây) / 449 đầy đủ
├── results/          eval_*.json · training_curve_*.json · hypotheses.md · figures/
├── report/           BAO_CAO.md + assets/ (bảng, provenance, hình — sinh tự động)
├── slides/           index.html — deck HTML, không gõ tay con số nào
├── docs/             SOLUTION.md · PLAN_TDD.md · ARCHITECTURE.md · SYSTEM_DESIGN.md
├── Dockerfile        một ảnh, bốn service
└── compose.yaml      app · tests · tests-full · fetch-data
```
