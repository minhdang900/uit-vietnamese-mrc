# Thiết kế hệ thống — `uit-vietnamese-mrc`

**Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt** (Vietnamese Extractive MRC)
**Môn:** CS116 · Đề tài T11 · UIT, ĐHQG-HCM · **Ngày:** 13/09/2026 · **Commit:** `dc85f58`

> Tài liệu này đi vào *thiết kế chi tiết*: yêu cầu, hợp đồng giữa các thành phần,
> mô hình dữ liệu, đường lỗi, hiệu năng, triển khai. Phần *quyết định kiến trúc và
> vì sao* nằm ở [`ARCHITECTURE.md`](ARCHITECTURE.md); tài liệu này giả định người
> đọc đã biết bốn tầng và bốn bất biến mô tả ở đó.

---

## 1. Yêu cầu

### 1.1 Yêu cầu chức năng

| # | Yêu cầu | Nguồn | Hiện trạng |
|---|---|---|---|
| F1 | Trích xuất answer span từ `(context, question)` tiếng Việt | Đề tài T11 | ✅ 4 model |
| F2 | Nhận ra câu **không có đáp án** và trả về rỗng | ViQuAD 2.0 có 30,4% câu impossible ở validation | ✅ qua `null_threshold` |
| F3 | Baseline TF-IDF retrieval làm sàn so sánh | Đề tài T11 | ✅ `baseline_tfidf.py` |
| F4 | Fine-tune encoder tiếng Việt + QA head | Đề tài T11 (nêu PhoBERT) | ✅ ViSoBERT — xem ADR-004 |
| F5 | Đánh giá EM + token-F1 theo quy ước SQuAD-2.0 | Đề tài T11 | ✅ `metrics.py` |
| F6 | Phân tích single-sentence vs multi-sentence | Đề tài T11 | ⚠️ heuristic tự gán, không phải nhãn ViQuAD |
| F7 | Demo web tương tác | Tiêu chí chấm | ✅ 6 màn hình Streamlit |
| F8 | Chạy được trên máy người chấm không cài Python | Tiêu chí nộp bài | ✅ Docker, một lệnh |

### 1.2 Yêu cầu phi chức năng

| Thuộc tính | Mục tiêu | Đo được | Ghi chú |
|---|---|---|---|
| **Độ trễ inference** | < 100 ms/câu để demo mượt | **12,3 ms** (mBERT, MPS) · 57,0 ms (CPU trong Docker) | Đo bằng `predict_timed`, không ước lượng |
| **Thời gian bộ test nhanh** | < 5 giây cho vòng lặp TDD | **~1 giây** · 423 test | Không tải model, không mạng |
| **Thời gian fine-tune** | Vừa một buổi | **~35 phút/epoch** (28.454 câu, MPS) | 2 epoch mBERT, 3 epoch ViSoBERT |
| **Dung lượng ảnh Docker** | Nhỏ nhất có thể | **2,48 GB** (nén 527 MB) | Từ 10,2 GB — xem §7.3 |
| **Bộ nhớ** | Chạy được trên máy 16 GB | Fine-tune dùng batch 12 × grad_accum 2 | Máy phát triển: 48 GB unified |
| **Tính tái lập** | Cùng seed → cùng số | `seed = 42` xuyên suốt | Mẫu con, split, shuffle |
| **Tính truy vết** | Mọi số → một commit | 4/4 file kết quả có đủ provenance | Bất biến #3 |

### 1.3 Ràng buộc

| Ràng buộc | Ảnh hưởng thiết kế |
|---|---|
| Không có ngân sách cloud GPU | Toàn bộ huấn luyện chạy trên MPS của máy cá nhân ⇒ chọn model `base` (~135M), không `large` |
| Test split của ViQuAD là blind set | Không thể dùng test; mọi số cuối cùng đến từ validation |
| PhoBERT không có fast tokenizer | Đổi sang ViSoBERT (ADR-004) |
| `transformers 5.17.0` giới hạn overflow ở 2 window | Tự cài `make_windows()` (ADR-003) |
| Người chấm chỉ có Docker | Gói `delivery/` gồm ảnh đã `docker save` + `run.sh` |
| Nhóm 4 người, ~2 tuần | TDD 9 phase, mỗi phase có đặc tả test trước (xem `PLAN_TDD.md`) |

---

## 2. Thiết kế mức cao

### 2.1 Hợp đồng giữa các thành phần

Hệ thống không có API mạng; "API" ở đây là các hợp đồng Python giữa tầng. Ba hợp
đồng quan trọng nhất:

#### Hợp đồng #1 — `Predictor`

```python
@runtime_checkable
class Predictor(Protocol):
    name: str
    def predict(self, context: str, question: str) -> str: ...
```

**Hậu điều kiện (được test cho mọi implementation):**
`predict(ctx, q) == "" or predict(ctx, q) in ctx`

Mở rộng tuỳ chọn, chỉ `TransformerQA` hiện thực:

```python
def predict_detailed(self, context, question, top_k=3) -> dict:
    # {answer, span, found, null_delta, start_prob, end_prob, top_k}
```

`demo.logic.answer()` phát hiện phần mở rộng bằng `getattr`, và **ẩn khối bằng
chứng** nếu predictor không có — thay vì bịa số để lấp chỗ trống.

#### Hợp đồng #2 — kết quả đánh giá

```jsonc
{
  "model": "mbert (fine-tuned)",
  "dataset": "UIT-ViQuAD 2.0", "split": "validation", "n": 500,
  "timestamp": "...", "commit": "590e775", "device": "mps",
  "env": { "torch": "2.14.0", "platform": "macOS-26.4.1-arm64", "mps_available": true },

  "overall":         { "EM": 50.80, "F1": 59.4886, "count": 500 },
  "answerable_only": { "EM": 54.5706, "F1": 66.6047, "count": 361 },
  "impossible_only": { "EM": 41.0072, "count": 139 },
  "avg_latency_ms": 12.264,

  "by_context_length": { "<100": { "EM": .., "F1": .., "count": 1, "unreliable": true }, ... },
  "by_question_type":  { "single-sentence": {...}, "multi-sentence": {...}, "_note": "..." },
  "sample_predictions": [ ... ]
}
```

Ba trường đầu của khối provenance là **bắt buộc** — không có đường code nào sinh
ra dict kết quả thiếu chúng. Trường `_note` là văn bản giải thích vì sao tổng của
`by_question_type` nhỏ hơn `n` (câu impossible không có phạm vi suy luận để phân
loại) — nó tồn tại vì một bảng có tổng không khớp mà không giải thích là một bảng
đáng ngờ.

#### Hợp đồng #3 — tầng trình bày chỉ đọc

```python
# demo/results.py — CHỖ DUY NHẤT chạm đĩa
load_eval(model_id, root=None) -> dict
load_all_evals(root=None)      -> dict[str, dict]
dataset_stats(root=None)       -> dict      # đo trực tiếp từ data/raw/
best_by(metric)                -> str       # TÍNH, không gõ tay
provenance(root=None)          -> Provenance
```

Mọi hàm nhận `root` tường minh để test trỏ vào thư mục tạm, không đụng
`results/` thật.

### 2.2 Lựa chọn lưu trữ

Không có cơ sở dữ liệu. Đây là quyết định, không phải thiếu sót:

| Dữ liệu | Nơi lưu | Định dạng | Vì sao không dùng DB |
|---|---|---|---|
| Dataset thô | `data/raw/*.json` | SQuAD-2.0 JSON | 16 MB, đọc một lần, bất biến |
| Kết quả đánh giá | `results/eval_*.json` | JSON có provenance | Phải **diff được bằng git** — đó chính là cơ chế truy vết |
| Đường cong huấn luyện | `results/training_curve_*.json` | JSON | Như trên |
| Checkpoint | `models/{mbert,visobert}/` | HuggingFace format | 3,4 GB, trong `.gitignore` |
| Hình cho báo cáo | `results/figures/*.png` | PNG | Sinh lại được từ JSON |

**Trade-off.** Một SQLite sẽ tiện hơn cho truy vấn, nhưng JSON trong git cho một
thứ quan trọng hơn: `git log results/` là lịch sử đầy đủ của mọi con số dự án
từng công bố. Với một dự án mà thất bại của phiên bản trước là *báo cáo số chưa
từng được sinh ra*, đó là đánh đổi đúng.

**Cái gì trong ảnh Docker, cái gì được mount:**

| | Trong ảnh | Mount lúc chạy |
|---|---|---|
| Mã nguồn, `results/`, `report/`, `slides/` | ✅ | — |
| `models/` (3,4 GB) | ❌ trong `.gitignore` | `./models:/app/models:ro` |
| `data/raw/` (16 MB) | ❌ trong `.gitignore` | `./data:/app/data:ro` |
| Cache HuggingFace | ❌ | named volume `hf-cache` |

Cả ba mount đều **read-only**: bất biến "demo không sinh ra con số nào" do kernel
bắt buộc chứ không chỉ là quy ước.

---

## 3. Thiết kế chi tiết

### 3.1 Mô hình dữ liệu

```python
@dataclass(frozen=True, eq=True)
class Example:
    qid: str                  # khoá xuyên suốt pipeline
    question: str
    context: str
    title: str = ""           # CHỈ để truy vết — không dùng làm đơn vị split
    answers: list[str] = []   # RỖNG = câu impossible
    answer_start: int = -1    # offset ký tự của answers[0]; -1 nếu impossible
    is_impossible: bool = False
```

**Bất biến ở `__post_init__`:** `is_impossible=True` và `answers` khác rỗng là
mâu thuẫn ⇒ `ValueError`.

**Ba trạng thái, không phải hai** — đây là chi tiết quyết định tính đúng đắn của
toàn bộ phần đánh giá:

| Tình huống | `answers` | `is_impossible` | Chấm được? | Đáp án đúng |
|---|---|---|---|---|
| Câu thường | có | `False` | ✅ | chuỗi gold |
| Câu impossible | rỗng | `True` | ✅ | chuỗi rỗng |
| **Blind split** | rỗng | `False` | ❌ | **không biết** (bị lược bỏ) |

Gộp hai hàng cuối — tức suy ra `is_impossible` từ việc gold rỗng — sẽ khiến một
model luôn trả về `""` đạt **EM 100%** trên test split của ViQuAD (7.301 câu), và
con số đó trông hoàn toàn hợp lý trong báo cáo. `Example.is_gradeable` phân biệt
ba trạng thái, và `assert_gradeable()` chặn ở cửa.

**Một cái bẫy nữa đã được xử lý:** với câu `is_impossible=True`, SQuAD-2.0 cung
cấp `plausible_answers` — một đáp án "nghe hợp lý nhưng sai". `parse_squad()`
**không bao giờ** dùng trường này làm gold; dùng nó sẽ âm thầm biến câu impossible
thành answerable.

### 3.2 Thống kê dataset (đo trực tiếp từ file tải về)

| Split | Questions | Contexts | Articles | Impossible | Có gold | **Chấm được** |
|---|---:|---:|---:|---:|---:|---:|
| train | 28.454 | 4.101 | 138 | 9.216 (32,4%) | 19.238 | 28.454 |
| validation | 3.814 | 557 | 19 | 1.161 (30,4%) | 2.653 | 3.814 |
| test | 7.301 | 1.241 | 48 | 0 | **0** | **0** |

`num_contexts` (4.101) và `num_articles` (138) được báo cáo **tách bạch** vì cơ
chế chống leakage dựa vào context; nhầm hai đơn vị làm vô hiệu hoá nó.

### 3.3 Thiết kế chuẩn hoá — ba quyết định đặc thù tiếng Việt

| | Quyết định | SQuAD tiếng Anh làm gì | Vì sao tiếng Việt phải khác |
|---|---|---|---|
| **A** | **Không loại mạo từ** | Loại `a/an/the` | Tiếng Việt **không có mạo từ**; `các`, `con`, `những` là **loại từ** và có thể là phần của đáp án đúng |
| **B** | **Token hoá theo khoảng trắng** (âm tiết) | Tách theo từ | Khớp eval chính thức ViQuAD ⇒ so sánh được với công trình khác; tránh phụ thuộc segmenter và tránh lỗi segmenter làm nhiễu metric |
| **C** | **Giữ dấu** | (không áp dụng) | `"hoà"` ≠ `"hoa"`; strip dấu sẽ làm sai EM ở hàng nghìn câu |

Quyết định C lan ra toàn hệ thống: nó là lý do `decode_span()` cắt chuỗi gốc thay
vì gọi `tokenizer.decode()` (ADR-005), và do đó là lý do fast tokenizer trở thành
ràng buộc cứng (ADR-004).

Chi tiết cài đặt: `_remove_punctuation` dùng `unicodedata.category(ch)` bắt nhóm
`P*` thay vì `string.punctuation`, để bắt được dấu câu Unicode hay gặp trong văn
bản Wikipedia (`–`, `"`, `"`, `…`). Chữ cái thuộc nhóm `L*` nên dấu tiếng Việt an
toàn.

### 3.4 Thiết kế metric

```
EM(pred, gold)  = 1.0 nếu normalize(pred) == normalize(gold)
F1(pred, gold)  = F1 trên BAG-OF-TOKENS, dùng Counter (không dùng set)
điểm của một câu = max qua tất cả gold  (ViQuAD có thể có nhiều đáp án đúng)
```

**Bốn quyết định nhỏ, mỗi cái chặn một lỗi cụ thể:**

1. `Counter` chứ không `set` — token lặp phải được đếm đúng số lần.
2. Trường hợp biên theo SQuAD: một bên rỗng ⇒ F1 = 1,0 **chỉ khi cả hai rỗng**,
   ngược lại 0,0. Không có điểm bán phần.
3. `evaluate()` **raise `KeyError`** nếu thiếu dự đoán cho bất kỳ qid nào. Âm
   thầm bỏ qua câu thiếu sẽ khiến EM được tính trên tập con trong khi báo cáo
   ghi `n` đầy đủ.
4. Kết quả trả về **phần trăm 0–100**, không phải tỉ lệ 0–1, để không có chỗ cho
   nhầm lẫn ×100 giữa các tầng.

### 3.5 Thiết kế chọn span — phần dễ sai âm thầm nhất

Ba lỗi kinh điển, cả ba **không crash** mà chỉ làm F1 thấp một cách khó hiểu:

| Lỗi | Cơ chế phòng |
|---|---|
| Span lấy từ vùng **question** (input là `[CLS] q [SEP] ctx [SEP]`) | Chỉ token có `offset_mapping is not None` mới là ứng viên |
| Span **đảo ngược** (`end < start`, vì hai argmax độc lập) | Vòng lặp chỉ xét cặp `end_idx >= start_idx` |
| Dùng `tokenizer.decode()` lấy lại chuỗi | `decode_span()` cắt context gốc; `ValueError` nếu khoảng không hợp lệ |

**Luồng chấm span trong một cửa sổ** (`score_spans`, hàm thuần):

```
start_logits, end_logits, offset_mapping
  │  với mọi cặp (start, end) hợp lệ:  score = start_logit + end_logit
  │    ràng buộc: end >= start · end - start + 1 <= max_answer_len
  ▼
best_pair  +  top-k (heap, khử trùng lặp theo KHOẢNG KÝ TỰ)
  +  null_score = start_logits[CLS] + end_logits[CLS]
  +  prob = softmax trên TOÀN BỘ cặp hợp lệ ∪ {null}   ← log-sum-exp chạy dòng
```

**Hai chi tiết hiệu năng/số học đáng nói:**

- **Không dựng ma trận.** Với cửa sổ 384 token, bảng điểm đầy đủ là 147.456 phần
  tử. `score_spans` dùng log-sum-exp **chạy dòng** — một biến `ceiling`, một biến
  `total` — nên không cấp phát mảng và không tràn số khi logit lớn.
- **Khử trùng lặp theo khoảng ký tự, không theo cặp token.** Nhiều cặp token khác
  nhau có thể trỏ về cùng một chuỗi; một danh sách "span xếp sau" lặp lại chính
  nó thì vô dụng trên màn hình demo.

**Quyết định trả lời hay từ chối:**

```
trả lời  ⟺  best.score > null_score + null_threshold
tương đương:  null_delta + threshold < 0,   với null_delta = null_score − best.score
```

Với context nhiều cửa sổ, `null_delta` lấy **giá trị nhỏ nhất** trên các cửa sổ,
vì model trả lời nếu *có* một cửa sổ vượt ngưỡng. Nhờ vậy dấu của biểu thức khớp
chính xác với quyết định mà `select_best_span` đưa ra từng cửa sổ — và thanh đo
trên demo không bao giờ nói "trả lời" đúng lúc model im lặng.

`demo.logic.margin()` / `abstains()` viết lại **cùng một phép so sánh** dưới dạng
hàm thuần, để giao diện giải thích được quyết định mà **không phải chạy lại
model**: kéo thanh ngưỡng tới đâu, thanh đo nhảy tới đó.

### 3.6 Thiết kế windowing

```python
make_windows(question, context, tokenizer, max_length=384, doc_stride=128) -> list[Window]
```

Ngân sách token: `budget = max_length − len(question_tokens) − num_special_tokens`.
Bước nhảy: `step = max(1, budget − doc_stride)`. Question **không bao giờ bị cắt**
(`truncation="only_second"`); nếu question dài đến mức `budget < 1` thì
`ValueError` — fail loud.

**Điểm thiết kế then chốt:** `Window.offset_mapping` chứa offset **TUYỆT ĐỐI**
trong context gốc, không phải trong chunk. Việc dịch toạ độ (`off + char_start`)
xảy ra đúng một lần, ngay tại nơi sinh ra window. Mọi tầng phía sau —
`score_spans`, `decode_span`, `features.prepare_*` — làm việc trên một hệ toạ độ
duy nhất.

### 3.7 Thiết kế huấn luyện

| Tham số | mBERT | ViSoBERT | Ghi chú |
|---|---|---|---|
| Model | `bert-base-multilingual-cased` | `uitnlp/visobert` | |
| Epoch | 2 | 3 | |
| Batch × grad_accum | 12 × 2 | 12 × 2 | batch hiệu dụng 24 |
| Learning rate | 3e-5 | **5e-5** | ViSoBERT cần cao hơn — xem §6.2 |
| `max_length` / `doc_stride` | 384 / 128 | 384 / 128 | |
| `max_answer_len` | 30 | **64** | ViSoBERT chia từ nhỏ hơn: p95 gold = 67 token so với 42 |
| warmup / weight decay | 0,1 / 0,01 | 0,1 / 0,01 | |
| seed | 42 | 42 | |
| Thời gian/epoch | ~2.066–2.100 s | ~2.416–2.475 s | trên MPS |

**Một quyết định nhỏ nhưng quyết định tính đúng:** `compute_schedule()` đếm theo
`optimizer.step()`, **không** theo batch. Đếm nhầm khiến learning rate giảm về 0
quá sớm — và loss vẫn giảm, nên lỗi không lộ ra.

**Gán nhãn:** câu impossible → `[CLS]`; window không chứa đáp án → `[CLS]`. Cùng
một quy ước cho hai tình huống khác nhau, vì với model chúng là cùng một thông
điệp: *ở đây không có đáp án*.

**Chẩn đoán tự động trên đường cong** (`training.py`, hàm thuần):

```python
COLLAPSE_TOLERANCE = 0.5   # |EM − F1| < 0.5  ⇒  nghi suy sụp về "luôn trả rỗng"
```

Cơ sở: F1 cho điểm bán phần nên bình thường phải cao hơn EM vài điểm. Khi hai
metric trùng nhau, mỗi câu chỉ có thể đúng-hoàn-toàn (impossible, đoán rỗng) hoặc
sai-hoàn-toàn (answerable, đoán rỗng). Đường cong ViSoBERT epoch 1 (`25,67 /
25,67`) và epoch 2 (`25,33 / 25,61`) rơi đúng vào đó.

### 3.8 Thiết kế demo — sáu màn hình

| Màn hình | URL | Nội dung | Nguồn số |
|---|---|---|---|
| Hỏi đáp | `/` | Nhập câu hỏi, chọn model, kéo ngưỡng, xem span + bằng chứng | `predict_detailed` (đo lúc chạy) |
| Kết quả | `/ket-qua` | Bảng EM/F1, breakdown, provenance | `results/eval_*.json` |
| Phân tích lỗi | `/phan-tich-loi` | Sai ở đâu: pred (terracotta) vs gold (sage) | `sample_predictions` |
| Dữ liệu | `/du-lieu` | Thống kê split, phân phối độ dài | đo từ `data/raw/` |
| So sánh model | `/so-sanh` | Bốn model cạnh nhau, ô tốt nhất in đậm | `best_by()` — **tính**, không gõ |
| Huấn luyện | `/huan-luyen` | Đường cong loss/EM/F1 | `training_curve_*.json` |

Mở bằng URL là mở một **phiên mới**, nên model và ngưỡng quay về mặc định (mBERT,
ngưỡng +0,0) — đúng thứ ta muốn khi nhảy thẳng vào một slide lúc thuyết trình.
Bấm điều hướng trong sidebar thì lựa chọn được giữ nguyên giữa các màn hình.

**Năm phán quyết, không phải hai** (`demo.logic.verdict`). Đúng/sai không nằm ở
chỗ model có trả lời hay không, mà ở chỗ câu hỏi **có** đáp án hay không:

| model từ chối? | câu impossible? | Phán quyết |
|---|---|---|
| ✅ | ✅ | Đúng — gold là rỗng |
| ✅ | ❌ | Bỏ sót — hạ ngưỡng để mạnh dạn hơn |
| ❌ | ✅ | Cảnh báo — đang trả lời một span trông hợp lý; tăng ngưỡng |
| ❌ | ❌ | Đúng — span nằm nguyên trong đoạn văn |
| — | **`None`** | Câu **tự nhập** — không có gold để đối chiếu |

Trạng thái `None` phải tách riêng: khẳng định "đúng, câu này impossible" cho một
câu người dùng vừa gõ là **bịa** — nhãn đó thuộc về câu hỏi gốc của đoạn văn, không
thuộc câu vừa gõ.

### 3.9 Hệ thiết kế

`demo/theme.py` khai báo toàn bộ hệ "Organic" dưới dạng **biến CSS**, không rải
số literal: đổi một token là đổi toàn bộ giao diện.

| Nhóm token | Giá trị |
|---|---|
| Nền / bề mặt / chữ | `#f5ead8` · `#ebddc5` · `#201e1d` |
| Accent (terracotta) / accent-2 (sage) | `#c67139` · `#7a8a5e` |
| Ramp | neutral 100–900, accent 100–900, accent-2 100–900 |
| Khoảng cách | 4,4 / 8,8 / 13,2 / 17,6 / 26,4 / 35,2 px |
| Bo góc | 8 / 16 / 28 / 32,2 px |
| Font | `Baloo 2` (tiêu đề) · `Nunito` (thân) |

Cặp màu pred/gold không phải trang trí: terracotta = *model nói gì*, sage = *đáp
án đúng model bỏ lỡ*. Đó là toàn bộ nội dung của màn hình phân tích lỗi, nên nó
được đặt tên (`mark(text, kind)`) chứ không rải class ở nơi gọi.

`resolve()` thay `var(--token)` bằng giá trị thật cho SVG nhúng dạng `data:` URI —
ảnh là một tài liệu riêng, không thấy `:root` của trang, nên biến CSS trong đó sẽ
rỗng và hình vẽ ra đen thui.

---

## 4. Xử lý lỗi

Nguyên tắc xuyên suốt: **fail loud** ở mọi chỗ mà một lỗi có thể biến thành một
con số trông hợp lý.

| Điều kiện | Phản ứng | Vì sao không âm thầm |
|---|---|---|
| Split có câu không chấm được | `ValueError` (`assert_gradeable`) | Model trả rỗng sẽ đạt **EM 100%** trên blind set |
| Hai split chia sẻ context | `AssertionError` (`assert_no_leakage`) | Kết quả cao giả tạo |
| Thiếu dự đoán cho một qid | `KeyError` (`metrics.evaluate`) | EM tính trên tập con nhưng báo cáo ghi `n` đầy đủ |
| `end_char < start_char` hoặc vượt độ dài | `ValueError` (`decode_span`) | Span sai là **bug**, không phải "không tìm thấy đáp án" |
| Model không có fast tokenizer | `RuntimeError` lúc `__init__` | Chạy được nhưng mất dấu tiếng Việt |
| `is_impossible=True` nhưng có `answers` | `ValueError` (`__post_init__`) | Dữ liệu mâu thuẫn |
| Question dài hơn `max_length` | `ValueError` (`make_windows`) | Cắt question âm thầm = hỏi một câu khác |
| Thiếu file `results/*.json` | `MissingResults` **kèm lệnh sinh lại** | Màn hình toàn dấu gạch ngang trông giống "model kém" chứ không giống "chưa chạy đánh giá" |

**Ba ngoại lệ có chủ đích** (fail soft, và mỗi cái có lý do):

1. `_git_commit()` trả `"unknown"` thay vì raise — để gói `delivery/` chạy được
   (`git archive` không mang theo `.git`).
2. `TfidfRetriever.predict` bắt `ValueError` của `TfidfVectorizer` khi vocabulary
   rỗng (context chỉ có stopword) và trả về câu đầu.
3. `demo.logic.answer` trả về dict có đủ khoá bằng chứng với giá trị `None` khi
   predictor không hỗ trợ — vì thiếu khoá là `KeyError` **lúc demo đang chạy
   trước mặt người chấm**.

---

## 5. Hiệu năng

### 5.1 Độ trễ đo được

| Model | MPS — trung bình trên 500 câu | CPU trong Docker — một câu hỏi mở màn | Ghi chú |
|---|---:|---:|---|
| TF-IDF Baseline | **0,54 ms** | — | Không có model để nạp |
| mBERT + QA | **12,26 ms** | ~57,0 ms | Nhanh nhất trong ba transformer |
| XLM-R squad2 | 14,35 ms | — | |
| ViSoBERT + QA | **22,73 ms** | — | Chậm nhất dù model **nhỏ nhất** |

**ViSoBERT nhỏ hơn 45% về tham số (97,0 M so với 177,3 M) nhưng chậm hơn 85%.**
Nguyên nhân không nằm ở model mà ở tokenizer: vocab 15.002 chia context validation
thành **326,5 token trung bình** so với 204,6 của mBERT (1,95 so với 1,22
token/từ). Nhiều token hơn ⇒ nhiều cửa sổ hơn ⇒ nhiều lần forward hơn. Ở ngưỡng
357 token, **154/557 context** của ViSoBERT cần nhiều cửa sổ, so với 16/557 của
mBERT.

Đây là một trong những quan sát hữu ích nhất của dự án: *kích thước vocab là một
quyết định về hiệu năng, không chỉ về chất lượng.*

### 5.2 Chi phí huấn luyện

| | Câu | Thời gian/epoch | Tổng |
|---|---:|---:|---:|
| mBERT, 2 epoch | 28.454 | ~2.083 s (~35 phút) | ~70 phút |
| ViSoBERT, 3 epoch | 28.454 | ~2.446 s (epoch 1–2) | ~2 giờ |

⚠️ `epoch_seconds` của epoch 3 ViSoBERT (6.798 s) **bị nhiễu**: một job chẩn đoán
`null_threshold` chạy song song tranh GPU. Con số đó được ghi chú ngay trong file
JSON (`timing_caveat`) và **không** được dùng làm số benchmark.

### 5.3 Bộ test

| Bộ | Số test | Thời gian | Mạng |
|---|---:|---:|---|
| `pytest -m "not slow"` | 398 | **~1 giây** | Không (`HF_HUB_OFFLINE=1`) |
| `pytest` (đầy đủ) | 449 | vài phút | Có — tải model thật |

Con số ~1 giây không phải may mắn: nó là hệ quả trực tiếp của việc 9/12 module
trong `src/mrc/` và toàn bộ `src/demo/` là hàm thuần (§3.1 của `ARCHITECTURE.md`).

### 5.4 Ý nghĩa thống kê — điều mà bảng kết quả không nói

Ở `n = 500`, quanh mức 40%, khoảng tin cậy 95% là khoảng **±4,3 điểm**. Đo trong
`evidence/ci.py`:

| So sánh | Chênh lệch | CI 95% | Có ý nghĩa? |
|---|---:|---:|---|
| mBERT − XLM-R, EM tổng | **+10,2** | ±6,14 | ✅ |
| mBERT − XLM-R, EM answerable | +8,86 | ±7,27 | ✅ (sát biên) |
| mBERT − XLM-R, EM impossible | **+13,67** | ±11,03 | ✅ |
| XLM-R − mBERT, **F1 answerable** | **+1,60** | — | ❌ **không** |
| mBERT − ViSoBERT, EM answerable | +47,65 | ±5,77 | ✅ rất mạnh |
| mBERT, single − multi sentence, EM | +10,38 | ±10,39 | ⚠️ sát biên |

Hàng thứ tư là hàng quan trọng nhất: **XLM-R zero-shot "tốt hơn" mBERT trên câu
answerable không phải một kết luận vững** — chênh lệch 1,60 điểm F1 nằm gọn trong
nhiễu (so sánh: chênh lệch EM answerable là 8,86 với CI ±7,27). Điều vững chắc là mBERT thắng ở **kỹ năng từ chối** (+13,67 điểm impossible
EM), và đó mới là thứ quyết định bảng xếp hạng.

---

## 6. Đánh đổi đã chọn

### 6.1 Bảng tóm tắt

| Quyết định | Được | Mất | Xem thêm |
|---|---|---|---|
| `Protocol` thay lớp cơ sở | Không có `isinstance` trong harness; test bằng stub | Không ép buộc lúc biên dịch | ADR-001 |
| Split theo context | Chống group leakage đúng đơn vị | Phức tạp hơn split theo dòng | ADR-002 |
| Tự cài windowing (402 dòng) | Không mất đuôi context; offset tuyệt đối | Phải bảo trì khi `transformers` nâng cấp | ADR-003 |
| ViSoBERT thay PhoBERT | Có fast tokenizer ⇒ giữ được dấu | Lệch miền (mạng xã hội ≠ Wikipedia) | ADR-004 |
| Cắt chuỗi gốc theo offset | Bảo toàn dấu; bất biến substring thành hệ quả cấu trúc | Buộc phải có fast tokenizer | ADR-005 |
| Tách `app/` ↔ `src/demo/` | Test ~1 giây, bắt cả `SyntaxError` lẫn lỗi logic | Phải viết `render.py` trả chuỗi HTML | ADR-006 |
| Một ảnh Docker | Không thể lệch thư viện giữa test và demo | Ảnh mang cả pytest lẫn Streamlit | ADR-007 |
| JSON trong git thay vì DB | `git log results/` là lịch sử mọi con số | Không truy vấn được bằng SQL | §2.2 |
| Provenance bắt buộc | Không thể báo cáo số không truy vết được | Mỗi lần eval gọi `git rev-parse` | ADR-008 |

### 6.2 Ba confound đã loại trừ trước khi kết luận về ViSoBERT

Đây là phần thiết kế **thực nghiệm**, không phải thiết kế phần mềm, nhưng nó là
thứ khiến kết luận đứng vững:

| Nghi vấn | Cách kiểm | Kết quả |
|---|---|---|
| `max_position_embeddings` < 384? | Đọc config | **514** — không phải nguyên nhân |
| `max_answer_len=30` quá ngắn? | Đo p95 gold answer theo token | **Đúng là confound**: p95 = 67 token, 27,8% gold vượt 30 ⇒ sửa thành **64** |
| Learning rate quá thấp? | Chạy lại 3e-5 → 5e-5 | Cải thiện thật (F1 12,25 → 30,48) nhưng vẫn kém xa |
| Ngưỡng null lệch? | Quét `null_threshold` | Tốt nhất F1 34,77; ép trả lời cho EM 11,50 |

Bốn nghi vấn trên đều được kiểm — nhưng bảng này **thiếu một trục**: `max_length`
giữ nguyên 384 (và `doc_stride` 128) cho cả hai model, dù tokenizer ViSoBERT sinh
chuỗi dài hơn ~60% (154/557 context vượt ngân sách, so với 16/557 của mBERT). Đó là
**nguyên nhân cấu hình chưa được bù trừ**, không phải giới hạn thật của model —
đừng nhầm nó với `max_answer_len`, trục đã được bù (30 → 64).

Giả thuyết đăng ký trước (`results/hypotheses.md`) dự đoán **PhoBERT** sẽ thắng.
PhoBERT chưa từng được chạy (không có fast tokenizer), nên giả thuyết đó vẫn **chưa
kiểm được**, không phải bị bác bỏ: ViSoBERT là mô hình *thay thế*, và lần huấn luyện
của nó đã suy sụp. Việc báo cáo nguyên vẹn cả kết quả lẫn giới hạn của phép kiểm
chính là điều pre-registration tồn tại để bảo vệ.

---

## 7. Triển khai

### 7.1 Ba cách chạy

```bash
# A. Máy thật (Apple Silicon → tự dùng MPS)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app/streamlit_app.py

# B. Docker — không cần cài gì ngoài Docker
docker compose run --rm tests      # 423 test, ~1 giây, KHÔNG cần mạng
docker compose up app              # demo tại http://localhost:8501
docker compose down

# C. Gói nộp bài — người chấm chỉ cần Docker
./run.sh          # nạp ảnh rồi mở demo
./run.sh test
./run.sh stop
```

### 7.2 Bốn service, một ảnh

| Service | Profile | Command | Mount ghi |
|---|---|---|---|
| `app` | (mặc định) | `streamlit run app/streamlit_app.py` | không — cả ba đều `:ro` |
| `tests` | `test` | `pytest -m "not slow"` | không |
| `tests-full` | `test-full` | `pytest` | không |
| `fetch-data` | `data` | `python scripts/fetch_data.py` | `./data` — **service duy nhất** |

Ba service sau nằm sau profile nên `docker compose up` chỉ dựng demo; `run` tự
bật profile của service nó gọi.

### 7.3 Tối ưu ảnh — một quyết định, 4 GB

| | Ảnh đầu | Sau khi ép CPU trên mọi kiến trúc |
|---|---:|---:|
| `site-packages` | 6,1 GB | **1,7 GB** |
| Dung lượng đĩa | 10,2 GB | **2,48 GB** |
| Dung lượng truyền (nén) | 3,52 GB | **527 MB** |

Bản dựng đầu tiên chỉ ép CPU cho `amd64`, vì tưởng wheel `aarch64` vốn đã
CPU-only. **Sai** — wheel aarch64 của torch 2.x *cũng* khai báo các gói
`nvidia-*`, nên chính máy arm64 dựng ra ảnh mang đủ 3,3 GB thư viện CUDA và 817 MB
Triton. Bỏ điều kiện theo kiến trúc là toàn bộ khác biệt.

Chi tiết PEP 440 khiến việc này hoạt động: index CPU đặt tên bản này là
`2.14.0+cpu`, và một specifier không có phần local (`==2.14.0` trong
`requirements.txt`) khớp với mọi nhãn local — nên `pip` ở bước sau thấy torch đã
thoả và không kéo lại bản PyPI.

### 7.4 Gói nộp bài

```bash
docker compose build app                # cần có ảnh trước
./scripts/make_delivery.sh              # → delivery/  (~1,5 GB, có checkpoint)
./scripts/make_delivery.sh --no-models  # → delivery/  (~500 MB, không checkpoint)
```

Gói **không** kèm `models/*/epoch*/` — đó là checkpoint giữa chừng, 1,3 GB mà demo
không bao giờ nạp (transformers đọc thẳng ở thư mục gốc của model). Bỏ chúng là
khác biệt giữa gói 1,5 GB và gói 4 GB.

`tests/test_docker.py` **ghim tag ảnh** giữa `make_delivery.sh` và
`compose.yaml`: lệch tag thì compose lặng lẽ dựng lại từ đầu ngay trên máy người
chấm — đúng thứ không được phép xảy ra lúc chấm.

### 7.5 Khả năng quan sát

Không có Prometheus, không có dashboard. Với hệ thống chạy một máy, một người
dùng, "monitoring" là bốn thứ sau:

| Tín hiệu | Nơi đọc |
|---|---|
| Kết quả từng lần chạy + provenance | `results/eval_*.json` (diff được bằng git) |
| Nhật ký huấn luyện | `results/finetune_*.log`, `training_curve_*.json` |
| Đối chiếu giả thuyết ↔ kết quả | `python scripts/check_hypotheses.py` |
| Sức khoẻ container | `HEALTHCHECK` gọi `/_stcore/health`; `up --wait` chờ đến khi trang thật sự phục vụ được |

`check_hypotheses.py` là thứ gần nhất với một hệ thống cảnh báo: nó tự phát hiện
các tín hiệu báo động đã đăng ký trước — TF-IDF EM > 10% (nghi bug metric hoặc
leakage), bất kỳ model EM > 85% (nghi leakage), model fine-tuned kém hơn
zero-shot, điểm impossible ≈ 100% kèm answerable ≈ 0% (suy sụp về trả rỗng).
Chính tín hiệu cuối cùng đã bắt được ViSoBERT.

---

## 8. Sẽ xem lại khi hệ thống lớn hơn

| Thời điểm | Thiết kế cần xem lại | Vì sao |
|---|---|---|
| Thêm model thứ 5–6 | `catalog.MODELS` là tuple tĩnh | Vẫn ổn tới ~10 model; sau đó nên nạp từ file cấu hình |
| Đánh giá trên full split (3.814 câu) | `evaluate()` giữ `per_item` cho mọi qid trong bộ nhớ | 3.814 dict nhỏ — vẫn ổn; ở quy mô 100k thì cần stream |
| Nhiều người dùng demo cùng lúc | Streamlit một tiến trình, model nạp trong session | Cần tách service inference riêng + hàng đợi |
| Chạy trên máy có CUDA | `pick_device()` ưu tiên MPS trước CUDA | Thứ tự đúng cho máy phát triển, sai cho máy server |
| `transformers` lên bản mới | `make_windows()` tự cài (ADR-003) | Nếu lỗi overflow được sửa, cân nhắc quay lại API chuẩn — nhưng chỉ sau khi đo lại bảng ở ADR-003 |
| Muốn so sánh với công trình khác | Đang dùng validation vì test là blind | Nộp lên leaderboard của ViQuAD mới có số trên test |

---

## 9. Giới hạn đã biết

1. **Đánh giá trên validation, không phải test** — vì test là blind split. Giới
   hạn của dataset, không phải của thiết kế.
2. **~30% câu validation là impossible**; metric tổng **trộn hai kỹ năng** — tìm
   đúng span, và biết khi nào nên trả lời rỗng. Đã tách trong `answerable_only` /
   `impossible_only`, nhưng mọi bảng chỉ có EM/F1 tổng đều đang che mất điều này.
3. **`question_type` là heuristic tự gán**, ngưỡng `0.6` chọn theo quan sát,
   **không** phải nhãn có sẵn của ViQuAD. Nó không phân biệt được suy luận nhiều
   bước thật sự với việc chỉ dùng từ đồng nghĩa.
4. **Kết quả `n = 500` là mẫu con** — CI 95% khoảng ±4,3 điểm (§5.4). Dùng
   `--full` cho số cuối cùng.
5. **Bucket `<100` có đúng 1 câu** ⇒ mọi F1 của nó là nhiễu. Cờ `unreliable` đã
   gắn, nhưng người đọc bảng vẫn phải để ý.
6. **mBERT chưa được huấn luyện đủ**: loss vẫn giảm và val vẫn tăng ở epoch 2 ⇒
   con số 50,80 EM là **cận dưới**, không phải trần của model.
