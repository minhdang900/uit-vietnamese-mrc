# Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt
### Vietnamese Extractive Machine Reading Comprehension trên UIT-ViQuAD 2.0

**CS116 — Lập trình Python cho Máy học · Đề tài T11 · UIT, ĐHQG-HCM**

Cho một đoạn văn (`context`) và một câu hỏi (`question`), hệ thống **trích xuất**
đoạn văn bản chứa câu trả lời (`answer span`) nằm trong chính context đó.

> **Mọi con số trong repo này đến từ một lần chạy thật**, ghi ra `results/*.json`
> kèm commit hash, timestamp và thiết bị. Không có số nào được viết tay.

---

## Kết quả

UIT-ViQuAD 2.0, **validation split**, n = 500 (cùng một mẫu ngẫu nhiên seed=42 cho
mọi model), thiết bị **MPS (Apple M5 Pro)**:

| Model | EM | F1 | answerable EM / F1 | impossible EM | Latency |
|---|---:|---:|---:|---:|---:|
| TF-IDF Baseline | 0.80 | 23.09 | 1.11 / 31.99 | 0.00 | 0.5 ms |
| ViSoBERT + QA (fine-tuned) | 27.80 | 31.31 | **6.93** / 11.78 | **82.01** | 22.7 ms |
| XLM-R squad2 (zero-shot) | 40.60 | 56.84 | 45.71 / **68.20** | 27.34 | 14.3 ms |
| **mBERT + QA (fine-tuned)** | **50.80** | **59.49** | **54.57** / 66.60 | 41.01 | 13.5 ms |

Đối chiếu với giả thuyết ghi **trước** khi chạy (`results/hypotheses.md`):
`python scripts/check_hypotheses.py`.

### Ba điều bảng này nói ra, mà con số tổng thì không

**1. Baseline: F1 23% nhưng EM 0,8%.** Nó trả về **cả một câu**, còn gold là **cụm
vài từ** — overlap token có, trùng khít thì không. Khoảng cách EM–F1 đó là bằng
chứng trực quan rằng hai metric đo hai thứ khác nhau.

**2. mBERT thắng XLM-R KHÔNG phải vì tìm span giỏi hơn.** Trên câu answerable,
XLM-R zero-shot thực ra **tốt hơn** (F1 68,20 so với 66,60). mBERT thắng tổng thể
vì **biết khi nào KHÔNG nên trả lời** tốt hơn hẳn (impossible EM 41,01 so với
27,34). Với ~30% câu là unanswerable, kỹ năng thứ hai quyết định bảng xếp hạng.
Đây là lý do báo cáo tách `answerable_only` và `impossible_only` — con số tổng
trộn hai kỹ năng và che mất điều này.

**3. ViSoBERT suy sụp về "luôn trả rỗng".** EM tổng 27,80 **bằng đúng tỉ lệ
impossible của tập (27,80%)**. Bóc tách ra: impossible EM **82,01** nhưng
answerable EM chỉ **6,93** — nó gần như chỉ ăn điểm từ việc từ chối trả lời.
`scripts/check_hypotheses.py` phát hiện tự động điều này.

### Vì sao ViSoBERT thất bại — và vì sao đó là kết quả hợp lệ

Đã kiểm bốn nghi vấn trước khi kết luận — nhưng xem dòng ngay dưới bảng: **một trục
vẫn chưa được bù trừ**, và nó không có trong bảng này.

| Nghi vấn | Kiểm tra | Kết quả |
|---|---|---|
| `max_position_embeddings` < 384? | đọc config | 514 — không phải nguyên nhân |
| `max_answer_len=30` quá ngắn? | đo p95 gold answer | **đúng là confound** — 25,2% gold vượt 30 token. Sửa thành 64 |
| learning rate quá thấp? | 3e-5 → 5e-5 | có cải thiện (F1 12,25 → 30,48) nhưng vẫn kém xa |
| ngưỡng null lệch? | quét `null_threshold` | tốt nhất F1 34,77; ép trả lời cho EM **11,50** |

Trục **chưa** được kiểm ở bảng trên là `max_length`: giữ nguyên 384 (và `doc_stride`
128) cho cả hai model, dù tokenizer ViSoBERT sinh chuỗi dài hơn ~60% — 154/557 context
vượt ngân sách so với 16/557 của mBERT. Đây là **nguyên nhân cấu hình chưa được bù
trừ**, không phải giới hạn thật của model. (`max_answer_len` thì đã bù, 30 → 64 —
hai tham số này rất dễ nói lẫn.)

**Giải thích:** ViSoBERT pretrain trên văn bản **mạng xã hội** với vocab **15.004**
(mBERT: 119.547). MRC trên Wikipedia đòi **biên span chính xác** trên văn phong
trang trọng dày đặc **tên riêng** — đúng thứ mà vocab nhỏ chia vụn nặng nhất:

```
"Hà Nội là thủ đô của nước Cộng hoà..."
  ViSoBERT (23 token): ['▁Hà','▁N','ội','▁là','▁thủ','▁đô','▁c','ủa',...]
  mBERT    (17 token): ['Hà','Nội','là','thủ','đô','của','nước','Cộng',...]
```

⇒ Đây là **lời giải thích phù hợp với bằng chứng**, chưa phải nhân quả đã chứng
minh. Yếu tố mạnh nhất là `max_length = 384` — đặt theo tokenizer mBERT và **chưa
được chỉnh lại** cho ViSoBERT (`max_answer_len` thì đã bù, 30 → 64: đừng nói lẫn
hai tham số). Cùng với loss chưa hội tụ, sức chứa nhỏ hơn 45% và hố cực tiểu do
32,39% câu impossible, **bốn yếu tố đó đã đủ** để giải thích sự suy sụp mà không
cần nói gì về ViSoBERT như một encoder.

⇒ Vì vậy **chưa kết luận được** rằng tiền huấn luyện tiếng Việt không giúp ích.
Câu hỏi "tiếng Việt chuyên biệt có giúp không?" vẫn **để ngỏ**: PhoBERT chưa từng
được chạy, và ViSoBERT chưa được huấn luyện lại sau chẩn đoán. Phép kiểm trực
tiếp: `max_length` 768, lr 3e-5, giữ nguyên mọi thứ khác.

### Đường cong huấn luyện

| | epoch | train loss | val EM | val F1 |
|---|---:|---:|---:|---:|
| **mBERT** (lr 3e-5) | 1 | 2.1316 | 46.00 | 58.39 |
| | 2 | 1.2960 | 52.00 | 59.75 |
| **ViSoBERT** (lr 5e-5) | 1 | 3.0512 | 25.67 | 25.67 |
| | 2 | 2.5678 | 25.33 | 25.61 |
| | 3 | 2.1908 | 27.00 | 30.48 |

mBERT: loss giảm, val tăng ⇒ **chưa overfit, thậm chí còn thiếu epoch**.
ViSoBERT epoch 1–2: `val_EM == val_F1` ⇒ dấu hiệu **suy sụp về luôn-trả-rỗng**;
epoch 3 mới bắt đầu thoát ra.

## Cài đặt

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Yêu cầu Python **3.12+** (PyTorch không có wheel cho 3.9). Trên Apple Silicon,
PyTorch tự dùng **MPS**; không cần cấu hình gì thêm.

## Chạy bằng Docker — không cần cài gì ngoài Docker

Dành cho lúc nộp bài và lúc chấm: một lệnh chạy test, một lệnh mở demo.

```bash
docker compose run --rm tests     # 423 test, ~1 giây, KHÔNG cần mạng
docker compose up app             # demo tại http://localhost:8501
docker compose down               # dọn
```

Cả hai dùng **chung một ảnh**, khác nhau ở lệnh chạy — nên không bao giờ có
chuyện test xanh trên một bộ thư viện còn demo chạy trên bộ khác.

| Service | Lệnh | Việc |
|---|---|---|
| `app` | `docker compose up app` | Demo web, cổng 8501 (đổi bằng `MRC_PORT=8600`) |
| `tests` | `docker compose run --rm tests` | Bộ test nhanh, `HF_HUB_OFFLINE=1` |
| `tests-full` | `docker compose run --rm tests-full` | Đủ 474 test (423 + 51 slow), có tải model thật |
| `fetch-data` | `docker compose run --rm fetch-data` | Tải UIT-ViQuAD 2.0 về `data/raw/` |

Ba service sau nằm sau profile nên `docker compose up` chỉ dựng demo; `run` tự
bật profile của service nó gọi.

Bản thân hợp đồng đóng gói cũng được ghim bằng test: `tests/test_docker.py` đọc
`Dockerfile`, `compose.yaml` và `.dockerignore` để khẳng định mount dữ liệu là
read-only, `up` không kéo theo pytest, torch đến từ index CPU trên mọi kiến
trúc, và thư mục cache thuộc về người dùng `app`. Ba file đó được COPY vào ảnh
nên bộ test **tự kiểm chính môi trường đang chạy nó**.

### Cái gì nằm trong ảnh, cái gì được mount

Ảnh chứa **mã nguồn và `results/`** — tức mọi thứ đã commit. Nó **không** chứa
`models/` (3,4 GB) và `data/raw/` (16 MB): cả hai đều trong `.gitignore` vì là
sản phẩm của một lần chạy trên một máy cụ thể, nướng vào ảnh thì ảnh vừa phình
lên vừa hết tái lập được. Compose mount chúng lúc chạy, **read-only** — bất biến
"demo không sinh ra con số nào" do đó được kernel bắt buộc chứ không chỉ là quy
ước.

Chưa fine-tune thì cứ chạy: thư mục `models/` rỗng làm sidebar báo checkpoint
còn thiếu, XLM-R zero-shot và baseline TF-IDF vẫn dùng được. Cache HuggingFace
nằm trong named volume `hf-cache`, nên XLM-R chỉ tải một lần (~1,1 GB).

### Đóng gói để nộp

```bash
docker compose build app                # cần có ảnh trước
./scripts/make_delivery.sh              # → delivery/  (~1,5 GB, có checkpoint)
./scripts/make_delivery.sh --no-models  # → delivery/  (~500 MB, không checkpoint)
```

Sinh ra `delivery/` gồm ảnh Docker đã `docker save`, mã nguồn lấy bằng
`git archive HEAD`, checkpoint, dữ liệu, `HUONG_DAN.md` và `run.sh`. Người chấm
chỉ cần Docker:

```bash
./run.sh            # nạp ảnh rồi mở demo tại http://localhost:8501
./run.sh test       # chạy bộ test
./run.sh stop
```

Không cần Python, không cần mạng, không dựng lại ảnh — `compose.yaml` khai báo
sẵn `image:` nên nó dùng luôn ảnh vừa nạp. `tests/test_docker.py` ghim tag ảnh
giữa `make_delivery.sh` và `compose.yaml`: lệch tag thì compose lặng lẽ dựng lại
từ đầu ngay trên máy người chấm.

Gói **không** kèm `models/*/epoch*/` — đó là checkpoint giữa chừng của quá trình
huấn luyện, 1,3 GB mà demo không bao giờ nạp (transformers đọc thẳng ở thư mục
gốc của model). Bỏ chúng là khác biệt giữa gói 1,5 GB và gói 4 GB.

### Trong container là CPU, không phải MPS

Container Linux không thấy Metal. Demo chạy trong Docker ghi `thiết bị cpu` ở
chân thẻ đáp án và **chậm hơn ~4 lần** bản chạy thẳng trên máy (đo được 57,0 ms
so với 13,5 ms cho cùng câu hỏi mở màn). Chỉ là độ trễ — span trả về y hệt.

Bảng kết quả trong `results/*.json` vẫn là của lần chạy **MPS**, và mỗi màn hình
đều in kèm thiết bị sinh ra nó, nên hai con số không thể bị nhầm lẫn với nhau.

Vì không service nào xin GPU, Dockerfile cài PyTorch từ **index CPU** của
PyTorch trước khi đọc `requirements.txt` — trên mọi kiến trúc, không chỉ amd64.
Bản dựng đầu tiên chỉ ép CPU cho amd64, và ảnh arm64 sinh ra mang theo 3,3 GB
thư viện CUDA của NVIDIA cùng 817 MB Triton: wheel aarch64 của torch 2.x **cũng**
khai báo các gói `nvidia-*`. Bỏ điều kiện theo kiến trúc, `site-packages` rút từ
6,1 GB xuống 1,7 GB:

| | ảnh đầu | sau khi ép CPU |
|---|---:|---:|
| Dung lượng đĩa | 10,2 GB | **2,48 GB** |
| Dung lượng truyền (nén) | 3,52 GB | **527 MB** |

## Chạy

```bash
# 1. Tải dữ liệu (ghi ra data/raw/viquad2_{train,validation,test}.json)
python scripts/fetch_data.py

# 2. Đánh giá — baseline và model zero-shot, không cần huấn luyện
python scripts/run_eval.py --models baseline xlmr --limit 500
python scripts/run_eval.py --models baseline xlmr --full      # toàn bộ split

# 3. Fine-tune trên MPS (~30 phút/epoch cho 28.454 câu hỏi)
python scripts/finetune.py --model bert-base-multilingual-cased --out models/mbert
python scripts/finetune.py --model uitnlp/visobert             --out models/visobert

# 4. Đánh giá model đã fine-tune
python scripts/run_eval.py --models baseline xlmr mbert visobert --full

# 5. Sinh toàn bộ hình cho báo cáo từ results/*.json
python scripts/make_figures.py

# 5b. Gom vật liệu báo cáo (bảng + xuất xứ + bản sao hình) → report/assets/
python scripts/make_report.py

# 6. Demo web — sáu màn hình, mỗi màn hình một URL
streamlit run app/streamlit_app.py
```

Demo mở ở màn hình **Hỏi đáp**; năm màn hình còn lại deep-link được, tiện lúc
thuyết trình: `/ket-qua`, `/phan-tich-loi`, `/du-lieu`, `/so-sanh`, `/huan-luyen`.

Mở bằng URL là mở một phiên mới, nên model và ngưỡng quay về mặc định (mBERT,
ngưỡng +0,0) — đúng thứ ta muốn khi nhảy thẳng vào một slide. Bấm điều hướng
trong sidebar thì lựa chọn được giữ nguyên giữa các màn hình.

Không cần `pip install -e .`: `app/streamlit_app.py` tự đưa `src/` vào `sys.path`.
Chưa fine-tune thì demo vẫn chạy — sidebar báo checkpoint nào còn thiếu, và
XLM-R zero-shot cùng baseline TF-IDF vẫn dùng được ngay.

## Kiểm thử

```bash
pytest -m "not slow"    # vòng lặp TDD nhanh, không tải model  (~1 giây)
pytest                  # đầy đủ, có tải model
pytest --cov=mrc        # kèm coverage
```

---

## Dữ liệu

**UIT-ViQuAD 2.0** (`taidng/UIT-ViQuAD2.0`), định dạng SQuAD-2.0. Số liệu **đo trực
tiếp từ file tải về**, không chép từ tài liệu:

| Split | Questions | Contexts | Articles | Impossible | Có gold | **Chấm được** |
|---|---:|---:|---:|---:|---:|---:|
| train | 28.454 | 4.101 | 138 | 9.216 (32,4%) | 19.238 | 28.454 |
| validation | 3.814 | 557 | 19 | 1.161 (30,4%) | 2.653 | 3.814 |
| test | 7.301 | 1.241 | 48 | 0 | **0** | **0** |

### ⚠️ Test split là blind set — không thể đánh giá trên đó

Cả **7.301 câu test có `answers.text` rỗng**, đồng thời `is_impossible = False`.
Hai điều này **mâu thuẫn nhau** nếu coi là nhãn thật (nếu không phải câu impossible
thì phải có đáp án) ⇒ đây là **placeholder**: đáp án bị lược bỏ để dùng cho
leaderboard.

Hàm `assert_gradeable()` **từ chối** chấm split này. Không có nó, một model luôn
trả về chuỗi rỗng sẽ đạt **EM 100%** trên test split và con số đó trông hoàn toàn
hợp lý trong báo cáo.

### Lưu ý về `num_contexts` vs `num_articles`

train có **138 article** nhưng **4.101 context**. Vì chống leakage dựa vào việc một
*context* chỉ thuộc một split, nhầm hai đơn vị này làm vô hiệu hoá chính cơ chế bảo
vệ. `compute_stats()` báo cáo tách bạch cả hai.

---

## Kiến trúc

```
src/mrc/
  normalize.py       chuẩn hoá chuỗi — 3 quyết định đặc thù tiếng Việt (xem dưới)
  metrics.py         Exact Match + token-F1 theo quy ước SQuAD-2.0
  data.py            parse, dedup/split theo CONTEXT, assert_no_leakage, assert_gradeable
  tagging.py         question_type (heuristic) + length_bucket
  predictor.py       Protocol chung — baseline và transformer dùng cùng interface
  baseline_tfidf.py  TF-IDF + cosine sentence retrieval
  windowing.py       chọn span, map token→ký tự, doc-stride windowing (tự cài)
  transformer_qa.py  inference với QA head
  features.py        map vị trí đáp án ký tự → token cho fine-tuning
  evaluate.py        harness + provenance + breakdown

src/demo/            logic của demo, test được mà không chạy Streamlit
  logic.py           gọi predictor, và quyết định trả lời/từ chối theo ngưỡng
  results.py         CHỖ DUY NHẤT đọc results/*.json và data/raw/
  render.py          hàm thuần dữ liệu → HTML cho từng khối giao diện
  theme.py           token hệ thiết kế + CSS + escape
  catalog.py         danh mục model, màn hình, thành viên nhóm (không có metric)
  text.py, vi.py     đếm âm tiết, cắt câu; định dạng số kiểu Việt

app/                 CHỈ gọi widget Streamlit — không có logic
  streamlit_app.py   entry, điều hướng sáu màn hình
  shell.py           theme, state dùng chung, sidebar
  screens/*.py       một module mỗi màn hình
```

Ranh giới `app/` ↔ `src/demo/` được giữ cố ý: một `SyntaxError` trong file app
từng làm hỏng demo mà không test nào bắt được. Mọi thứ quyết định *nội dung* nằm
trong `src/demo/` dưới dạng hàm thuần; `app/` chỉ ghép chúng lại. Test compile và
import **mọi** file trong `app/`, không riêng file entry.

### Ba quyết định đặc thù tiếng Việt (đều được pin bằng test)

| | Quyết định | Vì sao |
|---|---|---|
| **A** | **Không loại bỏ mạo từ** | SQuAD tiếng Anh loại `a/an/the`. Tiếng Việt **không có mạo từ**; `các`, `con`, `những` là **loại từ** và **có thể là phần của đáp án đúng** |
| **B** | **Token hoá theo khoảng trắng** (âm tiết) | Khớp eval chính thức ViQuAD ⇒ so sánh được với công trình khác; tránh phụ thuộc segmenter và tránh lỗi segmenter làm nhiễu metric |
| **C** | **Giữ dấu tiếng Việt** | `"hoà"` ≠ `"hoa"`. Đây cũng là lý do không dùng `tokenizer.decode()` để lấy lại đáp án — decode làm mất dấu |

---

## Hai phát hiện kỹ thuật trong quá trình xây dựng

### 1. `return_overflowing_tokens` của transformers 5.17.0 bị giới hạn ở 2 window

Đo được: context 210 / 420 / 700 / 1400 token đều chỉ sinh **2 window**, đáng lẽ
phải là 2 / 5 / 8 / 16 — bất kể `stride` bằng bao nhiêu. Phần đuôi context bị cắt
**âm thầm**: không exception, không cảnh báo, chỉ là đáp án nằm cuối đoạn văn thì
không bao giờ tìm được.

Dự án **tự cài `make_windows()`** trong `windowing.py`. Ảnh hưởng thực tế ở
`max_length=384` là nhỏ (2/557 context validation vượt ngưỡng), nhưng đây là lỗi
đúng-sai âm thầm nên được sửa tận gốc. Test
`test_at_least_one_window_of_long_context_finds_the_answer` là thứ bắt được nó.

### 2. PhoBERT không dùng được cho extractive QA ở đây

`vinai/phobert-base` và `phobert-base-v2` **không có fast tokenizer**
(`is_fast=False`), nên **không có `offset_mapping`** — thứ bắt buộc để map token
span về vị trí ký tự trong context gốc. Không có nó, cách duy nhất lấy lại chuỗi là
`tokenizer.decode()`, và decode làm mất dấu tiếng Việt.

Đây là ràng buộc **kỹ thuật**, không phải ràng buộc tính toán. `nguyenvulebinh/vi-mrc-base`
thì bị **gated** sau HuggingFace auth (đã xác nhận).

**Thay thế: `uitnlp/visobert`** — encoder tiếng Việt của chính UIT NLP, có fast
tokenizer. Hạn chế đã biết: ViSoBERT được pretrain trên văn bản **mạng xã hội**,
trong khi ViQuAD là **Wikipedia**, nên có domain shift; điều này được nêu trong
phần giới hạn của báo cáo.

---

## Bốn bất biến của dự án

1. **Extractive** — mọi `predict()` trả về **substring của context**. Test cho mọi model.
2. **Không leakage** — `assert_no_leakage()` chạy trong mọi đường split và **làm fail run**.
3. **Truy vết được** — mọi kết quả mang `commit`, `timestamp`, `device`, `split`, `n`.
4. **Có n kèm số** — nhóm `count < 30` tự gắn `unreliable: true`; không bảng nào có số thiếu n.

Demo web chịu đúng bốn bất biến đó: mọi con số trên màn hình đọc từ
`results/*.json` hoặc đo trực tiếp từ `data/raw/` qua `demo.results`, kể cả thứ
hạng "tốt nhất" và ô in đậm trong bảng so sánh — chúng được **tính** từ dữ liệu,
không gõ tay, nên huấn luyện lại là màn hình tự nói đúng. Thanh "độ tin cậy" và
biên độ từ chối lấy từ `TransformerQA.predict_detailed`, không phải số minh hoạ.

## Giới hạn (nêu rõ trong báo cáo)

- Đánh giá trên **validation**, không phải test — vì test là blind split (xem trên).
- **~30% câu validation là impossible**; metric tổng **trộn hai kỹ năng**: tìm đúng
  span, và biết khi nào nên trả lời rỗng. Kết quả được tách riêng trong
  `answerable_only` / `impossible_only`.
- `question_type` (single- vs multi-sentence) là **heuristic tự gán**, **không** phải
  nhãn có sẵn của ViQuAD.
- Kết quả `--limit 500` là mẫu con; ở n=500 quanh mức 40%, khoảng tin cậy 95% là
  khoảng **±4,3 điểm**. Dùng `--full` cho số cuối cùng.

## Tài liệu

- **`docs/HUONG_DAN_NHOM.md` — hướng dẫn cho thành viên nhóm: chạy, thuyết trình,
  bắt kịp báo cáo. Bắt đầu từ đây nếu bạn chưa theo sát phần code.**
- `docs/SOLUTION.md` — đề xuất giải pháp và tech stack, kèm lý do từng lựa chọn
- `docs/PLAN_TDD.md` — kế hoạch TDD 9 phase với đặc tả test từng phase
- `results/hypotheses.md` — giả thuyết đăng ký **trước** khi chạy thực nghiệm

## Tham khảo

- Nguyen, K. V. et al. *UIT-ViQuAD: A Vietnamese Dataset for Evaluating Machine Reading Comprehension.*
- UIT NLP datasets — https://nlp.uit.edu.vn/datasets/
- Rajpurkar, P. et al. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL.
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).*
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.*
