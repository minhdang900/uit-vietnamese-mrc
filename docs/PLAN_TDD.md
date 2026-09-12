# KẾ HOẠCH XÂY DỰNG T11 — TDD từ đầu
**Đề tài T11 — Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt**
Vietnamese Machine Reading Comprehension using Transformer Models
CS116 · UIT · Lập ngày 2026-09-12 · Thiết kế lại từ đầu theo Test-Driven Development

---

# 0. Ràng buộc và sự thật đã kiểm chứng

Mọi số dưới đây **đã kiểm trực tiếp** từ file dữ liệu trong repo, không lấy từ tài liệu thứ cấp.

## 0.1 Dataset — UIT-ViQuAD 2.0 (`taidng/UIT-ViQuAD2.0`)

Định dạng SQuAD-2.0 JSON:
```
{ "version": 2.0,
  "data": [ { "title": str,
              "paragraphs": [ { "context": str,
                                "qas": [ { "id": str,
                                           "question": str,
                                           "is_impossible": bool,
                                           "answers": {"text": [str], "answer_start": [int]},
                                           "plausible_answers": {...}   # chỉ khi is_impossible
                                         } ] } ] } ] }
```

| Split (sau dedup) | Questions | Contexts | Impossible | Gold answers |
|---|---|---|---|---|
| train | 28.454 | 4.101 | 9.216 (32,4%) | ✅ có |
| validation | 2.854 | 424 | 851 (29,8%) | ✅ có |
| test | 4.407 | 659 | 0 (0,0%) | ❌ **rỗng toàn bộ** |

## 0.2 ⚠️ Phát hiện quan trọng: test split là **blind test set**

Kiểm trực tiếp 4.407 câu hỏi trong `viquad2_deduped_test.json`: **cả 4.407 câu có `answers.text` rỗng**, đồng thời `is_impossible = False` cho toàn bộ.

**Diễn giải đúng:** đây là **held-out/blind split** — đáp án bị lược bỏ để dùng cho leaderboard, và `is_impossible=False` chỉ là **giá trị placeholder**, không phải nhãn thật.

**Hệ quả bắt buộc cho thiết kế:**
1. **Không thể** tính EM/F1 trên test split. Mọi số báo cáo phải ghi rõ là **validation**.
2. Đây **không phải** thiếu sót của nhóm — đó là tính chất của dataset. Nêu rõ trong báo cáo với lập luận này, nó trở thành điểm mạnh về hiểu dữ liệu.
3. Kế hoạch này dùng **train → fit, validation → report**, và **không** dùng test ngoài mục đích kiểm tra pipeline chạy end-to-end.

## 0.3 ⚠️ Mâu thuẫn trong file thống kê cũ — cần dọn

`viquad2_split_stats.json` ghi `train.num_contexts = 138` nhưng file thật có **4.101** paragraphs trên **138 articles**. Trường `num_contexts` thực chất đang đếm **article (title)**, không phải context.

→ Phase 2 phải định nghĩa **rõ ràng** đơn vị dedup là **context (paragraph)**, không phải article, và sinh lại thống kê bằng code có test.

## 0.4 Ràng buộc hạ tầng

| | |
|---|---|
| Thiết bị | **CPU-only** (không GPU) |
| Python hệ thống | 3.9.6 — **không đủ**, torch cần ≥3.10 |
| Công cụ có sẵn | `uv` 0.11.23 |
| Hệ quả | Fine-tune full trên 28.454 câu là **bất khả thi**. Chiến lược: pre-trained inference + fine-tune nhỏ có kiểm soát |

## 0.5 Yêu cầu từ đề bài T11

| Yêu cầu | Diễn giải thành tiêu chí nghiệm thu |
|---|---|
| Input: context + question → answer span | API `predict(context, question) -> str` |
| TF-IDF retrieval baseline | Bắt buộc có, làm sàn so sánh |
| mBERT / PhoBERT+QA head / XLM-R | "hoặc" ⇒ 1 là đủ, nhưng **≥2 để có so sánh** |
| Exact Match, token-level F1 | Metric chính, tự implement + test |
| Phân tích context length | Breakdown theo bucket độ dài |
| Phân tích single- vs multi-sentence reasoning | Breakdown theo loại câu hỏi |

---

# 1. Nguyên tắc TDD cho dự án này

## 1.1 Vì sao thứ tự phase là METRIC TRƯỚC, MODEL SAU

`AUDIT.md` ghi lại nguyên nhân tử vong của phiên bản v1: **báo cáo số liệu chưa từng được sinh ra** — 68,5% EM / 84,5% F1 là số bịa, mâu thuẫn nhau giữa README / báo cáo / app, và trái với artifact thật duy nhất (0,6% EM).

Đó không phải lỗi cẩu thả — đó là **lỗi kiến trúc**: không có thước đo đáng tin nào được xác lập *trước* khi có số để báo cáo.

> **Nguyên tắc dẫn đường:** thước đo phải được kiểm chứng **trước** vật được đo.
> Một model không có metric đã test là một model không có kết quả.

Vì vậy **Phase 1 là metrics**, trước cả data loader. Metric là hàm thuần (pure function) — dễ test nhất, không phụ thuộc gì, và mọi thứ khác phụ thuộc vào nó.

## 1.2 Vòng lặp TDD áp dụng cho mỗi hạng mục

```
1. RED     — Viết test trước. CHẠY nó. Xác nhận nó FAIL, và fail vì ĐÚNG lý do
             (AssertionError / ImportError mong đợi — không phải typo trong test).
2. GREEN   — Viết implementation TỐI THIỂU để test pass. Không thêm tính năng.
3. REFACTOR— Dọn code. Chạy lại toàn bộ test suite.
4. COMMIT  — Một commit cho một hành vi đã test.
```

**Quy tắc không thương lượng:**
- Không viết dòng code production nào mà không có test fail trước đó.
- Không có `test.skip`, `@pytest.mark.xfail` để che test hỏng.
- Không có số nào trong báo cáo mà không đến từ một artifact do code sinh ra.
- Mỗi con số báo cáo phải **truy vết được** về một file trong `results/` + commit hash.

## 1.3 Cấu trúc thư mục đích

```
t11_mrc/
├── pyproject.toml            # khai báo phụ thuộc + cấu hình pytest
├── README.md                 # hướng dẫn chạy (tiêu chí chấm #4)
├── src/mrc/
│   ├── __init__.py
│   ├── normalize.py          # Phase 1a — chuẩn hoá chuỗi
│   ├── metrics.py            # Phase 1b — EM, token-F1, aggregate
│   ├── schema.py             # Phase 2a — dataclass Example
│   ├── data.py               # Phase 2b — load, dedup, split, leakage guard
│   ├── tagging.py            # Phase 2c — question_type, length bucket
│   ├── predictor.py          # Phase 3a — Protocol (interface chung)
│   ├── baseline_tfidf.py     # Phase 3b — TF-IDF retrieval
│   ├── windowing.py          # Phase 4a — doc-stride + offset mapping
│   ├── transformer_qa.py     # Phase 4b — QA inference
│   └── evaluate.py           # Phase 5 — harness + breakdown
├── scripts/
│   ├── run_eval.py           # CLI đánh giá
│   ├── finetune_small.py     # Phase 6 — training curve
│   └── make_figures.py       # Phase 7 — hình cho báo cáo
├── app/streamlit_app.py      # Phase 8 — demo
├── tests/
│   ├── conftest.py           # fixtures: mini-dataset inline
│   ├── test_normalize.py
│   ├── test_metrics.py
│   ├── test_schema.py
│   ├── test_data.py
│   ├── test_tagging.py
│   ├── test_baseline_tfidf.py
│   ├── test_windowing.py
│   ├── test_transformer_qa.py
│   ├── test_evaluate.py
│   └── test_smoke_e2e.py     # chạy pipeline thật trên 20 câu
└── results/                  # artifact do code sinh, KHÔNG sửa tay
```

---

# 2. PHASE 0 — Môi trường & bộ khung (0,5 ngày)

## 2.1 Tiêu chí hoàn thành
`pytest` chạy được và báo **"no tests ran"** — nghĩa là hạ tầng test hoạt động, chưa có test nào.

## 2.2 Việc

```bash
uv venv --python 3.12 .venv          # 3.9.6 hệ thống KHÔNG đủ cho torch
source .venv/bin/activate
uv pip install pytest pytest-cov numpy pandas scikit-learn matplotlib
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # bản CPU, nhẹ hơn nhiều
uv pip install transformers streamlit
uv pip freeze > requirements.txt      # PIN phiên bản ⇒ tái lập
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers --cov=src/mrc --cov-report=term-missing"
markers = [
  "slow: cần tải model (bỏ qua bằng -m 'not slow')",
]
```

## 2.3 Cổng kiểm soát
- [ ] `pytest` exit code 5 (no tests collected) — hạ tầng OK
- [ ] `python -c "import torch; print(torch.__version__)"` chạy được
- [ ] `requirements.txt` có **pin phiên bản chính xác**, không phải `>=`

---

# 3. PHASE 1 — Metrics (1 ngày) ⭐ PHASE QUAN TRỌNG NHẤT

Không có model nào được viết trước khi phase này xanh hoàn toàn.

## 3.1 Ba quyết định thiết kế phải chốt bằng test

Đây là chỗ ViQuAD **khác** SQuAD tiếng Anh, và là chỗ dễ sai âm thầm nhất.

### Quyết định A — Không loại bỏ mạo từ (articles)
SQuAD tiếng Anh loại `a / an / the` khi chuẩn hoá. Tiếng Việt **không có mạo từ**; các từ tương tự về vị trí (`cái`, `con`, `những`, `các`) là **loại từ / lượng từ** và **có thể là phần của đáp án đúng**.
→ **Quyết định: KHÔNG loại bỏ từ nào.** Pin bằng test để người sau không "sửa cho giống SQuAD".

### Quyết định B — Token hoá theo khoảng trắng (âm tiết), không dùng word segmentation
ViQuAD official eval tách theo khoảng trắng ⇒ token là **âm tiết** ("Hà", "Nội"), không phải từ ("Hà_Nội").
→ **Quyết định: `text.split()`.** Lý do: (1) khớp với eval chính thức, so sánh được với công trình khác; (2) không thêm phụ thuộc `underthesea`/`VnCoreNLP`; (3) tránh việc segmenter sai làm nhiễu metric.
→ Pin bằng test, và **ghi vào báo cáo** như một quyết định có ý thức.

### Quyết định C — Quy ước cho câu hỏi impossible
Theo SQuAD 2.0: gold là chuỗi rỗng. Dự đoán rỗng ⇒ EM=1, F1=1. Dự đoán khác rỗng ⇒ EM=0, F1=0.

## 3.2 `tests/test_normalize.py` — viết TRƯỚC `normalize.py`

```python
from mrc.normalize import normalize_answer

def test_lowercases():
    assert normalize_answer("Hà Nội") == "hà nội"

def test_strips_punctuation():
    assert normalize_answer("Hà Nội.") == "hà nội"
    assert normalize_answer("(Hà Nội)") == "hà nội"

def test_collapses_whitespace():
    assert normalize_answer("  Hà   Nội \n") == "hà nội"

def test_preserves_vietnamese_diacritics():
    # KHÔNG được strip dấu — "hoà" ≠ "hoa"
    assert normalize_answer("Hoà Bình") == "hoà bình"

def test_does_NOT_remove_vietnamese_classifiers():
    # QUYẾT ĐỊNH A — pin hành vi: loại từ là phần của đáp án
    assert normalize_answer("các tỉnh miền Bắc") == "các tỉnh miền bắc"
    assert normalize_answer("con sông Hồng")    == "con sông hồng"

def test_empty_and_none_safe():
    assert normalize_answer("")   == ""
    assert normalize_answer(None) == ""
```

**Xác nhận RED đúng lý do:** chạy `pytest tests/test_normalize.py` → phải là `ModuleNotFoundError: No module named 'mrc.normalize'`. Nếu là `SyntaxError` trong test thì đó là lỗi test, sửa test trước.

## 3.3 `tests/test_metrics.py` — viết TRƯỚC `metrics.py`

```python
import pytest
from mrc.metrics import exact_match, token_f1, metric_max_over_ground_truths, evaluate

# ── EM ───────────────────────────────────────────────────────────
def test_em_exact_after_normalization():
    assert exact_match("Hà Nội.", "hà nội") == 1.0

def test_em_zero_on_different_answer():
    assert exact_match("Hà Nội", "Sài Gòn") == 0.0

def test_em_zero_on_superset():
    # EM là toàn-hoặc-không: thừa từ ⇒ 0
    assert exact_match("thủ đô Hà Nội", "Hà Nội") == 0.0

# ── token F1 ─────────────────────────────────────────────────────
def test_f1_perfect_overlap_is_one():
    assert token_f1("Hà Nội", "Hà Nội") == pytest.approx(1.0)

def test_f1_partial_overlap_exact_value():
    # pred = [thủ, đô, hà, nội] (4) ; gold = [hà, nội] (2) ; common = 2
    # P = 2/4 = 0.5 ; R = 2/2 = 1.0 ; F1 = 2(0.5)(1)/1.5 = 0.6667
    assert token_f1("thủ đô Hà Nội", "Hà Nội") == pytest.approx(2/3, abs=1e-4)

def test_f1_no_overlap_is_zero():
    assert token_f1("Sài Gòn", "Hà Nội") == 0.0

def test_f1_tokenizes_on_whitespace_not_words():
    # QUYẾT ĐỊNH B — "Hà Nội" là 2 token, không phải 1
    assert token_f1("Hà", "Hà Nội") == pytest.approx(2/3, abs=1e-4)

def test_f1_counts_token_multiplicity():
    # bag-of-tokens: token lặp phải được đếm đúng số lần (dùng Counter, không dùng set)
    # pred = [a, a, b] ; gold = [a, b] ; common = {a:1, b:1} = 2
    # P = 2/3 ; R = 2/2 = 1 ; F1 = 2(2/3)(1)/(5/3) = 0.8
    assert token_f1("a a b", "a b") == pytest.approx(0.8, abs=1e-4)

# ── quy ước impossible (QUYẾT ĐỊNH C) ────────────────────────────
def test_f1_both_empty_is_perfect():
    assert token_f1("", "") == 1.0

def test_f1_pred_nonempty_gold_empty_is_zero():
    assert token_f1("Hà Nội", "") == 0.0

def test_f1_pred_empty_gold_nonempty_is_zero():
    assert token_f1("", "Hà Nội") == 0.0

# ── nhiều đáp án đúng ────────────────────────────────────────────
def test_max_over_ground_truths_picks_best():
    assert metric_max_over_ground_truths(
        exact_match, "Hà Nội", ["Sài Gòn", "Hà Nội", "Đà Nẵng"]) == 1.0

def test_max_over_empty_ground_truths_treats_as_impossible():
    # gold list rỗng ⇒ câu impossible ⇒ chỉ dự đoán rỗng mới đúng
    assert metric_max_over_ground_truths(exact_match, "",        []) == 1.0
    assert metric_max_over_ground_truths(exact_match, "Hà Nội",  []) == 0.0

# ── tổng hợp ─────────────────────────────────────────────────────
def test_evaluate_returns_percentages_not_fractions():
    preds = {"q1": "Hà Nội", "q2": "sai"}
    golds = {"q1": ["Hà Nội"], "q2": ["Sài Gòn"]}
    r = evaluate(preds, golds)
    assert r["EM"] == pytest.approx(50.0)      # PHẦN TRĂM, không phải 0.5
    assert 0.0 <= r["F1"] <= 100.0
    assert r["count"] == 2

def test_evaluate_on_empty_input_returns_zero_not_nan():
    r = evaluate({}, {})
    assert r["EM"] == 0.0 and r["F1"] == 0.0 and r["count"] == 0

def test_evaluate_raises_on_missing_prediction():
    # Thà FAIL TO ỒN hơn âm thầm tính trên tập con
    with pytest.raises(KeyError):
        evaluate({"q1": "x"}, {"q1": ["x"], "q2": ["y"]})
```

> `test_evaluate_raises_on_missing_prediction` là test **chống lại chính căn bệnh của v1**: nếu thiếu dự đoán mà hàm âm thầm bỏ qua, bạn sẽ báo cáo EM trên 380 câu rồi ghi "n=400". Bắt nó nổ.

## 3.4 Cổng kiểm soát Phase 1
- [ ] 20+ test pass
- [ ] `--cov` cho `normalize.py` và `metrics.py` đạt **100%**
- [ ] Kiểm chứng chéo: tính tay `token_f1("thủ đô Hà Nội", "Hà Nội")` = 0,6667 ✅
- [ ] Ba quyết định A/B/C **đều có test pin hành vi**

---

# 4. PHASE 2 — Tầng dữ liệu (1,5 ngày)

## 4.1 `tests/conftest.py` — mini-dataset inline, không đọc file thật

```python
import pytest

@pytest.fixture
def mini_squad():
    """Dataset tối thiểu bao đủ các trường hợp biên."""
    return {
        "version": 2.0,
        "data": [
            {"title": "Hà Nội", "paragraphs": [
                {"context": "Hà Nội là thủ đô của Việt Nam. Dân số khoảng 8 triệu người.",
                 "qas": [
                    {"id": "q1", "question": "Thủ đô Việt Nam là gì?",
                     "is_impossible": False,
                     "answers": {"text": ["Hà Nội"], "answer_start": [0]}},
                    {"id": "q2", "question": "GDP Hà Nội là bao nhiêu?",
                     "is_impossible": True, "answers": {"text": [], "answer_start": []},
                     "plausible_answers": {"text": ["8 triệu"], "answer_start": [40]}},
                 ]}]},
            {"title": "Huế", "paragraphs": [
                {"context": "Huế là thành phố ở miền Trung Việt Nam.",
                 "qas": [
                    {"id": "q3", "question": "Huế ở đâu?", "is_impossible": False,
                     "answers": {"text": ["miền Trung Việt Nam", "miền Trung"],
                                 "answer_start": [22, 22]}},
                 ]}]},
        ]}
```

## 4.2 `tests/test_data.py`

```python
from mrc.data import (parse_squad, deduplicate_contexts, split_by_context,
                      assert_no_leakage, compute_stats)
import pytest

def test_parse_extracts_all_questions(mini_squad):
    ex = parse_squad(mini_squad)
    assert len(ex) == 3
    assert {e.qid for e in ex} == {"q1", "q2", "q3"}

def test_parse_attaches_context_and_title(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q1"].context.startswith("Hà Nội là thủ đô")
    assert ex["q1"].title == "Hà Nội"

def test_parse_reads_answers_as_dict_of_lists(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q1"].answers == ["Hà Nội"]
    assert ex["q3"].answers == ["miền Trung Việt Nam", "miền Trung"]   # giữ CẢ HAI

def test_impossible_question_has_empty_answers(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert ex["q2"].is_impossible is True
    assert ex["q2"].answers == []

def test_plausible_answers_are_NOT_used_as_gold(mini_squad):
    # BẪY: dùng plausible_answers làm gold sẽ làm metric sai một cách âm thầm
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    assert "8 triệu" not in ex["q2"].answers

def test_answer_start_points_at_the_answer(mini_squad):
    ex = {e.qid: e for e in parse_squad(mini_squad)}
    e = ex["q1"]
    assert e.context[e.answer_start : e.answer_start + len(e.answers[0])] == "Hà Nội"

# ── dedup: đơn vị là CONTEXT (paragraph), không phải ARTICLE ──────
def test_dedup_removes_repeated_context():
    raw = {"data": [
        {"title": "A", "paragraphs": [{"context": "X", "qas": []}, {"context": "X", "qas": []}]},
    ]}
    assert len(deduplicate_contexts(parse_squad(raw))) <= 1

def test_stats_counts_contexts_not_articles(mini_squad):
    # QUYẾT ĐỊNH: sửa lỗi của viquad2_split_stats.json (đếm article, gọi là context)
    s = compute_stats(parse_squad(mini_squad))
    assert s["num_contexts"] == 2     # 2 paragraph
    assert s["num_articles"] == 2
    assert s["num_questions"] == 3
    assert s["num_impossible"] == 1
    assert s["impossible_pct"] == pytest.approx(100 * 1/3, abs=0.01)

# ── chống leakage ────────────────────────────────────────────────
def test_split_puts_all_questions_of_a_context_together(mini_squad):
    tr, va = split_by_context(parse_squad(mini_squad), val_frac=0.5, seed=0)
    ctx_tr = {e.context for e in tr}
    ctx_va = {e.context for e in va}
    assert ctx_tr.isdisjoint(ctx_va)

def test_assert_no_leakage_passes_on_disjoint(mini_squad):
    tr, va = split_by_context(parse_squad(mini_squad), val_frac=0.5, seed=0)
    assert_no_leakage(tr, va)          # không được raise

def test_assert_no_leakage_RAISES_on_shared_context(mini_squad):
    ex = parse_squad(mini_squad)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(ex, ex)      # cùng tập ⇒ chắc chắn leak

def test_split_is_deterministic_with_seed(mini_squad):
    ex = parse_squad(mini_squad)
    a, _ = split_by_context(ex, val_frac=0.5, seed=42)
    b, _ = split_by_context(ex, val_frac=0.5, seed=42)
    assert [e.qid for e in a] == [e.qid for e in b]
```

## 4.3 `tests/test_tagging.py`

```python
from mrc.tagging import question_type, length_bucket

def test_length_bucket_boundaries_are_explicit():
    # Pin BIÊN. Bucket "100-200" phải nói rõ 200 thuộc bên nào.
    assert length_bucket(99)  == "<100"
    assert length_bucket(100) == "100-200"
    assert length_bucket(200) == "100-200"     # biên trên ĐÓNG
    assert length_bucket(201) == "200-300"
    assert length_bucket(301) == "300+"

def test_length_bucket_counts_WORDS_not_characters():
    assert length_bucket_from_text("a " * 150) == "100-200"

def test_question_type_returns_only_known_labels():
    assert question_type("Huế ở đâu?", "Huế là thành phố ở miền Trung.") in {
        "single-sentence", "multi-sentence"}

def test_high_lexical_overlap_with_one_sentence_is_single():
    ctx = "Hà Nội là thủ đô của Việt Nam. Paris là thủ đô của Pháp."
    assert question_type("Thủ đô của Việt Nam là gì?", ctx) == "single-sentence"

def test_question_spanning_two_sentences_is_multi():
    ctx = "Hà Nội là thủ đô. Thành phố này có 8 triệu dân."
    assert question_type("Thủ đô Việt Nam có bao nhiêu dân?", ctx) == "multi-sentence"
```

> **Ghi chú trung thực bắt buộc:** `question_type` là **heuristic**, không phải nhãn vàng của dataset. Báo cáo **phải** nói rõ điều này — nếu không, người chấm hiểu sai rằng ViQuAD có nhãn reasoning-scope. Test chỉ pin *hành vi của heuristic*, không chứng minh nó đúng.

## 4.4 Cổng kiểm soát Phase 2
- [ ] Toàn bộ test pass; coverage `data.py` + `tagging.py` ≥ 95%
- [ ] `compute_stats` chạy trên file thật, **tái tạo đúng** các số ở §0.1
- [ ] `scripts/make_stats.py` ghi `results/dataset_stats.json`, thay thế các file stats mâu thuẫn cũ
- [ ] Có test khẳng định **test split không có gold** → chốt §0.2 bằng code

---

# 5. PHASE 3 — Interface + Baseline TF-IDF (1 ngày)

## 5.1 Chốt interface trước, bằng test

```python
# src/mrc/predictor.py
from typing import Protocol

class Predictor(Protocol):
    name: str
    def predict(self, context: str, question: str) -> str: ...
```

```python
# tests/test_baseline_tfidf.py
from mrc.baseline_tfidf import TfidfRetriever
import pytest

def test_conforms_to_predictor_interface():
    m = TfidfRetriever()
    assert isinstance(m.name, str) and callable(m.predict)

def test_returns_a_sentence_from_the_context():
    ctx = "Hà Nội là thủ đô Việt Nam. Paris là thủ đô Pháp."
    out = TfidfRetriever().predict(ctx, "Thủ đô Việt Nam là gì?")
    assert out in ctx                       # phải là SPAN của context, không sinh mới

def test_picks_the_relevant_sentence():
    ctx = "Paris là thủ đô Pháp. Hà Nội là thủ đô Việt Nam."
    assert "Hà Nội" in TfidfRetriever().predict(ctx, "Thủ đô Việt Nam là gì?")

def test_empty_context_returns_empty_not_crash():
    assert TfidfRetriever().predict("", "câu hỏi?") == ""

def test_single_sentence_context_returns_that_sentence():
    assert TfidfRetriever().predict("Chỉ một câu.", "gì?") == "Chỉ một câu."

def test_deterministic():
    ctx, q = "A rồi B. C rồi D.", "B là gì?"
    m = TfidfRetriever()
    assert m.predict(ctx, q) == m.predict(ctx, q)
```

## 5.2 Dự đoán kết quả — và vì sao phải viết ra TRƯỚC khi chạy

**Giả thuyết đăng ký trước (pre-registered):** baseline sẽ có **F1 trung bình (~20–30%) nhưng EM gần 0**, vì nó trả về **cả câu** trong khi gold là **cụm vài từ** → overlap token có, trùng khít thì không.

Viết giả thuyết này vào `results/hypotheses.md` **trước** khi chạy. Nếu kết quả khớp ⇒ pipeline hoạt động đúng. Nếu EM cao bất thường (>10%) ⇒ **nghi ngờ bug hoặc leakage**, đi điều tra thay vì ăn mừng.

> Đây là cơ chế phòng vệ trực tiếp chống lại thất bại của v1: khi bạn *chưa biết* mình mong đợi gì, mọi con số đều trông hợp lý.

## 5.3 Cổng kiểm soát
- [ ] Test pass; `predict` **luôn** trả về substring của context (property test)
- [ ] `results/hypotheses.md` có giả thuyết ghi **trước** lần chạy đầu
- [ ] Số thật được sinh, và **so với giả thuyết** trong báo cáo

---

# 6. PHASE 4 — Transformer QA (2 ngày) — phase khó nhất

## 6.1 Vì sao tách `windowing.py` khỏi `transformer_qa.py`

Doc-stride + offset mapping là nơi phát sinh lỗi âm thầm nhiều nhất: span lệch vài ký tự, span lấy từ **question** thay vì **context**, span vắt qua hai window. Những lỗi này **không crash** — chúng chỉ làm F1 thấp một cách khó hiểu.

Tách ra thành module thuần (không cần model) ⇒ test được **không cần tải 1GB weights**.

## 6.2 `tests/test_windowing.py` — không cần model

```python
from mrc.windowing import make_windows, decode_span
import pytest

def test_short_context_yields_one_window():
    assert len(make_windows("ngắn", "q?", max_len=384, stride=128)) == 1

def test_long_context_yields_multiple_windows():
    assert len(make_windows("từ " * 2000, "q?", max_len=384, stride=128)) > 1

def test_windows_overlap_by_stride():
    w = make_windows("từ " * 2000, "q?", max_len=384, stride=128)
    assert w[1].context_char_start < w[0].context_char_end   # CÓ chồng lấp

def test_windows_cover_entire_context():
    ctx = "từ " * 2000
    w = make_windows(ctx, "q?", max_len=384, stride=128)
    assert w[0].context_char_start == 0
    assert w[-1].context_char_end >= len(ctx.rstrip())

def test_decode_span_returns_exact_original_substring():
    ctx = "Hà Nội là thủ đô của Việt Nam."
    span = decode_span(ctx, "Thủ đô?", start_char=0, end_char=6)
    assert span == "Hà Nội"          # CHÍNH XÁC, không lệch ký tự, không mất dấu

def test_decode_span_never_returns_question_tokens():
    ctx, q = "Hà Nội là thủ đô.", "Thủ đô Việt Nam là gì?"
    for w in make_windows(ctx, q, max_len=64, stride=16):
        assert w.context_token_range[0] > 0     # token context luôn SAU token question

def test_rejects_inverted_span():
    with pytest.raises(ValueError):
        decode_span("abc", "q?", start_char=5, end_char=2)

def test_rejects_span_longer_than_max_answer_length():
    with pytest.raises(ValueError):
        decode_span("từ " * 100, "q?", start_char=0, end_char=250, max_answer_len=30)
```

## 6.3 `tests/test_transformer_qa.py` — đánh dấu `slow`

```python
import pytest
from mrc.transformer_qa import TransformerQA

pytestmark = pytest.mark.slow      # bỏ qua bằng: pytest -m "not slow"

@pytest.fixture(scope="module")
def model():
    return TransformerQA("deepset/xlm-roberta-base-squad2")

def test_answer_is_substring_of_context(model):
    ctx = "Hà Nội là thủ đô của Việt Nam, có khoảng 8 triệu dân."
    out = model.predict(ctx, "Thủ đô của Việt Nam là gì?")
    assert out in ctx                 # EXTRACTIVE — bất biến cốt lõi

def test_answers_a_simple_factoid(model):
    ctx = "Hà Nội là thủ đô của Việt Nam."
    assert "Hà Nội" in model.predict(ctx, "Thủ đô của Việt Nam là gì?")

def test_handles_context_longer_than_max_len(model):
    ctx = ("Đây là câu nhồi. " * 400) + "Thủ đô Việt Nam là Hà Nội."
    assert model.predict(ctx, "Thủ đô Việt Nam là gì?") in ctx   # không crash, span hợp lệ

def test_deterministic_across_calls(model):
    ctx, q = "Hà Nội là thủ đô.", "Thủ đô?"
    assert model.predict(ctx, q) == model.predict(ctx, q)

def test_reports_latency(model):
    out, ms = model.predict_timed("Hà Nội là thủ đô.", "Thủ đô?")
    assert ms > 0                     # latency là số ĐO ĐƯỢC, không phải ước lượng
```

## 6.4 Chọn model — và trả lời trước câu hỏi của người chấm

| Model | Trạng thái | Ghi chú |
|---|---|---|
| `deepset/xlm-roberta-base-squad2` | ✅ mở, đã có QA head | **Model chính.** XLM-R có tiếng Việt trong 100 ngôn ngữ pretraining |
| `bert-base-multilingual-cased` + QA head | ⚠️ cần fine-tune | Đề bài nêu mBERT ⇒ **nên có** để so sánh |
| `vinai/phobert-base-v2` | ⚠️ **không có QA head** | Cần fine-tune → chặn bởi CPU-only. Nêu là giới hạn |
| `nguyenvulebinh/vi-mrc-base` | ❌ gated sau HF auth | Nêu là rào cản truy cập; để sẵn env `MRC_QA_MODEL` |

**Mục tiêu tối thiểu: 2 model + baseline** ⇒ có bảng so sánh 3 dòng, thay vì 2.

## 6.5 Cổng kiểm soát
- [ ] `test_windowing.py` pass **không cần tải model**
- [ ] `pytest -m "not slow"` chạy < 10 giây ⇒ vòng lặp TDD nhanh
- [ ] Bất biến "output ⊆ context" được test cho **mọi** predictor
- [ ] Latency được **đo**, không ước lượng

---

# 7. PHASE 5 — Harness đánh giá (1 ngày)

## 7.1 `tests/test_evaluate.py`

```python
from mrc.evaluate import run_evaluation, breakdown_by
import pytest

class StubPerfect:
    name = "stub-perfect"
    def predict(self, context, question): return "Hà Nội"

class StubEmpty:
    name = "stub-empty"
    def predict(self, context, question): return ""

def test_perfect_stub_scores_100_on_matching_data(one_answer_dataset):
    r = run_evaluation(StubPerfect(), one_answer_dataset)
    assert r["overall"]["EM"] == pytest.approx(100.0)

def test_empty_stub_scores_100_on_all_impossible_data(all_impossible_dataset):
    # kiểm ngược: quy ước impossible đi đúng qua toàn harness
    assert run_evaluation(StubEmpty(), all_impossible_dataset)["overall"]["EM"] == 100.0

def test_result_records_provenance(one_answer_dataset):
    r = run_evaluation(StubPerfect(), one_answer_dataset)
    for k in ("dataset", "split", "n", "model", "timestamp", "commit"):
        assert k in r            # mỗi con số TRUY VẾT ĐƯỢC — chống bệnh của v1

def test_breakdown_subgroup_counts_sum_to_total(mixed_dataset):
    r = run_evaluation(StubPerfect(), mixed_dataset)
    b = r["by_context_length"]
    assert sum(v["count"] for v in b.values()) == r["overall"]["count"]

def test_breakdown_records_n_for_every_bucket(mixed_dataset):
    for v in run_evaluation(StubPerfect(), mixed_dataset)["by_context_length"].values():
        assert "count" in v      # KHÔNG có số nào không kèm n

def test_small_buckets_are_flagged_as_unreliable(tiny_bucket_dataset):
    b = run_evaluation(StubPerfect(), tiny_bucket_dataset)["by_context_length"]
    small = [v for v in b.values() if v["count"] < 30]
    assert all(v.get("unreliable") is True for v in small)

def test_question_type_breakdown_excludes_impossible_and_says_so(mixed_dataset):
    r = run_evaluation(StubPerfect(), mixed_dataset)
    qt = r["by_question_type"]
    assert sum(v["count"] for v in qt.values()) == r["overall"]["count"] - r["n_impossible"]
    assert qt["_note"]                # ghi rõ vì sao tổng KHÁC n
```

> Hai test cuối sinh ra từ đúng chỗ báo cáo v1 dễ bị bắt lỗi: bucket `n=3` bị trình bày như kết quả, và `149+136=285≠400` không được giải thích. Giờ **code tự bắt buộc** phải nói ra.

## 7.2 Cổng kiểm soát
- [ ] Stub hoàn hảo ⇒ 100 EM; stub rỗng trên data impossible ⇒ 100 EM (kiểm hai chiều)
- [ ] Mọi kết quả JSON có `dataset / split / n / model / timestamp / commit`
- [ ] Bucket `count < 30` **tự động** gắn `unreliable: true`
- [ ] Tổng các subgroup **luôn** khớp, hoặc có `_note` giải thích

---

# 8. PHASE 6 — Fine-tune nhỏ + training curve (1 ngày)

## 8.1 Mục đích — nói rõ để không bị hiểu sai
Mục đích **không** phải đạt SOTA. Là để (a) đáp ứng tiêu chí "biểu đồ đường quá trình học" của thầy, và (b) **quan sát và giải thích overfitting** bằng dữ liệu thật của mình.

## 8.2 Tests

```python
def test_curve_has_one_record_per_epoch():
    c = finetune_small(train_size=8, val_size=4, epochs=2)
    assert [r["epoch"] for r in c] == [1, 2]

def test_curve_records_both_train_loss_and_val_metric():
    for r in finetune_small(train_size=8, val_size=4, epochs=1):
        assert "train_loss" in r and "val_f1" in r and "val_em" in r

def test_finetune_split_has_no_context_leakage():
    tr, va = _finetune_splits(train_size=8, val_size=4)
    assert {e.context for e in tr}.isdisjoint({e.context for e in va})

def test_detect_overfitting_flags_diverging_curve():
    # hàm chẩn đoán được TEST, không phải nhận định bằng mắt
    curve = [{"epoch":1,"train_loss":4.6,"val_f1":28.0},
             {"epoch":2,"train_loss":3.6,"val_f1":28.4},
             {"epoch":3,"train_loss":2.8,"val_f1":24.5}]
    d = detect_overfitting(curve)
    assert d["overfitting"] is True and d["best_epoch"] == 2
```

> `detect_overfitting` biến một nhận xét chủ quan thành **hàm có test**. Báo cáo trích kết luận của nó thay vì "chúng em quan sát thấy".

## 8.3 Cổng kiểm soát
- [ ] Curve sinh từ run thật, ghi ra `results/training_curve.json`
- [ ] `detect_overfitting` có test và được gọi trong báo cáo
- [ ] Ghi rõ `train_size` trong **mọi** chỗ nhắc tới curve — vì nó là nguyên nhân gốc

---

# 9. PHASE 7–8 — Hình vẽ & Demo (1 ngày)

## 9.1 Hình — mỗi hình sinh từ `results/*.json`, không tay

| Hình | Nội dung | Test |
|---|---|---|
| `model_comparison.png` | EM/F1 theo model, có `bar_label` | file tồn tại, size > 10KB |
| `em_f1_by_length.png` | Breakdown độ dài, **ghi n trên mỗi cột** | assert nhãn n có mặt |
| `em_f1_by_qtype.png` | single vs multi-sentence | như trên |
| `training_curve.png` | 2 trục y: loss + val F1 | như trên |
| `confusion_by_outcome.png` | Ma trận đúng/sai giữa 2 model | như trên |

```python
def test_every_figure_is_regenerable_from_results(tmp_path):
    # Xoá hình, chạy lại, hình phải xuất hiện ⇒ KHÔNG có hình "mồ côi" vẽ tay
    make_figures(results_dir="results", out_dir=tmp_path)
    assert (tmp_path / "model_comparison.png").stat().st_size > 10_000

def test_figures_fail_loudly_if_results_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_figures(results_dir=tmp_path, out_dir=tmp_path)
```

## 9.2 Demo Streamlit — tiêu chí chấm #1 của thầy

```python
def test_app_module_imports_without_error():
    # v1 chết vì SyntaxError trong app. Test này ngăn tái diễn.
    import importlib; importlib.import_module("app.streamlit_app")

def test_app_has_no_syntax_error():
    import py_compile; py_compile.compile("app/streamlit_app.py", doraise=True)

def test_answer_handler_is_pure_and_testable():
    # tách logic khỏi UI ⇒ test được mà không cần chạy Streamlit
    from app.streamlit_app import answer
    out = answer("Hà Nội là thủ đô.", "Thủ đô?", predictor=StubPerfect())
    assert out["answer"] == "Hà Nội" and "latency_ms" in out
```

## 9.3 `tests/test_smoke_e2e.py` — cổng cuối

```python
@pytest.mark.slow
def test_end_to_end_on_20_real_questions():
    """Chạy pipeline THẬT trên dữ liệu THẬT. Cổng chống mọi số bịa."""
    ex = load_split("validation")[:20]
    r  = run_evaluation(TransformerQA(), ex)
    assert r["overall"]["count"] == 20
    assert 0.0 <= r["overall"]["EM"] <= 100.0
    assert r["commit"]                       # truy vết được
    for s in r["sample_predictions"]:
        assert s["prediction"] in s["context"]   # extractive, luôn luôn
```

---

# 10. Tổng hợp — lịch và cổng kiểm soát

| Phase | Nội dung | Thời lượng | Cổng không được bỏ qua |
|---|---|---|---|
| 0 | Môi trường | 0,5 ngày | `pytest` chạy; requirements **pin** phiên bản |
| **1** | **Metrics** ⭐ | **1 ngày** | **100% coverage; 3 quyết định A/B/C có test pin** |
| 2 | Tầng dữ liệu | 1,5 ngày | leakage guard **raise**; stats tái tạo §0.1 |
| 3 | Baseline TF-IDF | 1 ngày | output ⊆ context; **giả thuyết ghi trước** khi chạy |
| 4 | Transformer QA | 2 ngày | windowing test **không cần model**; latency **đo** |
| 5 | Harness | 1 ngày | provenance đủ 6 trường; bucket nhỏ **tự gắn cờ** |
| 6 | Fine-tune + curve | 1 ngày | `detect_overfitting` có test |
| 7–8 | Hình + Demo | 1 ngày | app **import được**; hình sinh lại được từ results |
| — | Báo cáo | 2 ngày | 4 giới hạn có mặt; 30–50 trang |
| | **Tổng** | **~11 ngày** | |

## 10.1 Bốn bất biến xuyên suốt — vi phạm là chặn merge

1. **Extractive:** mọi `predict()` trả về substring của context. Test cho **mọi** predictor.
2. **Không leakage:** `assert_no_leakage` chạy trong mọi đường split và **làm fail run**.
3. **Truy vết được:** mọi số trong báo cáo ⟶ một file `results/*.json` ⟶ một commit hash.
4. **Có n kèm số:** không bảng nào có số mà thiếu `count`; `count < 30` tự gắn `unreliable`.

## 10.2 Bốn giới hạn bắt buộc nêu trong báo cáo

- [ ] Đánh giá trên **validation**, không phải test — vì **test split là blind, toàn bộ 4.407 gold rỗng** (§0.2)
- [ ] **29,8% câu validation là impossible** — metric tổng trộn hai kỹ năng: tìm span, và biết khi nào trả lời rỗng
- [ ] `question_type` là **heuristic tự gán**, không phải nhãn có sẵn của ViQuAD
- [ ] Nếu dùng subset: ghi `n` và sai số (ở n=400, quanh mức 40% thì CI 95% ≈ **±4,8 điểm**)

## 10.3 Ba việc làm ngay

| # | Việc | Tại sao trước |
|---|---|---|
| 1 | `uv venv --python 3.12` + cài phụ thuộc | Python 3.9.6 hệ thống không chạy được torch; `.venv` hiện tại đã hỏng (base Python biến mất khỏi máy) |
| 2 | Viết `tests/test_normalize.py`, chạy, **xác nhận FAIL đúng lý do** | Vòng RED đầu tiên. Nếu bỏ qua bước xác nhận, bạn không làm TDD |
| 3 | `results/hypotheses.md` — ghi kỳ vọng cho baseline **trước** khi chạy | Phòng vệ trực tiếp chống bệnh của v1: không biết mong đợi gì thì số nào cũng hợp lý |
