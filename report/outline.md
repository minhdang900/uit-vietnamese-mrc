# Dàn ý báo cáo — CS116 T11

**Không có con số nào trong file này.** Đó là chủ ý: số nằm ở `assets/`, và chỗ
nào cần số thì ghi rõ lấy từ đâu. Xem `README.md` cùng thư mục để biết vì sao.

---

## Danh sách kiểm bắt buộc

### Bốn giới hạn PHẢI nêu (`docs/PLAN_TDD.md` §10.2)

- [ ] **Đánh giá trên validation, không phải test.** Test split là blind set —
      toàn bộ gold rỗng, `is_impossible = False`. Không phải nhóm làm thiếu; đó
      là tính chất dataset. Nêu kèm lập luận thì thành điểm mạnh về hiểu dữ liệu.
      Bằng chứng: `mrc.data.assert_gradeable`, và số đếm ở màn hình "Dữ liệu".
- [ ] **Một phần đáng kể câu validation là impossible.** Tỉ lệ chính xác:
      dòng cuối `assets/provenance.md` (tính từ dữ liệu, không viết tay).
      Hệ quả: metric tổng **trộn hai kỹ năng** — tìm đúng span, và biết khi nào
      trả lời rỗng. Đây là lý do mọi bảng tách `answerable_only` /
      `impossible_only`.
- [ ] **`question_type` là heuristic tự gán**, KHÔNG phải nhãn có sẵn của
      ViQuAD. Test chỉ pin *hành vi của heuristic*, không chứng minh nó đúng.
      Thiếu câu này người chấm sẽ hiểu nhầm là dataset có nhãn reasoning-scope.
- [ ] **Nếu dùng subset: ghi `n` và sai số.** `n` có sẵn ở mọi dòng của
      `assets/table_results.md`.

### Bốn bất biến — nên nêu như cam kết kỹ thuật (§10.1)

- [ ] **Extractive:** mọi `predict()` trả về substring của context, có test cho
      *mọi* predictor.
- [ ] **Không leakage:** `assert_no_leakage` chạy trong mọi đường split và làm
      fail run.
- [ ] **Truy vết được:** mỗi số ⟶ `results/*.json` ⟶ commit hash. Phụ lục là
      `assets/provenance.md`.
- [ ] **Có `n` kèm số:** nhóm có `count < 30` tự gắn cờ `unreliable`; trong hình
      chúng hiện dấu `*`. Nói rõ trong chú thích hình.

---

## Cấu trúc đề xuất

### 1. Đặt vấn đề
Bài toán MRC trích xuất: cho `context` + `question`, trả về span nằm trong chính
context. Vì sao tiếng Việt khó hơn: không có nhãn reasoning-scope, tokenizer
tiếng Việt, dấu thanh.

### 2. Dữ liệu — UIT-ViQuAD 2.0
Ba split, thống kê **đo trực tiếp** (không chép từ tài liệu gốc).
→ Số: màn hình "Dữ liệu" của demo, hoặc `mrc.data.compute_stats`.
→ Cảnh báo: tài liệu ViQuAD ghi `num_contexts = 138` nhưng 138 là số **article**;
   nêu chuyện này như một ví dụ về việc tự đếm.
→ **Giới hạn #1** (test split là blind) thuộc về mục này.

### 3. Phương pháp
- Baseline TF-IDF (cosine sentence retrieval)
- mBERT + QA head, fine-tune
- ViSoBERT + QA head, fine-tune
- XLM-R squad2, zero-shot

Ba quyết định đặc thù tiếng Việt (mỗi quyết định đều được pin bằng test) —
xem `docs/SOLUTION.md`.

### 4. Thiết lập đánh giá
EM và F1; vì sao tách `answerable_only` / `impossible_only`;
`null_delta` và ngưỡng từ chối; cách đo latency.
→ **Giới hạn #2** (tỉ lệ impossible) thuộc về mục này.

### 5. Giả thuyết ghi trước
Chép nguyên `results/hypotheses.md`, rồi đối chiếu bằng
`python scripts/check_hypotheses.py results`. Nêu cả giả thuyết **sai** — đó là
phần có giá trị nhất.

### 6. Kết quả
→ Bảng: `assets/table_results.md`
→ Hình: `assets/figures/model_comparison.png`,
  `assets/figures/em_answerable_vs_impossible.png`

Ba điều bảng nói ra mà con số tổng thì không (xem README gốc để lấy lập luận):
1. Baseline F1 cao nhưng EM gần 0 — nó trả về cả câu, gold là cụm vài từ; khoảng
   cách đó là bằng chứng trực quan rằng hai metric đo hai thứ khác nhau.
2. mBERT thắng XLM-R **không** vì tìm span giỏi hơn — trên câu answerable XLM-R
   zero-shot thực ra tốt hơn. mBERT thắng vì biết khi nào KHÔNG nên trả lời.
3. ViSoBERT thất bại, và vì sao đó vẫn là kết quả hợp lệ.

### 7. Phân tích lỗi
→ Hình: `assets/figures/f1_by_context_length.png`,
  `assets/figures/f1_by_question_type.png`
→ Ví dụ cụ thể: `sample_predictions` trong `results/eval_*.json`
→ **Giới hạn #3** (`question_type` là heuristic) thuộc về mục này — đặt ngay
  dưới hình, không giấu xuống phần "Hạn chế" cuối bài.
→ Nhóm `n < 30` có dấu `*`: giải thích dấu đó trong chú thích.

### 8. Huấn luyện
→ Hình: `assets/figures/training_curves.png`
→ Kết luận overfitting: trích **output của `detect_overfitting`**, không viết
  "chúng em quan sát thấy".
→ ViSoBERT epoch 1–2 có `val_EM == val_F1` ⇒ dấu hiệu suy sụp về luôn-trả-rỗng;
  epoch 3 mới thoát ra. Đây là một phát hiện, nên kể như một phát hiện.

### 9. Demo
Sáu màn hình, deep-link từng màn. Ảnh chụp màn hình. Nêu rằng demo đọc cùng
`results/*.json` với báo cáo, nên không thể lệch số.
Chạy bằng Docker: `docker compose up app`.

### 10. Hạn chế và hướng phát triển
Gom lại bốn giới hạn ở trên (đã nêu tại chỗ, ở đây chỉ tóm tắt), cộng:
- Chỉ đánh giá tiếng Việt, chưa thử cross-lingual
- Chưa thử model lớn hơn vì giới hạn phần cứng

### 11. Phụ lục
- [ ] `assets/provenance.md` — commit, thiết bị, torch, ngày, `n` cho từng model
- [ ] Hướng dẫn chạy (trích README gốc) — tiêu chí chấm #4
- [ ] Số lượng test và cách chạy
- [ ] Phân công nhóm

---

## Trước khi nộp

- [ ] Mọi số trong bài đều tìm được trong `assets/` — không có ngoại lệ
- [ ] `python scripts/make_report.py` chạy lại được, kết quả không đổi
- [ ] Bốn giới hạn bắt buộc đều có mặt, **đúng chỗ** chứ không dồn cuối bài
- [ ] Mỗi hình có chú thích nói rõ `n` và ý nghĩa dấu `*`
- [ ] `split = validation` xuất hiện rõ ràng cạnh bảng kết quả chính
- [ ] Độ dài 30–50 trang
