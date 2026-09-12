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
