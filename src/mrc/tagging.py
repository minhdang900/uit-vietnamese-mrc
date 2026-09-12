"""Gán nhãn phục vụ phân tích lỗi: loại câu hỏi và bucket độ dài context.

⚠️ ``question_type`` là HEURISTIC tự gán, KHÔNG phải nhãn có sẵn của ViQuAD.
Đề tài T11 yêu cầu "phân tích câu hỏi single-sentence so với multi-sentence
reasoning" nhưng dataset không cung cấp nhãn đó, nên ta phải tự suy ra. Báo cáo
BẮT BUỘC nói rõ điều này — nếu không, người đọc sẽ hiểu sai rằng đây là nhãn vàng.
"""

from __future__ import annotations

from collections.abc import Iterable

from mrc.normalize import tokenize

__all__ = ["length_bucket", "question_type", "tag_examples", "BUCKETS", "QUESTION_TYPES"]

BUCKETS = ("<100", "100-200", "200-300", "300+")
QUESTION_TYPES = ("single-sentence", "multi-sentence")

#: Ngưỡng tỉ lệ token câu hỏi được bao phủ bởi MỘT câu đơn lẻ để coi là
#: single-sentence. 0.6 chọn theo quan sát; là hyperparameter của heuristic.
_SINGLE_SENTENCE_COVERAGE = 0.6


def length_bucket(n_words: int) -> str:
    """Xếp độ dài context (tính theo TỪ) vào một trong bốn bucket.

    Biên trên của mỗi bucket là ĐÓNG: 200 từ thuộc ``"100-200"``, 201 thuộc
    ``"200-300"``. Pin biên bằng test để bảng breakdown không mơ hồ.
    """
    if n_words < 0:
        raise ValueError(f"Độ dài không thể âm: {n_words}")
    if n_words < 100:
        return "<100"
    if n_words <= 200:
        return "100-200"
    if n_words <= 300:
        return "200-300"
    return "300+"


def _split_into_sentences(context: str) -> list[str]:
    """Tách câu thô cho mục đích đếm bằng chứng (evidence)."""
    from mrc.baseline_tfidf import split_sentences

    return split_sentences(context)


def question_type(question: str, context: str) -> str:
    """Suy ra câu hỏi cần bằng chứng từ MỘT câu hay NHIỀU câu của context.

    Heuristic: nếu tồn tại một câu đơn lẻ trong context bao phủ được ít nhất
    ``_SINGLE_SENTENCE_COVERAGE`` phần token nội dung của câu hỏi, coi là
    ``single-sentence``; ngược lại ``multi-sentence``.

    Đây là xấp xỉ, không phải chân lý. Nó không phân biệt được suy luận thật sự
    nhiều bước với việc chỉ dùng từ đồng nghĩa.
    """
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return "single-sentence"

    sentences = _split_into_sentences(context)
    if len(sentences) <= 1:
        return "single-sentence"

    best_coverage = max(
        (len(q_tokens & set(tokenize(s))) / len(q_tokens) for s in sentences),
        default=0.0,
    )
    return "single-sentence" if best_coverage >= _SINGLE_SENTENCE_COVERAGE else "multi-sentence"


def tag_examples(examples: Iterable) -> dict[str, dict]:
    """``{qid: {question_type, length_bucket, context_words}}``."""
    out: dict[str, dict] = {}
    for ex in examples:
        n_words = len(ex.context.split())
        out[ex.qid] = {
            "question_type": question_type(ex.question, ex.context),
            "length_bucket": length_bucket(n_words),
            "context_words": n_words,
        }
    return out
