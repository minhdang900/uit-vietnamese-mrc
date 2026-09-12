"""Baseline: truy hồi câu bằng TF-IDF + cosine similarity.

Đề tài T11 yêu cầu "TF-IDF retrieval baseline". Baseline định nghĩa SÀN: nếu một
model phức tạp không vượt được nó, model đó vô nghĩa. Nó cũng trả lời một câu hỏi
cụ thể — *bài toán này giải được bằng so khớp từ khoá thuần không?*

Cơ chế quan trọng cần hiểu để đọc kết quả: model trả về CẢ MỘT CÂU, trong khi đáp
án vàng thường là cụm vài từ. Vì vậy nó có overlap token (F1 trung bình) nhưng gần
như không bao giờ trùng khít (EM ~ 0). Khoảng cách EM–F1 đó là bằng chứng trực
quan rằng hai metric đo hai thứ khác nhau.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mrc.predictor import TimedPredictorMixin

__all__ = ["TfidfRetriever", "split_sentences"]

#: Tách câu tại . ? ! khi KHÔNG nằm giữa hai chữ số (tránh cắt "8.5 triệu").
_SENTENCE_END = re.compile(r"(?<!\d)[.!?]+(?:\s+|$)")


def split_sentences(text: str) -> list[str]:
    """Tách đoạn văn thành danh sách câu, giữ nguyên dấu kết thúc câu.

    Cố tình giữ dấu chấm để chuỗi trả về vẫn là **substring của context gốc** —
    bất biến extractive.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text.strip()]


class TfidfRetriever(TimedPredictorMixin):
    """Chọn câu trong context có cosine similarity TF-IDF cao nhất với câu hỏi."""

    name = "TF-IDF Baseline"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        #: ngram (1,2) để bắt được cụm hai âm tiết như "thủ đô", "Hà Nội" —
        #: quan trọng với tiếng Việt, nơi từ thường gồm nhiều âm tiết.
        self._ngram_range = ngram_range

    def predict(self, context: str, question: str) -> str:
        sentences = split_sentences(context)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        try:
            # fit trên chính các câu của context + câu hỏi: đây là truy hồi
            # trong-tài-liệu, không dùng corpus ngoài nên không có leakage.
            vec = TfidfVectorizer(ngram_range=self._ngram_range, lowercase=True)
            matrix = vec.fit_transform(sentences + [question])
        except ValueError:
            # vocabulary rỗng (context chỉ có stopword/dấu câu)
            return sentences[0]

        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        return sentences[int(sims.argmax())]
