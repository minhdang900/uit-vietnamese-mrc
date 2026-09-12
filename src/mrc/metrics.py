"""Exact Match + token-level F1 theo quy ước SQuAD-2.0, cho tiếng Việt.

Quy ước impossible (SQuAD-2.0): câu không có đáp án được biểu diễn bằng danh sách
gold RỖNG. Khi đó chỉ dự đoán rỗng mới được điểm; mọi dự đoán khác rỗng đều là 0.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from mrc.normalize import normalize_answer, tokenize

__all__ = [
    "exact_match",
    "token_f1",
    "metric_max_over_ground_truths",
    "evaluate",
]

Metric = Callable[[str, str], float]


def exact_match(prediction: str | None, ground_truth: str | None) -> float:
    """1.0 nếu hai chuỗi bằng nhau sau chuẩn hoá, ngược lại 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str | None, ground_truth: str | None) -> float:
    """F1 trên bag-of-tokens (âm tiết) giữa dự đoán và đáp án vàng.

    Dùng ``Counter`` chứ không phải ``set`` để token lặp được đếm đúng số lần.
    Trường hợp biên theo SQuAD: nếu một trong hai bên rỗng thì F1 = 1.0 khi CẢ HAI
    rỗng, ngược lại 0.0 — không có điểm bán phần.
    """
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens or not gold_tokens:
        return float(not pred_tokens and not gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric: Metric,
    prediction: str | None,
    ground_truths: Sequence[str],
) -> float:
    """Điểm cao nhất của ``metric`` trên tập các đáp án vàng.

    ViQuAD có thể cung cấp nhiều đáp án đúng cho một câu hỏi; theo quy ước SQuAD
    ta lấy đáp án nào cho điểm cao nhất.

    Danh sách rỗng nghĩa là câu impossible: so dự đoán với chuỗi rỗng.
    """
    if not ground_truths:
        return metric(prediction, "")
    return max(metric(prediction, gt) for gt in ground_truths)


def evaluate(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> dict:
    """Tổng hợp EM/F1 trên toàn bộ tập câu hỏi.

    Args:
        predictions: ``{qid: chuỗi dự đoán}``. Phải chứa MỌI qid có trong
            ``references`` — thiếu một cái là ``KeyError``.
        references: ``{qid: [các đáp án vàng]}``. Danh sách rỗng = impossible.

    Raises:
        KeyError: nếu thiếu dự đoán cho một qid có trong ``references``. Fail to
            ồn là có chủ đích: âm thầm bỏ qua câu thiếu sẽ khiến EM được tính
            trên tập con trong khi báo cáo ghi ``n`` đầy đủ.

    Returns:
        dict với ``EM``, ``F1`` (PHẦN TRĂM 0–100), ``count``, các chỉ số tách
        riêng cho nhóm answerable / impossible, và ``per_item``.
    """
    if not references:
        return {
            "EM": 0.0,
            "F1": 0.0,
            "count": 0,
            "n_impossible": 0,
            "n_answerable": 0,
            "EM_impossible": 0.0,
            "EM_answerable": 0.0,
            "F1_answerable": 0.0,
            "per_item": {},
        }

    missing = set(references) - set(predictions)
    if missing:
        raise KeyError(
            f"Thiếu dự đoán cho {len(missing)} câu hỏi, ví dụ: {sorted(missing)[:5]}. "
            "Mọi qid trong references phải có dự đoán — nếu không, EM/F1 sẽ được "
            "tính trên tập con và con số báo cáo sẽ sai."
        )

    per_item: dict[str, dict[str, float]] = {}
    for qid, golds in references.items():
        pred = predictions[qid]
        per_item[qid] = {
            "em": metric_max_over_ground_truths(exact_match, pred, golds),
            "f1": metric_max_over_ground_truths(token_f1, pred, golds),
            "is_impossible": not golds,
        }

    def _mean(values: list[float]) -> float:
        return 100.0 * sum(values) / len(values) if values else 0.0

    impossible = [v for v in per_item.values() if v["is_impossible"]]
    answerable = [v for v in per_item.values() if not v["is_impossible"]]

    return {
        "EM": _mean([v["em"] for v in per_item.values()]),
        "F1": _mean([v["f1"] for v in per_item.values()]),
        "count": len(per_item),
        "n_impossible": len(impossible),
        "n_answerable": len(answerable),
        "EM_impossible": _mean([v["em"] for v in impossible]),
        "EM_answerable": _mean([v["em"] for v in answerable]),
        "F1_answerable": _mean([v["f1"] for v in answerable]),
        "per_item": per_item,
    }
