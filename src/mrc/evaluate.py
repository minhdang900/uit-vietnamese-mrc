"""Harness đánh giá: chạy một ``Predictor`` trên một split và sinh kết quả có provenance.

Thiết kế của module này là phản ứng trực tiếp với thất bại của phiên bản trước dự
án (xem ``AUDIT.md`` của repo cũ), nơi báo cáo trình bày những con số chưa từng
được sinh ra. Bốn cơ chế phòng vệ được cài vào chính cấu trúc dữ liệu kết quả:

1. **Provenance bắt buộc.** Mọi kết quả mang ``commit``, ``timestamp``, ``device``,
   ``split``, ``n``. Một con số không truy vết được về commit là một con số không
   dùng được.
2. **Từ chối split không chấm được.** ``assert_gradeable`` chạy trước khi chấm, nên
   không thể vô tình báo cáo EM trên blind test set.
3. **Nhóm nhỏ tự gắn cờ.** Nhóm có ``count < 30`` nhận ``unreliable=True``.
4. **Tổng không khớp phải được giải thích.** Breakdown theo loại câu hỏi loại câu
   impossible, nên tổng nhỏ hơn ``n``; harness phát ra ``_note`` nói rõ lý do.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from mrc.data import Example, assert_gradeable, references_from
from mrc.metrics import evaluate as evaluate_metrics
from mrc.tagging import tag_examples

__all__ = ["run_evaluation", "breakdown", "MIN_RELIABLE_GROUP"]

#: Dưới ngưỡng này, trung bình của nhóm quá nhiễu để kết luận. Với n=3, mỗi câu
#: đúng/sai làm điểm nhảy 33 điểm — con số như vậy không được trình bày như kết quả.
MIN_RELIABLE_GROUP = 30


def _git_commit() -> str:
    """Hash commit hiện tại, để mọi con số truy vết được về một trạng thái code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def breakdown(
    per_item: Mapping[str, Mapping[str, float]],
    tags: Mapping[str, str],
) -> dict:
    """Gộp điểm theo nhóm, kèm ``count`` và cờ ``unreliable``.

    Args:
        per_item: ``{qid: {"em": .., "f1": ..}}``.
        tags: ``{qid: tên nhóm}``. qid không có tag sẽ bị bỏ qua.

    Returns:
        ``{tên nhóm: {"EM": %, "F1": %, "count": n, "unreliable": bool}}``.
    """
    groups: dict[str, list[Mapping[str, float]]] = {}
    for qid, scores in per_item.items():
        g = tags.get(qid)
        if g is None:
            continue
        groups.setdefault(g, []).append(scores)

    out: dict = {}
    for g, items in sorted(groups.items()):
        n = len(items)
        out[g] = {
            "EM": round(100.0 * sum(i["em"] for i in items) / n, 4),
            "F1": round(100.0 * sum(i["f1"] for i in items) / n, 4),
            "count": n,
            # Cờ này tồn tại để bảng trong báo cáo không thể trình bày nhóm n=3
            # như một kết quả mà không bị chú ý.
            "unreliable": n < MIN_RELIABLE_GROUP,
        }
    return out


def run_evaluation(
    predictor,
    examples: Sequence[Example],
    split: str = "validation",
    dataset: str = "UIT-ViQuAD 2.0",
    n_samples: int = 10,
) -> dict:
    """Chạy ``predictor`` trên ``examples`` và trả về kết quả đầy đủ provenance.

    Raises:
        ValueError: nếu split chứa câu không chấm được (gold bị lược bỏ). Đây là
            cổng chặn việc báo cáo EM/F1 trên blind test set.
    """
    # Cổng #1: không chấm cái không chấm được.
    assert_gradeable(examples)

    from mrc.device import device_info

    predictions: dict[str, str] = {}
    latencies: list[float] = []
    for ex in examples:
        if hasattr(predictor, "predict_timed"):
            pred, ms = predictor.predict_timed(ex.context, ex.question)
            latencies.append(ms)
        else:
            pred = predictor.predict(ex.context, ex.question)
        predictions[ex.qid] = pred

    scores = evaluate_metrics(predictions, references_from(examples))
    per_item = scores.pop("per_item")

    tags = tag_examples(examples)
    length_tags = {qid: t["length_bucket"] for qid, t in tags.items()}
    # Câu impossible không có "phạm vi suy luận" để phân loại — loại khỏi
    # breakdown theo loại câu hỏi, và NÓI RÕ điều đó.
    answerable_qids = {ex.qid for ex in examples if not ex.is_impossible}
    qtype_tags = {
        qid: t["question_type"] for qid, t in tags.items() if qid in answerable_qids
    }

    by_qtype = breakdown(per_item, qtype_tags)
    n_impossible = scores["n_impossible"]
    by_qtype["_note"] = (
        f"Tổng của bảng này là {sum(v['count'] for v in by_qtype.values() if isinstance(v, dict))} "
        f"chứ không phải n={len(examples)}, vì {n_impossible} câu impossible không được "
        f"gán question_type (chúng không có phạm vi suy luận để phân loại). "
        f"question_type là HEURISTIC tự gán, không phải nhãn có sẵn của ViQuAD."
    )

    samples = [
        {
            "qid": ex.qid,
            "question": ex.question,
            "gold": list(ex.answers),
            "prediction": predictions[ex.qid],
            "context": ex.context,
            "em": per_item[ex.qid]["em"],
            "f1": round(per_item[ex.qid]["f1"], 4),
            "is_impossible": ex.is_impossible,
        }
        for ex in examples[:n_samples]
    ]

    info = device_info()
    return {
        # ── provenance: mọi con số truy vết được ──
        "model": getattr(predictor, "name", type(predictor).__name__),
        "dataset": dataset,
        "split": split,
        "n": len(examples),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "device": info["device"],
        "env": info,
        # ── kết quả ──
        "overall": {
            "EM": round(scores["EM"], 4),
            "F1": round(scores["F1"], 4),
            "count": scores["count"],
        },
        "answerable_only": {
            "EM": round(scores["EM_answerable"], 4),
            "F1": round(scores["F1_answerable"], 4),
            "count": scores["n_answerable"],
        },
        "impossible_only": {
            "EM": round(scores["EM_impossible"], 4),
            "count": scores["n_impossible"],
        },
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "by_context_length": breakdown(per_item, length_tags),
        "by_question_type": by_qtype,
        "sample_predictions": samples,
    }
