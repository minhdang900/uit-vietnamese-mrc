"""Quyết định trong huấn luyện, tách hoàn toàn khỏi việc chạy torch.

Mọi hàm ở đây là hàm THUẦN: nhận số, trả về số. Không tải model, không cần GPU,
không đọc file. Nhờ vậy toàn bộ phần "ra quyết định" của quá trình huấn luyện —
tính lịch học, chọn epoch tốt nhất, chẩn đoán overfitting — được test trong vài
millisecond, thay vì phải chạy một epoch 35 phút mới biết đúng sai.

Đây là bài học rút ra từ lần viết trước: khi toàn bộ logic nằm trong ``main()``,
không có gì test được, và lỗi lấy mẫu thiên lệch trong đánh giá đã trôi qua không
bị phát hiện cho đến khi hai phép đo độc lập mâu thuẫn nhau.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "EpochRecord",
    "Schedule",
    "compute_schedule",
    "select_best_epoch",
    "detect_overfitting",
    "summarise_curve",
]

#: Ngưỡng chênh lệch |EM - F1| dưới mức này bị coi là dấu hiệu model suy sụp về
#: "luôn trả rỗng": khi đó mỗi câu chỉ có thể đúng-hoàn-toàn (impossible, đoán
#: rỗng) hoặc sai-hoàn-toàn (answerable, đoán rỗng), nên hai metric trùng nhau.
#: F1 cho điểm bán phần nên bình thường phải cao hơn EM vài điểm.
COLLAPSE_TOLERANCE = 0.5


@dataclass(frozen=True)
class EpochRecord:
    """Kết quả một epoch. ``val_em``/``val_f1`` tính bằng chính pipeline inference."""

    epoch: int
    train_loss: float
    val_em: float
    val_f1: float
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError(f"epoch phải >= 1, nhận {self.epoch}")

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 4),
            "val_em": round(self.val_em, 2),
            "val_f1": round(self.val_f1, 2),
            "seconds": round(self.seconds, 1),
        }


@dataclass(frozen=True)
class Schedule:
    """Lịch học đã tính sẵn, để optimizer và scheduler dùng chung một nguồn."""

    steps_per_epoch: int
    total_steps: int
    warmup_steps: int


def compute_schedule(
    n_features: int,
    batch_size: int,
    grad_accum: int = 1,
    epochs: int = 1,
    warmup_ratio: float = 0.1,
) -> Schedule:
    """Số bước optimizer mỗi epoch, tổng số bước, và số bước warmup.

    Một "bước" ở đây là một lần ``optimizer.step()``, tức sau khi đã gộp
    ``grad_accum`` batch. Scheduler phải đếm theo đơn vị này, không phải theo số
    batch — đếm nhầm khiến learning rate giảm về 0 quá sớm.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, nhận {batch_size}")
    if epochs <= 0:
        raise ValueError(f"epochs phải > 0, nhận {epochs}")
    if grad_accum <= 0:
        raise ValueError(f"grad_accum phải > 0, nhận {grad_accum}")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio phải trong [0,1], nhận {warmup_ratio}")

    n_batches = math.ceil(max(n_features, 1) / batch_size)
    # Luôn >= 1: dataset nhỏ hơn một batch vẫn phải chạy được một bước.
    steps_per_epoch = max(1, n_batches // grad_accum)
    total_steps = steps_per_epoch * epochs
    return Schedule(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=int(total_steps * warmup_ratio),
    )


def select_best_epoch(curve: list[EpochRecord]) -> int:
    """Epoch có ``val_f1`` cao nhất; hoà thì chọn epoch SỚM hơn.

    Chọn theo F1 chứ không theo loss: loss là đại lượng trên tập TRAIN, còn điều
    ta quan tâm là khả năng tổng quát. Hoà thì ưu tiên epoch sớm vì nó ít huấn
    luyện hơn mà cho kết quả tương đương.
    """
    if not curve:
        raise ValueError("Đường cong rỗng — không có epoch nào để chọn")
    best = max(curve, key=lambda r: (r.val_f1, -r.epoch))
    return best.epoch


def detect_overfitting(curve: list[EpochRecord]) -> dict:
    """Chẩn đoán đường cong huấn luyện, trả về kết luận có thể trích vào báo cáo.

    Ba tín hiệu:

    * ``overfitting`` — ``val_f1`` đạt đỉnh rồi GIẢM ở các epoch sau, trong khi
      ``train_loss`` vẫn giảm. Model học thuộc tập train thay vì học quy luật.
    * ``still_improving`` — ``val_f1`` vẫn tăng ở epoch cuối, tức còn THIẾU epoch.
    * ``degenerate_collapse`` — ``val_em`` ≈ ``val_f1`` ở mọi epoch. F1 cho điểm
      bán phần nên bình thường cao hơn EM; bằng nhau nghĩa là model chỉ sinh ra
      đúng-hoàn-toàn hoặc sai-hoàn-toàn, dấu hiệu nó luôn trả chuỗi rỗng.

    Biến nhận xét chủ quan thành hàm có test, để báo cáo trích kết luận thay vì
    viết "chúng em quan sát thấy".
    """
    if len(curve) < 2:
        return {
            "overfitting": False,
            "still_improving": False,
            "degenerate_collapse": _is_degenerate(curve),
            "best_epoch": curve[0].epoch if curve else None,
            "reason": "Cần ít nhất 2 epoch mới kết luận được xu hướng.",
        }

    best_epoch = select_best_epoch(curve)
    last = curve[-1]
    best = next(r for r in curve if r.epoch == best_epoch)

    overfitting = best_epoch < last.epoch and last.val_f1 < best.val_f1
    still_improving = last.epoch == best_epoch and last.val_f1 >= curve[-2].val_f1

    if overfitting:
        reason = (
            f"val_f1 đạt đỉnh {best.val_f1:.2f} ở epoch {best_epoch} rồi giảm còn "
            f"{last.val_f1:.2f} ở epoch {last.epoch}, trong khi train_loss vẫn giảm "
            f"({curve[0].train_loss:.4f} -> {last.train_loss:.4f}). Dừng ở epoch {best_epoch}."
        )
    elif still_improving:
        reason = (
            f"val_f1 vẫn tăng tới epoch cuối ({last.val_f1:.2f}) — model còn THIẾU "
            f"epoch, chưa overfit. Cân nhắc huấn luyện thêm."
        )
    else:
        reason = "val_f1 dao động, không thấy xu hướng overfit rõ ràng."

    return {
        "overfitting": overfitting,
        "still_improving": still_improving,
        "degenerate_collapse": _is_degenerate(curve),
        "best_epoch": best_epoch,
        "reason": reason,
    }


def _is_degenerate(curve: list[EpochRecord]) -> bool:
    """True nếu EM ≈ F1 ở MỌI epoch — dấu hiệu model luôn trả chuỗi rỗng."""
    if not curve:
        return False
    return all(abs(r.val_em - r.val_f1) <= COLLAPSE_TOLERANCE for r in curve)


def summarise_curve(curve: list[EpochRecord], config: dict | None = None) -> dict:
    """Gói đường cong + chẩn đoán thành dict ghi thẳng ra JSON."""
    return {
        "config": config or {},
        "curve": [r.as_dict() for r in curve],
        "best_epoch": select_best_epoch(curve) if curve else None,
        "diagnosis": detect_overfitting(curve),
    }


class QADataset:
    """Bọc dict feature thành ``torch.utils.data.Dataset``.

    Không dùng ``datasets`` của HuggingFace để tránh thêm một phụ thuộc nặng cho
    một việc chỉ vài dòng.
    """

    REQUIRED = ("input_ids", "attention_mask", "start_positions", "end_positions")

    def __init__(self, features: dict) -> None:
        missing = [k for k in self.REQUIRED if k not in features]
        if missing:
            raise ValueError(f"Thiếu trường feature: {missing}")

        lengths = {k: len(features[k]) for k in self.REQUIRED}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Các trường feature không khớp độ dài: {lengths}")

        self._n = next(iter(lengths.values()))
        if self._n == 0:
            raise ValueError("Feature rỗng — không có gì để huấn luyện")
        self._f = features

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        import torch

        # long: nhãn vị trí là CHỈ SỐ token, cross-entropy đòi kiểu nguyên.
        return {k: torch.tensor(self._f[k][i], dtype=torch.long) for k in self.REQUIRED}


def evaluate_checkpoint(
    examples,
    predictor,
    limit: int | None = 300,
    seed: int = 42,
) -> dict:
    """Chấm một checkpoint bằng CHÍNH pipeline inference dùng cho kết quả cuối.

    ``predictor`` được TIÊM VÀO thay vì tạo bên trong, nên hàm này test được bằng
    stub mà không cần tải model.

    Mẫu con lấy qua :func:`mrc.data.reproducible_subset` — cùng hàm và cùng seed
    với ``scripts/run_eval.py``, nên số trên đường cong huấn luyện SO SÁNH ĐƯỢC
    với bảng kết quả cuối.

    Bản trước lấy ``examples[:limit]`` tức n câu ĐẦU file. Các câu đầu thuộc vài
    article đầu tiên nên mẫu thiên lệch theo chủ đề: cùng một checkpoint mBERT cho
    EM 42,00 trên mẫu đó nhưng 50,80 trên mẫu ngẫu nhiên cùng cỡ.
    """
    from mrc.data import references_from, reproducible_subset
    from mrc.metrics import evaluate as evaluate_metrics

    subset = reproducible_subset(examples, limit, seed=seed)
    predictions = {ex.qid: predictor.predict(ex.context, ex.question) for ex in subset}
    scores = evaluate_metrics(predictions, references_from(subset))
    return {
        "em": scores["EM"],
        "f1": scores["F1"],
        "n": scores["count"],
        "qids": [ex.qid for ex in subset],
    }
