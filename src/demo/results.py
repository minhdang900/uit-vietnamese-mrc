"""Đọc mọi con số của demo từ ``results/`` và ``data/raw/``.

Bất biến đầu tiên của dự án: **không con số nào được viết tay**. Module này là
chỗ duy nhất đọc file, nên nếu một con số xuất hiện trên màn hình mà không đi qua
đây thì đó là số bịa.

Tất cả hàm nhận ``root`` tường minh (mặc định là gốc repo) để test trỏ vào thư
mục tạm, không đụng vào ``results/`` thật.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from demo.catalog import MODELS, Model
from demo.text import excerpt, length_bucket, syllable_count

__all__ = [
    "repo_root", "load_eval", "load_all_evals", "load_training_curves",
    "dataset_stats", "provenance", "sample_predictions", "passages",
    "Provenance", "Passage", "MissingResults", "best_by",
]

#: Tên split mà demo trình bày. Test split là blind set, không chấm được —
#: xem ``mrc.data.assert_gradeable``.
SPLIT = "validation"


class MissingResults(FileNotFoundError):
    """File kết quả chưa có. Ném ra kèm câu lệnh sinh ra nó.

    Ném lỗi ồn ào thay vì trả về dict rỗng: một màn hình kết quả toàn dấu gạch
    ngang trông giống "model kém" chứ không giống "chưa chạy đánh giá", và đó là
    đúng loại nhầm lẫn không được phép xảy ra khi đang chấm.
    """


def repo_root() -> Path:
    """Gốc repo, suy từ vị trí file này (``src/demo/results.py``)."""
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path, hint: str) -> dict:
    if not path.exists():
        raise MissingResults(f"Thiếu {path.name}. Sinh lại bằng:\n    {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def _cached_json(path_str: str, hint: str) -> dict:
    return _read_json(Path(path_str), hint)


def load_eval(model_id: str, root: Path | None = None) -> dict:
    """Nội dung ``results/eval_<model_id>_validation.json``."""
    root = root or repo_root()
    model = next((m for m in MODELS if m.id == model_id), None)
    if model is None:
        raise KeyError(f"Model không có trong danh mục: {model_id!r}")
    return _cached_json(
        str(root / "results" / model.eval_file),
        f"python scripts/run_eval.py --models {model_id} --full",
    )


def load_all_evals(root: Path | None = None) -> dict[str, dict]:
    """``{model_id: eval JSON}`` cho mọi model trong danh mục."""
    return {m.id: load_eval(m.id, root) for m in MODELS}


def load_training_curves(root: Path | None = None) -> dict[str, dict]:
    """``{model_id: training_curve JSON}`` cho các model có fine-tune.

    Chỉ mBERT và ViSoBERT có đường cong: XLM-R là zero-shot và baseline không
    huấn luyện, nên không có gì để vẽ.
    """
    root = root or repo_root()
    out: dict[str, dict] = {}
    for model_id in ("mbert", "visobert"):
        path = root / "results" / f"training_curve_{model_id}.json"
        if path.exists():
            out[model_id] = _cached_json(str(path), "")
    return out


@dataclass(frozen=True)
class Provenance:
    """Xuất xứ của những con số đang hiển thị.

    Hiện ở chân sidebar để người xem biết bảng kết quả sinh ra lúc nào, trên máy
    gì, ở commit nào — không có nó thì mọi màn hình đều là "một con số nào đó".
    """

    device: str
    torch: str
    platform: str
    split: str
    n: int
    commit: str
    timestamp: str
    dataset: str

    @property
    def date(self) -> str:
        """``2026-09-12T14:03:03+00:00`` → ``"12/09/2026"``."""
        head = self.timestamp[:10]
        try:
            year, month, day = head.split("-")
        except ValueError:
            return self.timestamp
        return f"{day}/{month}/{year}"


def provenance(model_id: str = "mbert", root: Path | None = None) -> Provenance:
    """Xuất xứ đọc từ chính file kết quả, không hard-code."""
    data = load_eval(model_id, root)
    env = data.get("env") or {}
    return Provenance(
        device=data.get("device", "?"),
        torch=env.get("torch", "?"),
        platform=env.get("platform", "?"),
        split=data.get("split", SPLIT),
        n=int(data.get("n", 0)),
        commit=data.get("commit", "?"),
        timestamp=data.get("timestamp", ""),
        dataset=data.get("dataset", "UIT-ViQuAD 2.0"),
    )


def sample_predictions(model_id: str = "mbert", root: Path | None = None) -> list[dict]:
    """Dự đoán mẫu kèm gold và điểm từng câu — nguyên văn từ file đánh giá."""
    return list(load_eval(model_id, root).get("sample_predictions", []))


@dataclass(frozen=True)
class Passage:
    """Một đoạn văn dùng làm ví dụ trên màn hình Hỏi đáp.

    Attributes:
        key: khoá ổn định (qid, hoặc ``"default"`` cho đoạn mở màn).
        title: tiêu đề article trong ViQuAD, dùng làm nhãn chip.
        context: đoạn văn.
        question: câu hỏi đi kèm, để chọn chip là có ngay một câu hỏi chạy được.
        gold: đáp án vàng. Rỗng nghĩa là câu impossible.
        impossible: cờ gốc của dataset.
    """

    key: str
    title: str
    context: str
    question: str
    gold: tuple[str, ...] = ()
    impossible: bool = False

    @property
    def syllables(self) -> int:
        return syllable_count(self.context)

    @property
    def bucket(self) -> str:
        return length_bucket(self.syllables)

    @property
    def excerpt(self) -> str:
        return excerpt(self.context)

    @property
    def chip(self) -> str:
        """Nhãn chip; câu impossible được đánh dấu vì đó là cả bài học."""
        return f"{self.title} · impossible" if self.impossible else self.title


def _titles_by_qid(root: Path) -> dict[str, str]:
    """``{qid: title}`` đọc từ split gốc.

    Dự đoán mẫu trong file đánh giá không mang theo ``title``; nối lại từ dataset
    cho nhãn chip là tên article thật chứ không phải một chuỗi cắt từ context.
    Không có ``data/raw/`` thì trả về rỗng và nơi gọi tự lùi về câu hỏi.
    """
    path = root / "data" / "raw" / f"viquad2_{SPLIT}.json"
    if not path.exists():
        return {}
    from mrc.data import load_squad_file

    return {ex.qid: ex.title for ex in load_squad_file(path)}


def passages(model_id: str = "mbert", root: Path | None = None,
             limit: int = 6) -> list[Passage]:
    """Các đoạn văn cho chip "Đoạn văn", lấy từ dự đoán mẫu thật.

    Đoạn mở màn (``catalog.DEFAULT_CONTEXT``) luôn đứng đầu; phần còn lại là các
    context khác nhau trong ``sample_predictions``, giữ nguyên thứ tự trong file
    để mỗi lần chạy đều như nhau.

    Khử trùng lặp theo NHÃN CHIP chứ không theo context: split validation có ba
    đoạn cùng thuộc article "Paris", và ba cái chip đều ghi "Paris" thì người xem
    không chọn được cái nào. Một đoạn answerable và một đoạn impossible cùng
    article vẫn được giữ cả hai — nhãn của chúng khác nhau, và cặp đó là minh hoạ
    trực tiếp nhất cho thanh ngưỡng từ chối.

    Ít nhất một câu impossible luôn có mặt: nếu chip nào cũng trả lời được thì
    ngưỡng từ chối chẳng có gì để minh hoạ.
    """
    from demo.catalog import DEFAULT_CONTEXT, DEFAULT_QUESTION

    root = root or repo_root()
    titles = _titles_by_qid(root)

    def make(sample: dict) -> Passage:
        qid = sample["qid"]
        return Passage(
            key=qid,
            title=titles.get(qid) or excerpt(sample["question"], 28),
            context=sample["context"],
            question=sample["question"],
            gold=tuple(sample.get("gold") or ()),
            impossible=bool(sample.get("is_impossible")),
        )

    first = Passage("default", "Hà Nội", DEFAULT_CONTEXT, DEFAULT_QUESTION, ("1999",))
    seen_chips = {first.chip}
    seen_contexts = {first.context}

    pool: list[Passage] = []
    for sample in sample_predictions(model_id, root):
        passage = make(sample)
        if passage.chip in seen_chips or passage.context in seen_contexts:
            continue
        seen_chips.add(passage.chip)
        seen_contexts.add(passage.context)
        pool.append(passage)

    chosen = pool[: max(0, limit - 1)]
    if chosen and not any(p.impossible for p in chosen):
        spare = next((p for p in pool[len(chosen):] if p.impossible), None)
        if spare is not None:
            chosen[-1] = spare

    return [first, *chosen]


def dataset_stats(root: Path | None = None) -> dict[str, dict]:
    """Thống kê ba split, ĐO trực tiếp từ file đã tải.

    Không chép từ tài liệu ViQuAD: dự án tiền nhiệm ghi ``num_contexts = 138``
    trong khi 138 là số *article*, và cách duy nhất bắt được sai lệch kiểu đó là
    tự đếm. Thiếu ``data/raw/`` thì trả về rỗng — màn hình Dữ liệu sẽ nói rõ cần
    chạy ``scripts/fetch_data.py``.
    """
    root = root or repo_root()
    raw = root / "data" / "raw"
    if not raw.exists():
        return {}

    from mrc.data import compute_stats, load_squad_file

    out: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        path = raw / f"viquad2_{split}.json"
        if path.exists():
            out[split] = compute_stats(load_squad_file(path))
    return out


def best_by(metric: str, evals: dict[str, dict]) -> tuple[Model, float]:
    """Model có ``overall[metric]`` cao nhất, kèm giá trị.

    Dùng cho các thẻ "tốt nhất": thứ hạng được TÍNH từ dữ liệu, nên nếu ai đó
    huấn luyện lại và model khác thắng thì màn hình tự nói đúng.
    """
    ranked = sorted(
        ((m, evals[m.id]["overall"][metric]) for m in MODELS if m.id in evals),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not ranked:
        raise MissingResults("Chưa có kết quả đánh giá nào.")
    return ranked[0]
