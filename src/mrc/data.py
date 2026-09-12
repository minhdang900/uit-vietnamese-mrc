"""Tầng dữ liệu: parse UIT-ViQuAD 2.0 (SQuAD-2.0 JSON) → ``Example``.

Hai điểm thiết kế quan trọng:

1. **Đơn vị dedup và split là CONTEXT (paragraph), không phải ARTICLE (title).**
   Dự án tiền nhiệm ghi ``train.num_contexts = 138`` trong khi dữ liệu thật có 138
   *article* chứa 4.101 *context* — trường đó thực chất đang đếm title. Vì
   chống leakage dựa vào việc một context chỉ thuộc một split, nhầm hai đơn vị
   này làm vô hiệu hoá chính cơ chế bảo vệ.

2. **``plausible_answers`` KHÔNG BAO GIỜ được dùng làm gold.** Với câu
   ``is_impossible=True``, SQuAD-2.0 cung cấp một đáp án "nghe hợp lý nhưng sai".
   Dùng nó làm gold sẽ biến câu impossible thành answerable và làm metric sai một
   cách âm thầm.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Example",
    "parse_squad",
    "load_squad_file",
    "deduplicate_contexts",
    "remove_contexts_present_in",
    "split_by_context",
    "assert_no_leakage",
    "assert_gradeable",
    "compute_stats",
    "references_from",
    "reproducible_subset",
]


@dataclass(frozen=True, eq=True)
class Example:
    """Một cặp (context, question) kèm đáp án vàng.

    Attributes:
        qid: id câu hỏi, khoá dùng xuyên suốt pipeline.
        question: câu hỏi.
        context: đoạn văn chứa (hoặc không chứa) đáp án.
        title: tiêu đề article — chỉ để truy vết, KHÔNG dùng làm đơn vị split.
        answers: danh sách đáp án vàng. RỖNG nghĩa là câu impossible.
        answer_start: offset ký tự của ``answers[0]`` trong ``context``; ``-1``
            nếu impossible.
        is_impossible: cờ gốc từ dataset.
    """

    qid: str
    question: str
    context: str
    title: str = ""
    answers: list[str] = field(default_factory=list)
    answer_start: int = -1
    is_impossible: bool = False

    def __post_init__(self) -> None:
        # Bất biến: cờ impossible và danh sách đáp án không được mâu thuẫn.
        if self.is_impossible and self.answers:
            raise ValueError(f"{self.qid}: is_impossible=True nhưng vẫn có answers")

    @property
    def is_gradeable(self) -> bool:
        """Câu này có chấm được EM/F1 hay không.

        Ba trường hợp, phân biệt bằng CẢ cờ ``is_impossible`` LẪN sự có mặt của
        gold — vì hai trường hợp sau đều có gold rỗng nhưng ý nghĩa trái ngược:

        =========================  ==========  =============================
        Tình huống                 Chấm được?  Đáp án đúng là gì
        =========================  ==========  =============================
        có gold                    có          chuỗi gold
        ``is_impossible=True``     có          chuỗi rỗng
        gold rỗng, cờ ``False``    **KHÔNG**   không biết (bị lược bỏ)
        =========================  ==========  =============================

        Trường hợp thứ ba là blind/held-out split. Chấm trên đó sẽ cho một model
        luôn trả về ``""`` số EM 100% hoàn toàn vô nghĩa.
        """
        return bool(self.answers) or self.is_impossible


def parse_squad(payload: dict) -> list[Example]:
    """Làm phẳng cấu trúc SQuAD-2.0 lồng nhau thành danh sách ``Example``.

    Thứ tự đầu ra ổn định (theo thứ tự xuất hiện trong file) để mọi thao tác phía
    sau tái lập được.
    """
    out: list[Example] = []
    for article in payload.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                ans = qa.get("answers") or {}
                texts = [t for t in ans.get("text", []) if t]
                starts = ans.get("answer_start", [])
                # Tôn trọng cờ của dataset. KHÔNG suy ra impossible từ việc gold
                # rỗng: blind split (test của ViQuAD) cũng có gold rỗng nhưng
                # is_impossible=False, và gộp hai trường hợp sẽ khiến model trả
                # về "" đạt EM 100%. Xem Example.is_gradeable.
                is_impossible = bool(qa.get("is_impossible", False))
                out.append(
                    Example(
                        qid=qa["id"],
                        question=qa["question"],
                        context=context,
                        title=title,
                        answers=[] if is_impossible else list(texts),
                        answer_start=int(starts[0]) if (texts and starts and not is_impossible) else -1,
                        is_impossible=is_impossible,
                    )
                )
    return out


def load_squad_file(path: str | Path) -> list[Example]:
    """Đọc file JSON SQuAD-2.0 từ đĩa."""
    return parse_squad(json.loads(Path(path).read_text(encoding="utf-8")))


def deduplicate_contexts(examples: Iterable[Example]) -> list[Example]:
    """Bỏ các câu hỏi trùng lặp hoàn toàn trên cùng một context.

    Context giống nhau xuất hiện dưới nhiều article là bình thường trong ViQuAD;
    hàm này KHÔNG bỏ câu hỏi của context đó — nó chỉ bỏ cặp
    ``(context, question)`` xuất hiện nhiều lần, giữ lần đầu.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[Example] = []
    for ex in examples:
        key = (ex.context, ex.question.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(ex)
    return kept


def remove_contexts_present_in(
    examples: Iterable[Example], other: Iterable[Example]
) -> list[Example]:
    """Bỏ khỏi ``examples`` mọi câu hỏi có context xuất hiện trong ``other``.

    Đây là bước dedup CHÉO SPLIT — cơ chế chống leakage khi hai split được cung
    cấp sẵn (như train/validation của ViQuAD) và có thể chia sẻ context.
    """
    forbidden = {ex.context for ex in other}
    return [ex for ex in examples if ex.context not in forbidden]


def split_by_context(
    examples: Sequence[Example], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Chia train/val theo CONTEXT — mọi câu hỏi của một context đi cùng nhau.

    Đây là ``GroupShuffleSplit`` với group = context. Chia theo câu hỏi thay vì
    theo context sẽ khiến model được "đọc" đoạn văn lúc train rồi bị hỏi về chính
    đoạn đó lúc test — group leakage, và kết quả cao giả tạo.
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac phải trong [0,1], nhận {val_frac}")

    # OrderedDict giữ thứ tự xuất hiện -> shuffle có seed là tái lập được.
    groups: OrderedDict[str, list[Example]] = OrderedDict()
    for ex in examples:
        groups.setdefault(ex.context, []).append(ex)

    contexts = list(groups)
    random.Random(seed).shuffle(contexts)

    n_val = int(len(contexts) * val_frac)
    val_contexts = set(contexts[:n_val])

    train = [ex for ctx in contexts if ctx not in val_contexts for ex in groups[ctx]]
    val = [ex for ctx in contexts if ctx in val_contexts for ex in groups[ctx]]
    return train, val


def assert_no_leakage(split_a: Iterable[Example], split_b: Iterable[Example]) -> None:
    """Raise nếu hai split chia sẻ bất kỳ context nào.

    Hàm này được gọi trong MỌI đường split và làm ``fail`` cả run khi vi phạm. Một
    assertion làm sập chương trình đáng tin hơn một đoạn văn trong báo cáo hứa
    rằng nhóm đã cẩn thận.
    """
    shared = {ex.context for ex in split_a} & {ex.context for ex in split_b}
    if shared:
        sample = next(iter(shared))
        raise AssertionError(
            f"Phát hiện data leakage: {len(shared)} context xuất hiện ở CẢ HAI split. "
            f"Ví dụ (80 ký tự đầu): {sample[:80]!r}"
        )


def assert_gradeable(examples: Sequence[Example]) -> None:
    """Raise nếu split chứa câu không chấm được (gold bị lược bỏ).

    Gọi hàm này TRƯỚC mọi lần đánh giá. Test split của ViQuAD 2.0 có toàn bộ
    7.301 câu với gold rỗng và ``is_impossible=False`` — đánh giá trên đó sẽ cho
    ra con số vô nghĩa thay vì báo lỗi.
    """
    ungradeable = [ex for ex in examples if not ex.is_gradeable]
    if ungradeable:
        raise ValueError(
            f"{len(ungradeable)}/{len(examples)} câu hỏi KHÔNG chấm được: gold rỗng "
            f"nhưng is_impossible=False (đáp án đã bị lược bỏ — đây là blind split). "
            f"Không thể tính EM/F1. Ví dụ qid: "
            f"{[ex.qid for ex in ungradeable[:5]]}"
        )


def compute_stats(examples: Sequence[Example]) -> dict:
    """Thống kê một split. Phân biệt rõ ``num_contexts`` và ``num_articles``."""
    n = len(examples)
    if n == 0:
        return {
            "num_questions": 0,
            "num_contexts": 0,
            "num_articles": 0,
            "num_impossible": 0,
            "num_with_gold": 0,
            "num_gradeable": 0,
            "impossible_pct": 0.0,
        }
    n_impossible = sum(ex.is_impossible for ex in examples)
    return {
        "num_questions": n,
        "num_contexts": len({ex.context for ex in examples}),
        "num_articles": len({ex.title for ex in examples}),
        "num_impossible": n_impossible,
        "num_with_gold": sum(bool(ex.answers) for ex in examples),
        "num_gradeable": sum(ex.is_gradeable for ex in examples),
        "impossible_pct": round(100.0 * n_impossible / n, 2),
    }


def references_from(examples: Iterable[Example]) -> dict[str, list[str]]:
    """``{qid: [đáp án vàng]}`` để đưa vào ``metrics.evaluate``."""
    return {ex.qid: list(ex.answers) for ex in examples}


def reproducible_subset(
    examples: Sequence[Example], n: int | None, seed: int = 42
) -> list[Example]:
    """Lấy mẫu con NGẪU NHIÊN và TÁI LẬP ĐƯỢC.

    Không lấy ``examples[:n]``: các câu đầu file thuộc vài article đầu tiên, nên
    mẫu đó thiên lệch theo chủ đề. Đo được trên mBERT: EM 42,00 trên 300 câu đầu
    so với EM 50,80 trên 300 câu ngẫu nhiên — chênh 8,8 điểm chỉ do cách lấy mẫu.

    Hàm này được dùng ở CẢ đường cong huấn luyện lẫn bảng kết quả cuối, nên hai
    nơi đó so sánh được với nhau.
    """
    if n is None or n >= len(examples):
        return list(examples)
    return random.Random(seed).sample(list(examples), n)
