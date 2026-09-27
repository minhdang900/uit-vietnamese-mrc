"""Chọn ngưỡng null ``τ`` OFFLINE, từ bản ghi từng cửa sổ — không chạy lại model.

Span tốt nhất và điểm null của mỗi cửa sổ KHÔNG phụ thuộc ``τ``: ``τ`` chỉ can
thiệp ở bước so sánh cuối cùng trong :meth:`mrc.transformer_qa.TransformerQA.predict_detailed`.
Vì vậy lưu ``(best_score, null_score, start_char, end_char, text)`` của mọi cửa
sổ là đủ để tái hiện CHÍNH XÁC quyết định online ở bất kỳ ``τ`` nào. Ba điều kiện
để "chính xác" đúng nghĩa đen, mỗi điều có test riêng:

(a) Cửa sổ giữ nguyên THỨ TỰ gốc và phép so sánh vẫn là ``>`` nghiêm ngặt, nên khi
    hai cửa sổ bằng điểm thì cửa sổ ĐẦU thắng — y như vòng lặp online.
(b) Cửa sổ có ``null_score`` là ``None`` (không có token ``[CLS]``) luôn qua cửa
    ngưỡng, ở mọi ``τ``.
(c) Cửa sổ mà online bỏ qua (``score_spans`` trả ``None`` hoặc không có ứng viên)
    được ghi ``{"skipped": true}`` và replay cũng bỏ qua.

Float đi qua JSON bằng ``repr`` nên round-trip không mất bit nào.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from mrc.metrics import exact_match, metric_max_over_ground_truths, token_f1

__all__ = [
    "replay",
    "sweep",
    "select_tau",
    "select_epoch_and_tau",
    "tau_grid",
    "DEFAULT_TAU_GRID",
]


def tau_grid(lo: float = -5.0, hi: float = 5.0, step: float = 0.25) -> list[float]:
    """Lưới ``τ`` đều, tính bằng số nguyên bước để không tích luỹ sai số float."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


#: Lưới đăng ký trước cho P3: [−5, 5] bước 0,25 (41 giá trị).
DEFAULT_TAU_GRID = tau_grid()


def replay(windows: Sequence[Mapping], tau: float) -> tuple[int, int] | None:
    """Tái hiện quyết định online ở ngưỡng ``tau``.

    Returns:
        ``(start_char, end_char)`` của span thắng cuộc, hoặc ``None`` nếu mọi cửa
        sổ đều nói "không có đáp án".
    """
    i = _replay_index(windows, tau)
    return None if i is None else (windows[i]["start_char"], windows[i]["end_char"])


def _replay_index(windows: Sequence[Mapping], tau: float) -> int | None:
    """Chỉ số cửa sổ thắng cuộc, hoặc ``None``. Lõi dùng chung của replay và sweep."""
    best_score, best = float("-inf"), None
    for i, w in enumerate(windows):
        if w.get("skipped"):
            continue  # (c)
        null_score = w.get("null_score")
        if null_score is not None and w["best_score"] <= null_score + tau:
            continue  # (b): null_score None thì không bao giờ bị loại
        if w["best_score"] > best_score:  # (a): ">" nghiêm ngặt, cửa sổ đầu thắng khi hoà
            best_score, best = w["best_score"], i
    return best


def _scores_for(record: Mapping) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """(em, f1) của từng cửa sổ và của đáp án rỗng — tính MỘT lần cho mọi ``τ``."""
    gold = list(record.get("gold") or [])  # rỗng = impossible, như metrics.evaluate

    def score(text: str) -> tuple[float, float]:
        return (metric_max_over_ground_truths(exact_match, text, gold),
                metric_max_over_ground_truths(token_f1, text, gold))

    per_window = [score(w.get("text", "")) if not w.get("skipped") else (0.0, 0.0)
                  for w in record["windows"]]
    return per_window, score("")


def sweep(records: Iterable[Mapping], taus: Sequence[float] = DEFAULT_TAU_GRID) -> list[dict]:
    """EM/F1/tỉ lệ rỗng (đều theo %) tại mỗi ``τ``, trên các bản ghi có ``windows``."""
    prepared = []
    for r in records:
        if "windows" not in r:
            raise ValueError(f"{r.get('qid')}: bản ghi không có 'windows' — không replay được")
        prepared.append((r["windows"], *_scores_for(r)))
    if not prepared:
        raise ValueError("Không có bản ghi nào để quét τ")

    n = len(prepared)
    out = []
    for tau in taus:
        em = f1 = empty = 0.0
        for windows, per_window, empty_score in prepared:
            i = _replay_index(windows, tau)
            if i is None or not windows[i].get("text"):
                e, f = empty_score
                empty += 1
            else:
                e, f = per_window[i]
            em += e
            f1 += f
        out.append({"tau": tau, "EM": 100.0 * em / n, "F1": 100.0 * f1 / n,
                    "empty_rate": 100.0 * empty / n, "n": n})
    return out


def select_tau(rows: Sequence[Mapping], criterion: str = "F1",
               tie: str = "closest_to_zero") -> dict:
    """Hàng có ``criterion`` cao nhất; hoà thì ``|τ|`` nhỏ nhất, rồi ``τ`` nhỏ hơn."""
    if not rows:
        raise ValueError("Bảng quét τ rỗng")
    if tie != "closest_to_zero":
        raise ValueError(f"tie rule không hỗ trợ: {tie!r}")
    return dict(max(rows, key=lambda r: (r[criterion], -abs(r["tau"]), -r["tau"])))


def select_epoch_and_tau(sweeps: Mapping[int, Sequence[Mapping]],
                         criterion: str = "F1") -> dict:
    """Chọn CẶP (epoch, τ) có dev ``criterion`` cao nhất.

    Hoà thì epoch SỚM hơn, rồi ``|τ|`` nhỏ hơn, rồi ``τ`` nhỏ hơn — quy tắc đã
    đăng ký trước, cố định để lựa chọn không phụ thuộc thứ tự duyệt.
    """
    if not sweeps:
        raise ValueError("Không có epoch nào để chọn")
    candidates = [(epoch, row) for epoch, rows in sweeps.items() for row in rows]
    if not candidates:
        raise ValueError("Mọi bảng quét τ đều rỗng")
    epoch, row = max(candidates, key=lambda c: (c[1][criterion], -c[0],
                                                -abs(c[1]["tau"]), -c[1]["tau"]))
    return {"epoch": epoch, **row}
