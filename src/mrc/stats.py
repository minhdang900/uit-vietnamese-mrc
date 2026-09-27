"""Thống kê suy diễn cho đánh giá full-validation (P2.2): CI, kiểm định cặp.

Ba công cụ, ba phạm vi áp dụng khác nhau — dùng sai chỗ là kết luận sai:

* :func:`wilson_ci` — khoảng tin cậy cho một TỈ LỆ (đếm được thành công/thất
  bại trên ``n`` phép thử độc lập). EM là tỉ lệ theo nghĩa đó; F1 KHÔNG PHẢI
  (nó là trung bình của một điểm liên tục 0–1) — không có ``wilson`` cho F1,
  CI của F1 chỉ đến từ :func:`cluster_bootstrap_ci`.
* :func:`cluster_bootstrap_ci` / :func:`paired_cluster_bootstrap_diff` — CI
  bằng bootstrap theo CỤM ĐOẠN VĂN (``paragraph_id``), không theo câu hỏi.
  Nhiều câu hỏi cùng một đoạn văn không độc lập (cùng phong cách viết, cùng độ
  khó, đôi khi cùng lỗi model); coi chúng độc lập (bootstrap theo câu hỏi, hay
  Wilson trên n câu hỏi) đánh giá THẤP độ bất định thật. Cụm hoá theo đoạn văn
  rồi resample CỤM (không phải câu hỏi) là cách sửa tối thiểu, không cần giả
  định gì về cấu trúc tương quan trong cụm.
* :func:`mcnemar_exact` — kiểm định CẶP (McNemar, nhị thức chính xác) cho hai
  model trên CÙNG một tập câu hỏi. ĐƯỢC GẮN NHÃN Ở ĐÂY LÀ MỘT KIỂM ĐỊNH IID:
  nó coi mỗi cặp (đúng/sai, đúng/sai) là một phép thử độc lập, bỏ qua cụm đoạn
  văn — cùng hạn chế như trên. Nó được báo cáo BÊN CẠNH CI cụm-bootstrap của
  hiệu số (``paired_cluster_bootstrap_diff``), không thay thế; McNemar cho p
  giá trị nhị phân "khác nhau hay không", cluster bootstrap cho khoảng của độ
  lớn khác biệt đã tính đến tương quan cụm.

Không có dependency mới: chỉ numpy (bootstrap) và ``scipy.stats.binomtest``
(nhị thức chính xác, đã có trong requirements từ trước).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from mrc.evaluate import empty_rate

__all__ = [
    "wilson_ci",
    "mcnemar_exact",
    "cluster_bootstrap_ci",
    "paired_cluster_bootstrap_diff",
    "empty_rate",
]

_ALTERNATIVES = ("two-sided", "greater", "less")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95% (mặc định ``z=1.96``) cho tỉ lệ ``k/n``.

    Trả về PHẦN TRĂM (0–100), làm tròn 2 chữ số thập phân — cùng thang với mọi
    số khác trong ``stats_validation.json``. Ổn định hơn Wald khi ``p`` gần 0
    hay 100, hoặc ``n`` nhỏ (xem ``scripts/ci.py`` cũ: Wald ±0,78 quanh EM
    0,80 % trên n=500 là xấp xỉ kém gần biên; Wilson cho ``[0,31 %; 2,04 %]``
    — 4/500 đúng theo nghĩa đen).

    Chỉ dùng cho tỉ lệ kiểu EM (biến Bernoulli đếm được). KHÔNG dùng cho F1.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    adjust = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, (center - adjust) / denom)
    hi = min(1.0, (center + adjust) / denom)
    return (round(100 * lo, 2), round(100 * hi, 2))


def mcnemar_exact(b01: int, b10: int, alternative: str = "two-sided") -> float:
    """Kiểm định McNemar CHÍNH XÁC (nhị thức, không xấp xỉ chi-square).

    ``b01``: số câu mà run A SAI còn run B ĐÚNG (B thắng). ``b10``: số câu mà A
    ĐÚNG còn B SAI (A thắng). Chỉ các cặp KHÔNG ĐỒNG THUẬN (``b01 + b10``) mang
    thông tin — dưới H0 mỗi cặp bất đồng thắng về phía nào là tung đồng xu công
    bằng, nên thống kê là ``Binomial(b01 + b10, 0.5)`` áp cho ``b10``.

    ``alternative``:

    * ``"two-sided"`` (mặc định) — H1: A và B khác nhau (không nói bên nào hơn).
    * ``"greater"`` — H1: ``b10 > b01`` (A thắng nhiều hơn B). Dùng cho gate
      một phía kiểu P4 (C2/C3): "run mới ăn đứt run kiểm soát trên tập HasAns".
    * ``"less"`` — H1: ``b10 < b01`` (B thắng nhiều hơn A).

    ĐƯỢC GẮN NHÃN LÀ KIỂM ĐỊNH IID: bỏ qua việc nhiều câu cùng đoạn văn không
    độc lập (xem docstring module). Không có cặp bất đồng (``b01 == b10 == 0``)
    thì không có thông tin để phân biệt A/B ⇒ trả ``1.0``.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative không hỗ trợ: {alternative!r} (có: {_ALTERNATIVES})")
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(binomtest(b10, n, 0.5, alternative=alternative).pvalue)


def _cluster_groups(clusters: Sequence[Hashable]) -> list[np.ndarray]:
    """``{cụm: mảng chỉ số}`` giữ nguyên thứ tự xuất hiện đầu tiên của mỗi cụm."""
    clusters = np.asarray(clusters)
    groups: dict[Hashable, list[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    return [np.array(idx, dtype=np.intp) for idx in groups.values()]


def _bootstrap_indices(groups: Sequence[np.ndarray], n_boot: int,
                       seed: int) -> list[np.ndarray]:
    """``n_boot`` tập chỉ số, mỗi tập là hợp của các cụm được resample CÓ HOÀN LẠI.

    Số cụm rút ra mỗi lần bằng đúng số cụm gốc (bootstrap chuẩn). Cùng ``seed``
    ⇒ cùng dãy chỉ số, nên kết quả tái lập được — test dùng đúng tính chất này.
    """
    n_clusters = len(groups)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        out.append(np.concatenate([groups[c] for c in chosen]))
    return out


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    seed: int = 0,
    stat: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI phần trăm ``(1 - alpha)`` của ``stat(values)`` bằng bootstrap theo cụm.

    ``clusters[i]`` là cụm (vd. ``paragraph_id``) của ``values[i]``; mỗi lần
    lặp resample CỤM có hoàn lại (không phải từng câu hỏi riêng lẻ) rồi gộp lại
    mọi câu hỏi của các cụm được chọn — nên tương quan trong cụm được giữ
    nguyên trong mỗi mẫu bootstrap. Trả về percentile 2,5/97,5 (mặc định),
    làm tròn 2 chữ số thập phân, CÙNG THANG với ``values`` (gọi với giá trị đã
    nhân 100 để ra phần trăm).

    CI này RỘNG HƠN CI coi từng câu hỏi độc lập khi dữ liệu có tương quan
    trong cụm mạnh (test ``test_stats.py`` dựng một tập cụm hoá mạnh để kiểm
    chính tính chất này).
    """
    values = np.asarray(values, dtype=float)
    groups = _cluster_groups(clusters)
    boot = np.array([stat(values[idx]) for idx in _bootstrap_indices(groups, n_boot, seed)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 2), round(float(hi), 2))


def paired_cluster_bootstrap_diff(
    a: Mapping[str, float],
    b: Mapping[str, float],
    clusters: Mapping[str, Hashable],
    n_boot: int = 2000,
    seed: int = 0,
    stat: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI bootstrap-theo-cụm của ``stat(a) - stat(b)`` cho HAI RUN trên cùng câu hỏi.

    ``a``, ``b``: ``{qid: giá trị}`` (vd. EM hoặc F1 đã nhân 100) của hai run.
    ``clusters``: ``{qid: cụm}`` (vd. ``paragraph_id``), phải chứa mọi qid của
    ``a``. Vì đây là so sánh CẶP (cùng câu hỏi, hai model), ``a`` và ``b`` BẮT
    BUỘC có đúng cùng tập qid, theo ĐÚNG CÙNG THỨ TỰ — khác thứ tự thường là
    dấu hiệu hai run không được ghép đúng cặp (vd. một run thiếu câu hỏi, hoặc
    preds JSONL bị sắp lại) và IM LẶNG cho qua sẽ ghép sai câu hỏi này với câu
    hỏi khác.

    Raises:
        ValueError: ``list(a) != list(b)`` (thiếu qid, thừa qid, hoặc thứ tự
            khác) — không tự động sắp lại, vì thứ tự khác nhau là bằng chứng
            đáng ngờ (xem trên), không phải chuyện vặt cần bỏ qua.

    Mỗi lần bootstrap resample CỤM (giữ nguyên cách cụm hoá của cả hai run,
    vì cùng một câu hỏi thuộc cùng một đoạn văn ở cả hai) rồi lấy hiệu số
    ``stat`` trên đúng tập câu hỏi đó ở cả hai run — nên tương quan giữa hai
    run (cùng câu dễ/khó cho cả hai) được giữ nguyên qua từng mẫu.
    """
    qids_a, qids_b = list(a), list(b)
    if qids_a != qids_b:
        raise ValueError(
            "paired_cluster_bootstrap_diff: hai run không có cùng tập qid theo cùng "
            "thứ tự — kiểm tra hai preds JSONL có ghép đúng cặp câu hỏi không "
            f"(n_a={len(qids_a)}, n_b={len(qids_b)})."
        )
    values_a = np.array([a[q] for q in qids_a], dtype=float)
    values_b = np.array([b[q] for q in qids_a], dtype=float)
    cluster_ids = [clusters[q] for q in qids_a]
    groups = _cluster_groups(cluster_ids)
    boot = np.array([stat(values_a[idx]) - stat(values_b[idx])
                     for idx in _bootstrap_indices(groups, n_boot, seed)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 2), round(float(hi), 2))
