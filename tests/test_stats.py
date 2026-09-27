"""Phase 2.2 — mrc.stats: Wilson (EM), McNemar chính xác, bootstrap theo cụm.

Mọi giá trị tham chiếu ở đây là số học đóng (Wilson) hoặc kiểm chứng chéo qua
``scipy.stats.binomtest`` / công thức nhị thức (McNemar) — không phụ thuộc
preds JSONL thật (chưa tồn tại, xem P2.3).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from mrc.stats import (
    cluster_bootstrap_ci,
    empty_rate,
    icc_oneway,
    mcnemar_exact,
    paired_cluster_bootstrap_diff,
    wilson_ci,
)

# ── wilson_ci ────────────────────────────────────────────────────────────


def test_wilson_ci_known_value_4_of_500():
    # scripts/ci.py (cũ): Wilson 95% của 4/500 = [0,31 %; 2,04 %].
    lo, hi = wilson_ci(4, 500)
    assert lo == pytest.approx(0.31, abs=0.01)
    assert hi == pytest.approx(2.04, abs=0.01)


def test_wilson_ci_symmetric_50_of_100():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(40.38, abs=0.01)
    assert hi == pytest.approx(59.62, abs=0.01)


def test_wilson_ci_zero_successes_upper_bound():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(27.75, abs=0.01)


def test_wilson_ci_never_escapes_0_100():
    lo, hi = wilson_ci(500, 500)
    assert 0.0 <= lo <= hi <= 100.0


def test_wilson_ci_empty_n_is_degenerate():
    assert wilson_ci(0, 0) == (0.0, 0.0)


# ── mcnemar_exact ────────────────────────────────────────────────────────


def test_mcnemar_two_sided_matches_binomial_reference():
    p = mcnemar_exact(0, 10, alternative="two-sided")
    ref = 2 * sum(math.comb(10, i) * 0.5**10 for i in range(0, 1))
    assert p == pytest.approx(min(1.0, ref), abs=1e-6)
    assert p == pytest.approx(0.001953, abs=1e-5)


def test_mcnemar_two_sided_symmetric_pairs_is_one():
    assert mcnemar_exact(5, 5, alternative="two-sided") == pytest.approx(1.0)


def test_mcnemar_two_sided_no_discordant_pairs_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_one_sided_greater_b10_wins():
    # b10=10, b01=0: A thắng tuyệt đối — H1 "b10 > b01" phải rất có ý nghĩa.
    p = mcnemar_exact(0, 10, alternative="greater")
    assert p == pytest.approx(0.5**10, abs=1e-6)
    assert p == pytest.approx(0.000977, abs=1e-5)


def test_mcnemar_one_sided_less_opposite_direction_is_one():
    assert mcnemar_exact(0, 10, alternative="less") == pytest.approx(1.0)


def test_mcnemar_matches_scipy_binomtest_directly():
    for b01, b10, alt in ((3, 12, "two-sided"), (7, 2, "greater"), (7, 2, "less")):
        expected = binomtest(b10, b01 + b10, 0.5, alternative=alt).pvalue
        assert mcnemar_exact(b01, b10, alternative=alt) == pytest.approx(expected)


def test_mcnemar_rejects_unknown_alternative():
    with pytest.raises(ValueError, match="alternative"):
        mcnemar_exact(1, 2, alternative="two-tailed")


# ── cluster_bootstrap_ci ─────────────────────────────────────────────────


def test_cluster_bootstrap_reproducible_with_seed():
    values = list(np.linspace(0, 100, 60))
    clusters = [i % 12 for i in range(60)]
    ci_a = cluster_bootstrap_ci(values, clusters, n_boot=300, seed=7)
    ci_b = cluster_bootstrap_ci(values, clusters, n_boot=300, seed=7)
    assert ci_a == ci_b


def test_cluster_bootstrap_different_seed_can_differ():
    values = list(np.linspace(0, 100, 60))
    clusters = [i % 12 for i in range(60)]
    ci_a = cluster_bootstrap_ci(values, clusters, n_boot=300, seed=1)
    ci_b = cluster_bootstrap_ci(values, clusters, n_boot=300, seed=2)
    assert ci_a != ci_b


def test_cluster_ci_wider_than_iid_on_strongly_clustered_data():
    """Mỗi cụm đồng nhất giá trị (0 hoặc 100), cụm khác nhau xen kẽ — tương
    quan trong cụm cực đại. Cluster bootstrap (đơn vị resample = cụm, ít cụm
    hiệu quả) phải cho CI RỘNG HƠN bootstrap coi từng câu hỏi độc lập."""
    rng = np.random.default_rng(0)
    n_clusters, per_cluster = 10, 20
    clusters, values = [], []
    for c in range(n_clusters):
        v = 100.0 if c % 2 == 0 else 0.0
        clusters += [c] * per_cluster
        values += [v] * per_cluster
    # cụm hoá theo item (mỗi item cụm riêng) == bootstrap "iid" (đơn vị nhỏ nhất).
    item_clusters = list(range(len(values)))

    cluster_lo, cluster_hi = cluster_bootstrap_ci(values, clusters, n_boot=2000, seed=0)
    iid_lo, iid_hi = cluster_bootstrap_ci(values, item_clusters, n_boot=2000, seed=0)

    assert (cluster_hi - cluster_lo) > (iid_hi - iid_lo)


def test_cluster_bootstrap_singleton_clusters_approx_item_level():
    """Mỗi câu hỏi một cụm riêng ⇒ cluster bootstrap suy biến về bootstrap
    thường (mọi cụm cùng cỡ 1, resample cụm == resample item)."""
    rng = np.random.default_rng(0)
    values = list(rng.normal(50, 10, size=200))
    clusters = list(range(len(values)))

    cluster_ci = cluster_bootstrap_ci(values, clusters, n_boot=2000, seed=0)

    mean = float(np.mean(values))
    assert cluster_ci[0] < mean < cluster_ci[1]


# ── icc_oneway ───────────────────────────────────────────────────────────


def test_icc_oneway_perfectly_clustered_is_one():
    """Mỗi cụm hằng số (0 hoặc 100), khác cụm khác giá trị ⇒ toàn bộ phương sai
    là GIỮA cụm, không có bên TRONG cụm ⇒ ICC đúng bằng 1.0 (MSW=0 chính xác)."""
    n_clusters, per_cluster = 8, 10
    values, clusters = [], []
    for c in range(n_clusters):
        v = 100.0 if c % 2 == 0 else 0.0
        values += [v] * per_cluster
        clusters += [c] * per_cluster

    assert icc_oneway(values, clusters) == pytest.approx(1.0)


def test_icc_oneway_no_clustering_is_near_zero():
    """Gán nhãn cụm NGẪU NHIÊN, không liên quan gì tới giá trị ⇒ ICC nhỏ."""
    rng = np.random.default_rng(0)
    values = rng.normal(50, 10, size=2000)
    clusters = rng.integers(0, 100, size=2000)  # cụm không mang thông tin gì

    icc = icc_oneway(values, clusters)

    assert abs(icc) < 0.05


def test_icc_oneway_scale_invariant():
    rng = np.random.default_rng(1)
    n_clusters, per_cluster = 20, 8
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    cluster_means = rng.normal(0, 5, size=n_clusters)
    values = np.repeat(cluster_means, per_cluster) + rng.normal(0, 3, size=n_clusters * per_cluster)

    icc = icc_oneway(values, clusters)
    icc_scaled = icc_oneway(values * 100.0 + 7.0, clusters)

    assert icc == pytest.approx(icc_scaled, abs=1e-9)


def test_icc_oneway_singleton_clusters_is_zero():
    values = list(np.linspace(0, 100, 50))
    clusters = list(range(50))  # mỗi câu hỏi một cụm riêng — không ước lượng được MSW
    assert icc_oneway(values, clusters) == 0.0


def test_icc_oneway_single_cluster_is_zero():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    clusters = ["p1"] * 5
    assert icc_oneway(values, clusters) == 0.0


def test_icc_oneway_constant_values_is_zero():
    values = [1.0] * 20
    clusters = [i % 4 for i in range(20)]
    assert icc_oneway(values, clusters) == 0.0


# ── paired_cluster_bootstrap_diff ────────────────────────────────────────


def test_paired_bootstrap_identical_arrays_is_zero():
    a = {f"q{i}": 100.0 * (i % 2) for i in range(40)}
    clusters = {f"q{i}": i % 8 for i in range(40)}
    lo, hi = paired_cluster_bootstrap_diff(a, dict(a), clusters, n_boot=500, seed=0)
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_reproducible_with_seed():
    a = {f"q{i}": float(i % 3 == 0) * 100 for i in range(50)}
    b = {f"q{i}": float(i % 5 == 0) * 100 for i in range(50)}
    clusters = {f"q{i}": i % 10 for i in range(50)}
    r1 = paired_cluster_bootstrap_diff(a, b, clusters, n_boot=400, seed=3)
    r2 = paired_cluster_bootstrap_diff(a, b, clusters, n_boot=400, seed=3)
    assert r1 == r2


def test_paired_bootstrap_raises_on_missing_qid():
    a = {"q1": 1.0, "q2": 0.0}
    b = {"q1": 1.0}
    clusters = {"q1": "p1", "q2": "p1"}
    with pytest.raises(ValueError, match="qid"):
        paired_cluster_bootstrap_diff(a, b, clusters)


def test_paired_bootstrap_raises_on_reordered_qids():
    a = {"q1": 1.0, "q2": 0.0}
    b = {"q2": 0.0, "q1": 1.0}  # cùng tập, khác thứ tự
    clusters = {"q1": "p1", "q2": "p1"}
    with pytest.raises(ValueError, match="qid"):
        paired_cluster_bootstrap_diff(a, b, clusters)


# ── empty_rate (re-export) ────────────────────────────────────────────────


def test_empty_rate_reexported_from_evaluate():
    from mrc.evaluate import empty_rate as evaluate_empty_rate

    assert empty_rate is evaluate_empty_rate
    assert empty_rate({"a": "", "b": "x", "c": "  "}) == pytest.approx(200.0 / 3)
