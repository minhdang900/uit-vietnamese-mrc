"""Phase 4a — chọn span và map token→ký tự. KHÔNG cần tải model.

Đây là nơi phát sinh lỗi âm thầm nhiều nhất trong extractive QA: span lệch vài
ký tự, span lấy từ vùng QUESTION thay vì CONTEXT, span vắt qua hai window. Các
lỗi này KHÔNG crash — chúng chỉ làm F1 thấp một cách khó hiểu.

Tách logic thuần ra khỏi tokenizer cho phép test toàn bộ phần dễ sai này mà không
tải 1GB weights, nên vòng lặp TDD chạy trong vài millisecond.
"""
import pytest

from mrc.windowing import decode_span, select_best_span

# Quy ước offset_mapping: None = token KHÔNG thuộc context (CLS/SEP/question).
# Chỉ token có offset (start_char, end_char) mới là ứng viên đáp án.


def test_picks_single_best_token():
    # context token duy nhất ở index 2, ứng với ký tự [0,6) của context
    off = [None, None, (0, 6), None]
    assert select_best_span([0, 0, 9, 0], [0, 0, 9, 0], off) == (0, 6)


def test_picks_multi_token_span():
    # "thủ đô" = token 2..3 -> ký tự 0..6
    off = [None, (0, 3), (0, 3), (4, 6), None]
    got = select_best_span([0, 0, 9, 0, 0], [0, 0, 0, 9, 0], off)
    assert got == (0, 6)


def test_never_selects_question_tokens():
    # index 1 có logit CAO NHẤT nhưng offset None (thuộc question) -> phải bỏ qua
    off = [None, None, (10, 14), None]
    start, end = select_best_span([0, 99, 1, 0], [0, 99, 1, 0], off)
    assert (start, end) == (10, 14)


def test_rejects_span_where_end_precedes_start():
    # logit đẩy start về token 3 và end về token 2 -> phải chọn cặp HỢP LỆ khác
    off = [None, (0, 2), (3, 5), (6, 8), None]
    start, end = select_best_span([0, 0, 0, 9, 0], [0, 0, 9, 0, 0], off)
    assert start <= end


def test_enforces_max_answer_length_in_tokens():
    # 5 context token; giới hạn 2 token -> không được trả span dài 5 token
    off = [None] + [(i * 2, i * 2 + 1) for i in range(5)] + [None]
    start_logits = [0, 9, 0, 0, 0, 0, 0]
    end_logits = [0, 0, 0, 0, 0, 9, 0]
    start, end = select_best_span(start_logits, end_logits, off, max_answer_len=2)
    assert end - start <= 2 * 2      # không thể trải hết 5 token


def test_returns_none_when_null_score_dominates():
    # CLS (index 0) có logit áp đảo -> câu impossible, trả None
    off = [None, (0, 4)]
    assert select_best_span([99, 0], [99, 0], off, null_threshold=0.0) is None


def test_returns_span_when_span_score_beats_null():
    off = [None, (0, 4)]
    assert select_best_span([0, 9], [0, 9], off, null_threshold=0.0) == (0, 4)


def test_null_threshold_shifts_the_decision():
    off = [None, (0, 4)]
    logits_s, logits_e = [5, 4], [5, 4]        # null=10, span=8
    assert select_best_span(logits_s, logits_e, off, null_threshold=0.0) is None
    # threshold âm lớn -> ưu tiên trả lời thay vì bỏ trống
    assert select_best_span(logits_s, logits_e, off, null_threshold=-10.0) == (0, 4)


def test_returns_none_when_no_context_tokens_at_all():
    assert select_best_span([1, 1], [1, 1], [None, None]) is None


def test_handles_empty_inputs():
    assert select_best_span([], [], []) is None


# ── decode_span: bất biến "đúng substring gốc" ───────────────────────
def test_decode_span_returns_exact_original_substring():
    ctx = "Hà Nội là thủ đô của Việt Nam."
    assert decode_span(ctx, 0, 6) == "Hà Nội"


def test_decode_span_preserves_vietnamese_diacritics():
    ctx = "Thành phố Hoà Bình nằm ở miền Bắc."
    out = decode_span(ctx, 10, 18)
    assert out == "Hoà Bình"
    assert "à" in out            # decode() của tokenizer thường làm mất dấu


def test_decode_span_result_is_substring_of_context():
    ctx = "Một đoạn văn bản tiếng Việt có dấu."
    assert decode_span(ctx, 4, 16) in ctx


def test_decode_span_rejects_inverted_range():
    with pytest.raises(ValueError):
        decode_span("abcdef", 5, 2)


def test_decode_span_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        decode_span("abc", 0, 99)


def test_decode_span_rejects_negative_start():
    with pytest.raises(ValueError):
        decode_span("abc", -1, 2)


def test_decode_span_empty_range_gives_empty_string():
    assert decode_span("abc", 1, 1) == ""


# ══════════════════════════════════════════════════════════════════════
# make_windows — windowing TỰ CÀI, không dùng return_overflowing_tokens
#
# transformers 5.17.0 GIỚI HẠN return_overflowing_tokens ở 2 window bất kể
# context dài bao nhiêu và stride bằng bao nhiêu (đo được: context 420/700/1400
# token đều chỉ sinh 2 window, đáng lẽ phải là 5/8/16). Hệ quả là phần đuôi của
# context bị cắt ÂM THẦM — không exception, không cảnh báo, chỉ là đáp án nằm ở
# cuối đoạn văn thì không bao giờ tìm được.
#
# Vì vậy dự án tự cài windowing: đúng, test được, và không phụ thuộc vào hành vi
# của một phiên bản thư viện cụ thể.
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)


@pytest.mark.slow
class TestMakeWindows:

    def test_short_context_gives_exactly_one_window(self, tok):
        from mrc.windowing import make_windows
        w = make_windows("Câu hỏi?", "Một đoạn văn ngắn.", tok, max_length=128, doc_stride=32)
        assert len(w) == 1

    def test_long_context_gives_many_windows_not_capped_at_two(self, tok):
        from mrc.windowing import make_windows
        ctx = "Câu nhồi dài. " * 200          # ~1400 token
        w = make_windows("Câu hỏi?", ctx, tok, max_length=128, doc_stride=32)
        assert len(w) > 5, f"chỉ có {len(w)} window — vẫn bị cap như transformers"

    def test_windows_cover_the_entire_context(self, tok):
        from mrc.windowing import make_windows
        ctx = "Câu nhồi dài. " * 200
        w = make_windows("Câu hỏi?", ctx, tok, max_length=128, doc_stride=32)
        covered = max(o[1] for win in w for o in win.offset_mapping if o is not None)
        # phải bao gần hết context (cho phép sai số vài ký tự khoảng trắng cuối)
        assert covered >= len(ctx.rstrip()) - 5, f"chỉ bao được {covered}/{len(ctx)} ký tự"

    def test_answer_at_very_end_is_inside_some_window(self, tok):
        from mrc.windowing import make_windows
        ctx = ("Câu nhồi dài. " * 200) + "Thủ đô là Hà Nội."
        start = ctx.index("Hà Nội"); end = start + len("Hà Nội")
        w = make_windows("Thủ đô?", ctx, tok, max_length=128, doc_stride=32)
        assert any(
            any(o is not None and o[0] <= start and o[1] >= end - 1 for o in win.offset_mapping)
            or (min(o[0] for o in win.offset_mapping if o) <= start
                and max(o[1] for o in win.offset_mapping if o) >= end)
            for win in w
        ), "đáp án ở cuối context không nằm trong window nào"

    def test_offsets_are_absolute_into_the_original_context(self, tok):
        from mrc.windowing import make_windows, decode_span
        ctx = "Hà Nội là thủ đô của Việt Nam. " + ("Câu nhồi. " * 100)
        w = make_windows("Thủ đô?", ctx, tok, max_length=128, doc_stride=32)
        # offset của window CUỐI phải trỏ vào vùng cuối context, không phải vùng đầu
        last = [o for o in w[-1].offset_mapping if o is not None]
        assert last[0][0] > 0
        # và cắt ra được substring hợp lệ
        assert decode_span(ctx, last[0][0], last[-1][1]) in ctx

    def test_context_tokens_never_include_question_region(self, tok):
        from mrc.windowing import make_windows
        w = make_windows("Câu hỏi rất dài để chiếm chỗ?", "Một đoạn văn.", tok,
                         max_length=64, doc_stride=16)
        for win in w:
            ctx_idx = [i for i, o in enumerate(win.offset_mapping) if o is not None]
            assert ctx_idx, "window không có token context nào"
            assert min(ctx_idx) > 1          # sau [CLS] + ít nhất 1 token question

    def test_every_window_fits_max_length(self, tok):
        from mrc.windowing import make_windows
        ctx = "Câu nhồi dài. " * 200
        for win in make_windows("Câu hỏi?", ctx, tok, max_length=128, doc_stride=32):
            assert len(win.input_ids) <= 128

    def test_empty_context_gives_no_windows(self, tok):
        from mrc.windowing import make_windows
        assert make_windows("Câu hỏi?", "", tok) == []

    def test_windows_overlap_by_doc_stride(self, tok):
        from mrc.windowing import make_windows
        ctx = "Câu nhồi dài. " * 200
        w = make_windows("Câu hỏi?", ctx, tok, max_length=128, doc_stride=32)
        a_end = max(o[1] for o in w[0].offset_mapping if o)
        b_start = min(o[0] for o in w[1].offset_mapping if o)
        assert b_start < a_end, "hai window liền kề không chồng lấp"

    def test_deterministic(self, tok):
        from mrc.windowing import make_windows
        ctx = "Câu nhồi. " * 80
        a = make_windows("q?", ctx, tok, max_length=128, doc_stride=32)
        b = make_windows("q?", ctx, tok, max_length=128, doc_stride=32)
        assert [w.offset_mapping for w in a] == [w.offset_mapping for w in b]


# ══════════════════════════════════════════════════════════════════════
# score_spans — bằng chứng mà select_best_span vứt đi
#
# ĐẶC TẢ: chấm mọi cặp (start, end) hợp lệ, trả về top-k kèm xác suất và điểm
# null, KHÔNG áp ngưỡng. Demo đọc những số này để giải thích quyết định của
# model; nếu chúng không khớp với span mà select_best_span chọn thì giao diện
# đang nói dối về chính model của nó.
# ══════════════════════════════════════════════════════════════════════
from mrc.windowing import score_spans  # noqa: E402

#: [CLS] q [SEP] "Hà" "Nội" "là" "thủ đô" [SEP] — hai token đầu là đáp án.
_OFFSETS = [None, None, None, (0, 2), (3, 6), (7, 9), (10, 16), None]


def _logits(*pairs):
    """Dựng (start_logits, end_logits) dài 8 từ các cặp (index, giá trị)."""
    start = [0.0] * 8
    end = [0.0] * 8
    for idx, s, e in pairs:
        start[idx], end[idx] = s, e
    return start, end


def test_score_spans_ranks_the_same_span_select_best_span_picks():
    start, end = _logits((3, 5.0, 0.0), (4, 0.0, 4.0))
    scores = score_spans(start, end, _OFFSETS)
    assert (scores.best.start_char, scores.best.end_char) == select_best_span(
        start, end, _OFFSETS
    )


def test_score_spans_returns_alternatives_below_the_winner():
    start, end = _logits((3, 5.0, 0.0), (4, 1.0, 4.0), (6, 0.5, 2.0))
    scores = score_spans(start, end, _OFFSETS, top_k=3)
    assert len(scores.candidates) > 1
    assert scores.candidates[0].score > scores.candidates[1].score


def test_score_spans_alternatives_are_distinct_character_ranges():
    start, end = _logits((3, 5.0, 1.0), (4, 2.0, 4.0), (5, 1.0, 3.0), (6, 0.5, 2.0))
    spans = score_spans(start, end, _OFFSETS, top_k=3).candidates
    ranges = [(s.start_char, s.end_char) for s in spans]
    assert len(ranges) == len(set(ranges))


def test_score_spans_probabilities_are_between_zero_and_one():
    start, end = _logits((3, 5.0, 0.0), (4, 0.0, 4.0), (6, 1.0, 1.0))
    for span in score_spans(start, end, _OFFSETS, top_k=3).candidates:
        assert 0.0 <= span.prob <= 1.0


def test_score_spans_probabilities_are_ordered_like_scores():
    start, end = _logits((3, 5.0, 0.0), (4, 1.0, 4.0), (6, 0.5, 2.0))
    spans = score_spans(start, end, _OFFSETS, top_k=3).candidates
    assert spans[0].prob > spans[1].prob


def test_score_spans_survives_large_logits_without_overflow():
    """log-sum-exp chạy dòng: logit 800 làm math.exp tràn nếu tính ngây thơ."""
    start, end = _logits((3, 800.0, 0.0), (4, 0.0, 800.0))
    assert score_spans(start, end, _OFFSETS).best.prob > 0.0


def test_score_spans_reports_the_null_score():
    start, end = _logits((0, 3.0, 2.0), (3, 5.0, 0.0), (4, 0.0, 4.0))
    assert score_spans(start, end, _OFFSETS).null_score == 5.0


def test_null_delta_is_negative_when_the_span_beats_null():
    start, end = _logits((0, 0.0, 0.0), (3, 5.0, 0.0), (4, 0.0, 4.0))
    assert score_spans(start, end, _OFFSETS).null_delta < 0


def test_null_delta_is_positive_when_null_beats_the_span():
    start, end = _logits((0, 9.0, 9.0), (3, 1.0, 0.0), (4, 0.0, 1.0))
    assert score_spans(start, end, _OFFSETS).null_delta > 0


def test_null_delta_sign_matches_the_abstain_decision_of_select_best_span():
    """Bất biến gắn thanh đo của demo với quyết định thật của model."""
    start, end = _logits((0, 4.0, 4.0), (3, 3.0, 0.0), (4, 0.0, 3.0))
    delta = score_spans(start, end, _OFFSETS).null_delta
    for threshold in (-4.0, -1.0, 0.0, 1.0, 4.0):
        abstains = select_best_span(start, end, _OFFSETS, null_threshold=threshold) is None
        assert abstains == (delta + threshold >= 0)


def test_score_spans_start_end_probabilities_favour_the_chosen_tokens():
    start, end = _logits((3, 8.0, 0.0), (4, 0.0, 8.0))
    scores = score_spans(start, end, _OFFSETS)
    assert scores.start_prob > 0.9 and scores.end_prob > 0.9


def test_score_spans_ignores_question_tokens_in_the_probabilities():
    """Token của question có logit cao vẫn không được vào mẫu số softmax."""
    start, end = _logits((1, 50.0, 50.0), (3, 8.0, 0.0), (4, 0.0, 8.0))
    assert score_spans(start, end, _OFFSETS).start_prob > 0.9


def test_score_spans_respects_max_answer_len():
    start, end = _logits((3, 5.0, 0.0), (6, 0.0, 5.0))
    best = score_spans(start, end, _OFFSETS, max_answer_len=1).best
    assert best.end_char - best.start_char <= 2


def test_score_spans_returns_none_without_context_tokens():
    start, end = _logits((0, 1.0, 1.0))
    assert score_spans(start, end, [None, None, None]) is None


def test_score_spans_returns_none_on_empty_logits():
    assert score_spans([], [], _OFFSETS) is None


def test_score_spans_null_score_is_none_without_a_cls_token():
    start, end = _logits((3, 5.0, 0.0), (4, 0.0, 4.0))
    scores = score_spans(start, end, _OFFSETS, cls_index=99)
    assert scores.null_score is None and scores.null_delta is None
