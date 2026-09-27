"""Test cho các script bằng chứng mới (Phụ lục A) và hàm thuần đứng sau chúng.

Hai đối tượng test ở đây có tốc độ khác hẳn:

- ``mrc.audit.null_label_rates`` cần một tokenizer FAST thật (offset_mapping,
  ``num_special_tokens_to_add``, ``sequence_ids``) nhưng KHÔNG cần một model thật
  nào cả — nên dùng tokenizer giả cấp KÝ TỰ (mỗi ký tự một token, offset
  ``(i, i+1)``) để test chạy trong mili-giây thay vì tải mBERT/ViSoBERT.
- ``scripts.window_table.n_windows_for_length`` là số học thuần trên số token,
  test trực tiếp bằng số nguyên, không cần tokenizer nào.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mrc.audit import null_label_rates
from mrc.data import Example


class FakeCharTokenizer:
    """Tokenizer giả cấp KÝ TỰ: mỗi ký tự là một token, offset ``(i, i+1)``.

    Đơn giản hoá tối đa cho test ``make_windows``/``_locate_answer_tokens`` (dùng
    xuyên qua ``null_label_rates``): hai hàm đó chỉ cần một ``offset_mapping`` hợp
    lệ và tuyệt đối trong context gốc, không quan tâm token là ký tự hay wordpiece.
    """

    cls_token_id = -100
    sep_token_id = -101
    pad_token_id = -1

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 2

    def __call__(self, *args, add_special_tokens=True, return_offsets_mapping=False,
                 truncation=None, max_length=None, padding=False):
        if len(args) == 1:
            text = args[0]
            ids = [ord(c) for c in text]
            offsets = [(i, i + 1) for i in range(len(text))]
            if add_special_tokens:
                ids = [self.cls_token_id, *ids, self.sep_token_id]
                offsets = [None, *offsets, None]
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offsets
            return out

        # Cặp (question, chunk) — đường dùng bởi make_windows khi tokenize từng cửa sổ.
        question, chunk = args
        qids = [ord(c) for c in question]
        budget = (max_length or 10**9) - len(qids) - 3     # 3 special: [CLS] .. [SEP] .. [SEP]
        if truncation == "only_second" and len(chunk) > budget:
            chunk = chunk[: max(0, budget)]
        cids = [ord(c) for c in chunk]

        input_ids = [self.cls_token_id, *qids, self.sep_token_id, *cids, self.sep_token_id]
        offsets = [None] * (len(qids) + 2) + [(i, i + 1) for i in range(len(chunk))] + [None]
        seq_ids = [None] * (len(qids) + 2) + [1] * len(chunk) + [None]

        class Enc(dict):
            def sequence_ids(self, batch_index: int = 0):
                return seq_ids

        return Enc(input_ids=input_ids, attention_mask=[1] * len(input_ids), offset_mapping=offsets)


@pytest.fixture
def fake_tokenizer() -> FakeCharTokenizer:
    return FakeCharTokenizer()


class TestNullLabelRates:
    """context = "ABCDEFGHIJKL" (12 ký tự), max_length=8, doc_stride=1.

    budget = 8 - len("Q") - 3 special = 4; step = max(1, 4-1) = 3 -> 4 cửa sổ:
    [0,4)="ABCD", [3,7)="DEFG", [6,10)="GHIJ", [9,12)="JKL" (đuôi ngắn hơn budget).
    """

    CONTEXT = "ABCDEFGHIJKL"

    def test_impossible_question_every_window_is_cls(self, fake_tokenizer):
        ex = [Example(qid="imp1", question="Q", context=self.CONTEXT,
                       answers=[], answer_start=-1, is_impossible=True)]
        got = null_label_rates(ex, fake_tokenizer, max_length=8, doc_stride=1)
        assert got == {
            "questions": 1, "features": 4,
            "cls_impossible": 4, "cls_answer_outside": 0, "positive": 0,
            "null_frac": 100.0, "q_multiwindow": 1,
        }

    def test_answerable_question_only_the_containing_window_is_positive(self, fake_tokenizer):
        # "HI" = context[7:9], nằm TRỌN chỉ trong cửa sổ [6,10)="GHIJ".
        ex = [Example(qid="pos1", question="Q", context=self.CONTEXT,
                       answers=["HI"], answer_start=7, is_impossible=False)]
        got = null_label_rates(ex, fake_tokenizer, max_length=8, doc_stride=1)
        assert got["features"] == 4
        assert got["positive"] == 1
        assert got["cls_answer_outside"] == 3
        assert got["cls_impossible"] == 0
        assert got["q_multiwindow"] == 1

    def test_mixed_batch_matches_hand_computed_totals(self, fake_tokenizer):
        examples = [
            Example(qid="imp1", question="Q", context=self.CONTEXT,
                     answers=[], answer_start=-1, is_impossible=True),
            Example(qid="pos1", question="Q", context=self.CONTEXT,
                     answers=["HI"], answer_start=7, is_impossible=False),
        ]
        got = null_label_rates(examples, fake_tokenizer, max_length=8, doc_stride=1)
        assert got == {
            "questions": 2, "features": 8,
            "cls_impossible": 4, "cls_answer_outside": 3, "positive": 1,
            "null_frac": 87.5, "q_multiwindow": 2,
        }

    def test_short_answerable_question_single_window_is_positive_no_null(self, fake_tokenizer):
        # context ngắn hơn budget -> đúng 1 cửa sổ, chứa trọn đáp án.
        ex = [Example(qid="short1", question="Q", context="AB",
                       answers=["AB"], answer_start=0, is_impossible=False)]
        got = null_label_rates(ex, fake_tokenizer, max_length=8, doc_stride=1)
        assert got == {
            "questions": 1, "features": 1,
            "cls_impossible": 0, "cls_answer_outside": 0, "positive": 1,
            "null_frac": 0.0, "q_multiwindow": 0,
        }

    def test_empty_batch_gives_zero_null_frac_not_a_crash(self, fake_tokenizer):
        got = null_label_rates([], fake_tokenizer, max_length=8, doc_stride=1)
        assert got["features"] == 0
        assert got["null_frac"] == 0.0


class TestWindowTableFormula:
    """``n_windows_for_length`` phải khớp bảng ADR-003 (đo tại 128/32, KHÔNG phải
    384 như caption gốc ghi nhầm — N4) và bảng đã tính lại tại cấu hình huấn
    luyện thật 384/128."""

    def test_matches_adr003_table_at_128_32(self):
        from window_table import n_windows_for_length

        expected = {210: 2, 420: 5, 700: 8, 1400: 16}
        for n_ctx, want in expected.items():
            assert n_windows_for_length(n_ctx, 128, 32, n_question_tokens=3) == want

    def test_recomputed_at_training_config_384_128(self):
        from window_table import n_windows_for_length

        expected = {210: 1, 420: 2, 700: 3, 1400: 6}
        for n_ctx, want in expected.items():
            assert n_windows_for_length(n_ctx, 384, 128, n_question_tokens=3) == want

    def test_single_window_when_context_fits_budget(self):
        from window_table import n_windows_for_length

        assert n_windows_for_length(50, 384, 128, n_question_tokens=3) == 1

    def test_zero_length_context_gives_zero_windows(self):
        from window_table import n_windows_for_length

        assert n_windows_for_length(0, 384, 128) == 0

    def test_raises_when_question_leaves_no_room_for_context(self):
        from window_table import n_windows_for_length

        with pytest.raises(ValueError):
            n_windows_for_length(100, 8, 2, n_question_tokens=10)
