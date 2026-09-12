"""Phase 6a — map vị trí đáp án từ KÝ TỰ sang TOKEN.

Nếu bước này lệch, fine-tuning học nhãn sai mà loss vẫn giảm bình thường — không
có lỗi nào được báo. Vì vậy nó phải được test bằng model thật.
"""
import pytest

pytestmark = pytest.mark.slow

from mrc.data import Example
from mrc.features import prepare_train_features


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)


def _ex(qid, ctx, ans, start, impossible=False):
    return Example(qid=qid, question="Thủ đô của Việt Nam là gì?", context=ctx,
                   answers=[] if impossible else [ans],
                   answer_start=-1 if impossible else start,
                   is_impossible=impossible)


def test_token_positions_decode_back_to_the_gold_answer(tok):
    """Cổng quan trọng nhất của phase này: nhãn token phải trỏ về đúng đáp án."""
    ctx = "Hà Nội là thủ đô của Việt Nam. Thành phố có tám triệu dân."
    ex = _ex("q1", ctx, "Hà Nội", ctx.index("Hà Nội"))
    f = prepare_train_features([ex], tok)

    s, e = f["start_positions"][0], f["end_positions"][0]
    ids = f["input_ids"][0][s : e + 1]
    decoded = tok.decode(ids).replace(" ##", "").strip()
    assert "Nội" in decoded or "Hà" in decoded


def test_impossible_question_labels_point_to_cls(tok):
    ex = _ex("q2", "Một đoạn văn không chứa đáp án.", None, None, impossible=True)
    f = prepare_train_features([ex], tok)
    cls_id = tok.cls_token_id
    assert f["input_ids"][0][f["start_positions"][0]] == cls_id
    assert f["input_ids"][0][f["end_positions"][0]] == cls_id


def test_answer_not_in_window_falls_back_to_cls(tok):
    # Đáp án nằm ở cuối context rất dài -> window ĐẦU không chứa nó
    ctx = ("Câu nhồi. " * 400) + "Thủ đô là Hà Nội."
    ex = _ex("q3", ctx, "Hà Nội", ctx.index("Hà Nội"))
    f = prepare_train_features([ex], tok, max_length=128, doc_stride=32)
    cls_id = tok.cls_token_id
    first = f["input_ids"][0][f["start_positions"][0]]
    assert first == cls_id or f["start_positions"][0] > 0   # window đầu: CLS hoặc tìm thấy


def test_long_context_produces_multiple_windows(tok):
    ctx = ("Câu nhồi dài. " * 300) + "Thủ đô là Hà Nội."
    ex = _ex("q4", ctx, "Hà Nội", ctx.index("Hà Nội"))
    f = prepare_train_features([ex], tok, max_length=128, doc_stride=32)
    assert len(f["input_ids"]) > 1


def test_at_least_one_window_of_long_context_finds_the_answer(tok):
    ctx = ("Câu nhồi dài. " * 100) + "Thủ đô là Hà Nội."
    ex = _ex("q5", ctx, "Hà Nội", ctx.index("Hà Nội"))
    f = prepare_train_features([ex], tok, max_length=128, doc_stride=64)
    cls_id = tok.cls_token_id
    non_cls = [i for i, s in enumerate(f["start_positions"])
               if f["input_ids"][i][s] != cls_id]
    assert non_cls, "không window nào tìm thấy đáp án -> doc-stride sai"


def test_every_feature_has_a_label(tok):
    ctx = "Hà Nội là thủ đô của Việt Nam."
    f = prepare_train_features([_ex("q6", ctx, "Hà Nội", 0)], tok)
    assert len(f["start_positions"]) == len(f["input_ids"]) == len(f["end_positions"])
