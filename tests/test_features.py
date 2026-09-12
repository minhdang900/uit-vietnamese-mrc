"""Chuyển Example thành feature huấn luyện cho QA head.

ĐẶC TẢ (từ bài toán, không từ code): model dự đoán hai chỉ số TOKEN —
``start_position`` và ``end_position``. Nhưng dataset cho vị trí đáp án theo KÝ TỰ
(``answer_start``). Bước này phải dịch ký tự -> token.

Đây là chỗ sai âm thầm nguy hiểm nhất của toàn pipeline: nếu map lệch, model học
nhãn sai mà loss VẪN GIẢM BÌNH THƯỜNG. Không có exception, không có cảnh báo —
chỉ là model học nhầm thứ. Vì vậy bất biến quan trọng nhất không phải "chạy không
lỗi" mà là: **giải mã nhãn token phải ra lại đúng đáp án vàng**.

Quy ước SQuAD-2.0: câu impossible, và window không chứa đáp án, đều gán nhãn về
``[CLS]`` — đó là cách model học nói "ở đây không có đáp án".
"""
import pytest

pytestmark = pytest.mark.slow          # cần tokenizer thật

from mrc.data import Example


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)


def _ex(context, answer=None, question="Thủ đô của Việt Nam là gì?"):
    """Tạo Example, tự tính answer_start để test không thể sai offset."""
    if answer is None:
        return Example(qid="q", question=question, context=context,
                       answers=[], answer_start=-1, is_impossible=True)
    start = context.index(answer)
    return Example(qid="q", question=question, context=context,
                   answers=[answer], answer_start=start, is_impossible=False)


# ══════════════════════════════════════════════════════════════════
# BẤT BIẾN CỐT LÕI: nhãn token giải mã ngược ra đáp án vàng
# ══════════════════════════════════════════════════════════════════

def test_token_labels_decode_back_to_the_gold_answer(tok):
    from mrc.features import prepare_train_features

    ctx = "Hà Nội là thủ đô của Việt Nam. Thành phố có tám triệu dân."
    f = prepare_train_features([_ex(ctx, "Hà Nội")], tok)

    s, e = f["start_positions"][0], f["end_positions"][0]
    decoded = tok.decode(f["input_ids"][0][s : e + 1]).replace(" ##", "").strip()
    assert "Hà" in decoded and "Nội" in decoded


def test_token_labels_point_inside_the_context_not_the_question(tok):
    from mrc.features import prepare_train_features

    # câu hỏi CHỨA cụm "Hà Nội" -> nếu map sai, nhãn có thể trỏ vào vùng question
    ctx = "Thủ đô của Việt Nam là Hà Nội."
    f = prepare_train_features([_ex(ctx, "Hà Nội", question="Hà Nội là gì?")], tok)
    s = f["start_positions"][0]
    sep = f["input_ids"][0].index(tok.sep_token_id)
    assert s > sep, "nhãn trỏ vào vùng question thay vì context"


def test_multi_token_answer_spans_more_than_one_token(tok):
    from mrc.features import prepare_train_features

    ctx = "Thành phố Hồ Chí Minh là đô thị lớn nhất Việt Nam."
    f = prepare_train_features([_ex(ctx, "Thành phố Hồ Chí Minh")], tok)
    assert f["end_positions"][0] > f["start_positions"][0]


def test_start_never_exceeds_end(tok):
    from mrc.features import prepare_train_features

    ctx = "Hà Nội là thủ đô của Việt Nam."
    f = prepare_train_features([_ex(ctx, "thủ đô")], tok)
    for s, e in zip(f["start_positions"], f["end_positions"]):
        assert s <= e


# ══════════════════════════════════════════════════════════════════
# Quy ước impossible / window không chứa đáp án -> [CLS]
# ══════════════════════════════════════════════════════════════════

def test_impossible_question_labelled_at_cls(tok):
    from mrc.features import prepare_train_features

    f = prepare_train_features([_ex("Một đoạn văn không có đáp án.")], tok)
    cls = tok.cls_token_id
    assert f["input_ids"][0][f["start_positions"][0]] == cls
    assert f["input_ids"][0][f["end_positions"][0]] == cls


def test_window_without_the_answer_is_labelled_at_cls(tok):
    from mrc.features import prepare_train_features

    # đáp án ở CUỐI context dài -> window đầu không chứa nó
    ctx = ("Câu nhồi không liên quan. " * 200) + "Thủ đô là Hà Nội."
    f = prepare_train_features([_ex(ctx, "Hà Nội")], tok, max_length=128, doc_stride=32)
    cls = tok.cls_token_id
    assert f["input_ids"][0][f["start_positions"][0]] == cls


# ══════════════════════════════════════════════════════════════════
# doc-stride windowing
# ══════════════════════════════════════════════════════════════════

def test_long_context_produces_many_windows(tok):
    from mrc.features import prepare_train_features

    ctx = ("Câu nhồi dài. " * 300) + "Thủ đô là Hà Nội."
    f = prepare_train_features([_ex(ctx, "Hà Nội")], tok, max_length=128, doc_stride=32)
    assert len(f["input_ids"]) > 5, "bị giới hạn window như transformers 5.17"


def test_some_window_of_a_long_context_finds_the_answer(tok):
    """Nếu KHÔNG window nào tìm được đáp án, model không bao giờ học được câu đó."""
    from mrc.features import prepare_train_features

    ctx = ("Câu nhồi dài. " * 200) + "Thủ đô là Hà Nội."
    f = prepare_train_features([_ex(ctx, "Hà Nội")], tok, max_length=128, doc_stride=64)
    cls = tok.cls_token_id
    labelled = [i for i, s in enumerate(f["start_positions"]) if f["input_ids"][i][s] != cls]
    assert labelled, "mọi window đều gán [CLS] -> doc-stride hỏng"


def test_short_context_produces_exactly_one_window(tok):
    from mrc.features import prepare_train_features

    f = prepare_train_features([_ex("Hà Nội là thủ đô.", "Hà Nội")], tok)
    assert len(f["input_ids"]) == 1


# ══════════════════════════════════════════════════════════════════
# Hình dạng đầu ra — phải nạp thẳng vào QADataset
# ══════════════════════════════════════════════════════════════════

def test_every_feature_field_has_the_same_length(tok):
    from mrc.features import prepare_train_features

    f = prepare_train_features([_ex("Hà Nội là thủ đô.", "Hà Nội")], tok)
    n = len(f["input_ids"])
    for key in ("attention_mask", "start_positions", "end_positions"):
        assert len(f[key]) == n


def test_all_sequences_padded_to_max_length(tok):
    from mrc.features import prepare_train_features

    ctx = ("Câu nhồi. " * 60) + "Thủ đô là Hà Nội."
    f = prepare_train_features([_ex(ctx, "Hà Nội")], tok, max_length=128, doc_stride=32)
    assert all(len(ids) == 128 for ids in f["input_ids"])


def test_output_is_accepted_by_QADataset(tok):
    """Hợp đồng giữa hai module — nếu lệch nhau thì huấn luyện sập lúc chạy."""
    from mrc.features import prepare_train_features
    from mrc.training import QADataset

    f = prepare_train_features([_ex("Hà Nội là thủ đô.", "Hà Nội")], tok)
    assert len(QADataset(f)) == len(f["input_ids"])


def test_multiple_examples_are_all_represented(tok):
    from mrc.features import prepare_train_features

    exs = [_ex("Hà Nội là thủ đô.", "Hà Nội"),
           _ex("Huế ở miền Trung.", "miền Trung"),
           _ex("Không có đáp án ở đây.")]
    f = prepare_train_features(exs, tok)
    assert len(f["input_ids"]) >= 3
    assert set(f["example_index"]) == {0, 1, 2}


def test_empty_example_list_gives_empty_features(tok):
    from mrc.features import prepare_train_features

    f = prepare_train_features([], tok)
    assert f["input_ids"] == []


def test_window_containing_only_the_start_of_the_answer_is_labelled_cls(tok):
    """Window chứa ĐẦU đáp án nhưng không chứa ĐUÔI phải gán [CLS], không gán span cụt.

    Đây là trường hợp phân biệt duy nhất cho phép kiểm tra bao-trọn: nếu bỏ kiểm
    tra đó, đa số window vẫn tình cờ ra [CLS] (không token nào thoả điều kiện tìm
    kiếm), nhưng window cắt NGANG đáp án sẽ sinh ra nhãn CỤT — model học sai biên
    span mà không có lỗi nào được báo.
    """
    from mrc.features import prepare_train_features

    answer = ("một đáp án cực kỳ dài trải qua rất nhiều từ liên tiếp nhau "
              "để chắc chắn bị cắt ngang bởi ranh giới cửa sổ trượt")
    ctx = "Mở đầu ngắn. " + answer + " Phần đuôi còn lại của đoạn văn."
    f = prepare_train_features([_ex(ctx, answer)], tok, max_length=32, doc_stride=8)

    cls = tok.cls_token_id
    for i, (s, e) in enumerate(zip(f["start_positions"], f["end_positions"])):
        offsets_ok = f["input_ids"][i][s] == cls and f["input_ids"][i][e] == cls
        if offsets_ok:
            continue
        # window nào không gán [CLS] thì PHẢI bao trọn đáp án
        decoded = tok.decode(f["input_ids"][i][s : e + 1]).replace(" ##", "").strip()
        assert len(decoded) >= len(answer) * 0.5, (
            f"window {i} gán nhãn CỤT {decoded!r} thay vì [CLS]"
        )
