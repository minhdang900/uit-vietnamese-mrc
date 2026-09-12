"""Lấy mẫu con cho đánh giá.

`subset()` là nền của mọi tuyên bố "kết quả tái lập được": nếu nó không xác định,
hai lần chạy cùng seed cho hai tập câu hỏi khác nhau và các con số không so sánh
được với nhau.

Nó cũng phải lấy mẫu NGẪU NHIÊN chứ không phải n câu đầu file — câu đầu file đều
thuộc vài article đầu tiên, nên một mẫu như vậy thiên lệch theo chủ đề.
"""
from mrc.data import Example
from scripts.run_eval import subset


def _mk(n):
    return [Example(qid=f"q{i}", question="Câu hỏi?", context=f"Đoạn văn số {i}.",
                    title=f"Bài {i // 10}", answers=["x"], answer_start=0)
            for i in range(n)]


def test_returns_requested_number():
    assert len(subset(_mk(100), 10)) == 10


def test_deterministic_for_same_seed():
    ex = _mk(100)
    assert [e.qid for e in subset(ex, 10, seed=7)] == [e.qid for e in subset(ex, 10, seed=7)]


def test_different_seeds_give_different_samples():
    ex = _mk(100)
    assert [e.qid for e in subset(ex, 10, seed=1)] != [e.qid for e in subset(ex, 10, seed=2)]


def test_does_not_simply_take_the_first_n():
    # n câu đầu thuộc vài article đầu -> mẫu thiên lệch theo chủ đề
    ex = _mk(200)
    assert [e.qid for e in subset(ex, 20, seed=42)] != [e.qid for e in ex[:20]]


def test_samples_without_replacement():
    qids = [e.qid for e in subset(_mk(100), 30)]
    assert len(qids) == len(set(qids))


def test_returns_everything_when_n_exceeds_size():
    assert len(subset(_mk(5), 50)) == 5


def test_none_means_full_split():
    assert len(subset(_mk(40), None)) == 40


def test_does_not_mutate_the_input_list():
    ex = _mk(50)
    before = [e.qid for e in ex]
    subset(ex, 10)
    assert [e.qid for e in ex] == before


def test_sample_spans_multiple_articles():
    # bằng chứng mẫu KHÔNG dồn vào một article
    titles = {e.title for e in subset(_mk(200), 30, seed=42)}
    assert len(titles) > 5
