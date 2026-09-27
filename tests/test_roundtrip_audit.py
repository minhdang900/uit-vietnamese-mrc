"""Kiểm khứ hồi ký tự (P6 gate) — trên fixture nhỏ, tokenizer từ cache."""
import importlib.util
import os
from pathlib import Path

import pytest

from mrc.data import Example

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rt():
    spec = importlib.util.spec_from_file_location("roundtrip_audit",
                                                  ROOT / "scripts" / "roundtrip_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ex(qid, ctx, gold, impossible=False):
    return Example(qid=qid, question="Câu hỏi?", context=ctx, title="T",
                   answers=[] if impossible else [gold],
                   answer_start=-1 if impossible else ctx.index(gold), is_impossible=impossible)


# Không có dấu câu ngay sau đáp án: fixture này đo cùng một điều cho mọi tokenizer.
CTX = "Thủ đô của Việt Nam là Hà Nội và thành phố có khoảng tám triệu dân sinh sống"
FIXTURE = [_ex("a", CTX, "Hà Nội"), _ex("b", CTX, "tám triệu"),
           _ex("c", CTX, "iệt Nam"),          # gold cắt giữa từ (lỗi gán nhãn kiểu ViQuAD)
           _ex("i", CTX, "", impossible=True)]


def test_categorize(rt):
    ctx = "Trung Quốc. Việt Nam"
    assert rt.categorize(ctx, 0, "Trung Quốc", "Trung Quốc.") == "trailing_punct_attached"
    assert rt.categorize(ctx, 13, "iệt Nam", "Việt Nam") == "gold_cuts_inside_word"
    assert rt.categorize(ctx, 0, "Trung Quốc", None) == "no_containing_window"


@pytest.mark.parametrize("name,ml,stride", [("bert-base-multilingual-cased", 384, 128),
                                            ("vinai/phobert-base-v2", 256, 64)])
def test_audit_on_fixture(rt, name, ml, stride):
    from mrc.tokenization import load_fast_tokenizer
    try:
        tok = load_fast_tokenizer(name)
    except OSError as e:
        pytest.skip(f"không có cache: {e}")
    r = rt.audit(FIXTURE, tok, ml, stride)
    assert r["answerable"] == 3                        # câu impossible bị bỏ
    assert r["exact"] == 2 and r["normalized"] == 2   # "iệt Nam" không khứ hồi được
    assert r["failure_categories"] == {"gold_cuts_inside_word": 1}
    assert r["failure_examples"]["gold_cuts_inside_word"][0]["qid"] == "c"


def test_phobert_trailing_punct_is_exact_miss_but_normalized_hit(rt):
    from mrc.tokenization import load_fast_tokenizer
    try:
        tok = load_fast_tokenizer("vinai/phobert-base-v2")
    except OSError as e:
        pytest.skip(f"không có cache: {e}")
    r = rt.audit([_ex("p", "Thủ đô là Hà Nội.", "Hà Nội")], tok, 256, 64)
    assert (r["exact"], r["normalized"]) == (0, 1)
    assert r["failure_categories"] == {"trailing_punct_attached": 1}


def test_p6_gate_rule(rt):
    def res(pho, mb):
        return {"phobert": {"normalized_rate": pho}, "mbert": {"normalized_rate": mb}}
    assert rt.p6_gate(res(99.303, 99.47))["passed"]
    assert not rt.p6_gate(res(98.9, 99.0))["passed"]      # < 99
    assert not rt.p6_gate(res(99.1, 99.7))["passed"]      # cách mBERT > 0,5
