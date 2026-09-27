"""Fast tokenizer cho mọi model, kể cả PhoBERT (bọc tokenizer.json) — P6.2.

Chỉ cần tokenizer trong cache HF (không tải weights), nên chạy trong bộ test nhanh;
thiếu cache thì skip thay vì đỏ.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PHOBERT = "vinai/phobert-base-v2"


@pytest.fixture(scope="module")
def phobert():
    from mrc.tokenization import load_fast_tokenizer
    try:
        return load_fast_tokenizer(PHOBERT)
    except OSError as e:  # không có cache
        pytest.skip(f"PhoBERT tokenizer không có trong cache: {e}")


def test_phobert_wrapper_is_fast_with_exact_offsets(phobert):
    ctx = "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam."
    enc = phobert(ctx, add_special_tokens=False, return_offsets_mapping=True)
    assert phobert.is_fast
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        assert ctx[a:b] == phobert.convert_ids_to_tokens(tid).replace("</w>", "")


def test_phobert_wrapper_matches_slow_ids_on_common_text(phobert):
    from transformers import AutoTokenizer
    slow = AutoTokenizer.from_pretrained(PHOBERT)
    q, ctx = "Thủ đô của Việt Nam là gì?", "Hà Nội là thủ đô của Việt Nam, dân số 8 triệu."
    assert phobert(q, ctx)["input_ids"] == slow(q, ctx)["input_ids"]
    assert phobert.num_special_tokens_to_add(pair=True) == 4


def test_phobert_windows_mark_context_tokens(phobert):
    from mrc.windowing import decode_span, make_windows
    ctx = "Câu đệm. " * 200 + "Thủ đô của Việt Nam là Hà Nội."
    windows = make_windows("Thủ đô?", ctx, phobert, max_length=256, doc_stride=64)
    assert len(windows) > 1 and all(len(w.input_ids) <= 256 for w in windows)
    last = windows[-1]
    spans = [o for o in last.offset_mapping if o is not None]
    assert decode_span(ctx, spans[-1][0], spans[-1][1]).endswith(".")


def test_saved_checkpoint_reloads_as_fast(phobert, tmp_path):
    from transformers import AutoTokenizer

    from mrc.tokenization import load_fast_tokenizer
    phobert.save_pretrained(tmp_path)
    for tok in (load_fast_tokenizer(str(tmp_path)), AutoTokenizer.from_pretrained(tmp_path)):
        assert tok.is_fast
        assert tok("Hà Nội")["input_ids"] == phobert("Hà Nội")["input_ids"]


def test_model_without_any_fast_path_is_refused(monkeypatch):
    import transformers

    from mrc.tokenization import load_fast_tokenizer

    class SlowOnly:
        is_fast = False

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: SlowOnly()))
    with pytest.raises(RuntimeError, match="fast|offset"):
        load_fast_tokenizer("mot-model-cham")


# ── ngân sách vị trí (finetune.py) ───────────────────────────────────
@pytest.fixture(scope="module")
def ft():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("finetune_tok", root / "scripts" / "finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("model_type,limit,max_length,ok", [
    ("roberta", 258, 256, True), ("roberta", 258, 384, False),
    ("xlm-roberta", 514, 512, True), ("bert", 512, 512, True), ("bert", 512, 513, False),
])
def test_position_budget(ft, model_type, limit, max_length, ok):
    args = SimpleNamespace(max_length=max_length, model="m")
    cfg = SimpleNamespace(model_type=model_type, max_position_embeddings=limit)
    if ok:
        ft.check_position_budget(args, cfg)
    else:
        with pytest.raises(SystemExit, match="max-length"):
            ft.check_position_budget(args, cfg)
