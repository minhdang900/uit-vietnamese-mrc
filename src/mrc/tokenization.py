"""Nạp tokenizer NHANH (có ``offset_mapping``) cho mọi model, kể cả PhoBERT.

``AutoTokenizer`` trả về tokenizer CHẬM cho ``vinai/phobert-base-v2`` (không có
offset ⇒ không map được span về ký tự gốc; xem :mod:`mrc.transformer_qa`). Nhưng
repo trên Hub có sẵn ``tokenizer.json`` (BPE, tách theo khoảng trắng) mà thư viện
``tokenizers`` đọc được và trả offset gốc. Bọc nó bằng ``PreTrainedTokenizerFast``
là đủ — không cần tự viết bộ căn offset.

Sai khác đã đo (P6.1, chỉ trên train): so với tokenizer chậm (bản dùng khi
pretrain), bản nhanh cho id KHÁC ở 69/4.101 context và 80/3.000 cặp câu hỏi +
context — đều ở những mẩu BPE hiếm mà bản chậm biến thành ``<unk>`` (vd. "Hokk").
Offset thì khớp tuyệt đối (0 lệch ký tự trên 500 context). Đây là một sai khác
được ghi nhận, không phải lỗi âm thầm.

Checkpoint fine-tune lưu bằng ``save_pretrained`` của bản bọc sẽ mang
``tokenizer.json`` + ``tokenizer_class = PreTrainedTokenizerFast``, nên nạp lại từ
thư mục checkpoint vẫn ra bản nhanh; hàm dưới đây vẫn kiểm ``is_fast`` mọi lần.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_fast_tokenizer", "PHOBERT_SPECIAL_TOKENS"]

#: Token đặc biệt kiểu RoBERTa/fairseq của PhoBERT (``<s>`` = CLS, ``</s>`` = SEP).
PHOBERT_SPECIAL_TOKENS = {
    "bos_token": "<s>", "eos_token": "</s>", "sep_token": "</s>", "cls_token": "<s>",
    "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>",
}


def _tokenizer_json(name: str) -> str | None:
    local = Path(name) / "tokenizer.json"
    if local.exists():
        return str(local)
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(name, "tokenizer.json")
    except Exception:
        return None


def load_fast_tokenizer(name: str):
    """Tokenizer nhanh cho ``name`` (tên Hub hoặc thư mục checkpoint), hoặc raise.

    Raises:
        RuntimeError: nếu không có đường nào ra được tokenizer nhanh. Fail to ồn:
            thay bằng ``tokenizer.decode()`` sẽ làm mất dấu tiếng Việt.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if getattr(tokenizer, "is_fast", False):
        return tokenizer

    path = _tokenizer_json(name)
    if path is not None and type(tokenizer).__name__ == "PhobertTokenizer":
        wrapped = PreTrainedTokenizerFast(tokenizer_file=path, **PHOBERT_SPECIAL_TOKENS)
        # Cửa sổ do make_windows tự cắt; model_max_length chỉ để transformers khỏi
        # cảnh báo sai. Giới hạn thật (258 vị trí) được finetune.py kiểm riêng.
        wrapped.model_max_length = getattr(tokenizer, "model_max_length", wrapped.model_max_length)
        if wrapped.is_fast:
            return wrapped

    raise RuntimeError(
        f"{name}: không có fast tokenizer ⇒ không có offset_mapping ⇒ không map được "
        "token span về ký tự gốc. Cách thay thế duy nhất là tokenizer.decode(), nhưng "
        "decode làm mất dấu tiếng Việt. Không dùng được model này cho extractive QA."
    )
