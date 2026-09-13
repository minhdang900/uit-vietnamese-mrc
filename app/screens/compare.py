"""Màn hình So sánh model — cùng một câu hỏi, bốn hệ thống trả lời."""

from __future__ import annotations

import streamlit as st

from app import shell
from demo import render as html
from demo.catalog import MODELS
from demo.logic import abstains, answer
from demo.results import MissingResults, load_all_evals, passages
from demo.vi import integer, join, number, signed

#: Nhận xét về hành vi TỔNG THỂ của từng model trên cả split, không phải nhận xét
#: về dòng dự đoán ngay bên cạnh. Câu chữ nói rõ điều đó ("trên cả split"), vì một
#: câu dễ như câu mở màn thì cả bốn model đều trả lời được — và một nhận xét kiểu
#: "suy sụp về luôn-trả-rỗng" đứng cạnh một đáp án đúng sẽ đọc như đang mâu thuẫn.
_VERDICTS = {
    "baseline": "Trên cả split: truy hồi nguyên một câu — overlap token có, "
                "trùng khít thì không, nên F1 {f1} mà EM chỉ {em}",
    "visobert": "Trên cả split: suy sụp về luôn-trả-rỗng, answerable EM chỉ {ans_em}",
    "xlmr": "Trên cả split: mạnh nhất ở câu answerable, F1 {ans_f1}",
    "mbert": "Trên cả split: biết khi nào không nên trả lời, impossible EM {imp_em}",
}


@st.cache_data(show_spinner="Đang chạy cả bốn model…")
def _predict_all(threshold: float, context: str, question: str) -> dict[str, dict]:
    """Chạy TẤT CẢ model trên cùng câu hỏi.

    Bản thiết kế chỉ minh hoạ hành vi bằng chữ; ở đây bốn model chạy thật, nên
    hàng nào cũng là dự đoán đo được chứ không phải mô tả.
    """
    out: dict[str, dict] = {}
    for model in MODELS:
        predictor = shell.predictor_for(model, threshold)
        if predictor is not None:
            out[model.id] = answer(context, question, predictor)
    return out


def render() -> None:  # pragma: no cover - lớp UI
    shell.set_width(1000)
    items = passages(st.session_state["model"])
    selected = st.session_state.get("passage")
    item = items[0 if selected is None else min(selected, len(items) - 1)]

    question = st.session_state["question"] or item.question
    threshold = st.session_state["threshold"]

    st.html(
        '<div class="om"><h1 style="margin:0 0 6px">So sánh model</h1>'
        '<p style="margin:0;color:var(--color-neutral-800)">Cùng một câu hỏi, bốn hệ '
        'thống. Đổi đoạn văn ở màn hình Hỏi đáp, đổi ngưỡng ở sidebar.</p></div>'
    )

    # Như màn hình Hỏi đáp: nhãn gold gắn với câu hỏi gốc của đoạn văn, không
    # gắn với câu người dùng tự gõ.
    preset = question.strip() == item.question.strip()
    if not preset:
        gold_line = "Gold: không có — câu tự nhập, không đối chiếu được"
    elif item.impossible:
        gold_line = "Gold: rỗng — câu này thuộc nhóm impossible"
    else:
        gold_line = f"Gold: {join(list(item.gold))}"
    st.html(
        '<div class="om om-card" style="gap:var(--space-2)">'
        '<div class="om-kicker">Câu hỏi</div>'
        f'<div class="om-display" style="font-size:22px;line-height:1.25">'
        f'{html.esc(question)}</div>'
        f'<div class="om-note">{html.esc(gold_line)}</div></div>'
    )

    try:
        evals = load_all_evals()
    except MissingResults as error:
        st.error(str(error))
        return

    predictions = _predict_all(threshold, item.context, question)

    for model in MODELS:
        data = evals.get(model.id)
        result = predictions.get(model.id)
        if data is None or result is None:
            continue

        null_delta = result["null_delta"]
        abstain = abstains(null_delta, threshold) if null_delta is not None \
            else not result["found"]

        note = _VERDICTS.get(model.id, "").format(
            em=number(data["overall"]["EM"]),
            f1=number(data["overall"]["F1"]),
            ans_em=number(data["answerable_only"]["EM"]),
            ans_f1=number(data["answerable_only"]["F1"]),
            imp_em=number(data["impossible_only"]["EM"]),
        )
        st.html(html.compare_row(
            name=model.name, note=model.note,
            prediction="— từ chối trả lời —" if abstain else result["answer"],
            verdict_text=(f"Từ chối ở ngưỡng {signed(threshold)}" if abstain else note),
            em=data["overall"]["EM"], f1=data["overall"]["F1"],
            muted=abstain,
        ))

    _tokenization_note()


def _tokenization_note() -> None:
    """Bằng chứng tokenization: vì sao ViSoBERT thua dù pretrain đúng tiếng Việt."""
    try:
        sizes = _vocab_sizes()
    except Exception:  # pragma: no cover - phụ thuộc môi trường
        sizes = {}

    viso = sizes.get("visobert")
    mbert = sizes.get("mbert")
    if viso is None or mbert is None:
        head = ("ViSoBERT pretrain trên văn bản mạng xã hội với vocab nhỏ hơn mBERT "
                "nhiều lần.")
        inset = ""
    else:
        head = (f"ViSoBERT pretrain trên văn bản mạng xã hội với vocab "
                f"{integer(viso['vocab'])} token, mBERT có {integer(mbert['vocab'])}.")
        inset = (
            '<div class="om-inset">'
            f'<div style="color:var(--color-neutral-700);margin-bottom:4px">'
            f'“{html.esc(_SAMPLE[:34])}…”</div>'
            f'<div>ViSoBERT · {integer(viso["tokens"])} token · '
            f'{html.esc(" · ".join(viso["pieces"]))}</div>'
            f'<div>mBERT · {integer(mbert["tokens"])} token · '
            f'{html.esc(" · ".join(mbert["pieces"]))}</div></div>'
        )

    st.html(html.note_card(
        "Vì sao ViSoBERT thất bại",
        [
            head + " MRC trên Wikipedia đòi biên span chính xác trên văn phong trang "
            "trọng dày đặc tên riêng — đúng thứ mà vocab nhỏ chia vụn nặng nhất.",
            "Pretraining đúng ngôn ngữ không bù được pretraining sai miền.",
        ],
        inset=inset,
    ))


_SAMPLE = "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam."


@st.cache_data(show_spinner=False)
def _vocab_sizes() -> dict[str, dict]:
    """Đếm token thật bằng chính tokenizer của hai model.

    Đo thay vì chép con số từ README: nếu ai đó đổi checkpoint, bằng chứng trên
    màn hình đổi theo.
    """
    from demo.catalog import MODEL_BY_ID
    from transformers import AutoTokenizer

    out: dict[str, dict] = {}
    for model_id in ("visobert", "mbert"):
        model = MODEL_BY_ID[model_id]
        tokenizer = AutoTokenizer.from_pretrained(model.path, use_fast=True)
        pieces = tokenizer.tokenize(_SAMPLE)
        out[model_id] = {"vocab": tokenizer.vocab_size, "tokens": len(pieces),
                         "pieces": pieces[:8]}
    return out
