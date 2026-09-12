"""Fine-tune một encoder + QA head trên UIT-ViQuAD 2.0, chạy trên MPS.

Đây là phần mà dự án tiền nhiệm KHÔNG làm được vì bị giới hạn CPU-only, nên
"PhoBERT encoder + QA head" — thứ đề tài T11 nêu tên — chưa từng được thực hiện.

Ví dụ:
    python scripts/finetune.py --model bert-base-multilingual-cased --out models/mbert
    python scripts/finetune.py --model vinai/phobert-base-v2 --out models/phobert
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mrc.data import assert_no_leakage, load_squad_file, reproducible_subset
from mrc.device import pick_device
from mrc.features import prepare_train_features


class QADataset(Dataset):
    """Dataset tối giản; không dùng `datasets` để tránh phụ thuộc thêm."""

    def __init__(self, feats: dict):
        self.n = len(feats["input_ids"])
        self.feats = feats

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": torch.tensor(self.feats["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(self.feats["attention_mask"][i], dtype=torch.long),
            "start_positions": torch.tensor(self.feats["start_positions"][i], dtype=torch.long),
            "end_positions": torch.tensor(self.feats["end_positions"][i], dtype=torch.long),
        }


def evaluate_quick(model_dir, val_examples, device, limit=300, seed=42, max_answer_len=30):
    """EM/F1 thật trên một mẫu con validation, dùng chính pipeline inference.

    Dùng CÙNG hàm lấy mẫu ``subset()`` với ``scripts/run_eval.py`` và cùng seed,
    nên con số trên đường cong huấn luyện SO SÁNH ĐƯỢC với bảng kết quả cuối.

    Trước đây hàm này lấy ``val_examples[:limit]`` — n câu ĐẦU file. Các câu đầu
    thuộc vài article đầu tiên nên mẫu thiên lệch theo chủ đề: mBERT chấm được
    EM 42,00 trên mẫu đó nhưng EM 50,80 trên mẫu ngẫu nhiên cùng kích thước.
    Chênh 8,8 điểm chỉ do cách lấy mẫu.
    """
    from mrc.evaluate import run_evaluation
    from mrc.transformer_qa import TransformerQA

    qa = TransformerQA(str(model_dir), device=device, name=str(model_dir),
                       max_answer_len=max_answer_len)
    r = run_evaluation(qa, reproducible_subset(val_examples, limit, seed=seed), split="validation")
    del qa
    return r["overall"]["EM"], r["overall"]["F1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-size", type=int, default=None, help="None = toàn bộ train")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--doc-stride", type=int, default=128)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--max-answer-len", type=int, default=30,
                    help="Do dai span toi da theo TOKEN; phu thuoc tokenizer. "
                         "Do tren 4000 gold answer: p95=40 voi mBERT nhung 64 "
                         "voi ViSoBERT (vocab 15k). De 30 cho ViSoBERT khien "
                         "25.2%% dap an vang khong the bieu dien duoc.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device={device}  model={args.model}")

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tok.is_fast:
        raise SystemExit(f"{args.model}: cần fast tokenizer để có offset_mapping")

    train_ex = load_squad_file("data/raw/viquad2_train.json")
    val_ex = load_squad_file("data/raw/viquad2_validation.json")
    # Cổng chống leakage: chạy TRƯỚC khi tốn giờ GPU.
    assert_no_leakage(train_ex, val_ex)
    print(f"assert_no_leakage: OK ({len({e.context for e in train_ex})} train ctx, "
          f"{len({e.context for e in val_ex})} val ctx, giao nhau = 0)")

    if args.train_size:
        train_ex = train_ex[: args.train_size]

    t0 = time.time()
    feats = prepare_train_features(train_ex, tok, args.max_length, args.doc_stride)
    print(f"{len(train_ex)} câu hỏi -> {len(feats['input_ids'])} feature "
          f"({time.time()-t0:.0f}s tokenize)")

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    loader = DataLoader(QADataset(feats), batch_size=args.batch_size, shuffle=True)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict] = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches, t_ep = 0.0, 0, time.time()
        optim.zero_grad()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            running += loss.item() * args.grad_accum
            n_batches += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} "
                      f"loss {running/n_batches:.4f} ({time.time()-t_ep:.0f}s)", flush=True)

        train_loss = running / max(1, n_batches)

        # Lưu checkpoint epoch này rồi đánh giá bằng pipeline inference THẬT.
        ck = out_dir / f"epoch{epoch}"
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        em, f1 = evaluate_quick(ck, val_ex, device, args.eval_limit,
                                max_answer_len=args.max_answer_len)

        curve.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                      "val_em": round(em, 2), "val_f1": round(f1, 2),
                      "epoch_seconds": round(time.time() - t_ep, 1)})
        print(f"epoch {epoch}: loss {train_loss:.4f}  val_EM {em:.2f}  val_F1 {f1:.2f}  "
              f"({time.time()-t_ep:.0f}s)", flush=True)

        # Early stopping thực chất: chỉ giữ epoch tốt nhất làm model cuối.
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"  -> epoch {epoch} là tốt nhất, lưu vào {out_dir}")

    (out_dir / "training_curve.json").write_text(
        json.dumps({"model": args.model, "config": vars(args), "curve": curve},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/training_curve_{out_dir.name}.json").write_text(
        json.dumps({"model": args.model, "config": vars(args), "curve": curve},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncurve -> results/training_curve_{out_dir.name}.json")


if __name__ == "__main__":
    main()
