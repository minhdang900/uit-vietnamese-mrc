"""Fine-tune một encoder + QA head trên UIT-ViQuAD 2.0 (MPS / CUDA / CPU).

Script này CỐ Ý mỏng: nó chỉ nối dây. Mọi quyết định — lịch học, chọn epoch tốt
nhất, chẩn đoán overfitting, lấy mẫu đánh giá — nằm trong ``mrc.training``, nơi
chúng là hàm thuần và được test không cần GPU.

Bản viết trước gộp tất cả vào ``main()``. Không phần nào test được, và lỗi lấy mẫu
thiên lệch trong đánh giá đã trôi qua không bị phát hiện cho tới khi hai phép đo
độc lập mâu thuẫn nhau — sai 10 điểm EM.

Ví dụ:
    python scripts/finetune.py --model bert-base-multilingual-cased --out models/mbert
    python scripts/finetune.py --model uitnlp/visobert --out models/visobert \
        --epochs 3 --lr 5e-5 --max-answer-len 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Đặt sys.path TRƯỚC mọi import của dự án. Chạy ``python scripts/x.py`` chỉ đưa
# ``scripts/`` vào path, không bối cảnh nào tự thấy ``src/``. Vài dòng ở đây rẻ
# hơn việc bắt người chấm phải ``pip install -e .`` trước khi README chạy được.
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import assert_no_leakage, load_squad_file
from mrc.device import pick_device
from mrc.features import prepare_train_features
from mrc.training import (
    EpochRecord,
    QADataset,
    compute_schedule,
    evaluate_checkpoint,
    select_best_epoch,
    summarise_curve,
)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="tên checkpoint HuggingFace")
    ap.add_argument("--out", required=True, help="thư mục lưu model tốt nhất")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--train-size", type=int, default=None, help="None = toàn bộ train")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--doc-stride", type=int, default=128)
    ap.add_argument(
        "--max-answer-len", type=int, default=30,
        help="Độ dài span tối đa theo TOKEN — phụ thuộc tokenizer. Đo trên 4.000 "
             "gold answer: p95 = 40 với mBERT nhưng 64 với ViSoBERT (vocab 15k).",
    )
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    import torch
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device={device}  model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(
            f"{args.model} không có fast tokenizer ⇒ không có offset_mapping ⇒ "
            "không map được token span về ký tự gốc. Không dùng được cho extractive QA."
        )

    data_dir = Path(args.data_dir)
    train_ex = load_squad_file(data_dir / "viquad2_train.json")
    val_ex = load_squad_file(data_dir / "viquad2_validation.json")

    # Cổng chống leakage chạy TRƯỚC khi tốn giờ GPU.
    assert_no_leakage(train_ex, val_ex)
    print(f"assert_no_leakage OK — {len({e.context for e in train_ex})} train ctx / "
          f"{len({e.context for e in val_ex})} val ctx, giao nhau = 0")

    if args.train_size:
        train_ex = train_ex[: args.train_size]

    t0 = time.time()
    features = prepare_train_features(train_ex, tokenizer, args.max_length, args.doc_stride)
    dataset = QADataset(features)
    print(f"{len(train_ex)} câu hỏi -> {len(dataset)} feature ({time.time() - t0:.0f}s)")

    schedule = compute_schedule(
        n_features=len(dataset), batch_size=args.batch_size, grad_accum=args.grad_accum,
        epochs=args.epochs, warmup_ratio=args.warmup_ratio,
    )
    print(f"lịch học: {schedule.steps_per_epoch} bước/epoch, "
          f"{schedule.total_steps} tổng, {schedule.warmup_steps} warmup")

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, schedule.warmup_steps, schedule.total_steps
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve: list[EpochRecord] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches, t_epoch = 0.0, 0, time.time()
        optimiser.zero_grad()

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
            total_loss += loss.item() * args.grad_accum
            n_batches += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} "
                      f"loss {total_loss / n_batches:.4f} "
                      f"({time.time() - t_epoch:.0f}s)", flush=True)

        checkpoint = out_dir / f"epoch{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)

        # Chấm bằng CHÍNH pipeline inference dùng cho bảng kết quả cuối.
        from mrc.transformer_qa import TransformerQA

        predictor = TransformerQA(str(checkpoint), device=device,
                                  max_answer_len=args.max_answer_len,
                                  name=str(checkpoint))
        scored = evaluate_checkpoint(val_ex, predictor, limit=args.eval_limit,
                                     seed=args.seed)
        del predictor

        curve.append(EpochRecord(epoch=epoch, train_loss=total_loss / max(1, n_batches),
                                 val_em=scored["em"], val_f1=scored["f1"],
                                 seconds=time.time() - t_epoch))
        print(f"epoch {epoch}: loss {curve[-1].train_loss:.4f}  "
              f"val_EM {scored['em']:.2f}  val_F1 {scored['f1']:.2f}  "
              f"(n={scored['n']}, {curve[-1].seconds:.0f}s)", flush=True)

    # Early stopping thực chất: chỉ epoch tốt nhất được lưu làm model cuối.
    best = select_best_epoch(curve)
    best_ckpt = out_dir / f"epoch{best}"
    for name in best_ckpt.iterdir():
        (out_dir / name.name).write_bytes(name.read_bytes())
    print(f"\nepoch tốt nhất = {best} -> sao chép vào {out_dir}")

    summary = summarise_curve(curve, config=vars(args))
    print(f"chẩn đoán: {summary['diagnosis']['reason']}")
    if summary["diagnosis"]["degenerate_collapse"]:
        print("  *** CẢNH BÁO: EM ≈ F1 ở mọi epoch — model có thể đã suy sụp về "
              "'luôn trả rỗng'. Kiểm tra tỉ lệ dự đoán rỗng trước khi báo cáo.")

    for path in (out_dir / "training_curve.json",
                 Path("results") / f"training_curve_{out_dir.name}.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"curve -> results/training_curve_{out_dir.name}.json")


if __name__ == "__main__":
    main()
