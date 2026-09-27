"""Fine-tune một encoder + QA head trên UIT-ViQuAD 2.0 (MPS / CUDA / CPU).

Script này CỐ Ý mỏng: nó chỉ nối dây. Mọi quyết định — lịch học, chọn epoch và
ngưỡng null, chẩn đoán overfitting, lấy mẫu đánh giá — nằm trong ``mrc.training``
và ``mrc.threshold``, nơi chúng là hàm thuần và được test không cần GPU.

Chọn epoch và τ trên tập DEV tách từ train theo article (``--dev-group title``),
KHÔNG BAO GIỜ trên validation: validation chỉ được tải để kiểm leakage, rồi bị
bỏ. Bản v1 chọn epoch trên 300 câu của chính validation rồi báo cáo trên
validation — con số đó lạc quan một cách không đo được.

Chạy thật (ghi vào ``results/``) đòi đăng ký trước: ``HEAD:results/hypotheses.json``
phải liệt kê ``run_id`` và file giả thuyết phải sạch (xem ``mrc.prereg``). Chạy
thử thì trỏ ``--results-dir`` và ``--out`` ra ngoài repo; khi đó không kiểm.

Ví dụ:
    python scripts/finetune.py --model bert-base-multilingual-cased \
        --out models/mbert-dev --epochs 3 --lr 3e-5
    python scripts/finetune.py --model uitnlp/visobert --out models/visobert-dev \
        --epochs 3 --lr 3e-5 --max-answer-len 64
    python scripts/finetune.py --model bert-base-multilingual-cased \
        --out $TMPDIR/smoke_model --results-dir $TMPDIR/smoke_results \
        --train-size 300 --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
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

from mrc.data import assert_no_leakage, compute_stats, load_squad_file, split_by_context
from mrc.evaluate import _git_commit
from mrc.threshold import DEFAULT_TAU_GRID, select_epoch_and_tau, sweep
from mrc.training import (
    EpochRecord,
    evaluate_checkpoint,
    select_best_epoch,
    summarise_curve,
)

#: Thư mục kết quả thật của repo. Chỉ khi ghi vào đây mới đòi đăng ký trước.
REPO_RESULTS = _ROOT / "results"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tên checkpoint HuggingFace")
    ap.add_argument("--out", required=True, help="thư mục lưu model tốt nhất")
    ap.add_argument("--run-id", default=None, help="mặc định = tên thư mục --out")
    ap.add_argument("--results-dir", default=str(REPO_RESULTS),
                    help="nơi ghi curve, dev preds, threshold, split_dev (chạy thử: "
                         "trỏ ra ngoài repo; khi đó bỏ qua kiểm đăng ký trước)")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--train-size", type=int, default=None,
                    help="cắt train SAU khi tách dev; None = toàn bộ")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--dev-group", choices=("title", "context"), default="title")
    ap.add_argument("--dev-seed", type=int, default=42)
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="số câu dev chấm mỗi epoch; None = toàn bộ dev")
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.run_id is None:
        args.run_id = Path(args.out).name
    return args


def is_real_results_dir(results_dir: str | Path) -> bool:
    return Path(results_dir).resolve() == REPO_RESULTS.resolve()


def preregistration(args) -> dict:
    """Kiểm đăng ký trước (chỉ khi ghi vào ``results/`` thật). Chạy TRƯỚC mọi thứ khác."""
    if not is_real_results_dir(args.results_dir):
        print(f"results-dir={args.results_dir} ngoài repo ⇒ chạy thử, bỏ qua kiểm đăng ký trước")
        return {"prereg_commit": None, "prereg_commit_time": None, "smoke": True}
    from mrc.prereg import PreregError, assert_preregistered

    try:
        info = assert_preregistered(args.run_id, _ROOT)
    except PreregError as e:
        raise SystemExit(f"TỪ CHỐI chạy {args.run_id!r}: {e}")
    print(f"đăng ký trước OK: {args.run_id} @ {info['prereg_commit'][:10]} "
          f"({info['prereg_commit_time']})")
    return {**info, "smoke": False}


def load_and_split(args, results_dir: Path) -> tuple[list, list]:
    """Tách dev khỏi train, kiểm leakage ba chiều, ghi ``split_dev.json``.

    Chỉ trả về (train, dev): validation được tải DUY NHẤT để kiểm leakage, nên
    không đường nào phía sau có thể vô tình chấm hay chọn trên nó.
    """
    data_dir = Path(args.data_dir)
    train_file = data_dir / "viquad2_train.json"
    train_all = load_squad_file(train_file)
    val_ex = load_squad_file(data_dir / "viquad2_validation.json")

    train_ex, dev_ex = split_by_context(train_all, val_frac=args.dev_frac,
                                        seed=args.dev_seed, group=args.dev_group)
    # Cổng chống leakage chạy TRƯỚC khi tốn giờ GPU — cả ba cặp.
    assert_no_leakage(train_ex, dev_ex)
    assert_no_leakage(train_ex, val_ex)
    assert_no_leakage(dev_ex, val_ex)
    print(f"assert_no_leakage OK (train/dev, train/val, dev/val) — "
          f"dev: {len({e.title for e in dev_ex})} article, "
          f"{len({e.context for e in dev_ex})} context, {len(dev_ex)} câu")
    del val_ex

    if args.train_size:
        train_ex = train_ex[: args.train_size]

    split_info = {
        "run_id": args.run_id,
        "group": args.dev_group,
        "dev_frac": args.dev_frac,
        "dev_seed": args.dev_seed,
        "train_file": str(train_file),
        "train_file_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "train_size_after_split": args.train_size,
        "train_stats": compute_stats(train_ex),
        "dev_stats": compute_stats(dev_ex),
        "dev_titles": sorted({e.title for e in dev_ex}),
        "dev_qids": [e.qid for e in dev_ex],
    }
    _write_json(results_dir / f"split_dev_{args.run_id}.json", split_info)
    return train_ex, dev_ex


def finalize_selection(args, curve: list[EpochRecord], dev_records: dict[int, list],
                       out_dir: Path, results_dir: Path, dev_n: int) -> dict:
    """Chọn (epoch, τ) CÙNG LÚC trên dev, ghi ``selection.json`` và ``threshold_{run}.json``."""
    sweeps = {epoch: sweep(recs, DEFAULT_TAU_GRID) for epoch, recs in dev_records.items()}
    chosen = select_epoch_and_tau(sweeps)
    at_zero = next(r for r in sweeps[chosen["epoch"]] if r["tau"] == 0.0)
    grid_edge = chosen["tau"] in (DEFAULT_TAU_GRID[0], DEFAULT_TAU_GRID[-1])

    selection = {
        "run_id": args.run_id,
        "epoch": chosen["epoch"],
        "tau": chosen["tau"],
        "selected_on": "dev",
        "criterion": "F1",
        "tie_rule": "earlier epoch, then smallest |tau|, then smaller tau",
        "dev_metrics": {k: chosen[k] for k in ("EM", "F1", "empty_rate", "n")},
        "dev_metrics_tau0": {k: at_zero[k] for k in ("EM", "F1", "empty_rate", "n")},
        "tau_at_grid_edge": grid_edge,
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "max_answer_len": args.max_answer_len,
        "model_name": args.model,
        "seed": args.seed,
        "dev_n": dev_n,
        "dev_seed": args.dev_seed,
        "dev_frac": args.dev_frac,
        "group": args.dev_group,
        "best_epoch_tau0": select_best_epoch(curve),
    }
    _write_json(out_dir / "selection.json", selection)
    _write_json(results_dir / f"threshold_{args.run_id}.json", {
        "run_id": args.run_id,
        "tau_grid": list(DEFAULT_TAU_GRID),
        "selection": selection,
        "sweeps": {str(k): v for k, v in sweeps.items()},
    })
    return selection


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    started_utc = datetime.now(timezone.utc).isoformat()
    results_dir = Path(args.results_dir)

    # Đăng ký trước kiểm TRƯỚC khi tải dữ liệu hay model.
    prereg = preregistration(args)

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from mrc.device import pick_device
    from mrc.features import prepare_train_features
    from mrc.training import QADataset, compute_schedule

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device={device}  model={args.model}  run_id={args.run_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(
            f"{args.model} không có fast tokenizer ⇒ không có offset_mapping ⇒ "
            "không map được token span về ký tự gốc. Không dùng được cho extractive QA."
        )

    train_ex, dev_ex = load_and_split(args, results_dir)

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
    dev_records: dict[int, list] = {}

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

        # Chấm bằng CHÍNH pipeline inference dùng cho bảng kết quả cuối, với CÙNG
        # cấu hình cửa sổ như lúc huấn luyện, trên DEV.
        from mrc.transformer_qa import TransformerQA

        predictor = TransformerQA(str(checkpoint), device=device,
                                  max_length=args.max_length, doc_stride=args.doc_stride,
                                  max_answer_len=args.max_answer_len,
                                  null_threshold=0.0, name=str(checkpoint))
        preds_path = results_dir / f"preds_{args.run_id}_dev_epoch{epoch}.jsonl"
        scored = evaluate_checkpoint(dev_ex, predictor, limit=args.eval_limit,
                                     seed=args.seed, preds_path=preds_path)
        del predictor
        dev_records[epoch] = scored["records"]

        curve.append(EpochRecord(epoch=epoch, train_loss=total_loss / max(1, n_batches),
                                 val_em=scored["em"], val_f1=scored["f1"],
                                 seconds=time.time() - t_epoch))
        print(f"epoch {epoch}: loss {curve[-1].train_loss:.4f}  "
              f"dev_EM {scored['em']:.2f}  dev_F1 {scored['f1']:.2f}  "
              f"rỗng {scored['empty_rate']:.1f}%  "
              f"(n={scored['n']}, {curve[-1].seconds:.0f}s)", flush=True)

    selection = finalize_selection(args, curve, dev_records, out_dir, results_dir,
                                   dev_n=len(dev_records[1]))
    best_ckpt = out_dir / f"epoch{selection['epoch']}"
    for name in best_ckpt.iterdir():
        (out_dir / name.name).write_bytes(name.read_bytes())
    print(f"\n(epoch, τ) chọn trên dev = ({selection['epoch']}, {selection['tau']}) "
          f"dev F1 {selection['dev_metrics']['F1']:.2f} -> sao chép vào {out_dir}")
    if selection["tau_at_grid_edge"]:
        print("  *** CẢNH BÁO: τ chọn được nằm ở MÉP lưới — lưới có thể quá hẹp.")

    config = {
        **vars(args),
        "run_id": args.run_id,
        "prereg_commit": prereg["prereg_commit"],
        "prereg_commit_time": prereg["prereg_commit_time"],
        "commit": _git_commit(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "eval_split": "dev",
        "dev_group": args.dev_group,
        "seed": args.seed,
    }
    summary = summarise_curve(curve, config=config)
    summary["selection"] = selection
    print(f"chẩn đoán (τ=0): {summary['diagnosis']['reason']}")
    if summary["diagnosis"]["degenerate_collapse"]:
        print("  *** CẢNH BÁO: EM ≈ F1 ở mọi epoch — model có thể đã suy sụp về "
              "'luôn trả rỗng'. Kiểm tra tỉ lệ dự đoán rỗng trước khi báo cáo.")

    for path in (out_dir / "training_curve.json",
                 results_dir / f"training_curve_{args.run_id}.json"):
        _write_json(path, summary)
    print(f"curve -> {results_dir / f'training_curve_{args.run_id}.json'}")


if __name__ == "__main__":
    main()
