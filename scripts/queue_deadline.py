"""Số học hạn chót cho ``scripts/mps_queue.sh`` — tách ra Python để test được.

Hạn chót khởi chạy: 2026-10-01 20:00 (+07) = 13:00Z. Một bước HUẤN LUYỆN chỉ được
bắt đầu nếu ``bây giờ + thời lượng ước tính`` ≤ hạn chót; không thì bỏ qua và ghi
"skipped: deadline". Bước chấm/thống kê ngắn, không bị chặn.

    python scripts/queue_deadline.py check --hours 2.5        # exit 0 = kịp
    python scripts/queue_deadline.py plan --start 2026-09-27T19:30:00Z
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

DEADLINE = datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)

#: (bước, phase, loại, giờ ước tính, nội dung). "train" bị kiểm hạn chót; train bị
#: bỏ thì không tốn giờ. eval/post của một phase chỉ chạy nếu phase đó có ít nhất
#: một run đã huấn luyện (P3 không có bước train ở đây: đã chạy trước hàng đợi).
PLAN = (
    ("eval_p3", "P3", "eval", 0.15, "run_eval --models mbert-dev visobert-dev --full --preds"),
    ("post_p3", "P3", "post", 0.05, "compute_stats + check_hypotheses + make_numbers"),
    ("p4_smoke_train", "P4", "train", 2.55,
     "P4 smoke 512 (~3 phút) + visobert-len512 (2,5 h) — kiểm hạn chung"),
    ("eval_p4", "P4", "eval", 0.1, "run_eval --models visobert-len512 --full --preds"),
    ("post_p4", "P4", "post", 0.05, "compute_stats + check_hypotheses + make_numbers"),
    ("p5_s43", "P5", "train", 1.75, "mbert-dev-s43 (--seed 43)"),
    ("p5_s44", "P5", "train", 1.75, "mbert-dev-s44 (--seed 44)"),
    ("eval_p5", "P5", "eval", 0.1, "run_eval --models <các seed đã chạy> --full --preds"),
    ("post_p5", "P5", "post", 0.05, "compute_stats + check_hypotheses + make_numbers"),
    ("p6_train", "P6", "train", 2.0, "phobert-dev (256/64, max_answer_len 30)"),
    ("eval_p6", "P6", "eval", 0.1, "run_eval --models phobert-dev --full --preds"),
    ("post_p6", "P6", "post", 0.05, "compute_stats + check_hypotheses + make_numbers"),
)


def parse_utc(text: str) -> datetime:
    if text == "now":
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"cần múi giờ tường minh: {text!r}")
    return dt.astimezone(timezone.utc)


def fits(now: datetime, hours: float, deadline: datetime = DEADLINE) -> bool:
    """Bước dài ``hours`` bắt đầu lúc ``now`` có xong trước ``deadline`` không (≤ là kịp)."""
    return now + timedelta(hours=hours) <= deadline


def simulate(start: datetime, plan=PLAN, deadline: datetime = DEADLINE) -> list[dict]:
    """Lịch dự kiến, cùng quy tắc với mps_queue.sh."""
    has_train = {phase for _, phase, kind, _, _ in plan if kind == "train"}
    trained: set[str] = set()
    rows, t = [], start
    for step, phase, kind, hours, what in plan:
        if kind == "train" and not fits(t, hours, deadline):
            status, end = "skipped: deadline", t
        elif kind != "train" and phase in has_train and phase not in trained:
            status, end = "skipped (no run)", t
        else:
            status, end = "run", t + timedelta(hours=hours)
            if kind == "train":
                trained.add(phase)
        rows.append({"step": step, "phase": phase, "start": t, "end": end,
                     "status": status, "what": what})
        t = end
    return rows


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%MZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--hours", type=float, required=True)
    # MPS_QUEUE_NOW chỉ để test luồng điều khiển không phụ thuộc đồng hồ thật.
    c.add_argument("--now", default=os.environ.get("MPS_QUEUE_NOW", "now"))
    p = sub.add_parser("plan")
    p.add_argument("--start", default="now")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        now = parse_utc(args.now)
        end = now + timedelta(hours=args.hours)
        ok = fits(now, args.hours)
        print(f"{'OK' if ok else 'QUÁ HẠN'}: {_fmt(now)} + {args.hours} h = {_fmt(end)} "
              f"(hạn {_fmt(DEADLINE)})")
        return 0 if ok else 1

    rows = simulate(parse_utc(args.start))
    print(f"Hạn chót khởi chạy: {_fmt(DEADLINE)} (= 2026-10-01 20:00 +07)")
    print(f"{'bước':16s} {'bắt đầu':18s} {'kết thúc':18s} {'trạng thái':18s} nội dung")
    for r in rows:
        print(f"{r['step']:16s} {_fmt(r['start']):18s} {_fmt(r['end']):18s} "
              f"{r['status']:18s} {r['what']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
