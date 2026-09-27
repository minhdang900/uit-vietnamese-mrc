#!/usr/bin/env bash
# Hàng đợi MPS NỐI TIẾP sau P3: headline eval P3 → P4 → P5 → P6.
#
# Chờ job nền hiện tại (mbert-dev rồi visobert-dev) xong, rồi chạy từng bước dưới
# `caffeinate -dims`, HF_HUB_OFFLINE=1, log vào results/logs/<bước>.log và ghi
# "<bước> exit=<mã> <utc>" vào results/logs/mps_queue.log.
#
# Quy tắc:
#   * Huấn luyện / chấm lỗi (exit ≠ 0) ⇒ DỪNG cả hàng đợi.
#   * Mỗi run chỉ được chấm MỘT lần (run_eval từ chối ghi đè ⇒ cũng là dừng).
#   * P4 smoke 512 OOM ⇒ DỪNG và ghi chú; KHÔNG tự đổi sang batch 6×4 (sai khác đó
#     phải được commit thành bản sửa đăng ký trước).
#   * Trước mỗi bước HUẤN LUYỆN: now + thời lượng ước tính phải ≤ 2026-10-01T13:00Z
#     (scripts/queue_deadline.py); không kịp ⇒ "skipped: deadline", sang bước sau.
#   * Sau mỗi lần chấm: compute_stats + check_hypotheses + make_numbers. Đây là hậu
#     xử lý CPU: check_hypotheses exit 1 nghĩa là CÓ BÁO ĐỘNG (vd. suy sụp — kết quả
#     hợp lệ), nên chỉ ghi lại, không dừng; stats/numbers lỗi cũng chỉ ghi WARN để
#     không giữ MPS rảnh hàng giờ vì một lỗi báo cáo.
#   * Script KHÔNG BAO GIỜ git commit.
#   * finetune.py tự từ chối nếu hypotheses.{md,json} đang sửa dở ⇒ đừng để file
#     giả thuyết bẩn khi hàng đợi đang chạy.
#
#   scripts/mps_queue.sh --dry-run [ISO-start]   # in kế hoạch + giờ dự kiến
#   scripts/mps_queue.sh                          # chạy thật (nên chạy nền)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOGDIR=results/logs
QLOG=$LOGDIR/mps_queue.log
SMOKE=${TMPDIR:-/tmp}/mps_queue_smoke512
export HF_HUB_OFFLINE=1
# Ghi đè được CHỈ để test luồng điều khiển (tests/test_mps_queue.py).
WAIT_PATTERN=${MPS_QUEUE_WAIT_PATTERN:-scripts/finetune.py}
read -r -a CAFFEINATE <<< "${MPS_QUEUE_CAFFEINATE:-caffeinate -dims}"

# caffeinate chỉ exec được chương trình thật, không gọi được hàm shell ⇒ mảng argv.
FT=("$PY" scripts/finetune.py)

# Cấu hình chung = đăng ký P3 (tập dev, lr, epoch).
COMMON=(--epochs 3 --lr 3e-5 --dev-frac 0.1 --dev-group title --dev-seed 42)

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Chờ: pgrep -f scripts/finetune.py rỗng VÀ $LOGDIR/p3_queue.log chứa 'visobert-dev exit='"
  $PY scripts/queue_deadline.py plan --start "${2:-now}"
  echo
  echo "Lệnh huấn luyện:"
  echo "  p4_smoke : finetune --model uitnlp/visobert --out $SMOKE/model --results-dir $SMOKE/results --max-length 512 --doc-stride 128 --max-answer-len 64 --train-size 300 --dev-size 300 --epochs 1 --lr 3e-5"
  echo "  p4_train : finetune --model uitnlp/visobert --out models/visobert-len512 --max-length 512 --doc-stride 128 --max-answer-len 64 ${COMMON[*]}"
  echo "  p5_s43   : finetune --model bert-base-multilingual-cased --out models/mbert-dev-s43 --seed 43 ${COMMON[*]}"
  echo "  p5_s44   : finetune --model bert-base-multilingual-cased --out models/mbert-dev-s44 --seed 44 ${COMMON[*]}"
  echo "  p6_train : finetune --model vinai/phobert-base-v2 --out models/phobert-dev --max-length 256 --doc-stride 64 --max-answer-len 30 ${COMMON[*]}"
  exit 0
fi

mkdir -p "$LOGDIR"
utc() { date -u +%FT%TZ; }
note() { echo "$* $(utc)" >> "$QLOG"; }

# run <bước> <lệnh…>: chạy dưới caffeinate, log riêng, ghi exit vào QLOG.
run() {
  local step=$1; shift
  "${CAFFEINATE[@]}" "$@" > "$LOGDIR/$step.log" 2>&1
  local code=$?
  note "$step exit=$code"
  return $code
}

# must: lỗi ⇒ dừng hàng đợi.
must() {
  local step=$1
  run "$@" || { note "STOP: $step thất bại — hàng đợi dừng"; exit 1; }
}

# deadline <bước> <giờ>: 0 nếu kịp; không kịp thì ghi skipped.
deadline() {
  local step=$1 hours=$2
  if $PY scripts/queue_deadline.py check --hours "$hours" >> "$QLOG" 2>&1; then
    return 0
  fi
  note "$step skipped: deadline"
  return 1
}

# post <phase>: hậu xử lý sau mỗi lần chấm (không dừng hàng đợi).
post() {
  local phase=$1
  run "stats_$phase" $PY scripts/compute_stats.py || note "WARN: stats_$phase lỗi (tiếp tục)"
  run "check_$phase" $PY scripts/check_hypotheses.py results \
    || note "check_$phase: có BÁO ĐỘNG hoặc lỗi — xem $LOGDIR/check_$phase.log (tiếp tục)"
  run "numbers_$phase" $PY scripts/make_numbers.py || note "WARN: numbers_$phase lỗi (tiếp tục)"
}

# ── 0. chờ P3 (mbert-dev → visobert-dev) xong ─────────────────────────
note "queue: chờ P3"
until ! pgrep -f "$WAIT_PATTERN" > /dev/null \
      && grep -q "visobert-dev exit=" "$LOGDIR/p3_queue.log" 2>/dev/null; do
  sleep "${MPS_QUEUE_POLL:-60}"
done
note "queue: bắt đầu"

# ── 1. headline eval P3, MỘT lần ─────────────────────────────────────
must eval_p3 $PY scripts/run_eval.py --models mbert-dev visobert-dev --full --preds
post p3

# ── 2–3. P4: smoke 512 rồi visobert-len512 ───────────────────────────
if deadline p4 2.55; then
  rm -rf "$SMOKE"
  if ! run p4_smoke "${FT[@]}" --model uitnlp/visobert --out "$SMOKE/model" \
        --results-dir "$SMOKE/results" --max-length 512 --doc-stride 128 \
        --max-answer-len 64 --train-size 300 --dev-size 300 --epochs 1 --lr 3e-5; then
    if grep -qiE "out of memory|MPS backend out of memory|OOM" "$LOGDIR/p4_smoke.log"; then
      note "STOP: P4 smoke 512 OOM — cần commit bản sửa đăng ký (batch 6×4) TRƯỚC khi chạy P4"
    else
      note "STOP: P4 smoke lỗi (không phải OOM) — xem $LOGDIR/p4_smoke.log"
    fi
    exit 1
  fi
  must p4_train "${FT[@]}" --model uitnlp/visobert --out models/visobert-len512 \
    --max-length 512 --doc-stride 128 --max-answer-len 64 "${COMMON[@]}"
  must eval_p4 $PY scripts/run_eval.py --models visobert-len512 --full --preds
  post p4
fi

# ── 4. P5: hai seed mBERT ────────────────────────────────────────────
P5_RUNS=()
for seed in 43 44; do
  if deadline "p5_s$seed" 1.75; then
    must "p5_s$seed" "${FT[@]}" --model bert-base-multilingual-cased \
      --out "models/mbert-dev-s$seed" --seed "$seed" "${COMMON[@]}"
    P5_RUNS+=("mbert-dev-s$seed")
  fi
done
if (( ${#P5_RUNS[@]} )); then
  must eval_p5 $PY scripts/run_eval.py --models "${P5_RUNS[@]}" --full --preds
  post p5
fi

# ── 5. P6: PhoBERT ───────────────────────────────────────────────────
if deadline p6 2.0; then
  must p6_train "${FT[@]}" --model vinai/phobert-base-v2 --out models/phobert-dev \
    --max-length 256 --doc-stride 64 --max-answer-len 30 "${COMMON[@]}"
  must eval_p6 $PY scripts/run_eval.py --models phobert-dev --full --preds
  post p6
fi

note "queue: xong"
