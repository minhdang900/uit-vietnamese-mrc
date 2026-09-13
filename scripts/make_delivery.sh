#!/usr/bin/env bash
# Đóng gói toàn bộ dự án thành một thư mục nộp được.
#
#     ./scripts/make_delivery.sh              # đầy đủ, có checkpoint (~1,7 GB)
#     ./scripts/make_delivery.sh --no-models  # gọn, không checkpoint (~600 MB)
#
# Sản phẩm là delivery/ — người chấm chỉ cần Docker, không cần Python, không cần
# mạng, không cần dựng ảnh. Xem delivery/HUONG_DAN.md sau khi chạy.
set -euo pipefail

IMAGE_TAG="vietnamese-mrc:latest"          # PHẢI khớp services.app.image trong
                                           # compose.yaml — pin bằng
                                           # tests/test_docker.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/delivery"
WITH_MODELS=1

[[ "${1:-}" == "--no-models" ]] && WITH_MODELS=0

cd "$ROOT"

# ── 0. Điều kiện cần ────────────────────────────────────────────────────────
command -v docker >/dev/null || { echo "Cần docker."; exit 1; }
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Chưa có ảnh $IMAGE_TAG. Dựng trước:  docker compose build app"; exit 1
fi

echo "==> Dọn $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/source"

# ── 1. Mã nguồn ─────────────────────────────────────────────────────────────
# Lấy từ ``git archive`` chứ không phải ``cp -r``: chỉ những gì đã commit mới ra,
# nên .venv/, __pycache__/, models/ và mọi rác cục bộ không lọt vào gói — và cái
# nộp đi đúng bằng cái nằm trong lịch sử git.
echo "==> Mã nguồn (git archive HEAD)"
git archive --format=tar HEAD | tar -x -C "$OUT/source"
echo "    commit $(git rev-parse --short HEAD)"

# ── 2. Ảnh Docker ───────────────────────────────────────────────────────────
# ``docker save`` + gzip: người chấm ``docker load`` là chạy được ngay, không
# dựng lại và không cần mạng. compose.yaml có khai báo ``image:`` nên
# ``docker compose up`` dùng luôn ảnh vừa load thay vì build.
echo "==> Ảnh Docker ($IMAGE_TAG) — bước này lâu nhất"
docker save "$IMAGE_TAG" | gzip -1 > "$OUT/vietnamese-mrc-image.tar.gz"

# ── 3. Checkpoint (tuỳ chọn) ────────────────────────────────────────────────
# CHỈ checkpoint cuối. models/*/epoch*/ là các bản giữa của quá trình huấn
# luyện — 1,3 GB mà demo không bao giờ nạp, vì transformers đọc thẳng ở thư mục
# gốc của model. Bỏ chúng đi là khác biệt giữa gói 1,7 GB và gói 4 GB.
if [[ $WITH_MODELS -eq 1 && -d models ]]; then
  echo "==> Checkpoint (bỏ qua epoch*/ — bản giữa, demo không dùng)"
  for model in models/*/; do
    name="$(basename "$model")"
    mkdir -p "$OUT/source/models/$name"
    find "$model" -maxdepth 1 -type f -exec cp {} "$OUT/source/models/$name/" \;
    echo "    $name: $(du -sh "$OUT/source/models/$name" | cut -f1)"
  done
else
  echo "==> Bỏ qua checkpoint — demo sẽ chạy với XLM-R zero-shot và TF-IDF"
  mkdir -p "$OUT/source/models"
fi

# ── 4. Dữ liệu ──────────────────────────────────────────────────────────────
# data/raw/ không nằm trong git (gitignore) nhưng màn hình "Dữ liệu" đếm trực
# tiếp từ nó. Chép vào để gói chạy được offline hoàn toàn.
if [[ -d data/raw ]]; then
  echo "==> Dữ liệu UIT-ViQuAD 2.0"
  mkdir -p "$OUT/source/data/raw"
  cp data/raw/*.json "$OUT/source/data/raw/"
fi

# ── 5. Script chạy ──────────────────────────────────────────────────────────
cat > "$OUT/run.sh" <<'RUNNER'
#!/usr/bin/env bash
# Chạy dự án. Chỉ cần Docker — không cần Python, không cần mạng.
#
#     ./run.sh            # mở demo  (mặc định)
#     ./run.sh test       # chạy bộ test nhanh
#     ./run.sh test-full  # chạy toàn bộ test
#     ./run.sh stop       # dừng demo
set -euo pipefail

IMAGE_TAG="vietnamese-mrc:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$HERE/vietnamese-mrc-image.tar.gz"

command -v docker >/dev/null || { echo "Chưa cài Docker: https://docs.docker.com/get-docker/"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker chưa chạy. Mở Docker Desktop rồi thử lại."; exit 1; }

# Nạp ảnh nếu máy chưa có. Có rồi thì bỏ qua, để chạy lần hai không mất một phút.
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "==> Nạp ảnh Docker (chỉ lần đầu, khoảng một phút)"
  docker load -i "$ARCHIVE"
fi

cd "$HERE/source"

case "${1:-demo}" in
  demo)
    echo "==> Khởi động demo"
    docker compose up -d --wait app
    echo
    echo "    Demo đang chạy:  http://localhost:8501"
    echo "    Dừng lại:        ./run.sh stop"
    ;;
  test)      docker compose run --rm tests ;;
  test-full) docker compose run --rm tests-full ;;
  stop)      docker compose down && echo "    Đã dừng." ;;
  *)
    echo "Dùng: ./run.sh [demo|test|test-full|stop]"; exit 1 ;;
esac
RUNNER
chmod +x "$OUT/run.sh"

# ── 6. Hướng dẫn ────────────────────────────────────────────────────────────
cat > "$OUT/HUONG_DAN.md" <<'GUIDE'
# Hệ thống đọc hiểu tiếng Việt — CS116 T11

Máy chấm **chỉ cần Docker**. Không cần cài Python, không cần mạng, không cần
dựng lại ảnh.

## Chạy

```bash
./run.sh            # mở demo tại http://localhost:8501
./run.sh test       # chạy bộ test nhanh
./run.sh test-full  # chạy toàn bộ test (có tải model)
./run.sh stop       # dừng demo
```

Lần đầu mất khoảng một phút để nạp ảnh Docker. Những lần sau chạy ngay.

Chưa có Docker: https://docs.docker.com/get-docker/ — mở Docker Desktop trước
khi chạy `./run.sh`.

## Demo có gì

Sáu màn hình, mỗi màn một địa chỉ riêng — mở thẳng được chỗ cần:

| Màn hình | Địa chỉ |
|---|---|
| Hỏi đáp | http://localhost:8501/ |
| Kết quả | http://localhost:8501/ket-qua |
| Phân tích lỗi | http://localhost:8501/phan-tich-loi |
| Dữ liệu | http://localhost:8501/du-lieu |
| So sánh model | http://localhost:8501/so-sanh |
| Huấn luyện | http://localhost:8501/huan-luyen |

Ở màn **Hỏi đáp**, kéo thanh "ngưỡng từ chối" từ +0,0 lên +1,5 rồi hỏi lại đoạn
có nhãn `impossible`: model chuyển từ trả lời sang từ chối. Đó là bài học trung
tâm của ViQuAD 2.0 — khoảng 30% câu không có đáp án, và biết khi nào **không**
trả lời là một kỹ năng riêng.

## Trong gói này có gì

| Đường dẫn | Nội dung |
|---|---|
| `run.sh` | Script chạy — điểm bắt đầu |
| `vietnamese-mrc-image.tar.gz` | Ảnh Docker đã dựng sẵn |
| `source/` | Toàn bộ mã nguồn, đúng bằng commit ghi ở `MANIFEST.txt` |
| `source/report/` | Vật liệu báo cáo: bảng, xuất xứ, hình |
| `source/results/` | Kết quả đánh giá thật, kèm commit và thiết bị |
| `source/models/` | Checkpoint đã fine-tune (nếu gói có kèm) |
| `MANIFEST.txt` | Commit, ngày đóng gói, danh sách tệp |

## Lưu ý khi chấm

- Số trong báo cáo đến từ `source/results/*.json`; mỗi file ghi kèm commit hash,
  thiết bị và thời điểm chạy. Demo đọc **cùng** những file đó, nên hai bên không
  thể lệch nhau.
- Kết quả được chấm trên **validation**, không phải test. Test split của ViQuAD
  2.0 là blind set — toàn bộ đáp án vàng rỗng, không chấm được.
- Trong Docker, suy luận chạy bằng **CPU** nên chậm hơn số đo trong báo cáo
  (đo trên MPS). Đáp án trả về y hệt, chỉ khác độ trễ.
- Nếu gói không kèm checkpoint, demo vẫn chạy với XLM-R zero-shot và TF-IDF
  baseline; sidebar sẽ báo model nào còn thiếu.
GUIDE

# ── 7. Manifest ─────────────────────────────────────────────────────────────
# Ghi commit vào chính gói: không có nó thì "gói này ứng với bản nào" là câu hỏi
# không trả lời được sau khi nộp.
{
  echo "Vietnamese MRC — CS116 T11"
  echo "commit:     $(git rev-parse HEAD)"
  echo "đóng gói:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ảnh:        $IMAGE_TAG"
  echo "checkpoint: $([[ $WITH_MODELS -eq 1 ]] && echo "có" || echo "không")"
  echo
  echo "--- tệp ---"
  cd "$OUT" && find . -type f | sort
} > "$OUT/MANIFEST.txt"

echo
echo "==> Xong: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "    Thử ngay:  cd delivery && ./run.sh"
