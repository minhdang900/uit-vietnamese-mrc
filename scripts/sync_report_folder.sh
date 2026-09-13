#!/usr/bin/env bash
# Gom tài liệu rải rác trong repo về thư mục nộp bài ../06_BaoCao_T11/.
#
#     ./scripts/sync_report_folder.sh
#
# Vì sao là script chứ không phải chép tay: 06_BaoCao_T11/ KHÔNG nằm trong git,
# nên mỗi lần chép tay là một cơ hội để hai bên lệch nhau — đúng kiểu hỏng đã
# giết bản v1 của đồ án. Chạy lại script là thư mục nộp khớp với repo.
#
# Script chỉ GHI ĐÈ những gì nó tạo ra. Nó KHÔNG đụng tới src/ (nguồn LaTeX),
# evidence/ hay bản PDF báo cáo — đó là phần viết tay, không sinh lại được.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(cd "$ROOT/.." && pwd)/06_BaoCao_T11"

[[ -d "$OUT/src/chapters" ]] || { echo "Không thấy $OUT/src/chapters — sai thư mục?"; exit 1; }
cd "$ROOT"

copy() {  # copy <nguồn> <đích tương đối trong OUT>
  [[ -e "$1" ]] || { echo "    bỏ qua (chưa có): $1"; return; }
  mkdir -p "$(dirname "$OUT/$2")"
  cp -R "$1" "$OUT/$2"
  echo "    $2"
}

echo "==> Slide"
# Bản nộp để ở gốc cho dễ thấy; nguồn dựng deck để riêng.
copy slides/CS116_T11_Slide_BaoCao.pptx  CS116_T11_Slide_BaoCao.pptx
copy slides/CS116_T11_Slide_BaoCao.pdf   CS116_T11_Slide_BaoCao.pdf
rm -rf "$OUT/slide-nguon"
copy slides/build_deck.js  slide-nguon/build_deck.js
copy slides/fonts          slide-nguon/fonts
copy slides/README.md      slide-nguon/README.md
copy slides/index.html     slide-nguon/index.html

echo "==> Số liệu (bảng + hình sinh từ results/)"
rm -rf "$OUT/so-lieu"
copy report/assets/table_results.md   so-lieu/table_results.md
copy report/assets/table_results.csv  so-lieu/table_results.csv
copy report/assets/provenance.md      so-lieu/provenance.md
copy report/assets/provenance.csv     so-lieu/provenance.csv
copy report/assets/figures            so-lieu/figures

# index.html trỏ hình bằng đường dẫn tương đối tới report/assets/ trong repo.
# Ở thư mục nộp, hình nằm ở so-lieu/figures, nên phải sửa lại — không sửa thì
# deck HTML mở ra là năm ô ảnh vỡ.
if [[ -f "$OUT/slide-nguon/index.html" ]]; then
  perl -pi -e 's{\.\./report/assets/figures/}{../so-lieu/figures/}g' \
    "$OUT/slide-nguon/index.html"
  echo "    (đã sửa đường dẫn hình trong slide-nguon/index.html)"
fi

echo "==> Hướng dẫn"
rm -rf "$OUT/huong-dan"
copy docs/HUONG_DAN_NHOM.md  huong-dan/HUONG_DAN_NHOM.md
copy README.md               huong-dan/README_repo.md

echo "==> Phụ lục kỹ thuật"
rm -rf "$OUT/phu-luc-ky-thuat"
copy docs/SOLUTION.md       phu-luc-ky-thuat/SOLUTION.md
copy docs/PLAN_TDD.md       phu-luc-ky-thuat/PLAN_TDD.md
copy docs/ARCHITECTURE.md   phu-luc-ky-thuat/ARCHITECTURE.md
copy docs/SYSTEM_DESIGN.md  phu-luc-ky-thuat/SYSTEM_DESIGN.md
copy results/hypotheses.md  phu-luc-ky-thuat/hypotheses.md

echo "==> Ghi nguồn"
{
  echo "Sinh bởi scripts/sync_report_folder.sh"
  echo "repo:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "nhánh:     $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "thời điểm: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Các thư mục slide-nguon/, so-lieu/, huong-dan/, phu-luc-ky-thuat/ được"
  echo "GHI ĐÈ mỗi lần chạy. Sửa ở repo rồi chạy lại, đừng sửa trực tiếp ở đây."
} > "$OUT/NGUON.txt"
echo "    NGUON.txt"

echo
echo "==> Xong: $OUT  ($(du -sh "$OUT" | cut -f1))"
