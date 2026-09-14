#!/usr/bin/env bash
# Gom tài liệu rải rác trong repo về thư mục nộp bài ../06_BaoCao_T11/.
#
#     ./scripts/sync_report_folder.sh
#
# Vì sao là script chứ không phải chép tay: 06_BaoCao_T11/ KHÔNG nằm trong git,
# nên mỗi lần chép tay là một cơ hội để hai bên lệch nhau — đúng kiểu hỏng đã
# giết bản v1 của đồ án. Chạy lại script là thư mục nộp khớp với repo.
#
# Script chỉ GHI ĐÈ những gì nó tạo ra. Nó KHÔNG đụng tới 02_NGUON_BAO_CAO/ (nguồn LaTeX),
# 05_BANG_CHUNG/ hay bản PDF báo cáo — đó là phần viết tay, không sinh lại được.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(cd "$ROOT/.." && pwd)/06_BaoCao_T11"

[[ -d "$OUT/02_NGUON_BAO_CAO/chapters" ]] \
  || { echo "Không thấy $OUT/02_NGUON_BAO_CAO/chapters — sai thư mục?"; exit 1; }
cd "$ROOT"

copy() {  # copy <nguồn> <đích tương đối trong OUT>
  [[ -e "$1" ]] || { echo "    bỏ qua (chưa có): $1"; return; }
  mkdir -p "$(dirname "$OUT/$2")"
  cp -R "$1" "$OUT/$2"
  echo "    $2"
}

echo "==> Slide"
# Bản nộp vào 01_NOP_BAI; nguồn dựng deck để riêng ở 03_NGUON_SLIDE.
copy slides/CS116_T11_Slide_BaoCao.pptx  01_NOP_BAI/CS116_T11_Slide_BaoCao.pptx
copy slides/CS116_T11_Slide_BaoCao.pdf   01_NOP_BAI/CS116_T11_Slide_BaoCao.pdf
rm -rf "$OUT/03_NGUON_SLIDE"
copy slides/build_deck.js  03_NGUON_SLIDE/build_deck.js
copy slides/fonts          03_NGUON_SLIDE/fonts
copy slides/README.md      03_NGUON_SLIDE/README.md
copy slides/index.html     03_NGUON_SLIDE/index.html

echo "==> Số liệu (bảng + hình sinh từ results/)"
rm -rf "$OUT/04_SO_LIEU"
copy report/assets/table_results.md   04_SO_LIEU/table_results.md
copy report/assets/table_results.csv  04_SO_LIEU/table_results.csv
copy report/assets/provenance.md      04_SO_LIEU/provenance.md
copy report/assets/provenance.csv     04_SO_LIEU/provenance.csv
copy report/assets/figures            04_SO_LIEU/figures

# index.html trỏ hình bằng đường dẫn tương đối tới report/assets/ trong repo.
# Ở thư mục nộp, hình nằm ở 04_SO_LIEU/figures, nên phải sửa lại — không sửa thì
# deck HTML mở ra là năm ô ảnh vỡ.
if [[ -f "$OUT/03_NGUON_SLIDE/index.html" ]]; then
  perl -pi -e 's{\.\./report/assets/figures/}{../04_SO_LIEU/figures/}g' \
    "$OUT/03_NGUON_SLIDE/index.html"
  echo "    (đã sửa đường dẫn hình trong 03_NGUON_SLIDE/index.html)"
fi

echo "==> Hướng dẫn"
rm -rf "$OUT/06_HUONG_DAN"
copy docs/HUONG_DAN_NHOM.md  06_HUONG_DAN/HUONG_DAN_NHOM.md
copy README.md               06_HUONG_DAN/README_repo.md

echo "==> Phụ lục kỹ thuật"
rm -rf "$OUT/07_PHU_LUC_KY_THUAT"
copy docs/SOLUTION.md       07_PHU_LUC_KY_THUAT/SOLUTION.md
copy docs/PLAN_TDD.md       07_PHU_LUC_KY_THUAT/PLAN_TDD.md
copy docs/ARCHITECTURE.md   07_PHU_LUC_KY_THUAT/ARCHITECTURE.md
copy docs/SYSTEM_DESIGN.md  07_PHU_LUC_KY_THUAT/SYSTEM_DESIGN.md
copy results/hypotheses.md  07_PHU_LUC_KY_THUAT/hypotheses.md

echo "==> Ghi nguồn"
{
  echo "Sinh bởi scripts/sync_report_folder.sh"
  echo "repo:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "nhánh:     $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "thời điểm: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Các thư mục 03_NGUON_SLIDE/, 04_SO_LIEU/, 06_HUONG_DAN/, 07_PHU_LUC_KY_THUAT/ được"
  echo "GHI ĐÈ mỗi lần chạy. Sửa ở repo rồi chạy lại, đừng sửa trực tiếp ở đây."
} > "$OUT/NGUON.txt"
echo "    NGUON.txt"

echo
echo "==> Xong: $OUT  ($(du -sh "$OUT" | cut -f1))"
