# Xuất xứ của mọi con số trong báo cáo

Bất biến #3: mỗi số ⟶ một file `results/*.json` ⟶ một commit hash.
Bảng này sinh tự động; đừng sửa tay.

| model | split | n | device | torch | platform | commit | date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF Baseline | validation | 500 | mps | 2.14.0 | macOS-26.4.1-arm64-arm-64bit | 590e775 | 12/09/2026 |
| visobert (fine-tuned) | validation | 500 | mps | 2.14.0 | macOS-26.4.1-arm64-arm-64bit | c63b28c | 12/09/2026 |
| XLM-R (squad2, zero-shot) | validation | 500 | mps | 2.14.0 | macOS-26.4.1-arm64-arm-64bit | c63b28c | 12/09/2026 |
| mbert (fine-tuned) | validation | 500 | mps | 2.14.0 | macOS-26.4.1-arm64-arm-64bit | 590e775 | 12/09/2026 |

Câu impossible chiếm **27.80%** mẫu đã chấm — đây là lý do báo cáo tách `answerable_only` và `impossible_only`: metric tổng trộn hai kỹ năng khác nhau.
