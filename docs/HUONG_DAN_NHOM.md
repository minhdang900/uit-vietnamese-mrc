# Hướng dẫn cho nhóm — chạy, demo, bắt kịp báo cáo

Dành cho thành viên chưa theo sát phần code. Đọc mục 1 là chạy được; mục 2 là
thuyết trình được; mục 3 là biết báo cáo đang ở đâu.

---

## 0. Bản đồ — cái gì nằm ở đâu

Dự án nằm ở **hai thư mục song song**, và đây là chỗ dễ nhầm nhất:

```
Python-ML/
├── uit-vietnamese-mrc/     ← code, demo, test, dữ liệu, hình  (CÓ trên GitHub)
└── 06_BaoCao_T11/          ← báo cáo LaTeX + PDF nộp          (KHÔNG trên GitHub)
```

> ⚠️ **`06_BaoCao_T11/` không nằm trong git.** Clone repo về sẽ **không** có báo
> cáo. Muốn sửa báo cáo phải lấy thư mục đó qua Drive/USB, không phải `git pull`.

| Cần gì | Vào đâu |
|---|---|
| Chạy demo, chạy test | `uit-vietnamese-mrc/` |
| Sửa báo cáo | `06_BaoCao_T11/02_NGUON_BAO_CAO/chapters/*.tex` |
| Slide thuyết trình | `uit-vietnamese-mrc/slides/index.html` |
| Bảng số, hình cho báo cáo | `uit-vietnamese-mrc/report/assets/` |
| Kết quả gốc (nguồn của mọi con số) | `uit-vietnamese-mrc/results/*.json` |

---

## 1. Chạy — 5 phút, chỉ cần Docker

Không cần cài Python, không cần tạo venv, không cần tải model.

```bash
cd uit-vietnamese-mrc

docker compose run --rm tests    # 423 test, ~2 giây
docker compose up app            # mở http://localhost:8501
docker compose down              # dừng
```

Lần đầu chạy `up` sẽ dựng ảnh (khoảng 5–10 phút, cần mạng). Những lần sau vài giây.

Chưa có Docker: <https://docs.docker.com/get-docker/> — cài Docker Desktop, mở
nó lên, rồi chạy lại.

### Không dùng Docker được thì sao

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/fetch_data.py          # tải dữ liệu, ~16 MB
streamlit run app/streamlit_app.py
pytest -m "not slow"
```

Chạy thẳng trên máy thì **nhanh hơn** Docker khoảng 4 lần (MPS so với CPU), nhưng
kết quả trả về y hệt.

### Model ở đâu

`models/` (3,4 GB) và `data/raw/` **không** nằm trong git. Không có chúng thì
demo vẫn chạy — thanh bên sẽ báo checkpoint nào còn thiếu, và XLM-R zero-shot
cùng TF-IDF baseline vẫn dùng được bình thường. Muốn có mBERT đã fine-tune thì
xin file từ người đã huấn luyện, hoặc lấy từ gói `delivery/`.

---

## 2. Demo — kịch bản thuyết trình

Mở slide trước: `slides/index.html` (mở thẳng bằng trình duyệt).
Phím `←` `→` chuyển slide, phím `S` bật ghi chú người nói, phím `P` in ra PDF.

Sáu màn hình, mỗi màn một địa chỉ riêng — **mở thẳng được**, không cần bấm lần lượt:

| Màn hình | Địa chỉ | Nói gì |
|---|---|---|
| Hỏi đáp | `/` | Màn chính. Xem mục 2.1 |
| Kết quả | `/ket-qua` | Bảng bốn mô hình, EM/F1 |
| Phân tích lỗi | `/phan-tich-loi` | Lỗi tập trung ở đâu |
| Dữ liệu | `/du-lieu` | Thống kê ba split, test là blind set |
| So sánh model | `/so-sanh` | Cùng câu hỏi, bốn mô hình trả lời |
| Huấn luyện | `/huan-luyen` | Đường cong loss và val EM/F1 |

### 2.1 Đoạn quan trọng nhất — tập trước cho quen

Đây là phần đáng giá nhất của cả demo:

1. Vào màn **Hỏi đáp**.
2. Bấm chip đoạn văn **"Đế quốc La Mã Thần thánh · impossible"**.
3. Để ngưỡng ở **+0,0**. Model **vẫn trả lời** ("24 bàn tay") — và ứng dụng tự
   cảnh báo rằng câu này thuộc nhóm impossible, gold là rỗng.
4. Kéo thanh **NGƯỠNG TỪ CHỐI** sang **+1,5**.
5. Model chuyển sang **"Model từ chối trả lời"**. Biên độ đổi dấu từ −0,6 sang
   +0,9, và kim chỉ vượt qua ranh giới quyết định.

**Câu nên nói:** *"Khoảng 30% câu hỏi trong ViQuAD 2.0 không có đáp án. Biết khi
nào KHÔNG trả lời là một kỹ năng riêng, tách hẳn với việc tìm đúng span. Thanh
này cho thấy mô hình đang đứng ở đâu so với ranh giới quyết định — chứ không chỉ
cho thấy đáp án cuối cùng."*

### 2.2 Ba điều nên nói ở màn Kết quả

Đừng đọc từng con số. Chỉ vào ba điều:

1. **Baseline F1 cao nhưng EM gần bằng 0** — nó trả về *cả một câu*, gold là *cụm
   vài từ*. Overlap thì có, trùng khít thì không. Đây là minh hoạ trực quan cho
   việc EM và F1 đo hai thứ khác nhau.
2. **mBERT thắng XLM-R nhưng KHÔNG phải vì đọc giỏi hơn** — trên câu *có đáp án*,
   XLM-R zero-shot thực ra tốt hơn. mBERT thắng vì biết khi nào không nên trả lời.
3. **ViSoBERT thất bại** — model tiếng Việt chuyên biệt thua cả model đa ngữ.
   Nguyên nhân: vocab 15.002 token, pretrain trên mạng xã hội, trong khi MRC
   Wikipedia đòi hỏi nhận diện tên riêng. *Pretrain đúng ngôn ngữ là điều kiện
   cần, không phải điều kiện đủ.*

### 2.3 Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn |
|---|---|
| Sao không đánh giá trên test split? | Test split của ViQuAD 2.0 là **blind set** — toàn bộ 7.301 câu có gold rỗng. Chấm trên đó thì model trả rỗng sẽ được EM 100%. Code có `assert_gradeable()` chặn việc này. |
| Số này lấy ở đâu ra? | `results/*.json`, mỗi file ghi kèm commit hash, thiết bị, thời điểm. Phụ lục A của báo cáo là bảng truy vết đầy đủ. |
| Nhãn single/multi-sentence lấy ở đâu? | **Nhóm tự gán bằng heuristic**, ViQuAD không có nhãn này. Báo cáo nói rõ ở §7.3. |
| Sao demo chậm hơn số trong báo cáo? | Trong Docker chạy CPU, báo cáo đo trên MPS. Đáp án y hệt, chỉ khác độ trễ. |

---

## 3. Báo cáo — đang ở đâu, sửa thế nào

**File nộp:** `06_BaoCao_T11/01_NOP_BAI/Bao_Cao_Do_An_CS116_T11.pdf` — A4, **55 trang**,
theo đúng cấu trúc mẫu báo cáo cuối kỳ của trường.

### 3.1 Sửa và biên dịch

```bash
cd 06_BaoCao_T11/02_NGUON_BAO_CAO
# sửa chapters/*.tex
latexmk -xelatex main.tex
cp main.pdf ../01_NOP_BAI/Bao_Cao_Do_An_CS116_T11.pdf
```

Cần TeX Live có XeLaTeX. Chương nào ở file nào:

| File | Nội dung |
|---|---|
| `00_front.tex` | Bìa trong, lời cảm ơn, tóm tắt |
| `01_gioi_thieu.tex` | Giới thiệu, đặt vấn đề, mục tiêu |
| `02_co_so_ly_thuyet.tex` | Cơ sở lý thuyết, các mô hình |
| `03_phuong_phap.tex` | Kiến trúc, bốn bất biến, TDD |
| `04_thuc_nghiem.tex` | Dữ liệu, kết quả, phân tích lỗi, demo |
| `05_ket_luan.tex` | Kết luận, giới hạn, hướng phát triển |
| `06`–`08` | Tài liệu tham khảo, phụ lục |

### 3.2 Vừa sửa gì (đợt kiểm toán gần nhất)

| Sửa | Từ | Thành |
|---|---|---|
| Tên thành viên trên bìa | Võ Cẩm Thu | **Vỏ Cẩm Thu** |
| Số test | 251 (210+41) | **474 (423+51)**, độ phủ 75% |
| Kích thước từ vựng ViSoBERT | 15.004 | **15.002** |
| Ảnh demo | bản một trang cũ | bản sáu màn hình, có minh hoạ ngưỡng |
| PDF nộp | cũ, 49 trang | dựng lại, **55 trang** |

> **Vì sao đổi 15.004 → 15.002:** `config.json` khai báo ma trận nhúng 15.004
> hàng, nhưng tokenizer chỉ sinh ra 15.002 token phân biệt — hai hàng nhúng không
> token nào ánh xạ tới. "Kích thước từ vựng" đúng là **15.002**. Bảng trong báo
> cáo giờ có cả hai dòng kèm giải thích.

### 3.3 Bắt kịp nội dung — đọc theo thứ tự này

1. `00_front.tex` phần **Tóm tắt** — 1 trang, nắm toàn bộ kết quả.
2. Báo cáo §6.3 và §6.4 — hai kết quả đáng chú ý nhất (mBERT thắng nhờ biết từ
   chối; ViSoBERT thất bại vì lệch miền).
3. §11.1 — **bốn giới hạn bắt buộc**. Thầy nhiều khả năng hỏi phần này.
4. Phụ lục A — bảng truy vết, để biết con số nào từ đâu.

---

## 4. Ba quy tắc không được phá

Dự án này có một tiền lệ: **phiên bản trước thất bại vì báo cáo trình bày những
con số chưa từng được sinh ra** — mâu thuẫn giữa README, báo cáo và ứng dụng.
Ba quy tắc dưới đây tồn tại để điều đó không lặp lại.

1. **Không gõ số bằng tay vào báo cáo hay slide.** Số phải đến từ
   `report/assets/` (nhúng bằng `<!-- include: -->`) hoặc từ hình sinh bởi
   `scripts/make_figures.py`. Có test tự động cấm việc này — gõ "EM 50,80" vào
   template là test đỏ ngay.
2. **Không sửa tay `results/*.json`.** Muốn số khác thì chạy lại
   `scripts/run_eval.py`. Mỗi file mang commit hash của lần chạy sinh ra nó.
3. **Test phải xanh trước khi commit.** `docker compose run --rm tests`.

Chi tiết bốn bất biến: `docs/PLAN_TDD.md` §10.1.

---

## 5. Việc còn mở

| Việc | Trạng thái |
|---|---|
| PR #2 (báo cáo + slide) | **Đang mở**, chưa merge |
| Độ dài báo cáo | 55 trang, vượt mốc 30–50 ghi trong `PLAN_TDD.md` §10.2. Mẫu của trường là 42 trang — có thể vẫn ổn, nên hỏi thầy |
| Tên trong `05_DoAn_T11_MRC/` | Vẫn là "Võ Cẩm Thu" ở 4 file. Đó là dự án v1 đã bỏ, chưa sửa |
| `docs/ARCHITECTURE.md`, `docs/SYSTEM_DESIGN.md` | Chưa commit, chưa ai rà lại |
| `results/streamlit.log` | Đang sửa đổi trong working tree, cố ý chưa commit (log chạy demo) |

---

## 6. Lỗi hay gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'mrc'` | Chạy bằng `docker compose`, hoặc đặt `PYTHONPATH=src` |
| Demo báo thiếu checkpoint | Bình thường khi không có `models/`. XLM-R và baseline vẫn chạy |
| `docker compose up` dựng lại ảnh rất lâu | Bình thường ở lần đầu. Lần sau dùng lại ảnh cũ |
| Cổng 8501 bị chiếm | `MRC_PORT=8600 docker compose up app` |
| LaTeX báo `Bad character code (-1)` | Có dấu gạch ngang trong `\code{}` ở cỡ chữ nhỏ. Viết lại câu cho tránh |
| Màn hình "Dữ liệu" trống | Chưa có `data/raw/`. Chạy `docker compose run --rm fetch-data` |

---

## 7. Đóng gói nộp bài

```bash
docker compose build app
./scripts/make_delivery.sh        # → delivery/ (~1,5 GB, kèm checkpoint)
```

Người chấm chỉ cần Docker: giải nén, chạy `./run.sh`, xong. Không cần Python,
không cần mạng, không dựng lại ảnh.
