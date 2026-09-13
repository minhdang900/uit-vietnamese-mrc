# syntax=docker/dockerfile:1.7

# Một ảnh cho cả ba việc: chạy test, chạy demo, tải dữ liệu.
#
# Ba ảnh riêng sẽ cần đúng một bộ thư viện, nên tách ra chỉ tạo chỗ cho chúng
# lệch nhau âm thầm — và "test xanh nhưng demo đỏ" là đúng loại lỗi không được
# phép xảy ra lúc chấm. Compose chọn việc bằng ``command``, không bằng ảnh.

FROM python:3.12-slim-bookworm AS base

# PYTHONDONTWRITEBYTECODE: mã nguồn được bind-mount, ghi .pyc vào đó chỉ làm bẩn
# cây thư mục của host. PYTHONUNBUFFERED: log của Streamlit và pytest ra thẳng
# ``docker compose logs`` thay vì kẹt trong buffer cho tới khi tiến trình chết.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# curl là cho HEALTHCHECK ở dưới. Không cài compiler: mọi pin trong
# requirements.txt đều có wheel dựng sẵn, và nếu một ngày nào đó không còn thì
# ta muốn build gãy ồn ào chứ không muốn nó im lặng biên dịch mất 20 phút.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch: ép bản CPU, trên MỌI kiến trúc ─────────────────────────────────
# ``torch==2.14.0`` lấy từ PyPI kéo theo cả chồng wheel CUDA của NVIDIA và
# Triton mà container này không bao giờ chạm tới — không service nào ở đây xin
# GPU. Đo trong ảnh trước khi sửa: nvidia/ 3.341 MB + triton/ 817 MB = 4,1 GB
# chết trong site-packages 6,1 GB.
#
# Và KHÔNG chỉ riêng amd64: bản dựng đầu tiên của Dockerfile này chỉ ép CPU khi
# TARGETARCH=amd64, vì tưởng wheel aarch64 vốn đã CPU-only. Sai — wheel aarch64
# linux của torch 2.x cũng khai báo các gói nvidia-*. Chính máy arm64 dựng ra ảnh
# mang đủ 4,1 GB CUDA đó. Nên điều kiện theo kiến trúc bị bỏ hẳn.
#
# Index CPU đặt tên bản này là ``2.14.0+cpu``. Theo PEP 440, một specifier không
# có phần local (``==2.14.0`` trong requirements.txt) khớp với mọi nhãn local —
# nên pip ở bước dưới thấy torch đã thoả và không kéo lại bản PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

# Lớp riêng cho requirements: sửa mã nguồn không phải cài lại 2 GB thư viện.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ``pyproject.toml`` copy kèm vì pytest đọc [tool.pytest.ini_options] từ đó —
# testpaths, pythonpath và marker ``slow`` đều nằm trong file này.
COPY pyproject.toml ./
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY results/ ./results/
COPY README.md ./

# Báo cáo và slide: nhẹ (~330 kB), và phải có mặt để hai test "cấm gõ tay số đo"
# CHẠY THẬT trong container. Thiếu chúng thì hai test đó lặng lẽ skip — đúng kiểu
# xanh giả đã gặp một lần với .dockerignore.
COPY report/ ./report/
COPY slides/ ./slides/

# Chính hai file đã dựng ra ảnh này. ``tests/test_docker.py`` đọc chúng để ghim
# các quyết định đóng gói, và test đó chạy BÊN TRONG container — không mang theo
# thì ``docker compose run --rm tests`` bỏ qua đúng phần cần kiểm nhất.
#
# Đặt sau cùng và tách khỏi mọi COPY khác: sửa compose.yaml chỉ dựng lại một lớp
# 4 kB, không đụng tới lớp pip.
COPY Dockerfile compose.yaml .dockerignore ./

# ``models/`` và ``data/raw/`` là mount point, không phải nội dung ảnh. Tạo sẵn
# thư mục rỗng để khi KHÔNG mount gì thì demo vẫn chạy: sidebar báo checkpoint
# còn thiếu, XLM-R zero-shot và baseline TF-IDF vẫn dùng được.
RUN mkdir -p models data/raw

# Chạy không cần root. UID 1000 khớp với người dùng đầu tiên trên đa số máy
# Linux, nên các bind-mount đọc-ghi (nếu có) không rơi vào tay root.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app

# Thư mục cache phải TỒN TẠI trong ảnh và thuộc về ``app`` TRƯỚC khi compose gắn
# named volume vào đó. Docker chỉ sao chép quyền sở hữu từ ảnh sang volume rỗng
# khi đường dẫn đã có sẵn; thiếu dòng này thì volume sinh ra thuộc root và
# container chạy dưới uid 1000 gặp PermissionError ngay lần đầu tải XLM-R —
# demo không chết, nhưng màn hình "So sánh model" mất một model.
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

USER app

# HF_HOME trỏ vào một named volume trong compose: XLM-R zero-shot tải từ Hub
# ~1,1 GB, và không ai muốn tải lại mỗi lần ``docker compose up``.
ENV HF_HOME=/home/app/.cache/huggingface \
    PYTHONPATH=/app:/app/src

EXPOSE 8501

# Endpoint sức khoẻ có sẵn của Streamlit. Compose dùng nó cho ``--wait``, nên
# "up xong" nghĩa là trang thật sự phục vụ được, không chỉ là tiến trình còn sống.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py"]
