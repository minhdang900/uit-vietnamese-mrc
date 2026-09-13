"""Hợp đồng đóng gói: Dockerfile, compose.yaml, .dockerignore.

Đây là đường nộp bài — người chấm chạy ``docker compose run --rm tests`` và
``docker compose up app`` chứ không dựng venv. Cho tới giờ đường đó chỉ được
kiểm bằng tay một lần, nên mọi quyết định trong đó đều có thể bị xoá mà không
ai biết. File này ghim chúng lại.

Toàn bộ test ở đây ĐỌC FILE CẤU HÌNH, không gọi docker: chúng phải chạy được
cả trên máy lẫn BÊN TRONG container (nơi không có docker daemon), và phải nằm
trong bộ test nhanh cùng 389 test còn lại.
"""

from __future__ import annotations

from pathlib import Path

#: Gốc repo, suy từ vị trí file này (``<root>/tests/test_docker.py``).
ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_and_compose_ship_inside_the_image():
    """Ảnh phải mang theo chính hai file đã dựng ra nó.

    Không phải để chạy chúng, mà để các test dưới đây ĐỌC được khi pytest chạy
    trong container. Thiếu chúng thì ``docker compose run --rm tests`` bỏ qua
    toàn bộ hợp đồng đóng gói — đúng lúc cần kiểm nhất.
    """
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "compose.yaml").is_file()


def dockerfile_instructions() -> list[str]:
    """Các chỉ thị của Dockerfile: bỏ chú thích, nối tiếp dòng ``\\``.

    Đọc thô từng dòng thì không dùng được, và điều đó được phát hiện lúc xem
    test đỏ: khẳng định "không có TARGETARCH" vớ phải chính đoạn chú thích GIẢI
    THÍCH lỗi, nên đỏ cả khi Dockerfile đã đúng. Còn khẳng định trên một dòng
    đơn thì BỎ LỌT lỗi thật, vì ``TARGETARCH`` nằm ở dòng ``RUN if`` phía trên
    còn lệnh pip nằm ở dòng nối tiếp.

    Một chỉ thị ``RUN`` là một đơn vị — phải ghép lại rồi mới xét.
    """
    out: list[str] = []
    buffer = ""
    for raw in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        buffer += " " + line.removesuffix("\\").strip() if buffer else line
        if line.endswith("\\"):
            buffer = buffer.removesuffix("\\").strip()
            continue
        out.append(" ".join(buffer.split()))
        buffer = ""
    if buffer:
        out.append(" ".join(buffer.split()))
    return out


def test_torch_comes_from_the_cpu_index_on_every_architecture():
    """Cài torch từ index CPU, KHÔNG rẽ nhánh theo kiến trúc.

    Hồi quy cho một lỗi có thật: bản Dockerfile đầu chỉ ép CPU khi
    ``TARGETARCH = amd64``, vì tưởng wheel aarch64 vốn đã CPU-only. Không phải —
    wheel aarch64 của torch 2.x cũng khai báo các gói ``nvidia-*``. Ảnh arm64 đo
    được 3.341 MB ``nvidia/`` cộng 817 MB ``triton/`` nằm chết trong
    site-packages 6,1 GB, cho một container không service nào xin GPU.

    Điều kiện theo kiến trúc quay lại = 4,1 GB quay lại, và không có gì kêu.
    """
    instructions = dockerfile_instructions()
    install = [i for i in instructions if "download.pytorch.org/whl/cpu" in i]
    assert install, "Dockerfile không cài torch từ index CPU"

    for instruction in install:
        assert "TARGETARCH" not in instruction, (
            f"torch CPU nằm sau điều kiện kiến trúc — ảnh arm64 sẽ kéo lại "
            f"4,1 GB thư viện CUDA: {instruction!r}"
        )
        assert " if " not in f" {instruction} ", (
            f"torch CPU nằm trong một nhánh điều kiện: {instruction!r}"
        )


def test_cache_dir_exists_and_is_owned_by_the_app_user_before_user_switch():
    """Thư mục HF_HOME phải được tạo và ``chown`` TRƯỚC chỉ thị ``USER``.

    Hồi quy cho lỗi thứ hai có thật: docker chỉ sao chép quyền sở hữu từ ảnh
    sang một named volume rỗng khi đường dẫn ĐÃ TỒN TẠI trong ảnh. Thiếu bước
    này, volume ``hf-cache`` sinh ra thuộc root, container chạy dưới uid 1000
    gặp PermissionError, và màn hình "So sánh model" lặng lẽ mất XLM-R — demo
    vẫn chạy nên không ai nhận ra trừ khi mở đúng màn hình đó.

    Phải nằm trước ``USER app``: sau đó thì không còn quyền ``chown``.
    """
    instructions = dockerfile_instructions()
    home = next((i for i in instructions if i.startswith("ENV HF_HOME=")), None)
    assert home is not None, "Dockerfile không đặt HF_HOME"
    cache_dir = home.removeprefix("ENV HF_HOME=").split()[0]

    user_at = next(
        (n for n, i in enumerate(instructions) if i.startswith("USER ")), None
    )
    assert user_at is not None, "Dockerfile không đổi sang người dùng thường"

    prepared = [
        n for n, i in enumerate(instructions)
        if i.startswith("RUN ") and cache_dir in i and "chown" in i
    ]
    assert prepared, (
        f"{cache_dir} không được tạo+chown trong ảnh — named volume gắn vào đó "
        f"sẽ thuộc root và người dùng app không ghi được"
    )
    assert min(prepared) < user_at, (
        "chown thư mục cache nằm SAU chỉ thị USER, lúc đó đã mất quyền"
    )


def compose() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


def dockerignore_patterns() -> set[str]:
    """Các mẫu loại trừ, bỏ dòng trống và chú thích."""
    return {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


#: Thư mục dữ liệu của dự án. ``fetch-data`` là service DUY NHẤT được ghi, và
#: chỉ vào ``./data`` — nó tồn tại chính để sinh ra thư mục đó.
DATA_DIRS = ("./models", "./data", "./results")


def test_demo_mounts_project_data_read_only():
    """Bất biến số một của dự án, do kernel bắt buộc chứ không phải quy ước.

    "Không con số nào được viết tay" chỉ vững khi demo KHÔNG ghi được vào
    ``results/`` và ``data/raw/``. Bỏ ``:ro`` thì bất biến đó chết lặng lẽ:
    không có gì hỏng ngay, chỉ là từ đó trở đi demo có thể sửa chính con số nó
    đang trình bày.
    """
    app = compose()["services"]["app"]
    mounted = [v for v in app["volumes"] if v.split(":")[0] in DATA_DIRS]
    assert len(mounted) == len(DATA_DIRS), (
        f"demo phải mount đủ {DATA_DIRS}, đang có {mounted}"
    )
    for volume in mounted:
        assert volume.endswith(":ro"), (
            f"mount của demo không read-only: {volume!r} — bất biến 'demo không "
            f"sinh ra con số nào' không còn được bảo đảm"
        )


def test_only_the_data_fetcher_may_write_to_the_repo():
    """Mọi service khác ``fetch-data`` chỉ được mount dữ liệu read-only."""
    for name, service in compose()["services"].items():
        for volume in service.get("volumes", []):
            source = volume.split(":")[0]
            if source not in DATA_DIRS:
                continue
            if name == "fetch-data" and source == "./data":
                continue
            assert volume.endswith(":ro"), (
                f"service {name!r} ghi được vào {source!r}: {volume!r}"
            )


def test_plain_up_starts_the_demo_and_nothing_else():
    """``docker compose up`` chỉ được dựng demo.

    Mọi service khác phải nằm sau profile. Không thì người chấm gõ ``up`` và
    nhận về một cửa sổ log trộn lẫn pytest, trình tải dữ liệu và Streamlit —
    còn ``fetch-data`` thì ghi đè ``data/raw/`` mà không ai yêu cầu.
    """
    services = compose()["services"]
    default = [n for n, s in services.items() if not s.get("profiles")]
    assert default == ["app"], (
        f"``docker compose up`` sẽ dựng {default}, chỉ được phép là ['app']"
    )


def test_every_service_runs_a_command_that_exists():
    """Không service nào trỏ vào một script đã bị đổi tên hoặc xoá."""
    for name, service in compose()["services"].items():
        for token in service.get("command", []):
            if token.endswith(".py"):
                assert (ROOT / token).is_file(), (
                    f"service {name!r} chạy {token!r} nhưng file không tồn tại"
                )


def test_build_context_excludes_the_heavy_directories():
    """``models/`` (3,4 GB) và ``data/`` không được lọt vào build context.

    Cả hai đều nằm trong .gitignore và được mount lúc chạy. Nếu .dockerignore
    quên chúng, mỗi lần build phải gửi 3,4 GB sang daemon rồi nướng luôn vào
    ảnh — hỏng cả tốc độ lẫn tính tái lập, mà không có thông báo lỗi nào, chỉ
    là build "tự dưng chậm".
    """
    for heavy in ("models/", "data/"):
        assert heavy in dockerignore_patterns(), f".dockerignore thiếu {heavy!r}"


def test_docker_files_are_not_excluded_from_the_build_context():
    """Nghịch lại test đầu tiên: hai file này PHẢI vào được ảnh.

    Test đầu kiểm chúng có mặt lúc chạy; test này kiểm không ai vô tình loại
    chúng lại trong .dockerignore, vì triệu chứng khi đó chỉ hiện ra lúc build
    ảnh mới chứ không phải lúc chạy test trên máy.
    """
    assert "Dockerfile" not in dockerignore_patterns()
    assert "compose.yaml" not in dockerignore_patterns()


def test_delivery_script_packages_the_same_image_compose_runs():
    """Gói nộp bài và compose.yaml phải nói về CÙNG một tag ảnh.

    ``make_delivery.sh`` chạy ``docker save <tag>``, còn người chấm chạy
    ``docker compose up`` — nếu hai bên lệch tag thì compose không thấy ảnh vừa
    load và lặng lẽ **dựng lại từ đầu**, mất nhiều phút và cần mạng. Đúng thứ
    không được phép xảy ra trên máy người chấm.
    """
    script = ROOT / "scripts" / "make_delivery.sh"
    assert script.is_file(), "thiếu scripts/make_delivery.sh"

    tag = compose()["services"]["app"]["image"]
    assert tag in script.read_text(encoding="utf-8"), (
        f"make_delivery.sh không đóng gói tag {tag!r} mà compose.yaml dùng"
    )
