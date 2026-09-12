"""Chọn thiết bị tính toán. Ưu tiên MPS (GPU của Apple Silicon)."""

from __future__ import annotations

__all__ = ["pick_device", "device_info"]


def pick_device(prefer: str | None = None) -> str:
    """Trả về ``"mps"``, ``"cuda"`` hoặc ``"cpu"``.

    Trên Apple Silicon, MPS cho tốc độ gấp nhiều lần CPU và là lý do dự án này
    fine-tune được thật, thay vì chỉ chạy inference như phiên bản trước.
    """
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info() -> dict:
    """Thông tin thiết bị để ghi vào provenance của kết quả."""
    import platform

    import torch

    return {
        "device": pick_device(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
