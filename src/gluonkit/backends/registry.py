"""Backend registration and automatic platform detection (NVIDIA CUDA vs MetaX C500 MACA)."""

import os

from gluonkit.interfaces.backend import Backend

SUPPORTED_BACKENDS = ("auto", "nvidia", "metax")


def detect_backend_name(device: str = "cuda:0") -> str:
    """Detect whether the runtime environment is MetaX MACA (C500) or NVIDIA CUDA."""
    if os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or os.environ.get("MCTRITON_HOME"):
        return "metax"
    try:
        import torch

        version_text = (getattr(torch, "__version__", "") or "").lower()
        if "maca" in version_text or "metax" in version_text:
            return "metax"
        if getattr(torch.version, "maca", None):
            return "metax"
        if torch.cuda.is_available():
            dev_name = (torch.cuda.get_device_name(device) or "").lower()
            if "metax" in dev_name or "c500" in dev_name or "maca" in dev_name:
                return "metax"
    except Exception:
        pass
    return "nvidia"


def get_backend(name: str | None = None, *, device: str = "cuda:0") -> Backend:
    """Instantiate the requested SIMT GPU backend (`auto`, `nvidia`, or `metax`)."""
    requested = (name or os.environ.get("GK_BACKEND") or "auto").strip().lower()
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {requested!r}; choose from {SUPPORTED_BACKENDS}")
    resolved = detect_backend_name(device) if requested == "auto" else requested
    if resolved == "metax":
        from gluonkit.backends.metax.backend import MetaxBackend

        return MetaxBackend(device=device)
    from gluonkit.backends.nvidia.backend import NvidiaBackend

    return NvidiaBackend(device=device)
