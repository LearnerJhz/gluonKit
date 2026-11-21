"""Static backend registry. Change `DEFAULT_BACKEND` when deploying on a MetaX C500 machine."""

from gluonkit.backends.metax.backend import MetaxBackend
from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.interfaces.backend import Backend

BACKENDS: dict[str, type[Backend]] = {
    "nvidia": NvidiaBackend,
    "metax": MetaxBackend,
}

# 当前机器使用的固定后端（迁移到沐曦 C500 机器时改为 "metax"）
DEFAULT_BACKEND = "nvidia"


def get_backend(name: str = DEFAULT_BACKEND, *, device: str = "cuda:0") -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; registered: {list(BACKENDS)}")
    return BACKENDS[name](device=device)
