"""Static backend registry. Change `DEFAULT_BACKEND` to select the active backend."""

from gluonkit.backends.metax.backend import MetaxBackend
from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.interfaces.backend import Backend

BACKENDS: dict[str, type[Backend]] = {
    "nvidia": NvidiaBackend,
    "metax": MetaxBackend,
}

# 当前机器使用的固定后端与设备
DEFAULT_BACKEND = "nvidia"
DEFAULT_DEVICE = "cuda:0"


def get_backend(name: str = DEFAULT_BACKEND, *, device: str = DEFAULT_DEVICE) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; registered: {list(BACKENDS)}")
    return BACKENDS[name](device=device)
