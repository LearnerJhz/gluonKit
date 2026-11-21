"""Structured diagnostics when GPU device initialization fails (sandbox vs env vs driver)."""

import os
from pathlib import Path
from typing import Any

_GPU_DEV_CANDIDATES = (
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/mxcd",
    "/dev/kfd",
    "/dev/dri/renderD128",
)


def classify_gpu_init_failure(dev_root: str | Path = "/dev", env: dict[str, str] | None = None) -> tuple[str, str]:
    """Classify why `torch.cuda` failed to initialize into (category, actionable_hint)."""
    environ = os.environ if env is None else env
    for var in ("CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        val = environ.get(var)
        if val is not None and val.strip() in ("", "-1", "none", "NoDevFiles"):
            return (
                "env_hidden",
                f"{var}={val!r} hides all GPUs; unset {var} or set it to a valid device index such as '0'.",
            )

    root = Path(dev_root)
    candidates = [
        root / "nvidia0",
        root / "nvidiactl",
        root / "mxcd",
        root / "kfd",
        root / "dri" / "renderD128",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return (
            "device_node_missing",
            "No GPU device nodes (/dev/nvidia*, /dev/mxcd*, /dev/dri/renderD*) are mounted or visible in /dev.",
        )

    inaccessible = [p for p in existing if not os.access(p, os.R_OK | os.W_OK)]
    nv0_only = (root / "nvidia0") in existing and not (root / "nvidiactl").exists()
    if inaccessible or nv0_only:
        blocked = ", ".join(str(p) for p in inaccessible) or str(root / "nvidiactl")
        return (
            "device_node_inaccessible",
            f"GPU device node access is blocked ({blocked}); if running in a sandbox, "
            "retry with sandbox bypass or grant read/write access to the GPU device nodes.",
        )

    return (
        "driver_init_failed",
        "GPU device nodes exist and are accessible, but the CUDA/MACA driver failed to initialize.",
    )


def get_device_properties_with_diagnosis(device: str = "cuda:0", *, platform: str = "nvidia") -> Any:
    """Query `torch.cuda.get_device_properties(device)` and attach structured diagnosis on failure."""
    import torch

    try:
        return torch.cuda.get_device_properties(device)
    except Exception as exc:
        category, hint = classify_gpu_init_failure()
        raise RuntimeError(f"[{platform} probe diagnosis: {category}] {exc} | Hint: {hint}") from exc
