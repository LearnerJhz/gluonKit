"""Backward-compatible NVIDIA dump_info wrapper."""

from pathlib import Path
from typing import Any, Callable

from gluonkit.backends.nvidia.backend import summarize_nvidia_kernel
from gluonkit.core.triton_dump import dump_triton_kernels


def dump_info(call: Callable[[], object], artifact_dir: str | Path) -> dict[str, Any]:
    return dump_triton_kernels(
        call,
        artifact_dir,
        summarize_fn=summarize_nvidia_kernel,
        expected_extensions=("ttgir", "llir", "ptx"),
    )
