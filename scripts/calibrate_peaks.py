"""One-off peak calibration: measure achievable DRAM bandwidth and matmul throughput.

Writes calibrated peaks directly to `./.gluonkit/peaks.json`::

    .venv/bin/python scripts/calibrate_peaks.py [--items gbps,tflops_fp16,...]
"""

import argparse
from contextlib import contextmanager
import os
from typing import Any, Callable

os.environ.setdefault("KINETO_LOG_LEVEL", "6")

from gluonkit.backends.registry import get_backend  # noqa: E402
from gluonkit.core.roofline import peaks_path, save_peaks  # noqa: E402
from gluonkit.core.storage import normalize_path  # noqa: E402
from gluonkit.interfaces.backend import Backend  # noqa: E402
from gluonkit.output import capture_run, emit_json  # noqa: E402

MiB = 1024 * 1024


def _measure_ms(fn: Callable[[], object], backend: Backend, *, iters: int, warmup: int) -> float:
    data = backend.profile(fn, iters, warmup)
    if not data.kernel_name or data.avg_ms <= 0:
        raise RuntimeError("calibration workload launched no measurable GPU kernel")
    return data.avg_ms


def _calibrate_gbps(backend: Backend) -> float:
    import torch

    props = torch.cuda.get_device_properties(backend.device)
    free_bytes, _ = torch.cuda.mem_get_info(backend.device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    nbytes = min(max(256 * MiB, 16 * l2_bytes), int(free_bytes * 0.4))
    nbytes -= nbytes % (16 * MiB)
    src = torch.ones(nbytes // 4, dtype=torch.float32, device=backend.device)
    dst = torch.empty_like(src)
    avg_ms = _measure_ms(lambda: torch.neg(src, out=dst), backend, iters=100, warmup=10)
    return (2 * nbytes) / (avg_ms * 1e-3) / 1e9


@contextmanager
def _matmul_precision(*, tf32: bool):
    import torch

    matmul = torch.backends.cuda.matmul
    saved = (
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
    )
    try:
        matmul.allow_tf32 = tf32
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_bf16_reduced_precision_reduction = False
        yield
    finally:
        (
            matmul.allow_tf32,
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction,
        ) = saved


def _calibrate_matmul(backend: Backend, dtype_name: str, size: int, *, tf32: bool) -> float:
    import torch

    dtype = getattr(torch, dtype_name)
    free_bytes, _ = torch.cuda.mem_get_info(backend.device)
    elem = torch.empty((), dtype=dtype).element_size()
    while size > 1024 and 3 * size * size * elem > free_bytes * 0.5:
        size //= 2
    gen = torch.Generator(device=backend.device).manual_seed(0)
    a = torch.randn((size, size), device=backend.device, dtype=dtype, generator=gen)
    b = torch.randn((size, size), device=backend.device, dtype=dtype, generator=gen)
    c = torch.empty((size, size), device=backend.device, dtype=dtype)
    with _matmul_precision(tf32=tf32):
        avg_ms = _measure_ms(lambda: torch.matmul(a, b, out=c), backend, iters=20, warmup=3)
    return (2 * size**3) / (avg_ms * 1e-3) / 1e12


ITEMS: dict[str, Callable[[Backend], float]] = {
    "gbps": _calibrate_gbps,
    "tflops_fp16": lambda b: _calibrate_matmul(b, "float16", 8192, tf32=False),
    "tflops_bf16": lambda b: _calibrate_matmul(b, "bfloat16", 8192, tf32=False),
    "tflops_tf32": lambda b: _calibrate_matmul(b, "float32", 8192, tf32=True),
    "tflops_fp32": lambda b: _calibrate_matmul(b, "float32", 4096, tf32=False),
}


def calibrate(items: list[str] | None = None, *, backend: Backend | None = None) -> dict[str, float]:
    """Run the selected (default: all) calibration items on the active single-machine backend."""
    import torch

    active = backend or get_backend()
    selected = list(dict.fromkeys(items)) if items else list(ITEMS)
    for name in selected:
        if name not in ITEMS:
            raise ValueError(f"unknown calibration item {name!r}; available: {list(ITEMS)}")
    results: dict[str, float] = {}
    with torch.cuda.device(active.device):
        for name in selected:
            results[name] = ITEMS[name](active)
            torch.cuda.empty_cache()
    return results


def _run(items_arg: str | None) -> dict[str, Any]:
    items = [i.strip() for i in items_arg.split(",") if i.strip()] if items_arg else None
    if items_arg and not items:
        raise ValueError("--items must name at least one calibration item")
    backend = get_backend()
    calibrated = calibrate(items, backend=backend)
    peaks = save_peaks(calibrated)
    return {
        "file": normalize_path(peaks_path()),
        "calibrated": calibrated,
        "peaks": peaks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate achievable peaks and save them to .gluonkit/peaks.json")
    parser.add_argument("--items", help="Comma-separated items to (re)calibrate; default: all")
    args = parser.parse_args(argv)
    return emit_json(capture_run(lambda: _run(args.items)))


if __name__ == "__main__":
    raise SystemExit(main())
