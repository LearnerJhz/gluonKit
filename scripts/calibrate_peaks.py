"""One-off NVIDIA peak calibration: measure achievable DRAM bandwidth and matmul throughput.

Usage (run from the repo root, since results go to ./.gluonkit/peaks.json)::

    .venv/bin/python scripts/calibrate_peaks.py [--items gbps,tflops_fp16,...]

Items: gbps, tflops_fp16, tflops_bf16 (sm80+), tflops_tf32 (sm80+), tflops_fp32. Results are
merged per device into peaks.json; `gk bench` reads them (compute peak chosen by the eval's
`compute_mode`, e.g. "fp16" -> tflops_fp16). Prints one JSON line like the gk commands.

Every item is timed with the same torch.profiler kernel-time protocol as `gk bench`
(`profiler.profile`, mean per-call device kernel time), so roofline percentages compare
like with like. Values are achievable peaks of vendor-tuned kernels, not datasheet numbers.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import platform
from typing import Any, Callable

os.environ.setdefault("KINETO_LOG_LEVEL", "6")  # Quiet Kineto before torch initialises.

from gluonkit.backends.nvidia.probe import probe  # noqa: E402
from gluonkit.backends.nvidia.profiler import profile  # noqa: E402
from gluonkit.core.peaks import PeakStore, device_key  # noqa: E402

MiB = 1024 * 1024


@dataclass(frozen=True)
class _Item:
    description: str
    min_sm: int  # major*10 + minor
    run: Callable[[str, Any], dict[str, Any]]


def _sm(props) -> int:
    return props.major * 10 + props.minor


def _theoretical_gbps(props) -> float | None:
    clock_khz = getattr(props, "memory_clock_rate", 0) or 0
    bus_bits = getattr(props, "memory_bus_width", 0) or 0
    if clock_khz <= 0 or bus_bits <= 0:
        return None
    # Double data rate: 2 transfers per reported memory clock.
    return 2 * clock_khz * 1e3 * (bus_bits / 8) / 1e9


def _measure(fn: Callable[[], object], device: str, iters: int, warmup: int) -> dict[str, Any]:
    data = profile(fn, iters, warmup, device=device)
    if not data.kernel_name or data.avg_ms <= 0:
        raise RuntimeError("calibration workload launched no measurable CUDA kernel")
    name = data.kernel_name
    if len(name) > 120:  # Demangled template names can run to several hundred characters.
        name = name[:117] + "..."
    return {"kernel_name": name, "avg_ms": data.avg_ms, "min_ms": data.min_ms,
            "std_ms": data.std_ms, "iters": iters, "warmup": warmup}


def _calibrate_gbps(device: str, props) -> dict[str, Any]:
    import torch

    free_bytes, _ = torch.cuda.mem_get_info(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    # Working set far beyond L2 so every byte comes from DRAM; two buffers must fit.
    nbytes = max(256 * MiB, 16 * l2_bytes)
    nbytes = min(nbytes, int(free_bytes * 0.4))
    nbytes -= nbytes % (16 * MiB)
    if nbytes < 16 * MiB:
        raise RuntimeError(f"not enough free device memory for bandwidth calibration ({free_bytes} B free)")
    numel = nbytes // 4
    src = torch.ones(numel, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    # Elementwise kernel (read N + write N bytes); a plain copy_ would lower to cudaMemcpy,
    # which is not a kernel and would not be timed under the bench protocol.
    timing = _measure(lambda: torch.neg(src, out=dst), device, iters=100, warmup=10)
    moved = 2 * nbytes
    value = moved / (timing["avg_ms"] * 1e-3) / 1e9
    result = {
        "value": value,
        "unit": "GB/s",
        "method": "torch.neg(fp32, out=dst) streaming read+write, bytes = 2 x buffer",
        "working_set_bytes": moved,
        **timing,
    }
    theoretical = _theoretical_gbps(props)
    if theoretical:
        result["theoretical_gbps"] = theoretical
        result["efficiency_pct"] = value / theoretical * 100
    return result


@contextmanager
def _matmul_precision(*, tf32: bool):
    import torch

    matmul = torch.backends.cuda.matmul
    saved = (matmul.allow_tf32, matmul.allow_fp16_reduced_precision_reduction,
             matmul.allow_bf16_reduced_precision_reduction)
    try:
        matmul.allow_tf32 = tf32
        # Keep fp32 accumulation/reduction so the peak matches fp16/bf16 -> fp32 kernels.
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_bf16_reduced_precision_reduction = False
        yield
    finally:
        (matmul.allow_tf32, matmul.allow_fp16_reduced_precision_reduction,
         matmul.allow_bf16_reduced_precision_reduction) = saved


def _matmul_item(dtype_name: str, n: int, *, tf32: bool, method: str):
    def run(device: str, props) -> dict[str, Any]:
        import torch

        dtype = getattr(torch, dtype_name)
        size = n
        free_bytes, _ = torch.cuda.mem_get_info(device)
        elem = torch.empty((), dtype=dtype).element_size()
        while size > 1024 and 3 * size * size * elem > free_bytes * 0.5:
            size //= 2
        generator = torch.Generator(device=device).manual_seed(0)
        a = torch.randn((size, size), device=device, dtype=dtype, generator=generator)
        b = torch.randn((size, size), device=device, dtype=dtype, generator=generator)
        c = torch.empty((size, size), device=device, dtype=dtype)
        with _matmul_precision(tf32=tf32):
            timing = _measure(lambda: torch.matmul(a, b, out=c), device, iters=20, warmup=3)
        flops = 2 * size ** 3
        return {
            "value": flops / (timing["avg_ms"] * 1e-3) / 1e12,
            "unit": "TFLOP/s",
            "method": method,
            "problem": f"{size}x{size}x{size} {dtype_name}",
            **timing,
        }

    return run


ITEMS: dict[str, _Item] = {
    "gbps": _Item("DRAM streaming bandwidth (GB/s)", 0, _calibrate_gbps),
    "tflops_fp16": _Item("FP16 tensor-core matmul, fp32 accumulate (TFLOP/s)", 70, _matmul_item(
        "float16", 8192, tf32=False, method="torch.matmul fp16 (cuBLAS, fp32 accumulate)")),
    "tflops_bf16": _Item("BF16 tensor-core matmul, fp32 accumulate (TFLOP/s)", 80, _matmul_item(
        "bfloat16", 8192, tf32=False, method="torch.matmul bf16 (cuBLAS, fp32 accumulate)")),
    "tflops_tf32": _Item("TF32 tensor-core matmul (TFLOP/s)", 80, _matmul_item(
        "float32", 8192, tf32=True, method="torch.matmul fp32 with allow_tf32=True (cuBLAS)")),
    "tflops_fp32": _Item("FP32 CUDA-core FMA matmul, no TF32 (TFLOP/s)", 0, _matmul_item(
        "float32", 4096, tf32=False, method="torch.matmul fp32 with allow_tf32=False (cuBLAS)")),
}


def supported_items(device: str) -> list[str]:
    import torch

    sm = _sm(torch.cuda.get_device_properties(device))
    return [name for name, item in ITEMS.items() if sm >= item.min_sm]


def calibrate(items: list[str] | None = None, device: str = "cuda:0") -> dict[str, dict[str, Any]]:
    """Run the requested (default: all supported) calibration items on `device`."""
    import torch

    supported = supported_items(device)
    selected = list(dict.fromkeys(items)) if items else supported
    for name in selected:
        if name not in ITEMS:
            raise ValueError(f"unknown calibration item {name!r}; supported here: {supported}")
        if name not in supported:
            raise ValueError(f"calibration item {name!r} requires sm{ITEMS[name].min_sm}+; "
                             f"supported here: {supported}")
    props = torch.cuda.get_device_properties(device)
    results: dict[str, dict[str, Any]] = {}
    with torch.cuda.device(device):
        for name in selected:
            results[name] = {"description": ITEMS[name].description, **ITEMS[name].run(device, props)}
            torch.cuda.empty_cache()
    return results



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate achievable peaks and save them to .gluonkit/peaks.json")
    parser.add_argument("--items", help="Comma-separated items to (re)calibrate; default: all supported")
    args = parser.parse_args(argv)
    try:
        items = [i.strip() for i in args.items.split(",") if i.strip()] if args.items else None
        if args.items and not items:
            raise ValueError("--items must name at least one calibration item")
        info = probe()
        store = PeakStore()
        store.load()  # Fail on a corrupt file before spending GPU time.
        results = calibrate(items)
        env = {"python": platform.python_version(), "torch": info.torch_version,
               "compiler": info.compiler_version, "device": "cuda:0"}
        entry = store.update(info, results, env=env)
    except Exception as exc:  # noqa: BLE001 -- single-line JSON contract, like gk
        print(json.dumps({"result": "failed", "info": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"result": "success", "info": {
        "file": store.display_path(), "device_key": device_key(info), "calibrated": list(results),
        "peaks": entry["peaks"],
        "details": {k: {d: v for d, v in r.items() if d != "value"} for k, r in results.items()},
    }}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
