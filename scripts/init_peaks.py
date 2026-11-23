"""One-off theoretical hardware peak initialization for `.gluonkit/peaks.json`.

Computes theoretical DRAM bandwidth (`mem_clock * 2 * bus_width / 8`) and dense compute
throughput (`sm_count * flops_per_sm_per_clock * max_sm_clock`) from GPU specs::

    .venv/bin/python scripts/init_peaks.py

For unsupported GPUs (e.g. MetaX C500) or locked-clock setups, author `.gluonkit/peaks.json` directly.
"""

import argparse
import os
import re
import subprocess
from typing import Any

os.environ.setdefault("KINETO_LOG_LEVEL", "6")

from gluonkit.backends.registry import get_backend  # noqa: E402
from gluonkit.core.baseline import device_key  # noqa: E402
from gluonkit.core.roofline import peaks_path, save_peaks  # noqa: E402
from gluonkit.core.storage import normalize_path  # noqa: E402
from gluonkit.interfaces.backend import Backend  # noqa: E402
from gluonkit.output import capture_run, emit_json  # noqa: E402

# Dense FLOPs per SM per clock (1 FMA = 2 FLOPs; Tensor Core FP16/BF16 use FP32 accumulation).
# Consumer GeForce sm75/sm86/sm89 have half-rate FP32-accum Tensor Cores vs datacenter/workstation SKUs.
_NVIDIA_FLOPS_PER_SM_PER_CLOCK: dict[tuple[str, bool], dict[str, int]] = {
    ("sm70", False): {"fp32": 128, "fp16": 1024},
    ("sm75", False): {"fp32": 128, "fp16": 1024},
    ("sm75", True): {"fp32": 128, "fp16": 512},
    ("sm80", False): {"fp32": 128, "tf32": 1024, "fp16": 2048, "bf16": 2048},
    ("sm86", False): {"fp32": 256, "tf32": 512, "fp16": 1024, "bf16": 1024},
    ("sm86", True): {"fp32": 256, "tf32": 256, "fp16": 512, "bf16": 512},
    ("sm89", False): {"fp32": 256, "tf32": 512, "fp16": 1024, "bf16": 1024},
    ("sm89", True): {"fp32": 256, "tf32": 256, "fp16": 512, "bf16": 512},
    ("sm90", False): {"fp32": 256, "tf32": 2048, "fp16": 4096, "bf16": 4096},
    ("sm100", False): {"fp32": 256, "tf32": 4096, "fp16": 8192, "bf16": 8192},
    ("sm120", True): {"fp32": 256, "tf32": 256, "fp16": 512, "bf16": 512},
}


def _nvidia_flops_per_sm_per_clock(arch: str, device_name: str) -> dict[str, int] | None:
    lower = device_name.lower()
    if re.search(r"\bh20\b", lower):
        return None
    return _NVIDIA_FLOPS_PER_SM_PER_CLOCK.get((arch, "geforce" in lower))


def _query_nvidia_max_sm_clock_mhz(props: Any) -> float | None:
    uuid = getattr(props, "uuid", None)
    gpu_id = (str(uuid) if str(uuid).startswith("GPU-") else f"GPU-{uuid}") if uuid else "0"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        val = float(out.splitlines()[0].strip())
        return val if val > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def compute_theoretical_peaks(*, backend: Backend | None = None) -> tuple[dict[str, float], str]:
    """Compute theoretical `{gbps, tflops_*}` peaks and `device_key` for the active GPU."""
    import torch

    active = backend or get_backend()
    info = active.probe()
    dev_key = device_key(info)
    props = torch.cuda.get_device_properties(active.device)

    mem_clock_mhz = (getattr(props, "memory_clock_rate", 0) or 0) / 1000.0
    bus_width_bits = getattr(props, "memory_bus_width", 0) or 0
    sm_clock_mhz = (
        (_query_nvidia_max_sm_clock_mhz(props) if info.platform == "nvidia" else None)
        or (getattr(props, "clock_rate", 0) or 0) / 1000.0
    )
    ops_table = _nvidia_flops_per_sm_per_clock(info.arch, info.device_name) if info.platform == "nvidia" else None

    if not (mem_clock_mhz > 0 and bus_width_bits > 0 and sm_clock_mhz > 0 and ops_table):
        raise ValueError(
            f"Cannot compute theoretical peaks automatically for {dev_key}; "
            "please author .gluonkit/peaks.json manually"
        )

    peaks: dict[str, float] = {
        "gbps": mem_clock_mhz * 2 * (bus_width_bits / 8.0) / 1000.0,
    }
    for mode, flops_per_sm in ops_table.items():
        peaks[f"tflops_{mode}"] = info.sm_count * flops_per_sm * sm_clock_mhz / 1e6
    return peaks, dev_key


def _run() -> dict[str, Any]:
    peaks, dev_key = compute_theoretical_peaks()
    saved = save_peaks(peaks, device_key=dev_key)
    return {"file": normalize_path(peaks_path()), **saved}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Compute theoretical hardware peaks and save to .gluonkit/peaks.json").parse_args(argv)
    return emit_json(capture_run(_run))


if __name__ == "__main__":
    raise SystemExit(main())
