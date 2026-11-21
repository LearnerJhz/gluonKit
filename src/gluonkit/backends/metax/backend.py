"""MetaX (沐曦) C500 SIMT Backend implementation (Wave64)."""

from pathlib import Path
from typing import Any, Callable

from gluonkit.backends.nvidia.bank_conflicts import bank_conflicts
from gluonkit.backends.nvidia.dump_info import _align_up, _metadata_value, derive_limiting_factor, dump_info
from gluonkit.backends.nvidia.layout import render_layout
from gluonkit.backends.nvidia.profiler import profile
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import ProbeResult, ProfileData


def _metax_kernel_summary(kernel: Any) -> dict[str, Any]:
    """Collect MetaX C500 (Wave64, 64KB LDS, 32 waves/CU) kernel resource summary."""
    metadata = getattr(kernel, "metadata", None)
    shared_bytes = _metadata_value(metadata, "shared")
    num_warps = _metadata_value(metadata, "num_warps")
    n_regs = getattr(kernel, "n_regs", None)
    allocated_smem = _align_up(int(shared_bytes), 256) if shared_bytes else 0
    blocks_per_sm, limiting_factor = derive_limiting_factor(
        num_warps=num_warps,
        n_regs=n_regs,
        allocated_shared_bytes=allocated_smem,
        warp_size=64,
        max_warps_per_cu=32,
        max_blocks_per_cu=16,
        max_regs_per_cu=65536,
        reg_alloc_granularity=256,
        max_smem_per_cu=65536,
    )
    occupancy_pct = (blocks_per_sm * num_warps) / 32 * 100.0 if blocks_per_sm and num_warps else None
    return {
        "n_regs": n_regs,
        "n_spills": getattr(kernel, "n_spills", None),
        "n_max_threads": getattr(kernel, "n_max_threads", None),
        "shared_bytes": shared_bytes,
        "num_warps": num_warps,
        "allocated_shared_bytes": allocated_smem,
        "blocks_per_sm": blocks_per_sm,
        "occupancy_pct": occupancy_pct,
        "limiting_factor": limiting_factor,
    }


class MetaxBackend(Backend):
    warp_size = 64
    timing_method = "torch.profiler device kernel mean per call"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def probe(self) -> ProbeResult:
        import torch
        import triton

        props = torch.cuda.get_device_properties(self.device)
        arch = getattr(props, "gcnArchName", None) or "maca_c500"
        if isinstance(arch, str):
            arch = arch.split(":")[0].strip()
        smem = getattr(props, "shared_memory_per_block_optin", None) or getattr(
            props, "shared_memory_per_block", 65536,
        )
        return ProbeResult(
            device_name=props.name,
            platform="metax",
            arch=arch,
            compiler_version=triton.__version__,
            torch_version=torch.__version__,
            smem_per_block_optin=smem,
            sm_count=props.multi_processor_count,
            vram_bytes=props.total_memory,
            warp_size=int(getattr(props, "warp_size", self.warp_size) or self.warp_size),
        )

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str:
        return render_layout(expression, shape, hardware, warp_size=self.warp_size)

    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]:
        return bank_conflicts(register_layout, shared_layout, shape, dtype)

    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, Any]:
        return dump_info(
            fn,
            output_dir,
            extensions=("ttgir", "llir", "maca"),
            summarize_fn=_metax_kernel_summary,
        )

    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData:
        return profile(
            fn, iters, warmup, device=self.device,
            flush_l2=flush_l2, trace_path=trace_path,
        )
