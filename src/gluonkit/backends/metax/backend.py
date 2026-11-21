"""MetaX (沐曦) C500 MACA SIMT GPU Backend implementation (Wave64)."""

from contextlib import contextmanager
from typing import Any, Iterator

from gluonkit.backends.base import SimtBackendBase
from gluonkit.core.device_diag import get_device_properties_with_diagnosis
from gluonkit.core.triton_dump import align_up, derive_limiting_factor, metadata_value
from gluonkit.interfaces.types import KernelSummary, ProbeResult

# MetaX C500 SIMT Compute Unit architectural defaults (Wave64, 64 KiB LDS, 32 waves/CU = 2048 threads/CU)
C500_WARP_SIZE = 64
C500_MAX_WAVES_PER_CU = 32
C500_MAX_BLOCKS_PER_CU = 16
C500_MAX_REGS_PER_CU = 65536
C500_REG_ALLOC_GRANULARITY = 256
C500_MAX_LDS_PER_CU = 65536
C500_LDS_ALLOC_GRANULARITY = 256


def _resolve_metax_arch(props: Any) -> str:
    gcn_arch = getattr(props, "gcnArchName", None)
    if isinstance(gcn_arch, str) and gcn_arch.strip():
        return gcn_arch.split(":")[0].strip()
    name = (getattr(props, "name", "") or "").lower()
    if "c500" in name:
        return "maca_c500"
    major = getattr(props, "major", None)
    minor = getattr(props, "minor", None)
    if major is not None and minor is not None:
        return f"maca_{major}{minor}"
    return "maca_c500"


def probe_metax(device: str = "cuda:0") -> ProbeResult:
    import torch
    import triton

    props = get_device_properties_with_diagnosis(device, platform="metax")
    arch = _resolve_metax_arch(props)
    warp_size = int(getattr(props, "warp_size", C500_WARP_SIZE) or C500_WARP_SIZE)
    default_smem = getattr(props, "shared_memory_per_block", C500_MAX_LDS_PER_CU)
    optin_smem = getattr(props, "shared_memory_per_block_optin", None) or default_smem
    max_smem = getattr(props, "max_shared_memory_per_block", optin_smem) or default_smem
    cu_count = int(getattr(props, "multi_processor_count", 0) or 0)
    extra: dict[str, Any] = {}
    if getattr(torch.version, "maca", None):
        extra["maca_version"] = str(torch.version.maca)

    return ProbeResult(
        device_name=props.name,
        platform="metax",
        arch=arch,
        compiler_version=triton.__version__,
        torch_version=torch.__version__,
        smem_per_block_optin=max_smem,
        sm_count=cu_count,
        vram_bytes=props.total_memory,
        compute_unit_count=cu_count,
        warp_size=warp_size,
        max_smem_per_block=max_smem,
        default_smem_per_block=default_smem,
        l2_cache_bytes=getattr(props, "L2_cache_size", None),
        supported_compute_modes=["fp16", "bf16", "tf32", "fp32"],
        knowledge_docs=[
            ".knowledge/gluon-api/gluon-api-common.md",
            ".knowledge/gluon-api/metax/metax.md",
            ".knowledge/gluon-api/troubleshooting.md",
        ],
        extra=extra,
    )


def summarize_metax_kernel(kernel: Any, *, warp_size: int = C500_WARP_SIZE) -> KernelSummary:
    """Collect MetaX C500 (MACA / mcTriton) kernel resource usage and Wave64 CU occupancy."""
    metadata = getattr(kernel, "metadata", None)
    shared_bytes = metadata_value(metadata, "shared")
    num_warps = metadata_value(metadata, "num_warps")
    n_regs = getattr(kernel, "n_regs", None)
    n_spills = getattr(kernel, "n_spills", None)
    n_max_threads = getattr(kernel, "n_max_threads", None)
    sgpr_count = metadata_value(metadata, "num_sgprs") or getattr(kernel, "n_sgprs", None)

    allocated_smem = (
        align_up(int(shared_bytes), C500_LDS_ALLOC_GRANULARITY)
        if shared_bytes is not None and int(shared_bytes) > 0
        else (0 if shared_bytes == 0 else None)
    )
    blocks_per_cu, limiting_factor = derive_limiting_factor(
        num_warps=num_warps,
        n_regs=n_regs,
        allocated_shared_bytes=allocated_smem,
        warp_size=warp_size,
        max_warps_per_cu=C500_MAX_WAVES_PER_CU,
        max_blocks_per_cu=C500_MAX_BLOCKS_PER_CU,
        max_regs_per_cu=C500_MAX_REGS_PER_CU,
        reg_alloc_granularity=C500_REG_ALLOC_GRANULARITY,
        max_smem_per_cu=C500_MAX_LDS_PER_CU,
    )
    occupancy_pct = (
        (blocks_per_cu * num_warps) / C500_MAX_WAVES_PER_CU * 100.0
        if blocks_per_cu is not None and num_warps
        else None
    )
    extra: dict[str, Any] = {}
    if sgpr_count is not None:
        extra["n_sgprs"] = sgpr_count

    return KernelSummary(
        n_regs=n_regs,
        n_spills=n_spills,
        n_max_threads=n_max_threads,
        shared_bytes=shared_bytes,
        num_warps=num_warps,
        allocated_shared_bytes=allocated_smem,
        blocks_per_sm=blocks_per_cu,
        blocks_per_cu=blocks_per_cu,
        occupancy_pct=occupancy_pct,
        warp_size=warp_size,
        limiting_factor=limiting_factor,
        occupancy_source="arch_model" if blocks_per_cu is not None else None,
        extra=extra,
    )


class MetaxBackend(SimtBackendBase):
    name = "metax"
    platform_label = "MetaX"
    warp_size = C500_WARP_SIZE
    expected_ir_extensions = ("ttgir", "llir")
    allow_event_fallback = True
    timing_method = "torch.profiler device kernel or MACA device event mean per call"

    def probe(self) -> ProbeResult:
        return probe_metax(self.device)

    def extra_layout_constructors(self) -> dict[str, type]:
        from triton.experimental.gluon import language as gl

        constructors: dict[str, type] = {}
        for attr in ("AMDMFMALayout", "MetaxMMALayout", "MFMADistributedLayout", "NVMMADistributedLayout"):
            if hasattr(gl, attr):
                constructors[attr] = getattr(gl, attr)
        return constructors

    def summarize_kernel(self, kernel: Any) -> KernelSummary:
        return summarize_metax_kernel(kernel, warp_size=self.warp_size)

    @contextmanager
    def matmul_precision_context(self, compute_mode: str) -> Iterator[None]:
        import torch

        matmul = getattr(torch.backends, "cuda", None)
        matmul_cfg = getattr(matmul, "matmul", None) if matmul is not None else None
        if matmul_cfg is None or not hasattr(matmul_cfg, "allow_tf32"):
            yield
            return
        saved = (
            matmul_cfg.allow_tf32,
            getattr(matmul_cfg, "allow_fp16_reduced_precision_reduction", False),
            getattr(matmul_cfg, "allow_bf16_reduced_precision_reduction", False),
        )
        try:
            matmul_cfg.allow_tf32 = compute_mode == "tf32"
            if hasattr(matmul_cfg, "allow_fp16_reduced_precision_reduction"):
                matmul_cfg.allow_fp16_reduced_precision_reduction = False
            if hasattr(matmul_cfg, "allow_bf16_reduced_precision_reduction"):
                matmul_cfg.allow_bf16_reduced_precision_reduction = False
            yield
        finally:
            matmul_cfg.allow_tf32 = saved[0]
            if hasattr(matmul_cfg, "allow_fp16_reduced_precision_reduction"):
                matmul_cfg.allow_fp16_reduced_precision_reduction = saved[1]
            if hasattr(matmul_cfg, "allow_bf16_reduced_precision_reduction"):
                matmul_cfg.allow_bf16_reduced_precision_reduction = saved[2]
