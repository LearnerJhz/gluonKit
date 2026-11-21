"""NVIDIA CUDA SIMT Backend implementation."""

from contextlib import contextmanager
from typing import Any, Iterator

from gluonkit.backends.base import SimtBackendBase
from gluonkit.core.device_diag import get_device_properties_with_diagnosis
from gluonkit.core.triton_dump import align_up, derive_limiting_factor, metadata_value
from gluonkit.interfaces.types import KernelSummary, ProbeResult


def _device_attr(driver: Any, attr: Any, device: Any, default: int | None = None) -> int | None:
    rc, val = driver.cuDeviceGetAttribute(attr, device)
    return int(val) if rc == driver.CUresult.CUDA_SUCCESS else default


def _func_attr(driver: Any, attr: Any, func: Any, default: int = 0) -> int:
    rc, val = driver.cuFuncGetAttribute(attr, func)
    return int(val) if rc == driver.CUresult.CUDA_SUCCESS else default


def _supported_nvidia_modes(sm: int) -> list[str]:
    modes = ["fp16"]
    if sm >= 80:
        modes.extend(["bf16", "tf32"])
    modes.append("fp32")
    return modes


def _nvidia_knowledge_docs(arch: str) -> list[str]:
    docs = [
        ".knowledge/gluon-api/gluon-api-common.md",
        ".knowledge/gluon-api/nvidia/nvidia.md",
    ]
    if arch.startswith("sm8"):
        docs.append(".knowledge/gluon-api/nvidia/nvidia-sm86.md")
    docs.append(".knowledge/gluon-api/troubleshooting.md")
    return docs


def probe_nvidia(device: str = "cuda:0") -> ProbeResult:
    import torch
    import triton

    props = get_device_properties_with_diagnosis(device, platform="nvidia")
    sm = props.major * 10 + props.minor
    arch = f"sm{props.major}{props.minor}"
    optin_smem = getattr(props, "shared_memory_per_block_optin", None)
    default_smem = getattr(props, "shared_memory_per_block", None)
    max_smem = optin_smem if optin_smem is not None else default_smem
    return ProbeResult(
        device_name=props.name,
        platform="nvidia",
        arch=arch,
        compiler_version=triton.__version__,
        torch_version=torch.__version__,
        smem_per_block_optin=optin_smem,
        sm_count=props.multi_processor_count,
        vram_bytes=props.total_memory,
        compute_unit_count=props.multi_processor_count,
        warp_size=int(getattr(props, "warp_size", 32) or 32),
        max_smem_per_block=max_smem,
        default_smem_per_block=default_smem,
        l2_cache_bytes=getattr(props, "L2_cache_size", None),
        supported_compute_modes=_supported_nvidia_modes(sm),
        knowledge_docs=_nvidia_knowledge_docs(arch),
    )


def summarize_nvidia_kernel(kernel: Any, *, warp_size: int = 32) -> KernelSummary:
    """Collect NVIDIA/Triton kernel resource summary and derive SM occupancy + limiting factor."""
    metadata = getattr(kernel, "metadata", None)
    shared_bytes = metadata_value(metadata, "shared")
    num_warps = metadata_value(metadata, "num_warps")
    n_regs = getattr(kernel, "n_regs", None)
    n_spills = getattr(kernel, "n_spills", None)
    n_max_threads = getattr(kernel, "n_max_threads", None)

    allocated_smem: int | None = None
    blocks_per_sm: int | None = None
    occupancy_pct: float | None = None
    limiting_factor: str | None = None
    occupancy_source: str | None = None

    func_ptr = getattr(kernel, "function", None)
    if func_ptr and num_warps is not None and num_warps > 0:
        try:
            from cuda.bindings import driver

            rc, device = driver.cuCtxGetDevice()
            if rc == driver.CUresult.CUDA_SUCCESS:
                attr = driver.CUdevice_attribute
                hw_warp = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device, warp_size) or warp_size
                cc_major = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device, 8) or 8
                max_threads_per_sm = _device_attr(
                    driver, attr.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device,
                )
                max_blocks_per_sm = _device_attr(
                    driver,
                    getattr(attr, "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR", None),
                    device,
                    16 if cc_major >= 8 else 32,
                ) if hasattr(attr, "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR") else (16 if cc_major >= 8 else 32)
                max_regs_per_sm = _device_attr(
                    driver, attr.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device, 65536,
                ) or 65536
                max_smem_per_sm = _device_attr(
                    driver, attr.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device, 102400,
                ) or 102400
                reserved_smem = _device_attr(
                    driver, attr.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, device, 0,
                ) or 0

                if max_threads_per_sm:
                    func = driver.CUfunction(func_ptr)
                    dynamic_smem = int(shared_bytes or 0)
                    static_smem = _func_attr(
                        driver, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func, 0,
                    )
                    block_threads = num_warps * hw_warp
                    rc_occ, active_blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        func, block_threads, dynamic_smem,
                    )
                    if rc_occ == driver.CUresult.CUDA_SUCCESS:
                        smem_gran = 128 if cc_major >= 8 else 256
                        allocated_smem = align_up(static_smem + dynamic_smem + reserved_smem, smem_gran)
                        blocks_per_sm = int(active_blocks)
                        occupancy_pct = (blocks_per_sm * block_threads) / max_threads_per_sm * 100
                        occupancy_source = "driver_api"
                        _, limiting_factor = derive_limiting_factor(
                            num_warps=num_warps,
                            n_regs=n_regs,
                            allocated_shared_bytes=allocated_smem,
                            warp_size=hw_warp,
                            max_warps_per_cu=max_threads_per_sm // hw_warp,
                            max_blocks_per_cu=int(max_blocks_per_sm or 16),
                            max_regs_per_cu=max_regs_per_sm,
                            reg_alloc_granularity=256,
                            max_smem_per_cu=max_smem_per_sm,
                        )
        except ImportError:
            pass

    if blocks_per_sm is None and num_warps is not None and num_warps > 0:
        allocated_smem = align_up(int(shared_bytes or 0), 128) if shared_bytes is not None else None
        est_blocks, limiting_factor = derive_limiting_factor(
            num_warps=num_warps,
            n_regs=n_regs,
            allocated_shared_bytes=allocated_smem,
            warp_size=warp_size,
        )
        if est_blocks is not None:
            blocks_per_sm = est_blocks
            occupancy_pct = (est_blocks * num_warps) / 48 * 100
            occupancy_source = "arch_model"

    return KernelSummary(
        n_regs=n_regs,
        n_spills=n_spills,
        n_max_threads=n_max_threads,
        shared_bytes=shared_bytes,
        num_warps=num_warps,
        allocated_shared_bytes=allocated_smem,
        blocks_per_sm=blocks_per_sm,
        blocks_per_cu=blocks_per_sm,
        occupancy_pct=occupancy_pct,
        warp_size=warp_size,
        limiting_factor=limiting_factor,
        occupancy_source=occupancy_source,
    )


class NvidiaBackend(SimtBackendBase):
    name = "nvidia"
    platform_label = "NVIDIA"
    warp_size = 32
    expected_ir_extensions = ("ttgir", "llir", "ptx")
    allow_event_fallback = False
    timing_method = "torch.profiler device kernel mean per call"

    def probe(self) -> ProbeResult:
        return probe_nvidia(self.device)

    def extra_layout_constructors(self) -> dict[str, type]:
        from triton.experimental.gluon import language as gl

        if hasattr(gl, "NVMMADistributedLayout"):
            return {"NVMMADistributedLayout": gl.NVMMADistributedLayout}
        return {}

    def summarize_kernel(self, kernel: Any) -> KernelSummary:
        return summarize_nvidia_kernel(kernel, warp_size=self.warp_size)

    @contextmanager
    def matmul_precision_context(self, compute_mode: str) -> Iterator[None]:
        import torch

        matmul = torch.backends.cuda.matmul
        saved = (
            matmul.allow_tf32,
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction,
        )
        try:
            matmul.allow_tf32 = compute_mode == "tf32"
            matmul.allow_fp16_reduced_precision_reduction = False
            matmul.allow_bf16_reduced_precision_reduction = False
            yield
        finally:
            (
                matmul.allow_tf32,
                matmul.allow_fp16_reduced_precision_reduction,
                matmul.allow_bf16_reduced_precision_reduction,
            ) = saved
