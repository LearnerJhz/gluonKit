from pathlib import Path
import re
from typing import Any, Callable


def _metadata_value(metadata: Any, attr: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(attr)
    return getattr(metadata, attr, None)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _device_attr(driver: Any, attr: Any, device: Any, default: int | None = None) -> int | None:
    rc, val = driver.cuDeviceGetAttribute(attr, device)
    return int(val) if rc == driver.CUresult.CUDA_SUCCESS else default


def _func_attr(driver: Any, attr: Any, func: Any, default: int = 0) -> int:
    rc, val = driver.cuFuncGetAttribute(attr, func)
    return int(val) if rc == driver.CUresult.CUDA_SUCCESS else default


def derive_limiting_factor(
    *,
    num_warps: int | None,
    n_regs: int | None,
    allocated_shared_bytes: int | None,
    warp_size: int = 32,
    max_warps_per_cu: int = 48,
    max_blocks_per_cu: int = 16,
    max_regs_per_cu: int = 65536,
    reg_alloc_granularity: int = 256,
    max_smem_per_cu: int = 102400,
) -> tuple[int | None, str | None]:
    """Compute theoretical max active blocks per SM/CU and identify the limiting resource."""
    if not num_warps or num_warps <= 0:
        return None, None
    limits: dict[str, int] = {
        "blocks": max(1, max_blocks_per_cu),
        "warps": max(0, max_warps_per_cu // num_warps),
    }
    if n_regs is not None and n_regs > 0:
        regs_per_warp = _align_up(n_regs * warp_size, reg_alloc_granularity)
        regs_per_block = regs_per_warp * num_warps
        limits["regs"] = max_regs_per_cu // regs_per_block if regs_per_block > 0 else max_blocks_per_cu
    if allocated_shared_bytes is not None and allocated_shared_bytes > 0 and max_smem_per_cu > 0:
        limits["shared_memory"] = max_smem_per_cu // allocated_shared_bytes

    active = min(limits.values())
    order = ("regs", "shared_memory", "warps", "blocks")
    bottlenecks = [name for name in order if name in limits and limits[name] == active]
    return active, "|".join(bottlenecks) if bottlenecks else None


def _occupancy_summary(
    kernel: Any,
    shared_bytes: int | None,
    num_warps: int | None,
    n_regs: int | None,
) -> dict[str, Any]:
    """Compute true SM occupancy, allocated shared memory, and limiting factor via the CUDA driver."""
    empty = {
        "allocated_shared_bytes": None,
        "blocks_per_sm": None,
        "occupancy_pct": None,
        "limiting_factor": None,
    }
    func_ptr = getattr(kernel, "function", None)
    if not func_ptr or num_warps is None or num_warps <= 0:
        return empty
    try:
        from cuda.bindings import driver
    except ImportError:
        return empty

    rc, device = driver.cuCtxGetDevice()
    if rc != driver.CUresult.CUDA_SUCCESS:
        return empty

    attr = driver.CUdevice_attribute
    warp_size = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device, 32)
    cc_major = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device, 8)
    max_threads_per_sm = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device)
    max_regs_per_sm = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device, 65536)
    max_smem_per_sm = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device, 102400)
    reserved_smem = _device_attr(driver, attr.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, device, 0)
    if None in (warp_size, max_threads_per_sm):
        return empty

    func = driver.CUfunction(func_ptr)
    dynamic_smem = int(shared_bytes or 0)
    static_smem = _func_attr(driver, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func, 0)
    block_threads = num_warps * warp_size

    rc, blocks_per_sm = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, block_threads, dynamic_smem)
    if rc != driver.CUresult.CUDA_SUCCESS:
        return empty

    smem_gran = 128 if (cc_major or 8) >= 8 else 256
    allocated_smem = _align_up(static_smem + dynamic_smem + (reserved_smem or 0), smem_gran)
    _, limiting_factor = derive_limiting_factor(
        num_warps=num_warps,
        n_regs=n_regs,
        allocated_shared_bytes=allocated_smem,
        warp_size=warp_size,
        max_warps_per_cu=max_threads_per_sm // warp_size,
        max_blocks_per_cu=16 if (cc_major or 8) >= 8 else 32,
        max_regs_per_cu=max_regs_per_sm or 65536,
        reg_alloc_granularity=256,
        max_smem_per_cu=max_smem_per_sm or 102400,
    )
    return {
        "allocated_shared_bytes": allocated_smem,
        "blocks_per_sm": blocks_per_sm,
        "occupancy_pct": (blocks_per_sm * block_threads) / max_threads_per_sm * 100,
        "limiting_factor": limiting_factor,
    }


def _kernel_summary(kernel: Any) -> dict[str, Any]:
    """Collect NVIDIA/Triton kernel resource summary after handle initialization."""
    metadata = getattr(kernel, "metadata", None)
    shared_bytes = _metadata_value(metadata, "shared")
    num_warps = _metadata_value(metadata, "num_warps")
    n_regs = getattr(kernel, "n_regs", None)
    return {
        "n_regs": n_regs,
        "n_spills": getattr(kernel, "n_spills", None),
        "n_max_threads": getattr(kernel, "n_max_threads", None),
        "shared_bytes": shared_bytes,
        "num_warps": num_warps,
        **_occupancy_summary(kernel, shared_bytes, num_warps, n_regs),
    }


def _write_kernel_ir(
    kernel: Any,
    directory: Path,
    extensions: tuple[str, ...] = ("ttgir", "llir", "ptx"),
) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}
    for extension in extensions:
        source = kernel.asm.get(extension)
        paths[extension] = None
        if source is not None:
            path = directory / f"kernel.{extension}"
            path.write_text(source, encoding="utf-8")
            paths[extension] = str(path)
    return paths


def dump_info(
    call: Callable[[], object],
    artifact_dir: str | Path,
    *,
    extensions: tuple[str, ...] = ("ttgir", "llir", "ptx"),
    summarize_fn: Callable[[Any], dict[str, Any]] = _kernel_summary,
) -> dict[str, Any]:
    """Run call() once, export Triton IR files, and return artifacts plus resource summary."""
    from triton.compiler.compiler import CompiledKernel

    artifacts: dict[str, dict[str, str | None]] = {}
    captured: dict[str, Any] = {}
    seen: set[str] = set()
    original = CompiledKernel.launch_metadata

    def export(kernel, *args, **kwargs):
        if kernel.hash not in seen:
            name = re.sub(r"[^\w.-]", "_", kernel.name)[:100].strip(".") or "kernel"
            key = name
            index = 2
            while key in artifacts:
                key = f"{name}_{index}"
                index += 1
            kernel_dir = Path(artifact_dir).resolve() / key
            if kernel.hash:
                kernel_dir = kernel_dir / str(kernel.hash)[:12]
            artifacts[key] = _write_kernel_ir(kernel, kernel_dir, extensions=extensions)
            captured[key] = kernel
            seen.add(kernel.hash)
        return original(kernel, *args, **kwargs)

    # Triton 3.8.0 invokes this for each launch, including cache hits.
    # Only the selected run() is wrapped in our single-task worker.
    CompiledKernel.launch_metadata = export
    try:
        call()
    finally:
        CompiledKernel.launch_metadata = original

    # CompiledKernel._init_handles() populates n_regs/n_spills/n_max_threads when
    # Triton accesses kernel.run right after launch_metadata returns.
    summary = {key: summarize_fn(kernel) for key, kernel in captured.items()}
    return {"artifacts": artifacts, "summary": summary}
