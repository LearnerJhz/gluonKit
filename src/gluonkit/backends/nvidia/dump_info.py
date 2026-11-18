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


def _occupancy_summary(
    kernel: Any,
    shared_bytes: int | None,
    num_warps: int | None,
) -> dict[str, Any]:
    """Compute true SM occupancy and allocated shared memory via the CUDA driver."""
    empty = {
        "allocated_shared_bytes": None,
        "blocks_per_sm": None,
        "occupancy_pct": None,
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
    return {
        "allocated_shared_bytes": allocated_smem,
        "blocks_per_sm": blocks_per_sm,
        "occupancy_pct": (blocks_per_sm * block_threads) / max_threads_per_sm * 100,
    }


def _kernel_summary(kernel: Any) -> dict[str, Any]:
    """Collect NVIDIA/Triton kernel resource summary after handle initialization."""
    metadata = getattr(kernel, "metadata", None)
    shared_bytes = _metadata_value(metadata, "shared")
    num_warps = _metadata_value(metadata, "num_warps")
    return {
        "n_regs": getattr(kernel, "n_regs", None),
        "n_spills": getattr(kernel, "n_spills", None),
        "n_max_threads": getattr(kernel, "n_max_threads", None),
        "shared_bytes": shared_bytes,
        "num_warps": num_warps,
        **_occupancy_summary(kernel, shared_bytes, num_warps),
    }


def _write_kernel_ir(kernel: Any, directory: Path) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}
    for extension in ("ttgir", "llir", "ptx"):
        source = kernel.asm.get(extension)
        paths[extension] = None
        if source is not None:
            path = directory / f"kernel.{extension}"
            path.write_text(source)
            paths[extension] = str(path)
    return paths


def dump_info(call: Callable[[], object], artifact_dir: str | Path) -> dict[str, Any]:
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
            artifacts[key] = _write_kernel_ir(kernel, Path(artifact_dir).resolve() / key)
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
    summary = {key: _kernel_summary(kernel) for key, kernel in captured.items()}
    return {"artifacts": artifacts, "summary": summary}
