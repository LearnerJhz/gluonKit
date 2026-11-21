"""Common Triton CompiledKernel interception and IR artifact exporter across SIMT GPU backends."""

from pathlib import Path
import re
from typing import Any, Callable

_DEFAULT_IR_EXTENSIONS = ("ttgir", "llir", "ptx")
_IGNORED_ASM_KEYS = frozenset({"source", "cubin", "hsaco", "json"})


def metadata_value(metadata: Any, attr: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(attr)
    return getattr(metadata, attr, None)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


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
    """Compute theoretical max active blocks per CU/SM and identify the limiting resource(s).

    Returns `(theoretical_blocks_per_cu, limiting_factor)` where `limiting_factor` is one of
    `"regs"`, `"shared_memory"`, `"warps"`, `"blocks"` (or pipe-joined on ties, e.g. `"regs|warps"`).
    """
    if not num_warps or num_warps <= 0:
        return None, None

    limits: dict[str, int] = {
        "blocks": max(1, max_blocks_per_cu),
        "warps": max(0, max_warps_per_cu // num_warps),
    }
    if n_regs is not None and n_regs > 0:
        regs_per_warp = align_up(n_regs * warp_size, reg_alloc_granularity)
        regs_per_block = regs_per_warp * num_warps
        limits["regs"] = max_regs_per_cu // regs_per_block if regs_per_block > 0 else max_blocks_per_cu
    if allocated_shared_bytes is not None and allocated_shared_bytes > 0 and max_smem_per_cu > 0:
        limits["shared_memory"] = max_smem_per_cu // allocated_shared_bytes

    active_blocks = min(limits.values())
    # Priority order when reporting ties: primary bottlenecks first.
    order = ("regs", "shared_memory", "warps", "blocks")
    bottlenecks = [name for name in order if name in limits and limits[name] == active_blocks]
    return active_blocks, "|".join(bottlenecks) if bottlenecks else None


def write_kernel_ir(
    kernel: Any,
    directory: Path,
    *,
    expected_extensions: tuple[str, ...] = _DEFAULT_IR_EXTENSIONS,
) -> dict[str, str | None]:
    """Export all textual IRs from `kernel.asm` to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    asm = getattr(kernel, "asm", None) or {}
    paths: dict[str, str | None] = {}

    for extension in expected_extensions:
        source = asm.get(extension)
        paths[extension] = None
        if isinstance(source, str):
            path = directory / f"kernel.{extension}"
            path.write_text(source, encoding="utf-8")
            paths[extension] = str(path)

    for extension, source in asm.items():
        if extension in paths or extension in _IGNORED_ASM_KEYS:
            continue
        if isinstance(source, str):
            path = directory / f"kernel.{extension}"
            path.write_text(source, encoding="utf-8")
            paths[extension] = str(path)
    return paths


def dump_triton_kernels(
    call: Callable[[], object],
    artifact_dir: str | Path,
    *,
    summarize_fn: Callable[[Any], Any],
    expected_extensions: tuple[str, ...] = _DEFAULT_IR_EXTENSIONS,
) -> dict[str, Any]:
    """Run `call()` once, intercept Triton CompiledKernel launches, export IRs, and summarize."""
    from triton.compiler.compiler import CompiledKernel

    artifacts: dict[str, dict[str, str | None]] = {}
    captured: dict[str, Any] = {}
    seen: set[str] = set()
    original = CompiledKernel.launch_metadata

    def export(kernel: Any, *args: Any, **kwargs: Any) -> Any:
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
            artifacts[key] = write_kernel_ir(kernel, kernel_dir, expected_extensions=expected_extensions)
            captured[key] = kernel
            seen.add(kernel.hash)
        return original(kernel, *args, **kwargs)

    CompiledKernel.launch_metadata = export
    try:
        call()
    finally:
        CompiledKernel.launch_metadata = original

    summary: dict[str, Any] = {}
    for key, kernel in captured.items():
        item = summarize_fn(kernel)
        summary[key] = item.to_dict() if hasattr(item, "to_dict") else item
    return {"artifacts": artifacts, "summary": summary}
