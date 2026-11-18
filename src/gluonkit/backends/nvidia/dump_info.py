from pathlib import Path
import re
from typing import Any, Callable


def _metadata_value(metadata: Any, attr: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(attr)
    return getattr(metadata, attr, None)


def _kernel_summary(kernel: Any) -> dict[str, Any]:
    """Collect NVIDIA/Triton kernel resource summary after handle initialization."""
    metadata = getattr(kernel, "metadata", None)
    return {
        "n_regs": getattr(kernel, "n_regs", None),
        "n_spills": getattr(kernel, "n_spills", None),
        "n_max_threads": getattr(kernel, "n_max_threads", None),
        "shared_bytes": _metadata_value(metadata, "shared"),
        "num_warps": _metadata_value(metadata, "num_warps"),
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
