from pathlib import Path
import re


def export_ir(call, artifact_dir: str | Path) -> dict[str, dict[str, str | None]]:
    from triton.compiler.compiler import CompiledKernel

    artifacts = {}
    seen = set()
    original = CompiledKernel.launch_metadata

    def export(kernel, *args, **kwargs):
        if kernel.hash not in seen:
            name = re.sub(r"[^\w.-]", "_", kernel.name)[:100].strip(".") or "kernel"
            key = name
            index = 2
            while key in artifacts:
                key = f"{name}_{index}"
                index += 1
            directory = Path(artifact_dir).resolve() / key
            directory.mkdir(parents=True, exist_ok=True)
            paths = {}
            for extension in ("ttgir", "llir", "ptx"):
                source = kernel.asm.get(extension)
                paths[extension] = None
                if source is not None:
                    path = directory / f"kernel.{extension}"
                    path.write_text(source)
                    paths[extension] = str(path)
            artifacts[key] = paths
            seen.add(kernel.hash)
        return original(kernel, *args, **kwargs)

    # Triton 3.8.0 invokes this for each launch, including cache hits.
    # Only the selected run() is wrapped in our single-task worker.
    CompiledKernel.launch_metadata = export
    try:
        call()
    finally:
        CompiledKernel.launch_metadata = original
    return artifacts
