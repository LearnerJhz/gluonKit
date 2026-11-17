from pathlib import Path


def export_ir(call, artifact_dir: str | Path) -> dict[str, dict[str, str | None]]:
    from triton.compiler.compiler import CompiledKernel

    artifacts = {}
    original = CompiledKernel.launch_metadata

    def export(kernel, *args, **kwargs):
        if kernel.hash not in artifacts:
            directory = Path(artifact_dir).resolve() / kernel.hash
            directory.mkdir(parents=True, exist_ok=True)
            paths = {}
            for extension in ("ttgir", "llir", "ptx"):
                source = kernel.asm.get(extension)
                paths[extension] = None
                if source is not None:
                    path = directory / f"kernel.{extension}"
                    path.write_text(source)
                    paths[extension] = str(path)
            artifacts[kernel.hash] = paths
        return original(kernel, *args, **kwargs)

    # Triton 3.8.0 invokes this for each launch, including cache hits.
    # Only the selected run() is wrapped in our single-task worker.
    CompiledKernel.launch_metadata = export
    try:
        call()
    finally:
        CompiledKernel.launch_metadata = original
    return artifacts
