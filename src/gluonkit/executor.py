"""Task loading and in-process execution for evaluation scripts."""

import ast
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tempfile

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.core.bencher import DEFAULT_ITERS, DEFAULT_WARMUP, bench_task
from gluonkit.core.checker import check_task
from gluonkit.core.history import History
from gluonkit.interfaces.types import Output
from gluonkit.output import capture_run


def load_task(eval_path: str, device: str) -> dict:
    """Load an evaluation script and call its prepare(device) entry point."""
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    namespace = runpy.run_path(str(path))
    prepare = namespace.get("prepare")
    if not callable(prepare):
        raise ValueError("Evaluation script must define prepare(device)")
    task = prepare(device)
    if not isinstance(task, dict) or not callable(task.get("run")):
        raise ValueError("prepare() must return a dict with callable 'run'")
    return task


def _prepare_task(eval_path: str) -> tuple[Path, NvidiaBackend, dict]:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    return path, backend, load_task(str(path), backend.device)


def run_dump_info(eval_path: str) -> Output:
    """Execute run() once and export backend analysis artifacts and summary."""
    def _run() -> dict:
        path, backend, task = _prepare_task(eval_path)
        return backend.dump_info(task["run"], Path("dumpInfo").resolve() / path.stem)

    return capture_run(_run)


def run_check(eval_path: str) -> Output:
    """Load task, JIT-compile via run(), and verify numerical correctness against ref()."""
    def _run() -> None:
        _, _, task = _prepare_task(eval_path)
        check_task(task)

    return capture_run(_run)


def read_kernel_path(eval_source: bytes, eval_path: Path) -> Path:
    """Return the kernel file named by the KERNEL_PATH string literal (relative to the eval script)."""
    tree = ast.parse(eval_source, filename=str(eval_path))
    value = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "KERNEL_PATH"):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError("KERNEL_PATH must be a string literal")
            value = node.value.value
    if value is None:
        raise ValueError(f"{eval_path} must declare KERNEL_PATH = \"<kernel file>\" for gk bench")
    kernel = (eval_path.parent / value).resolve()
    if not kernel.is_file():
        raise ValueError(f"KERNEL_PATH file not found: {kernel}")
    return kernel


def _module_loaded_from(path: Path) -> bool:
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() == path:
            return True
    return False


def run_bench(eval_path: str, *, trace: bool = False, no_baseline: bool = False,
              note: str | None = None, parent_id: str | None = None) -> Output:
    """Measure mean GPU kernel time, then record the result and both sources in history."""
    def _run() -> dict:
        path = Path(eval_path).resolve()
        if not path.is_file():
            raise ValueError(f"Evaluation script not found: {path}")
        eval_source = path.read_bytes()
        kernel_file = read_kernel_path(eval_source, path)
        kernel_source = kernel_file.read_bytes()
        history = History()
        parent = history.get(parent_id)["id"] if parent_id is not None else None

        _, backend, task = _prepare_task(str(path))
        if not _module_loaded_from(kernel_file):
            raise ValueError(f"KERNEL_PATH {kernel_file} was not imported by the evaluation script")
        trace_dir = None
        trace_path = None
        if trace:
            root = Path("runs").resolve()
            root.mkdir(exist_ok=True)
            trace_dir = Path(tempfile.mkdtemp(prefix="bench-", dir=root))
            trace_path = str(trace_dir / "trace.json")
        try:
            result = bench_task(task, backend, trace_path=trace_path, no_baseline=no_baseline)
        except BaseException:
            if trace_dir is not None:
                shutil.rmtree(trace_dir, ignore_errors=True)
            raise

        # The recorded snapshot must be exactly the code that was measured.
        if path.read_bytes() != eval_source or kernel_file.read_bytes() != kernel_source:
            raise RuntimeError("Kernel or evaluation source changed during bench; not recorded, rerun")
        info = backend.probe()
        record = history.save({
            "kernel_path": history.normalize_path(kernel_file),
            "kernel_source_hash": history.store_source(kernel_source),
            "eval_path": history.normalize_path(path),
            "eval_source_hash": history.store_source(eval_source),
            "parent_id": parent,
            "note": note,
            "platform": info.platform,
            "arch": info.arch,
            "device_info": info.to_dict(),
            "env": {
                "python": platform.python_version(),
                "torch": info.torch_version,
                "triton": info.compiler_version,
            },
            "measurement_options": {
                "warmup": DEFAULT_WARMUP,
                "iters": DEFAULT_ITERS,
                "flush_l2": False,
                "timing": "torch.profiler device kernel mean per call",
            },
            "bench": result.to_dict(),
        })
        return {"history_id": record["id"], **result.to_dict()}

    return capture_run(_run)
