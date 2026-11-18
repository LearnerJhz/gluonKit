"""Task loading and in-process execution for evaluation scripts."""

from pathlib import Path
import runpy
import shutil
import sys
import tempfile

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.core.bencher import bench_task
from gluonkit.core.checker import check_task
from gluonkit.interfaces.types import BenchResult, Output
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


def run_compile(eval_path: str) -> Output:
    """Load task and execute run() once to trigger JIT compilation."""
    def _run() -> None:
        _, _, task = _prepare_task(eval_path)
        task["run"]()

    return capture_run(_run)


def run_dump_info(eval_path: str) -> Output:
    """Execute run() once and export backend IR artifacts."""
    def _run() -> dict:
        path, backend, task = _prepare_task(eval_path)
        artifacts = backend.dump_info(task["run"], Path("dumpInfo").resolve() / path.stem)
        return {"artifacts": artifacts}

    return capture_run(_run)


def run_check(eval_path: str) -> Output:
    """Load task and verify numerical correctness against ref()."""
    def _run() -> None:
        _, _, task = _prepare_task(eval_path)
        check_task(task)

    return capture_run(_run)


def run_bench(eval_path: str, *, trace: bool = False,
              no_baseline: bool = False) -> Output:
    """Load task and measure mean GPU kernel execution time."""
    def _run() -> BenchResult:
        _, backend, task = _prepare_task(eval_path)
        trace_dir = None
        trace_path = None
        if trace:
            root = Path("runs").resolve()
            root.mkdir(exist_ok=True)
            trace_dir = Path(tempfile.mkdtemp(prefix="bench-", dir=root))
            trace_path = str(trace_dir / "trace.json")
        try:
            return bench_task(task, backend, trace_path=trace_path, no_baseline=no_baseline)
        except BaseException:
            if trace_dir is not None:
                shutil.rmtree(trace_dir, ignore_errors=True)
            raise

    return capture_run(_run)
