"""Task loading and in-process execution for evaluation scripts."""

import ast
from pathlib import Path, PurePosixPath
import platform
import runpy
import sys
import tempfile
import traceback

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.core.checker import check_task
from gluonkit.core.bencher import DEFAULT_ITERS, DEFAULT_WARMUP, bench_task
from gluonkit.core.history import History
from gluonkit.interfaces.types import BenchResult, CheckResult, CompileResult, DumpInfoResult


def load_task(eval_path: str, device: str) -> dict:
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


def run_compile(eval_path: str) -> CompileResult:
    try:
        path = Path(eval_path).resolve()
        if not path.is_file():
            raise ValueError(f"Evaluation script not found: {path}")
        backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            raise RuntimeError(reason)
        task = load_task(str(path), backend.device)
        task["run"]()
        return CompileResult(result="success")
    except Exception:
        return CompileResult(result="failed", info=traceback.format_exc())


def run_dump_info(eval_path: str) -> DumpInfoResult:
    """Run the selected task and ask its backend to export analysis artifacts."""
    try:
        path = Path(eval_path).resolve()
        backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            raise RuntimeError(reason)
        task = load_task(str(path), backend.device)
        artifacts = backend.dump_info(task["run"], Path("dumpInfo").resolve() / path.stem)
        return DumpInfoResult(result="success", artifacts=artifacts)
    except Exception:
        return DumpInfoResult(result="failed", info=traceback.format_exc())


def run_check(eval_path: str) -> CheckResult:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    try:
        backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            raise RuntimeError(reason)
        task = load_task(str(path), backend.device)
        return check_task(task)
    except Exception:
        return CheckResult(passed=False, error=traceback.format_exc())


def run_bench(eval_path: str, *, trace: bool = False,
              no_baseline: bool = False) -> BenchResult:
    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    task = load_task(eval_path, backend.device)
    trace_path = None
    if trace:
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        trace_path = str(Path(tempfile.mkdtemp(prefix="bench-", dir=root)) / "trace.json")
    return bench_task(task, backend, trace_path=trace_path, no_baseline=no_baseline)


def read_kernel_path(eval_source: bytes, eval_path: Path) -> Path:
    """Return KERNEL_PATH (a string literal, relative to the evaluation script's directory)."""
    tree = ast.parse(eval_source, filename=str(eval_path))
    value = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "KERNEL_PATH"):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError("KERNEL_PATH must be a string literal")
            value = node.value.value
    if value is None:
        raise ValueError(f"{eval_path} must declare KERNEL_PATH = \"<kernel file>\" for gk run")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("KERNEL_PATH must be a relative path inside the evaluation script's directory")
    return Path(*relative.parts)


def _drop_modules(name: str) -> None:
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[module]


def run_pipeline(eval_path: str, *, note: str | None = None, parent_id: str | None = None,
                 trace: bool = False, history: History | None = None) -> dict:
    """compile -> check -> bench on one locked source snapshot; always recorded in history."""
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)

    # Read each source exactly once: hashes, stored snapshots and executed code share these bytes.
    eval_source = path.read_bytes()
    kernel_relative = read_kernel_path(eval_source, path)
    kernel_file = path.parent / kernel_relative
    if not kernel_file.is_file():
        raise ValueError(f"KERNEL_PATH file not found: {kernel_file}")
    kernel_source = kernel_file.read_bytes()

    history = history or History()
    if parent_id is not None:
        parent_id = history.get(parent_id)["id"]
    info = backend.probe()

    root = Path("runs").resolve()
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    snapshot = run_dir / "snapshot"
    snapshot_eval = snapshot / path.name
    snapshot_kernel = snapshot / kernel_relative
    snapshot_kernel.parent.mkdir(parents=True, exist_ok=True)
    snapshot_eval.write_bytes(eval_source)
    snapshot_kernel.write_bytes(kernel_source)

    record = {
        "kernel_path": history.normalize_path(kernel_file),
        "kernel_source_hash": history.store_source(kernel_source),
        "eval_path": history.normalize_path(path),
        "eval_source_hash": history.store_source(eval_source),
        "parent_id": parent_id,
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
        "run_dir": str(run_dir),
        "status": None,
        "error": None,
        "compile": None,
        "check": None,
        "bench": None,
    }

    module = kernel_relative.parts[0].removesuffix(".py")
    _drop_modules(module)  # Never reuse a kernel module imported from the live working tree.
    try:
        try:
            task = load_task(str(snapshot_eval), backend.device)
            task["run"]()
            record["compile"] = CompileResult(result="success").to_dict()
        except Exception:
            record["compile"] = CompileResult(
                result="failed", info=traceback.format_exc()).to_dict()
            record["status"] = "compile_failed"
            return history.save(record)

        check = check_task(task)
        record["check"] = check.to_dict()
        if not check.passed:
            record["status"] = "check_failed"
            return history.save(record)

        try:
            trace_path = str(run_dir / "trace.json") if trace else None
            record["bench"] = bench_task(task, backend, trace_path=trace_path).to_dict()
            record["status"] = "ok"
        except Exception:
            record["status"] = "bench_failed"
            record["error"] = traceback.format_exc()
        return history.save(record)
    finally:
        _drop_modules(module)
        if str(snapshot) in sys.path:
            sys.path.remove(str(snapshot))
