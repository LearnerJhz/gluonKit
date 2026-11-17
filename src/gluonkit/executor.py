"""Subprocess evaluation worker. Evaluation code is imported only in the child process."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import traceback

from gluonkit.interfaces.types import CheckResult, CompileError, CompileResult


def run_compile(eval_path: str, seed: int = 0) -> dict:
    return _run_worker("compile", eval_path, seed)


def run_check(eval_path: str, seed: int = 0) -> dict:
    return _run_worker("check", eval_path, seed)


def _run_worker(command: str, eval_path: str, seed: int) -> dict:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    root = Path("runs").resolve()
    root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{command}-", dir=root))
    request = directory / "request.json"
    request.write_text(json.dumps({
        "command": command,
        "eval_path": str(path),
        "seed": seed,
    }))
    process = subprocess.run(
        [sys.executable, "-m", "gluonkit.executor", str(request)],
        capture_output=True, text=True,
    )
    # User scripts may print; keep the parent's JSON stdout clean.
    if process.stdout:
        print(process.stdout, file=sys.stderr, end="")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"{command.capitalize()} worker exited with code {process.returncode} without a result")
    result = json.loads(result_path.read_text())
    if process.returncode and _is_success(command, result):
        raise RuntimeError(f"{command.capitalize()} worker exited with code {process.returncode}")
    return result


def _load_task(eval_path: str, device: str, seed: int) -> dict:
    path = Path(eval_path)
    sys.path.insert(0, str(path.parent))
    namespace = runpy.run_path(str(path))
    prepare = namespace.get("prepare")
    if not callable(prepare):
        raise ValueError("Evaluation script must define prepare(device, seed)")
    task = prepare(device, seed)
    if not isinstance(task, dict) or not callable(task.get("run")):
        raise ValueError("prepare() must return a dict with callable 'run'")
    return task


def execute(request: dict, directory: Path) -> dict:
    from gluonkit.backends.nvidia.backend import NvidiaBackend

    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)

    command = request.get("command", "compile")
    seed = request["seed"]
    task = _load_task(request["eval_path"], backend.device, seed)

    if command == "check":
        from gluonkit.core.checker import check_task

        return check_task(task, seed).to_dict()

    from gluonkit.backends.nvidia.compiler import export_ir

    artifacts = export_ir(task["run"], directory / "artifacts")
    result = CompileResult(ok=True, errors=[], artifacts=artifacts).to_dict()
    return {**result, "seed": seed}


def _failure_result(command: str, message: str, seed: int) -> dict:
    if command == "check":
        return CheckResult(passed=False, error=message, seed=seed).to_dict()
    return CompileResult(ok=False, errors=[CompileError(message)]).to_dict()


def _is_success(command: str, result: dict) -> bool:
    return bool(result["passed" if command == "check" else "ok"])


def main() -> int:
    request_path = Path(sys.argv[1])
    request: dict = {}
    try:
        request = json.loads(request_path.read_text())
        result = execute(request, request_path.parent)
    except Exception:
        result = _failure_result(
            request.get("command", "compile"),
            traceback.format_exc(),
            request.get("seed", 0),
        )
    command = request.get("command", "compile")
    (request_path.parent / "result.json").write_text(json.dumps(result))
    return 0 if _is_success(command, result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
