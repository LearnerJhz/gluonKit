"""Minimal compile worker. Evaluation code is imported only in the child process."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import traceback

from gluonkit.interfaces.types import CompileError, CompileResult


def run_compile(eval_path: str, case: str | None = None, seed: int = 0) -> dict:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    root = Path("runs").resolve()
    root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="compile-", dir=root))
    request = directory / "request.json"
    request.write_text(json.dumps({"eval_path": str(path), "case": case, "seed": seed}))
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
        raise RuntimeError(f"Compile worker exited with code {process.returncode} without a result")
    result = json.loads(result_path.read_text())
    if process.returncode and result["ok"]:
        raise RuntimeError(f"Compile worker exited with code {process.returncode}")
    return result


def execute(request: dict, directory: Path) -> dict:
    from gluonkit.backends.nvidia.backend import NvidiaBackend
    from gluonkit.backends.nvidia.compiler import export_ir

    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    path = Path(request["eval_path"])
    sys.path.insert(0, str(path.parent))
    namespace = runpy.run_path(str(path))
    prepare = namespace.get("prepare")
    if not callable(prepare):
        raise ValueError("Evaluation script must define prepare(device, seed)")
    cases = prepare(backend.device, request["seed"])
    if not isinstance(cases, list) or not cases:
        raise ValueError("prepare() must return a non-empty list of cases")
    names = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"]:
            raise ValueError("Each case must have a non-empty name")
        if case["name"] in names or not callable(case.get("run")):
            raise ValueError("Case names must be unique and each case must provide callable run")
        names.append(case["name"])
    selected = request["case"] if request["case"] is not None else names[0]
    if selected not in names:
        raise ValueError(f"Unknown case {selected!r}; available: {', '.join(names)}")
    case = cases[names.index(selected)]
    artifacts = export_ir(case["run"], directory / "artifacts")
    result = CompileResult(ok=True, errors=[], artifacts=artifacts).to_dict()
    return {**result, "case_name": selected, "seed": request["seed"]}


def main() -> int:
    request_path = Path(sys.argv[1])
    try:
        result = execute(json.loads(request_path.read_text()), request_path.parent)
    except Exception:
        result = CompileResult(False, [CompileError(traceback.format_exc())]).to_dict()
    (request_path.parent / "result.json").write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
