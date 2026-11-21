import argparse
import os
from pathlib import Path
import sys
from typing import Any

from gluonkit.backends.registry import get_backend
from gluonkit.executor import (
    run_bank_conflicts,
    run_baseline,
    run_bench,
    run_check,
    run_dump_info,
    run_history,
    run_layout,
    run_probe,
    run_restore,
)
from gluonkit.interfaces.backend import Backend
from gluonkit.output import capture_run, emit_json


def _configure_env() -> None:
    """Configure local Triton cache and suppress Kineto USDT stderr logs before torch/triton init."""
    os.environ["TRITON_CACHE_DIR"] = str((Path.cwd() / ".gluonkit" / "triton_cache").resolve())
    os.environ.setdefault("KINETO_LOG_LEVEL", "6")


def _parse_shape(shape_arg: str) -> list[int]:
    return [int(dim.strip()) for dim in shape_arg.split(",")]


def _run_api(args: argparse.Namespace) -> int:
    """Plain-text Gluon API introspection: stdout for results, stderr + exit 1 for unresolved symbols."""
    from gluonkit.core import gluon_api

    try:
        mod = gluon_api.resolve_module(args.module)
    except ImportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.api_command == "list":
        print(gluon_api.list_module(mod))
        return 0
    text, errors = gluon_api.show_symbols(
        mod,
        args.symbols,
        fallback=args.module is None,
        full=getattr(args, "full", False),
    )
    if text:
        print(text)
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


def _dispatch(args: argparse.Namespace, backend: Backend) -> Any:
    if args.command == "probe":
        return run_probe(backend)
    if args.command == "layout":
        return run_layout(args.expression, _parse_shape(args.shape), backend, hardware=args.hardware)
    if args.command == "bank-conflicts":
        return run_bank_conflicts(
            args.register_layout, args.shared_layout, _parse_shape(args.shape), args.dtype, backend,
        )
    if args.command == "dump-info":
        return run_dump_info(args.eval_path, backend)
    if args.command == "check":
        return run_check(args.eval_path, backend)
    if args.command == "baseline":
        return run_baseline(args.eval_path, backend, refresh=getattr(args, "refresh", False))
    if args.command == "bench":
        return run_bench(
            args.eval_path, backend,
            trace=args.trace, note=args.note, parent_id=args.parent,
        )
    if args.command == "history":
        return run_history(args.target, limit=args.limit)
    if args.command == "restore":
        return run_restore(args.record_id)
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gk", description="Gluon kernel tools")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("probe", help="Show GPU information and calibrated peaks for the device")

    layout_parser = commands.add_parser("layout", help="Show a Gluon layout mapping (no GPU required)")
    layout_parser.add_argument("expression", help="BlockedLayout(...) or SliceLayout(...)")
    layout_parser.add_argument("--shape", required=True, help="Tensor dimensions, e.g. 64,64")
    layout_parser.add_argument("--hardware", action="store_true", help="Group coordinates by warp/thread")

    bank_parser = commands.add_parser(
        "bank-conflicts", help="Count shared-memory bank conflicts (no GPU required)",
        description="Report excess shared-memory accesses per wavefront; 0 means no bank conflicts.",
    )
    bank_parser.add_argument(
        "--register-layout", required=True,
        help="BlockedLayout(...), SliceLayout(...), DotOperandLayout(...) or NVMMADistributedLayout(...)",
    )
    bank_parser.add_argument("--shared-layout", required=True, help="SwizzledSharedLayout(...)")
    bank_parser.add_argument("--shape", required=True, help="Tensor dimensions, e.g. 64,32")
    bank_parser.add_argument("--dtype", required=True, help="Gluon element type, e.g. float16 or float32")

    dump_parser = commands.add_parser("dump-info",
                                      help="Run once and export backend analysis information")
    dump_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    check_parser = commands.add_parser("check",
                                       help="Compile (JIT) and check correctness against ref")
    check_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    baseline_parser = commands.add_parser(
        "baseline",
        help="Measure and store ref baseline once if missing",
    )
    baseline_parser.add_argument("eval_path", help="Python script (eval_<name>.py) defining prepare(device)")
    baseline_parser.add_argument("--refresh", action="store_true", help="Re-measure and overwrite stored baseline")

    bench_parser = commands.add_parser(
        "bench",
        help="Measure mean GPU kernel time and record it in history (roofline uses calibrated peaks)",
    )
    bench_parser.add_argument("eval_path", help="Python script (eval_<name>.py) defining prepare(device)")
    bench_parser.add_argument("--note", help="What changed and why (stored in history)")
    bench_parser.add_argument("--parent", help="Parent history record id")
    bench_parser.add_argument("--trace", action="store_true", help="Export the measured run's Chrome trace")

    history_parser = commands.add_parser("history", help="List recorded bench experiments")
    history_parser.add_argument("target", nargs="?", help="Optional kernel or eval path filter")
    history_parser.add_argument("--limit", type=int, help="Return only the most recent N records")

    restore_parser = commands.add_parser("restore", help="Restore a kernel file from a history snapshot")
    restore_parser.add_argument("record_id", help="History record id (or unique prefix)")

    api_parser = commands.add_parser(
        "api", help="Inspect live Gluon APIs (plain text output, no GPU required)",
    )
    api_commands = api_parser.add_subparsers(dest="api_command", required=True)
    api_list = api_commands.add_parser("list", help="List public APIs and submodules of a module")
    api_list.add_argument("module", nargs="?", default=None, help="Module to list (default: gl)")
    api_show = api_commands.add_parser("show", help="Show signature and doc of symbols")
    api_show.add_argument("symbols", nargs="+", help="Symbol names, e.g. max sum exp load")
    api_show.add_argument("-m", "--module", default=None, help="Search only this module (default: gl, falling back to gluon)")
    api_show.add_argument("--full", action="store_true", help="Show full docstring and source file location")

    args = parser.parse_args(argv)
    _configure_env()
    if args.command == "api":
        return _run_api(args)
    backend: Backend = get_backend()
    return emit_json(capture_run(lambda: _dispatch(args, backend)))
