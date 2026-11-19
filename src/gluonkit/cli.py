import argparse
import os
from pathlib import Path
from typing import Any

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.executor import (
    run_bank_conflicts,
    run_bench,
    run_check,
    run_dump_info,
    run_layout,
    run_probe,
)
from gluonkit.interfaces.backend import Backend
from gluonkit.output import capture_run, emit_json


def _configure_env() -> None:
    """Configure local Triton cache and suppress Kineto USDT stderr logs before torch/triton init."""
    os.environ["TRITON_CACHE_DIR"] = str((Path.cwd() / ".gluonkit" / "triton_cache").resolve())
    os.environ.setdefault("KINETO_LOG_LEVEL", "6")


def _parse_shape(shape_arg: str) -> list[int]:
    return [int(dim.strip()) for dim in shape_arg.split(",")]


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
    if args.command == "bench":
        return run_bench(
            args.eval_path, backend,
            trace=args.trace, no_baseline=args.no_baseline,
            note=args.note, parent_id=args.parent,
        )
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gk", description="Gluon kernel tools")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("probe", help="Show NVIDIA GPU information")

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

    dump_parser = commands.add_parser("dump-info", help="Run once and export backend analysis information")
    dump_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    check_parser = commands.add_parser("check", help="Compile (JIT) and check correctness against ref")
    check_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    bench_parser = commands.add_parser("bench", help="Measure mean GPU kernel time and record it in history")
    bench_parser.add_argument("eval_path", help="Python script defining prepare(device) and KERNEL_PATH")
    bench_parser.add_argument("--note", help="What changed and why (stored in history)")
    bench_parser.add_argument("--parent", help="Parent history record id")
    bench_parser.add_argument("--trace", action="store_true", help="Export measured run's Chrome trace")
    bench_parser.add_argument("--no-baseline", action="store_true", help="Do not measure ref")

    args = parser.parse_args(argv)
    _configure_env()
    backend: Backend = NvidiaBackend("cuda:0")
    return emit_json(capture_run(lambda: _dispatch(args, backend)))
