import argparse
import os
from pathlib import Path

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import ProbeResult
from gluonkit.output import capture_run, emit_json


def _configure_env() -> None:
    """Configure local Triton cache and suppress Kineto USDT stderr logs before torch/triton init."""
    os.environ["TRITON_CACHE_DIR"] = str((Path.cwd() / ".gluonkit" / "triton_cache").resolve())
    os.environ.setdefault("KINETO_LOG_LEVEL", "6")


def _run_probe() -> ProbeResult:
    backend: Backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    return backend.probe()


def _run_layout(expression: str, shape_arg: str, hardware: bool) -> str:
    shape = [int(dim.strip()) for dim in shape_arg.split(",")]
    return NvidiaBackend().layout(expression, shape, hardware)


def _run_bank_conflicts(
    register_layout: str, shared_layout: str, shape_arg: str, dtype: str,
) -> dict[str, int]:
    shape = [int(dim.strip()) for dim in shape_arg.split(",")]
    return NvidiaBackend().bank_conflicts(register_layout, shared_layout, shape, dtype)


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

    if args.command == "probe":
        return emit_json(capture_run(_run_probe))

    if args.command == "layout":
        return emit_json(capture_run(lambda: _run_layout(args.expression, args.shape, args.hardware)))

    if args.command == "bank-conflicts":
        return emit_json(capture_run(lambda: _run_bank_conflicts(
            args.register_layout, args.shared_layout, args.shape, args.dtype,
        )))

    if args.command == "dump-info":
        from gluonkit.executor import run_dump_info

        return emit_json(run_dump_info(args.eval_path))

    if args.command == "check":
        from gluonkit.executor import run_check

        return emit_json(run_check(args.eval_path))

    if args.command == "bench":
        from gluonkit.executor import run_bench

        return emit_json(run_bench(args.eval_path, trace=args.trace, no_baseline=args.no_baseline,
                                   note=args.note, parent_id=args.parent))

    return 1
