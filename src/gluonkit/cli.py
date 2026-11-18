import argparse

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import ProbeResult
from gluonkit.output import capture_run, emit_json


def _run_probe() -> ProbeResult:
    backend: Backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    return backend.probe()


def _run_layout(expression: str, shape_arg: str, hardware: bool) -> str:
    shape = [int(dim.strip()) for dim in shape_arg.split(",")]
    return NvidiaBackend().layout(expression, shape, hardware)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gk", description="Gluon kernel tools")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("probe", help="Show NVIDIA GPU information")

    layout_parser = commands.add_parser("layout", help="Show a Gluon layout mapping (no GPU required)")
    layout_parser.add_argument("expression", help="BlockedLayout(...) or SliceLayout(...)")
    layout_parser.add_argument("--shape", required=True, help="Tensor dimensions, e.g. 64,64")
    layout_parser.add_argument("--hardware", action="store_true", help="Group coordinates by warp/thread")

    compile_parser = commands.add_parser("compile", help="Run an evaluation script to trigger JIT compilation")
    compile_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    dump_parser = commands.add_parser("dumpInfo", help="Run once and export backend analysis information")
    dump_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    check_parser = commands.add_parser("check", help="Check an evaluation script for correctness")
    check_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    bench_parser = commands.add_parser("bench", help="Measure mean GPU kernel time")
    bench_parser.add_argument("eval_path", help="Python script defining prepare(device)")
    bench_parser.add_argument("--trace", action="store_true", help="Export measured run's Chrome trace")
    bench_parser.add_argument("--no-baseline", action="store_true", help="Do not measure ref")

    args = parser.parse_args(argv)

    if args.command == "probe":
        return emit_json(capture_run(_run_probe))

    if args.command == "layout":
        return emit_json(capture_run(lambda: _run_layout(args.expression, args.shape, args.hardware)))

    if args.command == "compile":
        from gluonkit.executor import run_compile

        return emit_json(run_compile(args.eval_path))

    if args.command == "dumpInfo":
        from gluonkit.executor import run_dump_info

        return emit_json(run_dump_info(args.eval_path))

    if args.command == "check":
        from gluonkit.executor import run_check

        return emit_json(run_check(args.eval_path))

    if args.command == "bench":
        from gluonkit.executor import run_bench

        return emit_json(run_bench(args.eval_path, trace=args.trace, no_baseline=args.no_baseline))

    return 1
