import argparse
import json
import sys

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.interfaces.backend import Backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gk", description="Gluon kernel tools")
    commands = parser.add_subparsers(dest="command", required=True)

    probe_parser = commands.add_parser("probe", help="Show NVIDIA GPU information")
    probe_parser.add_argument("--json", action="store_true", help="Output JSON")

    layout_parser = commands.add_parser("layout", help="Show a Gluon layout mapping (no GPU required)")
    layout_parser.add_argument("expression", help="BlockedLayout(...) or SliceLayout(...)")
    layout_parser.add_argument("--shape", required=True, help="Tensor dimensions, e.g. 64,64")
    layout_parser.add_argument("--hardware", action="store_true", help="Group coordinates by warp/thread")

    compile_parser = commands.add_parser("compile", help="Run an evaluation script to trigger JIT compilation")
    compile_parser.add_argument("eval_path", help="Python script defining prepare(device)")
    compile_parser.add_argument("--json", action="store_true", help="Output JSON")

    check_parser = commands.add_parser("check", help="Check an evaluation script for correctness")
    check_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    bench_parser = commands.add_parser("bench", help="Measure mean GPU kernel time")
    bench_parser.add_argument("eval_path", help="Python script defining prepare(device)")
    bench_parser.add_argument("--trace", action="store_true", help="Export measured run's Chrome trace")
    bench_parser.add_argument("--no-baseline", action="store_true", help="Do not measure ref")
    bench_parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    if args.command == "probe":
        backend: Backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            print(f"gk: {reason}", file=sys.stderr)
            return 1
        try:
            result = backend.probe()
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            smem = result.smem_per_block_optin
            print(f"Device: {result.device_name}")
            print(f"Platform: {result.platform} / {result.arch}")
            print(f"SMs: {result.sm_count}")
            print(f"VRAM: {result.vram_bytes / 1024**3:.2f} GiB")
            print(f"Shared memory / block (opt-in): {smem if smem is not None else 'unknown'} bytes")
            print(f"PyTorch: {result.torch_version}")
            print(f"Triton: {result.compiler_version}")
        return 0

    if args.command == "layout":
        backend = NvidiaBackend()
        try:
            shape = [int(dim.strip()) for dim in args.shape.split(",")]
            print(backend.layout(args.expression, shape, args.hardware), end="")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "compile":
        from gluonkit.executor import run_compile

        try:
            result = run_compile(args.eval_path)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        if not result.ok:
            for error in result.errors:
                print(error.message, file=sys.stderr)
            return 1
        if not args.json:
            print("Compile succeeded")
            for key, paths in result.artifacts.items():
                print(f"IR ({key}):")
                for extension, path in paths.items():
                    print(f"  {extension}: {path or 'unavailable'}")
        return 0

    if args.command == "check":
        from gluonkit.executor import run_check

        try:
            result = run_check(args.eval_path)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        if result.error:
            print(result.error, file=sys.stderr)
        print("Check passed" if result.passed else "Check failed")
        return 0 if result.passed else 1

    if args.command == "bench":
        from gluonkit.executor import run_bench

        try:
            result = run_bench(args.eval_path, trace=args.trace,
                               no_baseline=args.no_baseline)
        except Exception as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False))
        else:
            print(f"Device kernel time (mean): {result.device_time_ms:.6f} ms")
            for label, value, unit in (
                ("Baseline (ref)", result.baseline_time_ms, "ms"),
                ("Speedup vs baseline", result.speedup_vs_baseline, "x"),
                ("Compute throughput", result.tflops, "TFLOP/s"),
                ("Estimated bandwidth", result.gbps, "GB/s"),
                ("Roofline", result.roofline_pct, "%"),
            ):
                print(f"{label}: {value:.6f} {unit}" if value is not None else f"{label}: n/a")
            print(f"Bound by: {result.bound_by or 'n/a'}")
            print("Kernels (mean per launch):")
            for kernel in result.kernels:
                print(f"  {kernel.name}: {kernel.avg_ms:.6f} ms")
            if result.trace_path:
                print(f"Trace: {result.trace_path}")
        return 0

    return 1
