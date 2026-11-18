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

    dump_parser = commands.add_parser("dumpInfo", help="Run once and export backend analysis information")
    dump_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    check_parser = commands.add_parser("check", help="Check an evaluation script for correctness")
    check_parser.add_argument("eval_path", help="Python script defining prepare(device)")

    bench_parser = commands.add_parser("bench", help="Measure mean GPU kernel time")
    bench_parser.add_argument("eval_path", help="Python script defining prepare(device)")
    bench_parser.add_argument("--trace", action="store_true", help="Export measured run's Chrome trace")
    bench_parser.add_argument("--no-baseline", action="store_true", help="Do not measure ref")
    bench_parser.add_argument("--json", action="store_true", help="Output JSON")

    run_parser = commands.add_parser("run", help="compile -> check -> bench and record in history")
    run_parser.add_argument("eval_path", help="Python script defining prepare(device) and KERNEL_PATH")
    run_parser.add_argument("--note", help="What changed and why")
    run_parser.add_argument("--parent", help="Parent history record id")
    run_parser.add_argument("--trace", action="store_true", help="Export measured run's Chrome trace")

    history_parser = commands.add_parser("history", help="Inspect experiment history")
    history_commands = history_parser.add_subparsers(dest="history_command", required=True)

    list_parser = history_commands.add_parser("list", help="List records")
    list_parser.add_argument("--kernel", help="Only records for this kernel file")

    show_parser = history_commands.add_parser("show", help="Show one record as JSON")
    show_parser.add_argument("id")

    best_parser = history_commands.add_parser("best", help="Fastest checked comparable record")
    best_parser.add_argument("kernel", nargs="?", help="Kernel file, e.g. examples/gemm/gemm.py")
    best_parser.add_argument("--like", help="Rank only records comparable with this id")

    diff_parser = history_commands.add_parser("diff", help="Source and metric differences")
    diff_parser.add_argument("id_a")
    diff_parser.add_argument("id_b")
    
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
        if result.result == "failed":
            if not args.json:
                print(result.info, file=sys.stderr)
            return 1
        if not args.json:
            print("Compile succeeded")
        return 0

    if args.command == "dumpInfo":
        from gluonkit.executor import run_dump_info

        result = run_dump_info(args.eval_path)
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return 0 if result.result == "success" else 1

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

    if args.command == "run":
        from gluonkit.executor import run_pipeline

        try:
            record = run_pipeline(args.eval_path, note=args.note, parent_id=args.parent,
                                  trace=args.trace)
        except Exception as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        compile_result, check, bench = record["compile"], record["check"], record["bench"]
        print(f"Compile: {'ok' if compile_result['result'] == 'success' else 'failed'}")
        if compile_result["result"] == "failed":
            print(compile_result["info"], file=sys.stderr)
        if check is not None:
            print(f"Check: {'passed' if check['passed'] else 'failed'}")
            if check["error"]:
                print(check["error"], file=sys.stderr)
        if bench is not None:
            print(f"Device kernel time (mean): {bench['device_time_ms']:.6f} ms")
            for label, name, unit in (
                ("Baseline (ref)", "baseline_time_ms", "ms"),
                ("Speedup vs baseline", "speedup_vs_baseline", "x"),
                ("Compute throughput", "tflops", "TFLOP/s"),
                ("Estimated bandwidth", "gbps", "GB/s"),
            ):
                value = bench[name]
                print(f"{label}: {value:.6f} {unit}" if value is not None else f"{label}: n/a")
            if bench["trace_path"]:
                print(f"Trace: {bench['trace_path']}")
        if record["error"]:
            print(record["error"], file=sys.stderr)
        print(f"Recorded {record['id']} ({record['status']})")
        return 0 if record["status"] == "ok" else 1

    if args.command == "history":
        from gluonkit.core.history import METRICS, History

        history = History()
        try:
            if args.history_command == "list":
                records = history.list(args.kernel)
                if not records:
                    print("No history records")
                for record in records:
                    _print_record_line(record)
            elif args.history_command == "show":
                print(json.dumps(history.get(args.id), ensure_ascii=False, indent=2))
            elif args.history_command == "best":
                record = history.best(args.kernel, like_id=args.like)
                if record is None:
                    print("gk: no checked bench record matches", file=sys.stderr)
                    return 1
                _print_record_line(record)
            elif args.history_command == "diff":
                result = history.diff(args.id_a, args.id_b)
                print(f"Diff {result['a']} -> {result['b']}")
                if result["comparable"]:
                    print("Comparable: yes")
                else:
                    print("Comparable: NO - performance change is not a direct optimization gain")
                    for reason in result["reasons"]:
                        print(f"  - {reason}")
                print("Metrics (a -> b):")
                for name in METRICS:
                    metric = result["metrics"][name]
                    change = metric["change_pct"]
                    suffix = f" ({change:+.2f}%)" if change is not None else ""
                    print(f"  {name}: {_fmt(metric['a'])} -> {_fmt(metric['b'])}{suffix}")
                for label, text in (("Kernel", result["kernel_diff"]),
                                    ("Eval", result["eval_diff"])):
                    print(f"{label} source: {'changed' if text else 'unchanged'}")
                    if text:
                        print(text, end="" if text.endswith("\n") else "\n")
        except (OSError, ValueError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        return 0

    return 1


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _print_record_line(record: dict) -> None:
    bench = record.get("bench") or {}
    time_ms = bench.get("device_time_ms")
    print("  ".join([
        record["id"],
        record.get("timestamp", ""),
        f"{record.get('status')}",
        f"{time_ms:.6f} ms" if time_ms is not None else "n/a",
        f"kernel {record.get('kernel_source_hash', '')[:8]}",
        f"eval {record.get('eval_source_hash', '')[:8]}",
        record.get("note") or "",
    ]).rstrip())
