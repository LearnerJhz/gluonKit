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
    args = parser.parse_args(argv)

    backend: Backend = NvidiaBackend()
    if args.command == "layout":
        try:
            shape = [int(dim.strip()) for dim in args.shape.split(",")]
            print(backend.layout(args.expression, shape, args.hardware), end="")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"gk: {exc}", file=sys.stderr)
            return 1
        return 0

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
