from typing import Callable
from pathlib import Path

from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import ProbeResult, ProfileData

from .probe import probe


class NvidiaBackend(Backend):
    name = "nvidia"
    device = "cuda:0"

    def available(self) -> tuple[bool, str]:
        try:
            import torch
            import triton  # noqa: F401 -- probe requires both dependencies

            if not torch.cuda.is_available():
                return False, "CUDA is unavailable; check GPU access and the NVIDIA driver."
        except (ImportError, OSError, RuntimeError) as exc:
            return False, str(exc)
        return True, ""

    def probe(self) -> ProbeResult:
        return probe(self.device)

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str:
        from .layout import render_layout

        return render_layout(expression, shape, hardware)

    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, dict[str, str | None]]:
        from .dump_info import export_ir

        return export_ir(fn, output_dir)

    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData:
        from .profiler import profile

        return profile(fn, iters, warmup, device=self.device,
                       flush_l2=flush_l2, trace_path=trace_path)

    def flush_cache(self) -> None:
        from .profiler import flush_cache

        flush_cache(self.device)
