"""Base class for SIMT GPU backends sharing core Gluon layout, IR dump, and profiler primitives."""

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Callable

from gluonkit.core.simt_layout import compute_bank_conflicts, render_layout
from gluonkit.core.torch_profiler import profile_with_events, profile_with_kineto
from gluonkit.core.triton_dump import dump_triton_kernels
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import KernelSummary, ProbeResult, ProfileData


class SimtBackendBase(Backend):
    name: str = "simt"
    platform_label: str = "SIMT"
    warp_size: int = 32
    expected_ir_extensions: tuple[str, ...] = ("ttgir", "llir", "ptx")
    allow_event_fallback: bool = False

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def probe(self) -> ProbeResult:
        raise NotImplementedError

    def extra_layout_constructors(self) -> dict[str, type]:
        return {}

    def summarize_kernel(self, kernel: Any) -> KernelSummary | dict[str, Any]:
        raise NotImplementedError

    def matmul_precision_context(self, compute_mode: str) -> AbstractContextManager[None]:
        return nullcontext()

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str:
        return render_layout(
            expression,
            shape,
            hardware=hardware,
            warp_size=self.warp_size,
            platform_label=self.platform_label,
        )

    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]:
        return compute_bank_conflicts(
            register_layout,
            shared_layout,
            shape,
            dtype,
            extra_register_constructors=self.extra_layout_constructors(),
        )

    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, Any]:
        return dump_triton_kernels(
            fn,
            output_dir,
            summarize_fn=self.summarize_kernel,
            expected_extensions=self.expected_ir_extensions,
        )

    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData:
        try:
            return profile_with_kineto(
                fn,
                iters,
                warmup,
                device=self.device,
                flush_l2=flush_l2,
                trace_path=trace_path,
            )
        except RuntimeError:
            if not self.allow_event_fallback or trace_path is not None:
                raise
            return profile_with_events(
                fn,
                iters,
                warmup,
                device=self.device,
                flush_l2=flush_l2,
                trace_path=None,
            )
