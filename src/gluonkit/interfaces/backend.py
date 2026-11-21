from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .types import KernelSummary, ProbeResult, ProfileData


@runtime_checkable
class Backend(Protocol):
    name: str
    device: str
    warp_size: int

    def probe(self) -> ProbeResult: ...

    def extra_layout_constructors(self) -> dict[str, type]: ...

    def summarize_kernel(self, kernel: Any) -> KernelSummary | dict[str, Any]: ...

    def matmul_precision_context(self, compute_mode: str) -> AbstractContextManager[None]: ...

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...

    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]: ...

    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, Any]: ...

    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...
