from typing import Any, Callable, Protocol
from pathlib import Path

from .types import ProbeResult, ProfileData


class Backend(Protocol):
    device: str

    def probe(self) -> ProbeResult: ...

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
