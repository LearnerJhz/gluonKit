from typing import Callable, Protocol
from pathlib import Path

from .types import ProbeResult, ProfileData


class Backend(Protocol):
    name: str
    device: str

    # 执行前检查 backend 能不能用
    def available(self) -> tuple[bool, str]: ...

    def probe(self) -> ProbeResult: ...

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...

    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, dict[str, str | None]]: ...

    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...

    def flush_cache(self) -> None: ...
