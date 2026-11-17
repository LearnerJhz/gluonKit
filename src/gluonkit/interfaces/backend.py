from typing import Protocol

from .types import ProbeResult


class Backend(Protocol):
    name: str
    device: str

    # 执行前检查 backend 能不能用
    def available(self) -> tuple[bool, str]: ...

    def probe(self) -> ProbeResult: ...

    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...
