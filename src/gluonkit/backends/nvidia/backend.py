from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import ProbeResult

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
