import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _softmax(X, Y, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    row = gl.program_id(0)
    cols = gl.arange(0, N, layout=layout)
    offsets = row * N + cols
    values = gl.load(X + offsets).to(gl.float32)
    numerators = gl.exp(values - gl.max(values, axis=0))
    probabilities = numerators / gl.sum(numerators, axis=0)
    gl.store(Y + offsets, probabilities.to(gl.float16))


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply Softmax along the last axis of a contiguous 4096 × 4096 CUDA FP16 tensor."""
    if x.shape != (4096, 4096) or x.dtype != torch.float16:
        raise ValueError("Expected a 4096 × 4096 FP16 tensor")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("Expected a contiguous CUDA tensor")

    rows, cols = x.shape
    output = torch.empty_like(x)
    _softmax[(rows,)](x, output, cols, num_warps=4)
    return output
