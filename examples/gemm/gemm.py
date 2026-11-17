import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout, SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2


@g.jit
def _gemm(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    BM: gl.constexpr = 32
    BN: gl.constexpr = 64
    BK: gl.constexpr = 32
    load_layout: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = NVMMADistributedLayout([2, 0], [4, 1], [16, 8])
    rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=SliceLayout(1, load_layout))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, layout=SliceLayout(0, load_layout))
    ak = gl.arange(0, BK, layout=SliceLayout(0, load_layout))
    bk = gl.arange(0, BK, layout=SliceLayout(1, load_layout))
    a_smem = gl.allocate_shared_memory(gl.float16, [BM, BK], SwizzledSharedLayout(8, 1, 4, [1, 0]))
    b_smem = gl.allocate_shared_memory(gl.float16, [BK, BN], SwizzledSharedLayout(8, 1, 8, [1, 0]))
    acc = gl.full([BM, BN], 0, gl.float32, mma_layout)
    for offset in range(gl.cdiv(K, BK)):
        a_k = offset * BK + ak
        b_k = offset * BK + bk
        async_copy.async_load(a_smem, A + rows[:, None] * K + a_k[None, :],
                              (rows[:, None] < M) & (a_k[None, :] < K))
        async_copy.async_load(b_smem, B + b_k[:, None] * N + cols[None, :],
                              (b_k[:, None] < K) & (cols[None, :] < N))
        async_copy.commit_group()
        async_copy.wait_group(0)
        a = a_smem.load(DotOperandLayout(0, mma_layout, 2))
        b = b_smem.load(DotOperandLayout(1, mma_layout, 2))
        acc = mma_v2(a, b, acc)
    result = gl.convert_layout(acc.to(gl.float16), load_layout)
    gl.store(C + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))


def gemm(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible 2D matrices")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("This example supports FP16 inputs only")
    if not a.is_cuda or a.device != b.device or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Inputs must be contiguous and on the same CUDA device")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](a, b, c, m, n, k, num_warps=4)
    return c
