import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout, SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2


@g.jit
def _gemm(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
          BM: gl.constexpr = 64, BN: gl.constexpr = 64, BK: gl.constexpr = 32,
          NUM_STAGES: gl.constexpr = 2):
    NUM_PREFETCH: gl.constexpr = NUM_STAGES - 1
    a_load_layout: gl.constexpr = BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
    b_load_layout: gl.constexpr = BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = NVMMADistributedLayout([2, 0], [2, 2], [16, 8])
    rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=SliceLayout(1, a_load_layout))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, layout=SliceLayout(0, b_load_layout))
    ak = gl.arange(0, BK, layout=SliceLayout(0, a_load_layout))
    bk = gl.arange(0, BK, layout=SliceLayout(1, b_load_layout))
    a_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BM, BK], SwizzledSharedLayout(8, 2, 4, [1, 0]))
    b_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BK, BN], SwizzledSharedLayout(8, 1, 8, [1, 0]))
    acc = gl.full([BM, BN], 0, gl.float32, mma_layout)
    num_tiles = gl.cdiv(K, BK)

    for s in gl.static_range(NUM_PREFETCH):
        if s < num_tiles:
            a_k = s * BK + ak
            b_k = s * BK + bk
            async_copy.async_load(a_smem.index(s), A + rows[:, None] * K + a_k[None, :],
                                  (rows[:, None] < M) & (a_k[None, :] < K))
            async_copy.async_load(b_smem.index(s), B + b_k[:, None] * N + cols[None, :],
                                  (b_k[:, None] < K) & (cols[None, :] < N))
            async_copy.commit_group()

    for offset in range(0, num_tiles - NUM_PREFETCH):
        load_idx = offset + NUM_PREFETCH
        load_stage = load_idx % NUM_STAGES
        a_k = load_idx * BK + ak
        b_k = load_idx * BK + bk
        async_copy.async_load(a_smem.index(load_stage), A + rows[:, None] * K + a_k[None, :],
                              (rows[:, None] < M) & (a_k[None, :] < K))
        async_copy.async_load(b_smem.index(load_stage), B + b_k[:, None] * N + cols[None, :],
                              (b_k[:, None] < K) & (cols[None, :] < N))
        async_copy.commit_group()
        async_copy.wait_group(NUM_PREFETCH)

        comp_stage = offset % NUM_STAGES
        a = a_smem.index(comp_stage).load(DotOperandLayout(0, mma_layout, 2))
        b = b_smem.index(comp_stage).load(DotOperandLayout(1, mma_layout, 2))
        acc = mma_v2(a, b, acc)

    for d in gl.static_range(NUM_PREFETCH):
        comp_idx = num_tiles - (NUM_PREFETCH - d)
        if comp_idx >= 0:
            async_copy.wait_group(NUM_PREFETCH - 1 - d)
            comp_stage = comp_idx % NUM_STAGES
            a = a_smem.index(comp_stage).load(DotOperandLayout(0, mma_layout, 2))
            b = b_smem.index(comp_stage).load(DotOperandLayout(1, mma_layout, 2))
            acc = mma_v2(a, b, acc)

    c_rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=SliceLayout(1, b_load_layout))
    result = gl.convert_layout(acc.to(gl.float16), b_load_layout)
    gl.store(C + c_rows[:, None] * N + cols[None, :], result,
             (c_rows[:, None] < M) & (cols[None, :] < N))


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
    BM, BN, BK = 64, 64, 32
    _gemm[(triton.cdiv(m, BM), triton.cdiv(n, BN))](a, b, c, m, n, k, BM=BM, BN=BN, BK=BK, num_warps=4)
    return c
