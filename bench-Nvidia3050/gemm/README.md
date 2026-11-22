# GEMM

RTX 3050 6GB Laptop GPU（sm86），计算 `A[1024, 512] @ B[512, 1024]`，输出 `C[1024, 1024]`。输入和输出均为行主序连续 FP16，三个算子均使用 FP32 累加。用同一份[评测](eval_gemm.py)的输入和 `torch.matmul` 参考结果校验，三个算子的最大绝对误差均为 `0`。

计时使用 `gk bench` 的 NVIDIA profiler。每轮预热 10 次、采样 100 次，每次采样前清空 L2，记录 GPU kernel 耗时。下表是同机交替复测三轮的平均值。

| 实现 | 平均耗时 | 源码 |
| --- | ---: | --- |
| Gluon，`64×64×32` CTA tile，4 warps，双缓冲 | **76.74 µs** | [gemm.py](gemm.py) |
| Triton 3.8.0，`64×64×32` CTA tile，4 warps，3 stages，`GROUP_SIZE_M=8` | 78.52 µs | [上游矩阵乘源码](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/03-matrix-multiplication.py) |
| CUTLASS 3.9.2，`64×64×32` CTA tile，`32×32×32` warp tile，3 stages | 78.95 µs | [上游 CUDA GEMM 源码](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/device/gemm.h) |
| `torch.matmul`，同轮参考 | 103.30 µs | [评测中的参考调用](eval_gemm.py) |

筛选了 23 组 Triton 配置和 18 组 CUTLASS 配置，再复测各自最快的一组。Gluon 达到约 14.0 TFLOP/s，是同轮 `torch.matmul` 的 1.35 倍。
