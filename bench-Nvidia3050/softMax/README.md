
# Softmax

RTX 3050 6GB Laptop GPU（sm86），`4096 × 4096` 连续 FP16，按行做 Softmax。三版都用同一份[评测](eval_softmax.py)通过 `gk check`。`gk bench` 和 `gk baseline` 均预热 10 次、采样 100 次，每次采样前驱逐 L2，记录 GPU kernel 平均耗时。

| 实现 | 平均耗时 | 源码 |
| --- | ---: | --- |
| Gluon，4 个 warp 处理一行 | **422.98–423.02 µs** | [softmax.py](softmax.py) |
| AITemplate，128 位向量读写 + 共享内存归约 | 428.29 µs | [上游 CUDA 源码](https://github.com/facebookincubator/AITemplate/blob/main/python/aitemplate/backend/cuda/softmax/softmax.cuh) |
| OneFlow，每线程成对读写 + 共享内存归约 | 429.05 µs | [上游 CUDA 源码](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh) |
| PyTorch 基线，`torch.softmax(x, dim=-1)` | 696.50 µs | [评测参考实现](eval_softmax.py) |

Gluon 耗时范围来自两次复测。两个开源实现均用上游代码编译为 `sm_86` 后在本机测量。PyTorch 基线由 `gk baseline` 单独测得，版本为 `2.14.1+cu130`。
