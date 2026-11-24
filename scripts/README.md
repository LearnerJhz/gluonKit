# Hardware Peak Initialization

本目录存放一次性硬件理论峰值初始化脚本，用于生成 `.gluonkit/peaks.json` 供 `gk bench` 计算 Roofline。

## 用法

```bash
.venv/bin/python scripts/init_peaks.py
```

`init_peaks.py` 当前针对 NVIDIA GPU 按硬件规格自动计算理论上限（显存带宽 `mem_clock × 2 × bus_width / 8`，计算算力 `sm_count × flops_per_sm_per_clock × max_sm_clock`）。

## 输出格式（`.gluonkit/peaks.json`）

只需包含 `device_key` 和 `peaks` 即可：

```json
{
  "device_key": "nvidia:sm86:NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "peaks": {
    "gbps": 168.024,
    "tflops_bf16": 21.504,
    "tflops_fp16": 21.504,
    "tflops_fp32": 10.752,
    "tflops_tf32": 10.752
  }
}
```

- `gbps`：理论显存带宽（GB/s），对应评测脚本中的 `bytes_moved`。
- `tflops_<compute_mode>`：理论稠密算力（TFLOP/s，FP16/BF16 Tensor Core 按 FP32 累加口径），对应评测脚本中的 `flops` + `compute_mode`。

对于其他平台、未入表型号或锁频环境，按上述格式直接手写 `.gluonkit/peaks.json` 即可。
