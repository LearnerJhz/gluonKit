import torch

from gemm import gemm

def prepare(device):
    generator = torch.Generator(device=device).manual_seed(0)
    a = torch.randn((1024, 512), device=device, dtype=torch.float16, generator=generator)
    b = torch.randn((512, 1024), device=device, dtype=torch.float16, generator=generator)
    return {
        "run": lambda: gemm(a, b),
        "ref": lambda: torch.matmul(a, b),
        "flops": 2 * 1024 * 1024 * 512,
        "compute_mode": "fp16",
        "bytes_moved": (1024 * 512 + 512 * 1024 + 1024 * 1024) * 2,
    }
