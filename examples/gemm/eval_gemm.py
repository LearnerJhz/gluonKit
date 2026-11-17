import torch

from gemm import gemm

KERNEL_PATH = "gemm.py"


def prepare(device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((1024, 512), device=device, dtype=torch.float16, generator=generator)
    b = torch.randn((512, 1024), device=device, dtype=torch.float16, generator=generator)
    return [{"name": "1024x1024x512-f16", "run": lambda: gemm(a, b)}]
