import torch

from softmax import softmax


def prepare(device):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((4096, 4096), device=device, dtype=torch.float16, generator=generator)
    return {
        "run": lambda: softmax(x),
        "ref": lambda: torch.softmax(x, dim=-1),
        "atol": 1e-7,
        "rtol": 1e-3,
        # Count one FP16 input read and one FP16 output write.
        "bytes_moved": 2 * x.numel() * x.element_size(),
    }
