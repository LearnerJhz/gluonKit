from gluonkit.interfaces.types import ProbeResult


def probe(device: str = "cuda:0") -> ProbeResult:
    import torch
    import triton

    props = torch.cuda.get_device_properties(device)
    return ProbeResult(
        device_name=props.name,
        platform="nvidia",
        arch=f"sm{props.major}{props.minor}",
        compiler_version=triton.__version__,
        torch_version=torch.__version__,
        smem_per_block_optin=getattr(props, "shared_memory_per_block_optin", None),
        sm_count=props.multi_processor_count,
        vram_bytes=props.total_memory,
        warp_size=int(getattr(props, "warp_size", 32) or 32),
    )
