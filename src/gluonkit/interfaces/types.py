from dataclasses import asdict, dataclass


@dataclass
class ProbeResult:
    device_name: str
    platform: str
    arch: str
    compiler_version: str
    torch_version: str
    smem_per_block_optin: int | None
    sm_count: int
    vram_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)
