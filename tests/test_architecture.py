"""Unit and integration tests for GluonKit multi-backend SIMT architecture (NVIDIA + MetaX C500)."""

from pathlib import Path
from types import SimpleNamespace
import pytest

from gluonkit.backends.metax.backend import MetaxBackend
from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.backends.registry import detect_backend_name, get_backend
from gluonkit.core.baseline import BaselineStore
from gluonkit.core.checker import CheckStore
from gluonkit.core.device_diag import classify_gpu_init_failure
from gluonkit.core.gluon_api import resolve_module, show_symbols
from gluonkit.core.roofline import calculate_roofline
from gluonkit.core.torch_profiler import compute_p95
from gluonkit.core.triton_dump import derive_limiting_factor, write_kernel_ir
from gluonkit.executor import run_baseline, run_bench, run_check, run_history
from gluonkit.interfaces.types import ProbeResult, ProfileData


class DummySimtBackend(MetaxBackend):
    """In-memory SIMT backend for testing experiment control plane without launching GPU kernels."""

    def __init__(self, *, platform: str = "metax", arch: str = "maca_c500", device_name: str = "MetaX C500"):
        super().__init__(device="cpu")
        self._platform = platform
        self._arch = arch
        self._device_name = device_name

    def probe(self) -> ProbeResult:
        return ProbeResult(
            device_name=self._device_name,
            platform=self._platform,
            arch=self._arch,
            compiler_version="3.8.0",
            torch_version="2.14.1",
            smem_per_block_optin=65536,
            sm_count=64,
            vram_bytes=64 * 1024**3,
            compute_unit_count=64,
            warp_size=64,
            max_smem_per_block=65536,
            default_smem_per_block=65536,
            l2_cache_bytes=32 * 1024**2,
            supported_compute_modes=["fp16", "bf16", "tf32", "fp32"],
            knowledge_docs=[".knowledge/gluon-api/metax/metax.md"],
        )

    def profile(self, fn, iters: int, warmup: int, *, flush_l2: bool = False, trace_path: str | None = None) -> ProfileData:
        fn()
        return ProfileData(
            kernel_name="_dummy_kernel",
            avg_ms=0.5,
            std_ms=0.01,
            min_ms=0.49,
            median_ms=0.5,
            p95_ms=0.52,
            iters=iters,
            timing_method="device_event",
            trace_path=trace_path,
        )

    def dump_info(self, fn, output_dir):
        fn()
        summary = self.summarize_kernel(
            SimpleNamespace(
                n_regs=64,
                n_spills=0,
                n_max_threads=1024,
                metadata={"shared": 4096, "num_warps": 4},
            )
        )
        return {"artifacts": {}, "summary": {"_dummy_kernel": summary.to_dict()}}


def test_backend_registry_and_detection(monkeypatch):
    nv = get_backend("nvidia")
    assert isinstance(nv, NvidiaBackend)
    assert nv.warp_size == 32

    mx = get_backend("metax")
    assert isinstance(mx, MetaxBackend)
    assert mx.warp_size == 64

    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    assert detect_backend_name() == "metax"
    monkeypatch.delenv("MACA_PATH", raising=False)

    monkeypatch.setenv("GK_BACKEND", "metax")
    assert isinstance(get_backend(None), MetaxBackend)
    monkeypatch.delenv("GK_BACKEND", raising=False)


def test_wave32_vs_wave64_layout_validation():
    nv = NvidiaBackend()
    mx = MetaxBackend()

    wave32_expr = "BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])"
    wave64_expr = "BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])"

    assert "Warp0:" in nv.layout(wave32_expr, [16, 128], hardware=True)
    with pytest.raises(ValueError, match="MetaX threads_per_warp must multiply to 64"):
        mx.layout(wave32_expr, [16, 128], hardware=True)

    assert "Warp0:" in mx.layout(wave64_expr, [16, 256], hardware=True)
    with pytest.raises(ValueError, match="NVIDIA threads_per_warp must multiply to 32"):
        nv.layout(wave64_expr, [16, 256], hardware=True)


def test_offline_bank_conflicts_both_backends():
    nv = NvidiaBackend()
    mx = MetaxBackend()

    res_nv = nv.bank_conflicts(
        "BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])",
        "SwizzledSharedLayout(8, 1, 1, [1, 0])",
        [16, 64],
        "float16",
    )
    res_mx = mx.bank_conflicts(
        "BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])",
        "SwizzledSharedLayout(8, 1, 1, [1, 0])",
        [32, 64],
        "float16",
    )
    assert "bank_conflicts_count" in res_nv
    assert "bank_conflicts_count" in res_mx


def test_limiting_factor_and_metax_occupancy_model():
    # NVIDIA sm86 Softmax case from REPORT.md: 51 regs, 4 warps, 1152B shared -> 9 blocks/SM limited by regs
    blocks, factor = derive_limiting_factor(
        num_warps=4,
        n_regs=51,
        allocated_shared_bytes=1152,
        warp_size=32,
        max_warps_per_cu=48,
        max_blocks_per_cu=16,
        max_regs_per_cu=65536,
        reg_alloc_granularity=256,
        max_smem_per_cu=102400,
    )
    assert blocks == 9
    assert factor == "regs"

    # MetaX C500 Wave64 kernel constrained by 64 KiB LDS (24 KiB per block -> 2 blocks/CU)
    mx = MetaxBackend()
    summary = mx.summarize_kernel(
        SimpleNamespace(
            n_regs=32,
            n_spills=0,
            n_max_threads=256,
            metadata={"shared": 24576, "num_warps": 4, "num_sgprs": 40},
        )
    )
    assert summary.warp_size == 64
    assert summary.blocks_per_cu == 2
    assert summary.occupancy_pct == 25.0
    assert summary.limiting_factor == "shared_memory"
    assert summary.extra["n_sgprs"] == 40


def test_write_kernel_ir_exports_custom_isa_and_ignores_source(tmp_path: Path):
    fake_kernel = SimpleNamespace(
        asm={
            "ttgir": "module { ttgir }",
            "llir": "; llvm ir",
            "maca": "// metax c500 isa",
            "source": "internal python wrapper",
            "cubin": b"\x00\x01",
        }
    )
    exported = write_kernel_ir(fake_kernel, tmp_path, expected_extensions=("ttgir", "llir"))
    assert set(exported.keys()) == {"ttgir", "llir", "maca"}
    assert Path(exported["maca"]).read_text(encoding="utf-8") == "// metax c500 isa"
    assert "source" not in exported


def test_baseline_collision_and_Check_gate(tmp_path: Path):
    backend = DummySimtBackend()
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    for d in (dir_a, dir_b):
        (d / "op.py").write_text(
            "import torch\n"
            "def run_op(x):\n"
            "    return x * 2.0\n",
            encoding="utf-8",
        )
        (d / "eval_op.py").write_text(
            "import torch\n"
            "from op import run_op\n"
            "def prepare(device):\n"
            "    x = torch.ones(16)\n"
            "    return {\n"
            "        'run': lambda: run_op(x),\n"
            "        'ref': lambda: x * 2.0,\n"
            "        'atol': 1e-6,\n"
            "        'rtol': 1e-6,\n"
            "        'bytes_moved': 128,\n"
            "        'peaks': {'gbps': 100.0},\n"
            "    }\n",
            encoding="utf-8",
        )

    eval_a = str(dir_a / "eval_op.py")
    eval_b = str(dir_b / "eval_op.py")

    # 1. Measure baseline for dir_a
    base_a = run_baseline(eval_a, backend, root=tmp_path)
    assert base_a["status"] == "measured"
    assert base_a["eval_path"] == "dir_a/eval_op.py"

    # Re-running on dir_a reuses the stored baseline
    reused_a = run_baseline(eval_a, backend, root=tmp_path)
    assert reused_a["status"] == "reused"

    # 2. Running baseline on dir_b/eval_op.py (same stem!) must detect collision
    with pytest.raises(ValueError, match="Baseline collision"):
        run_baseline(eval_b, backend, root=tmp_path)

    # 3. Running bench with --require-check before `gk check` must fail
    with pytest.raises(RuntimeError, match="has not passed `gk check`"):
        run_bench(eval_a, backend, require_check=True, root=tmp_path)

    # 4. Run `gk check` and verify structured receipt
    chk = run_check(eval_a, backend, root=tmp_path)
    assert chk["passed"] is True
    assert chk["max_abs_err"] == 0.0
    assert chk["eval_path"] == "dir_a/eval_op.py"
    assert CheckStore(tmp_path).path.is_file()

    # 5. Now `gk bench --require-check` succeeds and marks verified=True
    bench_1 = run_bench(eval_a, backend, note="v1", require_check=True, root=tmp_path)
    assert bench_1["verified"] is True
    assert bench_1["metrics"]["roofline_model"] == "memory_only"
    assert bench_1["kernel"]["p95"] == 0.52

    # 6. Modify kernel source -> `gk bench --require-check` must reject stale check receipt
    (dir_a / "op.py").write_text(
        "import torch\n"
        "def run_op(x):\n"
        "    return (x + 0.0) * 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="has not passed `gk check`"):
        run_bench(eval_a, backend, require_check=True, root=tmp_path)

    # Re-check and run child bench to test history delta_vs_parent_pct
    run_check(eval_a, backend, root=tmp_path)
    bench_2 = run_bench(
        eval_a, backend, note="v2", parent_id=bench_1["history_id"], require_check=True, root=tmp_path,
    )
    hist = run_history(eval_a, root=tmp_path)
    assert hist["total"] == 2
    assert hist["records"][1]["parent_id"] == bench_1["history_id"]
    assert hist["records"][1]["delta_vs_parent_pct"] == 0.0
    assert hist["records"][1]["verified"] is True

    # 7. Device mismatch detection on baseline
    other_device_backend = DummySimtBackend(platform="nvidia", arch="sm86", device_name="RTX 3050")
    with pytest.raises(ValueError, match="Baseline device mismatch"):
        BaselineStore(tmp_path).lookup(eval_a, device_key="nvidia:sm86:RTX 3050")

    # Refresh overwrites baseline with the new device
    refreshed = run_baseline(eval_a, other_device_backend, refresh=True, root=tmp_path)
    assert refreshed["status"] == "measured"
    assert refreshed["device_key"] == "nvidia:sm86:RTX 3050"


def test_roofline_models_and_p95():
    r_mem = calculate_roofline(0.5, bytes_moved=50e6, peaks={"gbps": 100.0})
    assert r_mem["roofline_model"] == "memory_only"
    assert r_mem["bound_by"] == "memory"

    r_comp = calculate_roofline(0.5, flops=10e9, peaks={"tflops": 20.0})
    assert r_comp["roofline_model"] == "compute_only"
    assert r_comp["bound_by"] == "compute"

    r_dual = calculate_roofline(0.5, flops=10e9, bytes_moved=50e6, peaks={"tflops": 20.0, "gbps": 200.0})
    assert r_dual["roofline_model"] == "dual_roofline"
    assert r_dual["bound_by"] == "compute"

    assert compute_p95([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(4.8)


def test_device_diag_and_api_full_introspection(tmp_path: Path):
    cat, _ = classify_gpu_init_failure(tmp_path, env={"CUDA_VISIBLE_DEVICES": ""})
    assert cat == "env_hidden"

    cat, _ = classify_gpu_init_failure(tmp_path, env={})
    assert cat == "device_node_missing"

    mod = resolve_module("gl")
    short_text, errs = show_symbols(mod, ["load"], full=False)
    assert not errs
    assert "source:" not in short_text

    full_text, errs = show_symbols(mod, ["load"], full=True)
    assert not errs
    assert "cache_modifier" in full_text
    assert "source:" in full_text
