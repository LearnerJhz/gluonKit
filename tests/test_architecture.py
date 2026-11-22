"""Tests for the GluonKit SIMT backend registry, Wave32/Wave64 layout, baseline guard, and roofline."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from gluonkit.backends.metax.backend import MetaxBackend, _metax_kernel_summary
from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.backends.nvidia.dump_info import derive_limiting_factor
from gluonkit.backends.nvidia.profiler import compute_p95
from gluonkit.backends.registry import BACKENDS, DEFAULT_DEVICE, get_backend
from gluonkit.core.bencher import bench_task
from gluonkit.core.gluon_api import resolve_module, show_symbols
from gluonkit.core.roofline import calculate_roofline, load_peaks, resolve_peaks, save_peaks
from gluonkit.executor import run_baseline, run_check, run_probe
from gluonkit.interfaces.types import Output, ProbeResult, ProfileData
from gluonkit.output import render_json


class DummyBackend(MetaxBackend):
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
            warp_size=64,
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
            trace_path=trace_path,
        )


def test_static_backend_registry():
    assert set(BACKENDS) == {"nvidia", "metax"}
    nv = get_backend("nvidia")
    assert isinstance(nv, NvidiaBackend)
    assert nv.warp_size == 32
    assert nv.device == DEFAULT_DEVICE

    mx = get_backend("metax")
    assert isinstance(mx, MetaxBackend)
    assert mx.warp_size == 64
    assert mx.device == DEFAULT_DEVICE

    assert get_backend("nvidia", device="cuda:1").device == "cuda:1"

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("unknown")

    probe_info = run_probe(DummyBackend())
    assert probe_info == DummyBackend().probe().to_dict()
    assert "peaks" not in probe_info


def test_wave32_vs_wave64_layout():
    nv = get_backend("nvidia")
    mx = get_backend("metax")

    wave32_expr = "BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])"
    wave64_expr = "BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])"

    assert "Warp0:" in nv.layout(wave32_expr, [16, 128], hardware=True)
    with pytest.raises(ValueError, match="threads_per_warp must multiply to 64"):
        mx.layout(wave32_expr, [16, 128], hardware=True)

    assert "Warp0:" in mx.layout(wave64_expr, [16, 256], hardware=True)
    with pytest.raises(ValueError, match="threads_per_warp must multiply to 32"):
        nv.layout(wave64_expr, [16, 256], hardware=True)


def test_limiting_factor_and_metax_summary():
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

    summary = _metax_kernel_summary(
        SimpleNamespace(
            n_regs=32,
            n_spills=0,
            n_max_threads=256,
            metadata={"shared": 24576, "num_warps": 4},
        )
    )
    assert summary["blocks_per_sm"] == 2
    assert summary["occupancy_pct"] == 25.0
    assert summary["limiting_factor"] == "shared_memory"


def test_baseline_collision_and_refresh(tmp_path: Path):
    backend = DummyBackend()
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    for d in (dir_a, dir_b):
        (d / "eval_op.py").write_text(
            "import torch\n"
            "def prepare(device):\n"
            "    x = torch.ones(16)\n"
            "    return {'run': lambda: x * 2.0, 'ref': lambda: x * 2.0}\n",
            encoding="utf-8",
        )

    eval_a = str(dir_a / "eval_op.py")
    eval_b = str(dir_b / "eval_op.py")

    base_a = run_baseline(eval_a, backend, root=tmp_path)
    assert base_a["eval_path"] == "dir_a/eval_op.py"

    with pytest.raises(ValueError, match="Baseline collision"):
        run_baseline(eval_b, backend, root=tmp_path)

    refreshed = run_baseline(eval_b, backend, refresh=True, root=tmp_path)
    assert refreshed["eval_path"] == "dir_b/eval_op.py"

    chk = run_check(eval_a, backend)
    assert chk["passed"] is True
    assert chk["max_abs_err"] == 0.0


def test_roofline_and_api_full():
    r_mem = calculate_roofline(0.5, bytes_moved=50e6, peaks={"gbps": 100.0})
    assert r_mem["roofline_model"] == "memory_only"
    assert r_mem["bound_by"] == "memory"
    assert r_mem["roofline"] == pytest.approx(100.0)
    assert "roofline_pct" not in r_mem

    r_dual = calculate_roofline(0.5, flops=5e9, bytes_moved=50e6, peaks={"tflops": 20.0, "gbps": 100.0})
    assert r_dual["roofline_model"] == "dual_roofline"
    assert r_dual["bound_by"] == "memory"
    assert r_dual["roofline"] == pytest.approx(100.0)

    bench = bench_task(
        {"run": lambda: None, "bytes_moved": 50e6},
        DummyBackend(),
        calibrated_peaks={"gbps": 100.0},
    )
    assert bench.metrics["roofline"] == pytest.approx(100.0)
    assert bench.metrics["bound_by"] == "memory"
    assert bench.metrics["roofline_model"] == "memory_only"
    assert bench.peaks == {"tflops": None, "gbps": 100.0}

    assert compute_p95([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(4.8)

    mod = resolve_module("gl")
    full_text, errs = show_symbols(mod, ["load"], full=True)
    assert not errs
    assert "cache_modifier" in full_text
    assert "source:" in full_text


def test_peaks_hub_and_resolve(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Peaks file not found"):
        load_peaks(tmp_path)

    save_peaks({"gbps": 157.14}, tmp_path)
    peaks = save_peaks({"tflops_fp16": 25.3}, tmp_path)
    assert peaks == {"gbps": 157.14, "tflops_fp16": 25.3}

    raw = json.loads((tmp_path / ".gluonkit" / "peaks.json").read_text(encoding="utf-8"))
    assert raw == {"gbps": 157.14, "tflops_fp16": 25.3}
    assert load_peaks(tmp_path) == {"gbps": 157.14, "tflops_fp16": 25.3}

    resolved = resolve_peaks({"flops": 1e9, "compute_mode": "fp16", "bytes_moved": 1e6}, peaks)
    assert resolved == {"tflops": 25.3, "gbps": 157.14}


def test_output_rounding_and_calibrate_envelope(tmp_path: Path, monkeypatch, capsys):
    rendered = json.loads(render_json(Output.ok({
        "tflops": 12.3456,
        "gbps": 157.136,
        "roofline": 88.1234,
        "tflops_fp16": 12.3456789,
    })))
    assert rendered["info"] == {
        "tflops": 12.35,
        "gbps": 157.14,
        "roofline": 88.12,
        "tflops_fp16": 12.35,
    }

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_peaks.py"
    spec = importlib.util.spec_from_file_location("calibrate_peaks", script_path)
    assert spec is not None and spec.loader is not None
    calibrate_peaks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrate_peaks)

    monkeypatch.chdir(tmp_path)
    assert calibrate_peaks.main(["--items", " , "]) == 1
    fail_out = json.loads(capsys.readouterr().out)
    assert fail_out["result"] == "failed"
    assert "--items" in fail_out["info"]

    monkeypatch.setattr(calibrate_peaks, "get_backend", lambda: DummyBackend())
    monkeypatch.setattr(
        calibrate_peaks,
        "calibrate",
        lambda items, backend=None: {"gbps": 157.136},
    )
    assert calibrate_peaks.main(["--items", "gbps"]) == 0
    ok_out = json.loads(capsys.readouterr().out)
    assert ok_out["result"] == "success"
    assert ok_out["info"]["peaks"] == {"gbps": 157.14}
    assert (tmp_path / ".gluonkit" / "peaks.json").is_file()
