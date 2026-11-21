"""Platform-independent correctness check and verification receipt store (.gluonkit/checks.json)."""

from pathlib import Path
from typing import Any

from gluonkit.core.storage import (
    atomic_write_json,
    gluonkit_dir,
    normalize_path,
    now_iso,
    read_json,
    resolve_root,
)
from gluonkit.interfaces.types import CheckResult

CHECKS_FILENAME = "checks.json"
CHECKS_VERSION = 1


def _compute_error_stats(actual: Any, expected: Any) -> tuple[float, float]:
    """Compute (max_abs_err, max_rel_err) across tensor or nested tuple/list outputs."""
    import torch

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.numel() == 0:
            return 0.0, 0.0
        a = actual.detach().to(torch.float32)
        e = expected.detach().to(torch.float32)
        finite_mask = torch.isfinite(a) & torch.isfinite(e)
        if not bool(finite_mask.any()):
            return 0.0, 0.0
        diff = (a[finite_mask] - e[finite_mask]).abs()
        max_abs = float(diff.max().item())
        denom = e[finite_mask].abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item())
        return max_abs, max_rel
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        pairs = [_compute_error_stats(x, y) for x, y in zip(actual, expected)]
        if not pairs:
            return 0.0, 0.0
        return max(p[0] for p in pairs), max(p[1] for p in pairs)
    if isinstance(actual, dict) and isinstance(expected, dict):
        pairs = [_compute_error_stats(actual[k], expected[k]) for k in actual if k in expected]
        if not pairs:
            return 0.0, 0.0
        return max(p[0] for p in pairs), max(p[1] for p in pairs)
    return 0.0, 0.0


def check_task(
    task: dict,
    *,
    eval_path: str | None = None,
    eval_source_hash: str | None = None,
    kernel_path: str | None = None,
    kernel_source_hash: str | None = None,
    device_key: str | None = None,
) -> CheckResult:
    """Run task['run']() and compare against task['ref']() using assert_close, returning CheckResult."""
    import torch

    ref = task.get("ref")
    if not callable(ref):
        raise ValueError("Evaluation script must provide callable ref")
    atol = float(task.get("atol", 1e-2))
    rtol = float(task.get("rtol", 1e-2))
    equal_nan = bool(task.get("equal_nan", False))
    actual = task["run"]()
    expected = ref()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_abs_err, max_rel_err = _compute_error_stats(actual, expected)
    return CheckResult(
        passed=True,
        atol=atol,
        rtol=rtol,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        eval_path=eval_path,
        eval_source_hash=eval_source_hash,
        kernel_path=kernel_path,
        kernel_source_hash=kernel_source_hash,
        device_key=device_key,
        checked_at=now_iso(),
    )


class CheckStore:
    """Persist and verify `gk check` receipts in `.gluonkit/checks.json`."""

    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.path = gluonkit_dir(self.root, CHECKS_FILENAME)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": CHECKS_VERSION, "checks": {}}
        data = read_json(self.path, recovery_hint="delete .gluonkit/checks.json and rerun `gk check`")
        if not isinstance(data.get("checks"), dict):
            return {"version": CHECKS_VERSION, "checks": {}}
        return data

    def save_receipt(self, result: CheckResult) -> dict[str, Any]:
        if not result.eval_path:
            return result.to_dict()
        data = self.load()
        key = normalize_path(result.eval_path, self.root)
        entry = result.to_dict()
        entry["eval_path"] = key
        data["checks"][key] = entry
        atomic_write_json(self.path, data)
        return entry

    def is_verified(
        self,
        eval_path: str | Path,
        *,
        eval_source_hash: str,
        kernel_source_hash: str,
        device_key: str | None = None,
    ) -> bool:
        """Check if the exact (eval_path, eval_source_hash, kernel_source_hash) passed `gk check`."""
        key = normalize_path(eval_path, self.root)
        entry = self.load()["checks"].get(key)
        if not isinstance(entry, dict) or not entry.get("passed"):
            return False
        if entry.get("eval_source_hash") != eval_source_hash:
            return False
        if entry.get("kernel_source_hash") != kernel_source_hash:
            return False
        if device_key and entry.get("device_key") and entry.get("device_key") != device_key:
            return False
        return True
