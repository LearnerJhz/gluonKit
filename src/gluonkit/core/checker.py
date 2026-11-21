"""Platform-independent correctness check."""

from typing import Any


def _max_errors(actual: Any, expected: Any) -> tuple[float, float]:
    import torch

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor) and actual.numel() > 0:
        a = actual.detach().to(torch.float32)
        e = expected.detach().to(torch.float32)
        mask = torch.isfinite(a) & torch.isfinite(e)
        if bool(mask.any()):
            diff = (a[mask] - e[mask]).abs()
            max_abs = float(diff.max().item())
            max_rel = float((diff / e[mask].abs().clamp_min(1e-12)).max().item())
            return max_abs, max_rel
    return 0.0, 0.0


def check_task(task: dict) -> dict[str, Any]:
    """Run task['run']() and compare against task['ref']() using assert_close."""
    import torch

    ref = task.get("ref")
    if not callable(ref):
        raise ValueError("Evaluation script must provide callable ref")
    atol = float(task.get("atol", 1e-2))
    rtol = float(task.get("rtol", 1e-2))
    actual = task["run"]()
    expected = ref()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs_err, max_rel_err = _max_errors(actual, expected)
    return {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
    }
