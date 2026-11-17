"""Platform-independent correctness check."""

import torch

from gluonkit.interfaces.types import CheckResult


def check_task(task: dict) -> CheckResult:
    """Run task['run']() and compare against task['ref']() using assert_close."""
    ref = task.get("ref")
    if not callable(ref):
        return CheckResult(passed=False, error="Evaluation script must provide callable ref")
    atol = task.get("atol", 1e-2)
    rtol = task.get("rtol", 1e-2)
    try:
        actual = task["run"]()
        expected = ref()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return CheckResult(passed=True)
    except Exception as exc:
        return CheckResult(passed=False, error=str(exc))
