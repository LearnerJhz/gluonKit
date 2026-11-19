"""Platform-independent correctness check."""


def check_task(task: dict) -> None:
    """Run task['run']() and compare against task['ref']() using assert_close."""
    import torch

    ref = task.get("ref")
    if not callable(ref):
        raise ValueError("Evaluation script must provide callable ref")
    atol = task.get("atol", 1e-2)
    rtol = task.get("rtol", 1e-2)
    actual = task["run"]()
    expected = ref()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
