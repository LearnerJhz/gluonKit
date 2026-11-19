"""Unified JSON output layer for agent consumption."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from math import isfinite
import sys
import traceback
from typing import Any, Callable, TextIO

from gluonkit.interfaces.types import Output

_MS_KEYS = frozenset({"avg", "median", "min", "std", "avg_ms", "baseline_ms", "baseline_time_ms"})
_PCT_KEYS = frozenset({"roofline", "roofline_pct", "occupancy_pct"})
_RATIO_KEYS = frozenset({"speedup", "speedup_vs_baseline"})
_RATE_KEYS = frozenset({"tflops", "gbps"})


def _is_rate_key(key: str | None) -> bool:
    # tflops/gbps plus calibrated item names such as tflops_fp16 or theoretical_gbps.
    return key is not None and (key in _RATE_KEYS or key.startswith(("tflops_", "gbps_"))
                                or key.endswith(("_tflops", "_gbps")))


def _round_float(value: float, key: str | None = None) -> float:
    """Round finite floats for CLI JSON readability without altering non-finite errors."""
    if not isfinite(value):
        return value
    if key in _MS_KEYS or (key is not None and key.endswith("_ms")):
        ndigits = 6
    elif key in _PCT_KEYS or (key is not None and key.endswith("_pct")):
        ndigits = 2
    elif key in _RATIO_KEYS:
        ndigits = 3
    elif _is_rate_key(key):
        ndigits = 2 if abs(value) >= 1 else 4
    else:
        ndigits = 6
    rounded = round(value, ndigits)
    return 0.0 if rounded == 0 else rounded


def _format_floats(value: Any, key: str | None = None) -> Any:
    if isinstance(value, float):
        return _round_float(value, key)
    if isinstance(value, dict):
        return {k: _format_floats(v, k if isinstance(k, str) else None) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_floats(item) for item in value]
    return value


def format_error(
    error: BaseException | str,
    *,
    stdout: str = "",
    stderr: str = "",
) -> str:
    """Combine error/traceback details with any captured terminal output."""
    parts: list[str] = []
    if isinstance(error, BaseException):
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
    else:
        text = str(error).rstrip()
    if text:
        parts.append(text)
    if stdout.strip():
        parts.append(f"[stdout]\n{stdout.rstrip()}")
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    return "\n\n".join(parts)


def capture_run(fn: Callable[[], Any]) -> Output:
    """Run fn() with stdio captured; on failure pack traceback and terminal output into info."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            value = fn()
    except Exception as exc:
        return Output.fail(
            format_error(exc, stdout=stdout_buf.getvalue(), stderr=stderr_buf.getvalue())
        )
    if isinstance(value, Output):
        if value.result == "failed":
            return Output.fail(
                format_error(
                    "" if value.info is None else str(value.info),
                    stdout=stdout_buf.getvalue(),
                    stderr=stderr_buf.getvalue(),
                )
            )
        return value
    return Output.ok(value)


def render_json(output: Output) -> str:
    """Serialize an Output envelope as a single JSON string."""
    payload = _format_floats(output.to_dict())
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def emit_json(output: Output, stream: TextIO | None = None) -> int:
    """Write one JSON object to stdout (or stream) and return process exit code."""
    target = sys.stdout if stream is None else stream
    print(render_json(output), file=target)
    return 0 if output.result == "success" else 1
