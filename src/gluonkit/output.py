"""Unified JSON output layer for agent consumption."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import traceback
from typing import Any, Callable, TextIO

from gluonkit.interfaces.types import Output


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
    return json.dumps(output.to_dict(), ensure_ascii=False, allow_nan=False)


def emit_json(output: Output, stream: TextIO | None = None) -> int:
    """Write one JSON object to stdout (or stream) and return process exit code."""
    target = sys.stdout if stream is None else stream
    print(render_json(output), file=target)
    return 0 if output.result == "success" else 1
