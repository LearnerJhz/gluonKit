"""Backward-compatible NVIDIA profiler wrapper delegating to core.torch_profiler."""

from gluonkit.core.torch_profiler import profile_with_kineto as profile

__all__ = ["profile"]
