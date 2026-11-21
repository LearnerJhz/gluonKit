"""Backward-compatible NVIDIA bank_conflicts wrapper delegating to core.simt_layout."""

from gluonkit.core.simt_layout import compute_bank_conflicts as bank_conflicts

__all__ = ["bank_conflicts"]
