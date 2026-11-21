"""Backward-compatible NVIDIA layout wrapper delegating to core.simt_layout."""

from gluonkit.core.simt_layout import parse_layout, render_layout

__all__ = ["parse_layout", "render_layout"]
