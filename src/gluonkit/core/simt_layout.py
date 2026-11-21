"""SIMT layout rendering and offline shared-memory bank conflict analysis (parameterized by warp_size)."""

import ast
import math
from typing import Any


def parse_layout(expression: str, constructors: dict[str, type]) -> Any:
    def parse(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [parse(item) for item in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in constructors and all(kw.arg is not None for kw in node.keywords):
                args = [parse(arg) for arg in node.args]
                kwargs = {kw.arg: parse(kw.value) for kw in node.keywords}
                if len(kwargs) != len(node.keywords):
                    raise ValueError("Duplicate layout keyword argument")
                return constructors[node.func.id](*args, **kwargs)
        raise ValueError(f"Only {', '.join(constructors)}, integer literals and lists/tuples are allowed")

    return parse(ast.parse(expression, mode="eval").body)


def render_layout(
    expression: str,
    shape: list[int],
    hardware: bool = False,
    *,
    warp_size: int = 32,
    platform_label: str = "NVIDIA",
) -> str:
    """Validate and render a Gluon BlockedLayout or nested SliceLayout for a target SIMT warp_size."""
    from triton.experimental.gluon.language import BlockedLayout, SliceLayout

    constructors = {"BlockedLayout": BlockedLayout, "SliceLayout": SliceLayout}

    def powers_of_two(values: Any) -> bool:
        return isinstance(values, list) and bool(values) and all(
            type(value) is int and value > 0 and value & (value - 1) == 0
            for value in values
        )

    def validate(layout: Any) -> None:
        if isinstance(layout, BlockedLayout):
            vectors = (layout.size_per_thread, layout.threads_per_warp, layout.warps_per_cta)
            if not all(powers_of_two(vector) for vector in vectors):
                raise ValueError("BlockedLayout dimensions must be positive powers of two")
            if any(len(vector) != layout.rank for vector in vectors):
                raise ValueError("BlockedLayout vectors must have equal rank")
            if not isinstance(layout.order, list) or any(type(i) is not int for i in layout.order):
                raise ValueError("order must be a permutation of dimension indices")
            if sorted(layout.order) != list(range(layout.rank)):
                raise ValueError("order must be a permutation of dimension indices")
            if math.prod(layout.threads_per_warp) != warp_size:
                raise ValueError(f"{platform_label} threads_per_warp must multiply to {warp_size}")
            if layout.cga_layout != []:
                raise ValueError("Multi-CTA cga_layout is not supported yet")
        elif isinstance(layout, SliceLayout):
            if not isinstance(layout.parent, (BlockedLayout, SliceLayout)):
                raise ValueError("SliceLayout parent must be a supported layout")
            validate(layout.parent)
            if type(layout.dim) is not int or not 0 <= layout.dim < layout.parent.rank:
                raise ValueError("SliceLayout dim is outside the parent rank")
        else:
            raise ValueError("Expression must construct a supported layout")

    try:
        layout = parse_layout(expression, constructors)
        validate(layout)
        if not powers_of_two(shape):
            raise ValueError("shape dimensions must be positive powers of two")
        if len(shape) != layout.rank:
            raise ValueError(f"shape rank {len(shape)} does not match layout rank {layout.rank}")
        if hardware:
            return layout.format_hardware_view(shape)
        return layout.format_tensor_view(shape)
    except (SyntaxError, TypeError, AssertionError) as exc:
        raise ValueError(f"Invalid layout arguments: {exc or 'inconsistent dimensions'}") from exc


def compute_bank_conflicts(
    register_layout: str,
    shared_layout: str,
    shape: list[int],
    dtype: str,
    *,
    extra_register_constructors: dict[str, type] | None = None,
    extra_shared_constructors: dict[str, type] | None = None,
) -> dict[str, int]:
    """Return Gluon's excess shared-memory accesses per wavefront without a GPU."""
    from triton._C.libtriton import gluon_ir, ir
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language._semantic import GluonSemantic

    reg_constructors: dict[str, type] = {
        "BlockedLayout": gl.BlockedLayout,
        "SliceLayout": gl.SliceLayout,
        "DotOperandLayout": gl.DotOperandLayout,
    }
    if hasattr(gl, "NVMMADistributedLayout"):
        reg_constructors["NVMMADistributedLayout"] = gl.NVMMADistributedLayout
    if extra_register_constructors:
        reg_constructors.update(extra_register_constructors)

    sh_constructors: dict[str, type] = {
        "SwizzledSharedLayout": gl.SwizzledSharedLayout,
    }
    if extra_shared_constructors:
        sh_constructors.update(extra_shared_constructors)

    register = parse_layout(register_layout, reg_constructors)
    shared = parse_layout(shared_layout, sh_constructors)
    element_ty = getattr(gl, dtype)
    register_ty = gl.distributed_type(element_ty, shape, register)
    shared_ty = gl.shared_memory_descriptor_type(element_ty, shape, shared, shape)

    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    module = builder.create_module()
    function = builder.get_or_insert_function(
        module, "bank_conflicts", builder.get_function_ty([], []), "private", False,
    )
    module.push_back(function)
    builder.set_insertion_point_to_start(function.add_entry_block())
    count = gl.bank_conflicts(register_ty, shared_ty, _semantic=GluonSemantic(builder))
    return {"bank_conflicts_count": count}
