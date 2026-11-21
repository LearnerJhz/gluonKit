import ast
import math


def parse_layout(expression: str, constructors: dict[str, type]):
    def parse(node):
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
    expression: str, shape: list[int], hardware: bool = False, *, warp_size: int = 32,
) -> str:
    from triton.experimental.gluon.language import BlockedLayout, SliceLayout

    constructors = {"BlockedLayout": BlockedLayout, "SliceLayout": SliceLayout}

    def powers_of_two(values):
        return isinstance(values, list) and bool(values) and all(
            type(value) is int and value > 0 and value & (value - 1) == 0
            for value in values
        )

    def validate(layout):
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
                raise ValueError(f"threads_per_warp must multiply to {warp_size}")
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
