from .layout import parse_layout


def bank_conflicts(
    register_layout: str, shared_layout: str, shape: list[int], dtype: str,
) -> dict[str, int]:
    """Return Gluon's excess shared-memory accesses per wavefront without a GPU."""
    from triton._C.libtriton import gluon_ir, ir
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language._semantic import GluonSemantic

    register = parse_layout(register_layout, {
        "BlockedLayout": gl.BlockedLayout,
        "SliceLayout": gl.SliceLayout,
        "DotOperandLayout": gl.DotOperandLayout,
        "NVMMADistributedLayout": gl.NVMMADistributedLayout,
    })
    shared = parse_layout(shared_layout, {"SwizzledSharedLayout": gl.SwizzledSharedLayout})
    element_ty = getattr(gl, dtype)
    register_ty = gl.distributed_type(element_ty, shape, register)
    shared_ty = gl.shared_memory_descriptor_type(element_ty, shape, shared, shape)

    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    module = builder.create_module()
    # The native analysis looks up the module through the current function block.
    function = builder.get_or_insert_function(
        module, "bank_conflicts", builder.get_function_ty([], []), "private", False,
    )
    module.push_back(function)
    builder.set_insertion_point_to_start(function.add_entry_block())
    count = gl.bank_conflicts(register_ty, shared_ty, _semantic=GluonSemantic(builder))
    return {"bank_conflicts_count": count}
