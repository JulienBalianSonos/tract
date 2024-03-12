use crate::internal::*;
use crate::ser::*;
use tract_core::ops::quant::BitUnpack;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_bit_unpack);
    registry.register_primitive(
        "tract_core_dyn_bit_unpack",
        &[TypeName::Scalar.tensor().named("input"), TypeName::Integer.named("bit_width")],
        &[("output", TypeName::Scalar.tensor())],
        de_bit_unpack,
    );
}

fn ser_bit_unpack(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &BitUnpack,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_dyn_bit_unpack",
        &[input],
        &[("bit_width", numeric(op.bit_width))],
    )))
}

fn de_bit_unpack(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let bit_width = invocation.named_arg_as(builder, "bit_width")?;
    builder.wire(BitUnpack { bit_width }, &[input])
}
