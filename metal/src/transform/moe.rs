use super::metal_cast_new;
use crate::{kernels, ops};
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantStorage, Q4_0};
use tract_gpu::sync::{DeviceSyncKind, sync_outlet_if_required};
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::moe_ffn::{
    ExpertLayout, MoeFfn, RouteTopK, RoutedInputMode, transpose_block_quant_experts,
};

/// Enable command-buffer boundaries for large routed MoE dispatches.
pub(crate) fn moe_sync_after_dispatch() -> bool {
    !env_flag("TRACT_METAL_MOE_UNSAFE_NOSYNC")
}

fn sync_f32_input(
    target: &mut TypedModel,
    name: impl Into<String>,
    outlet: OutletId,
    fact: &TypedFact,
) -> TractResult<OutletId> {
    let name = name.into();
    let device = sync_outlet_if_required(
        target,
        format!("{name}.to-device"),
        outlet,
        DeviceSyncKind::ToDevice,
    )?;
    if fact.datum_type == f32::datum_type() {
        Ok(device)
    } else {
        Ok(target.wire_node(
            format!("{name}.cast-f32"),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[device],
        )?[0])
    }
}

fn add_routed_bias(
    target: &mut TypedModel,
    name: impl Into<String>,
    value: OutletId,
    bias: OutletId,
    route_expert_ids: OutletId,
) -> TractResult<OutletId> {
    let name = name.into();
    Ok(target.wire_node(
        format!("{name}.bias-add"),
        ops::MetalRoutedBiasAdd,
        &[value, bias, route_expert_ids],
    )?[0])
}

/// Lower supported linear Q40 MoE blocks to device routing, fused expert
/// projections and weighted combination, preserving CPU fallback otherwise.
pub(super) fn convert_q40_moe_ffn_to_metal(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &HashMap<OutletId, OutletId>,
    op: &MoeFfn,
) -> TractResult<Option<TVec<OutletId>>> {
    let log_lowering = env_flag("TRACT_METAL_LOG_Q40_MOE");
    if env_flag("TRACT_METAL_DISABLE_Q40_MOE") {
        return Ok(None);
    }
    if op.expert_layout != ExpertLayout::Linear
        || !q40_moe_activation_supported(op)
        || (op.act_limit_bits.is_some() && !op.has_w3)
    {
        if log_lowering {
            eprintln!(
                "Metal Q40 MoE skip {}: metadata layout={:?} activation={} has_w3={} biases=[wg:{} w1:{} w3:{} w2:{}] act_limit={}",
                node.name,
                op.expert_layout,
                op.activation,
                op.has_w3,
                op.has_wg_bias,
                op.has_w1_bias,
                op.has_w3_bias,
                op.has_w2_bias,
                op.act_limit_bits.is_some()
            );
        }
        return Ok(None);
    }

    let facts = source.node_input_facts(node.id)?;
    let indexes = moe_input_indexes(op);
    let facts_supported = q40_moe_facts_supported(op, &facts, ExpertLayout::Linear);
    if log_lowering {
        let w1_q40 = as_quant_fact(facts[2], &Q4_0).is_some();
        let w2_q40 = as_quant_fact(facts[3], &Q4_0).is_some();
        let w3_q40 =
            !op.has_w3 || indexes.w3.is_some_and(|ix| as_quant_fact(facts[ix], &Q4_0).is_some());
        eprintln!(
            "Metal Q40 MoE candidate {}: x_rank={} x_dt={:?} wg_dt={:?} w1_q40={} w2_q40={} w3_q40={} shapes x={:?} wg={:?} w1={:?} w2={:?}{}",
            node.name,
            facts[0].rank(),
            facts[0].datum_type,
            facts[1].datum_type,
            w1_q40,
            w2_q40,
            w3_q40,
            facts[0].shape,
            facts[1].shape,
            facts[2].shape,
            facts[3].shape,
            indexes.w3.map(|ix| format!(" w3={:?}", facts[ix].shape)).unwrap_or_default()
        );
    }
    if !facts_supported {
        if log_lowering {
            eprintln!("Metal Q40 MoE skip {}: unsupported facts", node.name);
        }
        return Ok(None);
    }

    if log_lowering {
        eprintln!("Metal Q40 MoE lowering: {}", node.name);
    }

    // has_w3 blocks fuse w1/w3/(biases)/activation into one op: a single
    // gemv-pair dispatch at decode, one shared expert sort + gather at
    // prefill. Both bias inputs must be present or absent together for the
    // fused kernel; mixed-bias models fall back to the unfused chain.
    let fused_swiglu = op.has_w3
        && indexes.w1_bias.is_some() == indexes.w3_bias.is_some()
        && !env_flag("TRACT_METAL_DISABLE_ROUTED_SWIGLU");
    let cpu_route_topk = env_flag("TRACT_METAL_DISABLE_ROUTE_TOPK");

    let x_device = sync_outlet_if_required(
        target,
        format!("{}.x.to-device", node.name),
        mapping[&node.inputs[0]],
        DeviceSyncKind::ToDevice,
    )?;
    // f16 x feeds the router and the fused swiglu directly: both read f16
    // through exact per-element conversion (scores and expert math stay f32,
    // bit-identical to upcasting first), dropping a per-layer per-step cast
    // dispatch. The unfused expert chain and the CPU route-topk fallback
    // still want f32; so does the escape hatch.
    let f16_x_direct = facts[0].datum_type == DatumType::F16
        && fused_swiglu
        && !cpu_route_topk
        && !env_flag("TRACT_METAL_DISABLE_MOE_F16_X");
    let x_device = if facts[0].datum_type != f32::datum_type() && !f16_x_direct {
        target.wire_node(
            format!("{}.x.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let wg_device = sync_outlet_if_required(
        target,
        format!("{}.wg.to-device", node.name),
        mapping[&node.inputs[1]],
        DeviceSyncKind::ToDevice,
    )?;
    let wg_device = if facts[1].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.wg.cast-f32", node.name),
            metal_cast_new(f32::datum_type()).context("unsupported Metal cast")?,
            &[wg_device],
        )?[0]
    } else {
        wg_device
    };
    let wg_bias_device = indexes
        .wg_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.wg_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;

    let routes = if env_flag("TRACT_METAL_DISABLE_ROUTE_TOPK") {
        if indexes.wg_bias.is_some() {
            if log_lowering {
                eprintln!(
                    "Metal Q40 MoE skip {}: CPU RouteTopK does not support wg_bias",
                    node.name
                );
            }
            return Ok(None);
        }
        let x_host = sync_outlet_if_required(
            target,
            format!("{}.route-x-to-cpu", node.name),
            mapping[&node.inputs[0]],
            DeviceSyncKind::ToHost,
        )?;
        let wg_host = sync_outlet_if_required(
            target,
            format!("{}.route-wg-to-cpu", node.name),
            mapping[&node.inputs[1]],
            DeviceSyncKind::ToHost,
        )?;
        target.wire_node(
            format!("{}.route_topk", node.name),
            RouteTopK { k: op.k, gate: op.gate.clone() },
            &[x_host, wg_host],
        )?
    } else {
        let mut route_inputs = tvec![x_device, wg_device];
        if let Some(wg_bias_device) = wg_bias_device {
            route_inputs.push(wg_bias_device);
        }
        target.wire_node(
            format!("{}.route_topk", node.name),
            ops::MetalRouteTopK { k: op.k, gate: op.gate.clone() },
            &route_inputs,
        )?
    };
    let route_token_ids = sync_outlet_if_required(
        target,
        format!("{}.route_token_ids.to-device", node.name),
        routes[0],
        DeviceSyncKind::ToDevice,
    )?;
    let route_expert_ids = sync_outlet_if_required(
        target,
        format!("{}.route_expert_ids.to-device", node.name),
        routes[1],
        DeviceSyncKind::ToDevice,
    )?;
    let route_weights = sync_outlet_if_required(
        target,
        format!("{}.route_weights.to-device", node.name),
        routes[2],
        DeviceSyncKind::ToDevice,
    )?;

    let x_shape_like = x_device;
    let x_expert_input = if facts[0].rank() == 3 {
        target.wire_node(
            format!("{}.x.rm-batch", node.name),
            tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Rm(0)),
            &[x_device],
        )?[0]
    } else {
        x_device
    };
    let w1_device = sync_outlet_if_required(
        target,
        format!("{}.w1.to-device", node.name),
        mapping[&node.inputs[2]],
        DeviceSyncKind::ToDevice,
    )?;
    let w2_device = sync_outlet_if_required(
        target,
        format!("{}.w2.to-device", node.name),
        mapping[&node.inputs[3]],
        DeviceSyncKind::ToDevice,
    )?;
    let w1_bias_device = indexes
        .w1_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w1_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;
    let w3_bias_device = indexes
        .w3_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w3_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;
    let w2_bias_device = indexes
        .w2_bias
        .map(|ix| {
            sync_f32_input(
                target,
                format!("{}.w2_bias", node.name),
                mapping[&node.inputs[ix]],
                facts[ix],
            )
        })
        .transpose()?;

    // fused_swiglu computed above, next to the f16-x decision it gates
    // (w1_bias/w3_bias devices map 1:1 from the indexes used there).
    let hidden = if fused_swiglu {
        let w3_device = sync_outlet_if_required(
            target,
            format!("{}.w3.to-device", node.name),
            mapping[&node.inputs[indexes.w3.context("missing w3 input index")?]],
            DeviceSyncKind::ToDevice,
        )?;
        let has_bias = w1_bias_device.is_some();
        let mut inputs =
            tvec![x_expert_input, w1_device, w3_device, route_token_ids, route_expert_ids];
        if let (Some(b1), Some(b3)) = (w1_bias_device, w3_bias_device) {
            inputs.push(b1);
            inputs.push(b3);
        }
        target.wire_node(
            format!("{}.w1w3_swiglu", node.name),
            ops::MetalRoutedQ40SwiGlu {
                input_mode: RoutedInputMode::TokenRows,
                act_alpha_bits: op.act_alpha_bits,
                act_limit_bits: op.act_limit_bits,
                has_bias,
                sync_after_dispatch: moe_sync_after_dispatch(),
            },
            &inputs,
        )?[0]
    } else {
        let mut h1 = target.wire_node(
            format!("{}.w1", node.name),
            ops::MetalRoutedQ40MatMul {
                input_mode: RoutedInputMode::TokenRows,
                sync_after_dispatch: moe_sync_after_dispatch(),
            },
            &[x_expert_input, w1_device, route_token_ids, route_expert_ids],
        )?[0];
        if let Some(w1_bias_device) = w1_bias_device {
            h1 = add_routed_bias(
                target,
                format!("{}.w1", node.name),
                h1,
                w1_bias_device,
                route_expert_ids,
            )?;
        }

        if op.has_w3 {
            let w3_device = sync_outlet_if_required(
                target,
                format!("{}.w3.to-device", node.name),
                mapping[&node.inputs[indexes.w3.context("missing w3 input index")?]],
                DeviceSyncKind::ToDevice,
            )?;
            let mut up = target.wire_node(
                format!("{}.w3", node.name),
                ops::MetalRoutedQ40MatMul {
                    input_mode: RoutedInputMode::TokenRows,
                    sync_after_dispatch: moe_sync_after_dispatch(),
                },
                &[x_expert_input, w3_device, route_token_ids, route_expert_ids],
            )?[0];
            if let Some(w3_bias_device) = w3_bias_device {
                up = add_routed_bias(
                    target,
                    format!("{}.w3", node.name),
                    up,
                    w3_bias_device,
                    route_expert_ids,
                )?;
            }
            if let Some(limit_bits) = op.act_limit_bits {
                let alpha = op.act_alpha_bits.map(f32::from_bits).unwrap_or(1.0);
                target.wire_node(
                    format!("{}.clamped_swiglu", node.name),
                    ops::MetalClampedSwiGlu::new(alpha, f32::from_bits(limit_bits)),
                    &[h1, up],
                )?[0]
            } else {
                let activated = target.wire_node(
                    format!("{}.activation", node.name),
                    kernels::element_wise::metal_element_wise_op(Box::new(
                        tract_core::ops::nn::Silu {},
                    )),
                    &[h1],
                )?[0];
                target.wire_node(
                    format!("{}.swiglu_mul", node.name),
                    kernels::bin_ops::metal_bin_op(Box::new(tract_core::ops::math::Mul)),
                    &[activated, up],
                )?[0]
            }
        } else {
            target.wire_node(
                format!("{}.activation", node.name),
                kernels::element_wise::metal_element_wise_op(Box::new(
                    tract_core::ops::nn::Silu {},
                )),
                &[h1],
            )?[0]
        }
    };

    let mut route_values = target.wire_node(
        format!("{}.w2", node.name),
        ops::MetalRoutedQ40MatMul {
            input_mode: RoutedInputMode::RouteRows,
            sync_after_dispatch: moe_sync_after_dispatch(),
        },
        &[hidden, w2_device, route_token_ids, route_expert_ids],
    )?[0];
    if let Some(w2_bias_device) = w2_bias_device {
        route_values = add_routed_bias(
            target,
            format!("{}.w2", node.name),
            route_values,
            w2_bias_device,
            route_expert_ids,
        )?;
    }
    let output = target.wire_node(
        format!("{}.combine", node.name),
        ops::MetalRoutedCombine { routes_per_token: op.k as u32 },
        &[x_shape_like, route_values, route_token_ids, route_weights],
    )?[0];
    let output = if facts[0].datum_type != f32::datum_type() {
        target.wire_node(
            format!("{}.out.cast", node.name),
            metal_cast_new(facts[0].datum_type).context("unsupported Metal cast")?,
            &[output],
        )?[0]
    } else {
        output
    };

    Ok(Some(tvec![output]))
}

fn q40_moe_activation_supported(op: &MoeFfn) -> bool {
    matches!(op.activation.as_str(), "silu")
        || (op.has_w3 && matches!(op.activation.as_str(), "swiglu"))
}

pub(super) fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
}

#[derive(Default)]
struct MoeInputIndexes {
    w3: Option<usize>,
    wg_bias: Option<usize>,
    w1_bias: Option<usize>,
    w3_bias: Option<usize>,
    w2_bias: Option<usize>,
}

fn moe_input_indexes(op: &MoeFfn) -> MoeInputIndexes {
    let mut next = 4;
    let mut take = |present: bool| {
        if present {
            let ix = next;
            next += 1;
            Some(ix)
        } else {
            None
        }
    };
    MoeInputIndexes {
        w3: take(op.has_w3),
        wg_bias: take(op.has_wg_bias),
        w1_bias: take(op.has_w1_bias),
        w3_bias: take(op.has_w3_bias),
        w2_bias: take(op.has_w2_bias),
    }
}

fn fact_is_f16_or_f32(fact: &TypedFact) -> bool {
    matches!(fact.datum_type, DatumType::F16 | DatumType::F32)
}

/// Shared fact gate for the Metal Q40 MoE fast path. `layout` tells how to
/// read the expert dims out of w1: [E,H,D] for linear, [E,D,H] for canonical
/// (the repack pre-pass checks canonical facts before committing to the
/// transpose; the lowering itself only ever sees linear).
fn q40_moe_facts_supported(op: &MoeFfn, facts: &[&TypedFact], layout: ExpertLayout) -> bool {
    let indexes = moe_input_indexes(op);
    let x_rank_ok =
        facts[0].rank() == 2 || (facts[0].rank() == 3 && facts[0].shape.dims()[0] == 1.to_dim());
    let x_dt_ok = fact_is_f16_or_f32(facts[0]);
    let wg_dt_ok = fact_is_f16_or_f32(facts[1]);
    let w1_q40 = as_quant_fact(facts[2], &Q4_0).is_some();
    let w2_q40 = as_quant_fact(facts[3], &Q4_0).is_some();
    let w3_q40 =
        !op.has_w3 || indexes.w3.is_some_and(|ix| as_quant_fact(facts[ix], &Q4_0).is_some());
    if facts[2].rank() != 3 {
        return false;
    }
    let num_experts = facts[2].shape[0].clone();
    let (d_hidden, d_model) = match layout {
        ExpertLayout::Linear => (facts[2].shape[1].clone(), facts[2].shape[2].clone()),
        ExpertLayout::Canonical => (facts[2].shape[2].clone(), facts[2].shape[1].clone()),
    };
    let wg_bias_ok = indexes.wg_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix]) && facts[ix].rank() == 1 && facts[ix].shape[0] == num_experts
    });
    let w1_bias_ok = indexes.w1_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_hidden
    });
    let w3_bias_ok = indexes.w3_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_hidden
    });
    let w2_bias_ok = indexes.w2_bias.is_none_or(|ix| {
        fact_is_f16_or_f32(facts[ix])
            && facts[ix].rank() == 2
            && facts[ix].shape[0] == num_experts
            && facts[ix].shape[1] == d_model
    });
    // The Metal route-topk kernel scores at most 256 experts per token.
    let experts_ok = num_experts
        .as_i64()
        .is_some_and(|e| (1..=256).contains(&e) && op.k > 0 && op.k <= 16 && op.k <= e as usize);
    x_rank_ok
        && x_dt_ok
        && wg_dt_ok
        && w1_q40
        && w2_q40
        && w3_q40
        && experts_ok
        && wg_bias_ok
        && w1_bias_ok
        && w3_bias_ok
        && w2_bias_ok
}

/// Requantize exclusive canonical Q40 expert constants into linear layout.
/// Callers must opt into the precision loss and temporary projection storage.
pub(super) fn repack_canonical_q40_moe_experts(model: &mut TypedModel) -> TractResult<()> {
    if env_flag("TRACT_METAL_DISABLE_Q40_MOE") || env_flag("TRACT_METAL_DISABLE_Q40_MOE_REPACK") {
        return Ok(());
    }
    Rewriter::default()
        .with_rule_for("repack-canonical-q40-moe", repack_canonical_q40_moe_rule)
        .rewrite(&(), model)
}

fn repack_canonical_q40_moe_rule(
    _: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &MoeFfn,
) -> TractResult<Option<TypedModelPatch>> {
    let log_lowering = env_flag("TRACT_METAL_LOG_Q40_MOE");
    if op.expert_layout != ExpertLayout::Canonical
        || !q40_moe_activation_supported(op)
        || (op.act_limit_bits.is_some() && !op.has_w3)
    {
        return Ok(None);
    }

    let facts = model.node_input_facts(node.id)?;
    if !q40_moe_facts_supported(op, &facts, ExpertLayout::Canonical) {
        return Ok(None);
    }
    // Canonical expert shapes must be concrete and mutually consistent:
    // w1/w3 [E,D,H], w2 [E,H,D]. Both D and H must be quantizable along
    // the new innermost axis after the transpose.
    let indexes = moe_input_indexes(op);
    let w1_shape = facts[2].shape.as_concrete().map(|s| s.to_vec());
    let w2_shape = facts[3].shape.as_concrete().map(|s| s.to_vec());
    let w3_shape = indexes.w3.map(|ix| facts[ix].shape.as_concrete().map(|s| s.to_vec()));
    let (Some(w1_shape), Some(w2_shape)) = (w1_shape, w2_shape) else { return Ok(None) };
    let (e, d_model, d_hidden) = (w1_shape[0], w1_shape[1], w1_shape[2]);
    if w2_shape != [e, d_hidden, d_model]
        || w3_shape.is_some_and(|s| s.as_deref() != Some(&w1_shape[..]))
        || d_model % Q4_0.block_len() != 0
        || d_hidden % Q4_0.block_len() != 0
    {
        return Ok(None);
    }
    // Every expert projection must be an exclusive block-quant Const.
    let mut expert_slots = tvec![2usize, 3];
    expert_slots.extend(indexes.w3);
    let mut const_ids: TVec<usize> = tvec![];
    for &slot in &expert_slots {
        let outlet = node.inputs[slot];
        let knode = model.node(outlet.node);
        let exclusive_bq_const = outlet.slot == 0
            && knode
                .op_as::<Const>()
                .is_some_and(|k| k.val().storage_as::<BlockQuantStorage>().is_some())
            && knode.outputs[0].successors.len() == 1
            && !model.outputs.contains(&outlet);
        if !exclusive_bq_const {
            if log_lowering {
                eprintln!(
                    "Metal Q40 MoE canonical repack skip {}: input {slot} is not an exclusive block-quant const",
                    node.name
                );
            }
            return Ok(None);
        }
        const_ids.push(outlet.node);
    }

    if log_lowering {
        eprintln!(
            "Metal Q40 MoE canonical repack {}: transposing {} expert tensors to linear layout",
            node.name,
            const_ids.len()
        );
    }
    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;
    for (slot, cid) in expert_slots.into_iter().zip(const_ids) {
        let old = model.node(cid).op_as::<Const>().context("expected expert constant")?.val();
        let repacked = transpose_block_quant_experts(old)?;
        let exotic_fact = repacked.exotic_fact()?.context("missing repacked expert fact")?;
        let konst = Const::new_with_exotic_fact(Arc::new(repacked), exotic_fact)?;
        inputs[slot] = patch.wire_node(format!("{name}.linear.{slot}"), konst, &[])?[0];
    }
    let mut linear_op = op.clone();
    linear_op.expert_layout = ExpertLayout::Linear;
    let outputs = patch.wire_node(format!("{name}.linear"), linear_op, &inputs)?;
    for (slot, output) in outputs.into_iter().enumerate() {
        patch.shunt_outside(model, OutletId::new(node.id, slot), output)?;
    }
    Ok(Some(patch))
}

#[cfg(test)]
mod q40_moe_lowering_tests {
    use super::*;
    use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};
    use tract_transformers::ops::moe_ffn::{ExpertLayout, GateMode, MoeFfn};

    fn add_q40_const(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
        let shape = tensor.shape().to_vec();
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0);
        let m: usize = shape[..shape.len() - 1].iter().product();
        let quant = Q4_0.quant_f32(tensor.try_as_plain_ram()?.as_slice::<f32>()?)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
        let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(
            name,
            tract_core::ops::konst::Const::new_with_exotic_fact(packed, Box::new(fact))?,
            &[],
        )?[0])
    }

    /// Qwen-shaped bias-free SwiGLU MoE with Q40 experts in the requested
    /// layout. d_model != d_hidden on purpose: a transposition bug in the
    /// repack would show up as a shape mismatch, not just as noise.
    fn build_qwen_moe(
        layout: ExpertLayout,
        activation: &str,
        tokens: usize,
    ) -> TractResult<(TypedModel, Tensor)> {
        let (d_model, d_hidden, experts, k) = (64, 96, 8, 2);
        let mut rng_state: u64 = 1717;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut make = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| next_f32() * 0.5).collect();
            tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data).unwrap().into_tensor()
        };

        let wg_data = make(&[experts, d_model]);
        let (w1_shape, w2_shape) = match layout {
            // canonical: w1/w3 [E,D,H], w2 [E,H,D]
            ExpertLayout::Canonical => ([experts, d_model, d_hidden], [experts, d_hidden, d_model]),
            // linear: w1/w3 [E,H,D], w2 [E,D,H]
            ExpertLayout::Linear => ([experts, d_hidden, d_model], [experts, d_model, d_hidden]),
        };
        let w1_data = make(&w1_shape);
        let w2_data = make(&w2_shape);
        let w3_data = make(&w1_shape);
        let x_data = make(&[1, tokens, d_model]).cast_to::<f16>()?.into_owned();

        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact([1, tokens, d_model]))?;
        let wg = model.add_const("wg", wg_data)?;
        let w1 = add_q40_const(&mut model, "w1", w1_data)?;
        let w2 = add_q40_const(&mut model, "w2", w2_data)?;
        let w3 = add_q40_const(&mut model, "w3", w3_data)?;

        let op = MoeFfn {
            k,
            activation: activation.to_string(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
            expert_layout: layout,
        };
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&outputs)?;
        Ok((model, x_data))
    }

    fn moe_ffn_layout(model: &TypedModel) -> Option<ExpertLayout> {
        model.nodes().iter().find_map(|n| n.op_as::<MoeFfn>()).map(|op| op.expert_layout)
    }

    fn w1_const_val(model: &TypedModel) -> Arc<Tensor> {
        model
            .nodes()
            .iter()
            .find(|n| &*n.name == "w1")
            .and_then(|n| n.op_as::<Const>())
            .map(|k| k.val().clone())
            .expect("w1 const")
    }

    #[test]
    fn canonical_repack_preserves_graph_contract() -> TractResult<()> {
        let (mut model, _) = build_qwen_moe(ExpertLayout::Canonical, "silu", 4)?;
        let outputs = model.output_fact(0)?.clone();
        let nodes = model.nodes().len();
        repack_canonical_q40_moe_experts(&mut model)?;
        ensure!(moe_ffn_layout(&model) == Some(ExpertLayout::Linear));
        ensure!(model.nodes().len() == nodes);
        ensure!(!model.nodes().iter().any(|node| &*node.name == "w1"));
        ensure!(model.output_fact(0)?.clone() == outputs);
        let before = model
            .nodes()
            .iter()
            .filter_map(|node| node.op_as::<Const>())
            .map(|op| op.val().clone())
            .collect::<Vec<_>>();
        repack_canonical_q40_moe_experts(&mut model)?;
        let after = model
            .nodes()
            .iter()
            .filter_map(|node| node.op_as::<Const>())
            .map(|op| op.val().clone())
            .collect::<Vec<_>>();
        ensure!(before.len() == after.len());
        ensure!(before.iter().zip(&after).all(|(a, b)| Arc::ptr_eq(a, b)));
        Ok(())
    }

    /// The repack pass must not touch linear-layout models: same consts (by
    /// pointer), same layout. This is what keeps existing linear exports
    /// byte-identical through the transform.
    #[test]
    fn linear_q40_moe_repack_is_noop() -> TractResult<()> {
        let (mut model, _) = build_qwen_moe(ExpertLayout::Linear, "silu", 4)?;
        let w1_before = w1_const_val(&model);
        repack_canonical_q40_moe_experts(&mut model)?;
        ensure!(moe_ffn_layout(&model) == Some(ExpertLayout::Linear));
        ensure!(
            Arc::ptr_eq(&w1_before, &w1_const_val(&model)),
            "repack pass rebuilt a linear-layout const"
        );
        Ok(())
    }

    /// A canonical op the Metal lowering would decline anyway (unsupported
    /// activation) must not be repacked: its fallback-path numerics stay
    /// exactly as today.
    #[test]
    fn canonical_q40_moe_repack_skips_unsupported_activation() -> TractResult<()> {
        let (mut model, _) = build_qwen_moe(ExpertLayout::Canonical, "gelu", 4)?;
        let w1_before = w1_const_val(&model);
        repack_canonical_q40_moe_experts(&mut model)?;
        ensure!(moe_ffn_layout(&model) == Some(ExpertLayout::Canonical));
        ensure!(
            Arc::ptr_eq(&w1_before, &w1_const_val(&model)),
            "repack pass rebuilt consts for an op the lowering cannot take"
        );
        Ok(())
    }

    /// The repacked canonical weights must dequantize to (approximately) the
    /// transposed original: catches axis mixups that the end-to-end check
    /// could mask through routing.
    #[test]
    fn canonical_q40_expert_transpose_roundtrip() -> TractResult<()> {
        let (experts, a, b) = (3, 64, 32);
        let mut data = vec![0f32; experts * a * b];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0;
        }
        let quant = Q4_0.quant_f32(&data)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), experts * a, b, Arc::new(quant))?;
        let tensor = storage.into_tensor_with_shape(f32::datum_type(), &[experts, a, b]);

        let transposed = transpose_block_quant_experts(&tensor)?;
        ensure!(transposed.shape() == [experts, b, a]);

        let orig_bqs = tensor.try_storage_as::<BlockQuantStorage>()?;
        let tr_bqs = transposed.try_storage_as::<BlockQuantStorage>()?;
        let orig = Q4_0.dequant_f32(orig_bqs.value())?;
        let tr = Q4_0.dequant_f32(tr_bqs.value())?;
        let orig = orig.try_as_plain_ram()?.as_slice::<f32>()?;
        let tr = tr.try_as_plain_ram()?.as_slice::<f32>()?;
        for e in 0..experts {
            for i in 0..a {
                for j in 0..b {
                    let x = orig[e * a * b + i * b + j];
                    let y = tr[e * a * b + j * a + i];
                    // One Q4_0 step of a block with amax ~1 is 0.125; the
                    // requantization can be off by up to about one step.
                    ensure!(
                        (x - y).abs() <= 0.13 + 0.05 * x.abs(),
                        "expert {e} [{i},{j}]: canonical {x} vs transposed {y}"
                    );
                }
            }
        }
        Ok(())
    }
}
