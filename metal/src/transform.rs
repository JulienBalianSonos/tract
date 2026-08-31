use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::OnceLock;

use crate::context::metal_context;
use crate::kernels::matmul::{GemmKernel, GgmlGemm, MetalGemmImplKind, MfaGemm, MlxGemm};
use crate::{kernels, ops};
use tract_core::dyn_clone::clone_box;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::cnn::conv::rewrite_kernel_conv_in_oihw;
use tract_core::ops::cnn::{Conv, rewrite_conv_with_n_axis};
use tract_core::ops::einsum::prefix_matmul::{PrefixMatMul, rewrite_einsum_to_prefix_matmul};
use tract_core::ops::konst::Const;
use tract_core::ops::nn::Reduce;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantStorage};
use tract_core::transform::ModelTransform;
use tract_gpu::fact::{DeviceFact, DeviceTypedFactExt};
use tract_gpu::rewrite_rules::rewire_syncs::rewire_syncs;
use tract_gpu::rewrite_rules::rms_norm::remove_rms_norm_cast;
use tract_gpu::sync::{
    DeviceSync, DeviceSyncKind, sync_inputs_if_required, sync_model_outputs_if_required,
};
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_gpu::utils::as_quant_fact;
use tract_transformers::ops::moe_ffn::{
    ExpertLayout, MoeFfn, RouteTopK, RoutedInputMode, transpose_block_quant_experts,
};

use crate::rewrite_rules;

/// A registered translator that can convert a core op into a Metal GPU op.
/// Each kernel module submits one (or more) of these via [`register_metal_op!`].
#[allow(clippy::type_complexity)]
pub struct MetalOpTranslator {
    pub type_id: TypeId,
    pub try_make: fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>,
}

inventory::collect!(MetalOpTranslator);

/// Register a translator for a core op type. The closure receives `(source, node, op)`
/// where `op` is already downcast to `$op_type`. Return `Ok(Some(gpu_op))` to translate,
/// `Ok(None)` to skip.
#[macro_export]
macro_rules! register_metal_op {
    ($op_type:ty, |$source:ident, $node:ident, $op:ident| $body:expr) => {
        inventory::submit! {
            $crate::transform::MetalOpTranslator {
                type_id: std::any::TypeId::of::<$op_type>(),
                try_make: |$source, $node| {
                    let Some($op) = $node.op_as::<$op_type>() else {
                        return Ok(None);
                    };
                    $body
                },
            }
        }
    };
}

/// Metal-local SDPA flattening: explode only the `Sdpa` nodes neither fused
/// kernel can take (MLX port first, vendored MFA metallib second), leaving
/// fusable ones for the chooser translator in `kernels::matmul::mlx_sdpa`.
/// (The shared `tract_gpu` `rewire_sdpa` explodes all of them; cuda still
/// uses it.)
fn flatten_unfused_sdpa(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &tract_transformers::ops::sdpa::Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    if crate::kernels::matmul::mlx_sdpa::mlx_sdpa_supported(op, &in_facts)
        || crate::kernels::matmul::mfa::mfa_sdpa_supported(op, &in_facts)
    {
        Ok(None) // leave intact for the fused-Sdpa translator
    } else {
        op.patch_sdpa(model, node) // explode (same as the shared rewire_sdpa)
    }
}

/// An exported causal LLM feeds `Sdpa` an f32 mask next to f16 activations, but
/// the fused kernels template the mask on the activation type. Cast it to the
/// query dtype so the constant folds and the node stays fusable.
fn cast_sdpa_mask_to_query_dt(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    _op: &tract_transformers::ops::sdpa::Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    if in_facts.len() != 4 {
        return Ok(None);
    }
    let (q_dt, mask_dt) = (in_facts[0].datum_type, in_facts[3].datum_type);
    if mask_dt == q_dt || !q_dt.is_float() || !mask_dt.is_float() {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;
    inputs[3] = patch.wire_node(
        format!("{name}.mask_cast"),
        tract_core::ops::cast::cast(q_dt),
        &[inputs[3]],
    )?[0];
    let out = patch.wire_node(&node.name, node.op.clone(), &inputs)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

fn rewire_sdpa_metal(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("cast-sdpa-mask-to-query-dt", cast_sdpa_mask_to_query_dt)
        .with_rule_for("flatten-unfused-sdpa", flatten_unfused_sdpa)
        .rewrite(&(), model)
}

impl MetalGemmImplKind {
    pub fn variants() -> Vec<MetalGemmImplKind> {
        vec![Self::Mlx, Self::Mfa, Self::Ggml]
    }

    pub fn variants_str() -> Vec<&'static str> {
        Self::variants().into_iter().map(|it| it.to_str()).collect()
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Mfa => "mfa",
            Self::Ggml => "ggml",
        }
    }
}

#[derive(Debug, Default)]
pub struct MetalTransform {
    pub gemm_impl: Option<MetalGemmImplKind>,
}

impl ModelTransform for MetalTransform {
    fn name(&self) -> StaticName {
        "metal-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        // The pool translators live in `ops::pool`, which nothing else calls
        // into; without a reference the linker drops the module and with it the
        // inventory registrations.
        crate::ops::pool::link_translators();

        self.transform_up_to_phase(model, usize::MAX)
    }
}

impl FromStr for MetalTransform {
    type Err = TractError;
    fn from_str(str: &str) -> TractResult<Self> {
        let gemm_impl = match str {
            "mlx" => Some(MetalGemmImplKind::Mlx),
            "ggml" => Some(MetalGemmImplKind::Ggml),
            "mfa" => Some(MetalGemmImplKind::Mfa),
            "" => None,
            _ => bail!("Unknown backend"),
        };
        Ok(MetalTransform { gemm_impl })
    }
}

impl MetalTransform {
    pub fn transform_up_to_phase(
        &self,
        model: &mut TypedModel,
        stop_at_phase: usize,
    ) -> TractResult<()> {
        // Init Metal Context if not done previously
        metal_context();

        rewire_sdpa_metal(model)?;
        rewrite_einsum_to_prefix_matmul(model, false)?;
        if stop_at_phase == 0 {
            return Ok(());
        }

        Rewriter::<MetalTransform>::default()
            .with_rule_for("untranspose-matmul-output", rewrite_rules::untranspose_matmul_output)
            .with_rule_for("add-broadcast-pre-matmul", rewrite_rules::add_broadcast_pre_matmul)
            .rewrite(self, model)?;

        Rewriter::default()
            .with_rule_for("rewrite_kernel_conv_in_oihw", rewrite_kernel_conv_in_oihw)
            .with_rule_for("rewrite_conv_with_n_axis", rewrite_conv_with_n_axis)
            .with_rule_for("remove_rms_norm_cast", remove_rms_norm_cast)
            .with_rule_for("split_multi_axis_reduce", split_multi_axis_reduce)
            .rewrite(&(), model)?;

        // Canonical-layout Q40 MoE exports: transpose the expert weights once
        // so the routed-Q40 lowering (which requires the linear layout) can
        // take them. Must run pre-translation, while the consts are host-side.
        repack_canonical_q40_moe_experts(model)?;

        // TRACT_METAL_DUMP_OPS=5: full node listing at the end of phase 1
        // (pre-translation), for debugging rules that should fire here.
        if std::env::var("TRACT_METAL_DUMP_OPS").is_ok_and(|v| v == "5") {
            for node in model.nodes() {
                let ins: Vec<String> =
                    node.inputs.iter().map(|i| format!("{}:{}", i.node, i.slot)).collect();
                eprintln!("  phase1-node {} [{}] {} <- {:?}", node.id, node.op.name(), node.name, ins);
            }
        }

        if stop_at_phase == 1 {
            return Ok(());
        }

        *model = self.translate_model(model)?;

        if stop_at_phase == 2 {
            return Ok(());
        }

        Rewriter::default()
            .with_rule_for("fuse_move_axis", rewrite_rules::fuse_move_axis)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse_axis_op", rewrite_rules::fuse_axis_op)
            .rewrite(&(), model)?;

        rewire_syncs(model)?;
        Ok(())
    }
}

/// Looks up the node's op TypeId in the inventory of registered `MetalOpTranslator`s.
/// Returns `Some(gpu_op)` if a translator matches and succeeds, `None` otherwise.
fn try_make_metal_op(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    type TranslateFn = fn(&TypedModel, &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>>;
    static MAP: OnceLock<HashMap<TypeId, Vec<TranslateFn>>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m: HashMap<TypeId, Vec<TranslateFn>> = HashMap::new();
        for t in inventory::iter::<MetalOpTranslator> {
            m.entry(t.type_id).or_default().push(t.try_make);
        }
        m
    });

    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type)));

    // Copy-based ops are fully generic (no backend-specific dispatch needed).
    if let Some(op) = tract_gpu::ops::copy_based::try_make_copy_based_op(source, node)? {
        return Ok(Some(op));
    }

    if let Some(fns) = map.get(&(*node.op).type_id()) {
        for f in fns {
            if let Some(op) = f(source, node)? {
                return Ok(Some(op));
            }
        }
    }
    Ok(None)
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for MetalTransform {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        // Special multi-node ops handled first
        let input_facts = source.node_input_facts(node.id)?;
        if let Some(op) = node.op_as::<PrefixMatMul>() {
            let facts: Vec<TypedFact> = input_facts.iter().map(|f| (*f).clone()).collect();
            if !op.transpose_c && op.quantize_output.is_none() && check_matmul_in_dts(&facts) {
                let mut device_inputs =
                    sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
                let outlet_ids = convert_matmul_to_metal(
                    source,
                    node,
                    target,
                    &mut device_inputs,
                    op,
                    self.gemm_impl,
                )?;
                return sync_model_outputs_if_required(source, node, target, outlet_ids);
            }
        }
        if let Some(conv) = node.op_as::<Conv>()
            && input_facts.iter().all(|f| DeviceTensor::is_supported_dt(f.datum_type))
            && matches!(input_facts[0].datum_type, DatumType::F16 | DatumType::F32)
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                ops::conv::wire_metal_conv(source, node, target, &device_inputs, conv)?;
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }
        // Resize bakes its plan, so it keeps only the data input: the scales /
        // sizes input it drops is a TDim const with no device equivalent.
        if let Some(gpu_op) = crate::kernels::array::metal_resize(source, node)? {
            let mut input = mapping[&node.inputs[0]];
            if target.outlet_fact(input)?.as_device_fact().is_none() {
                input = target.wire_node(
                    format!("{}.to-device-0", node.name),
                    DeviceSync::new(DeviceSyncKind::ToDevice),
                    &[input],
                )?[0];
            }
            let outlet_ids = target.wire_node(node.name.clone(), gpu_op, &[input])?;
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }
        // Const: inline conversion, not a GPU op
        if let Some(op) = node.op_as::<Const>()
            && DeviceTensor::is_supported_dt(op.val().datum_type())
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids =
                target.wire_node(node.name.clone(), convert_const(op)?, &device_inputs)?;
            return sync_model_outputs_if_required(source, node, target, outlet_ids);
        }

        // Single-op translation.  See the matching CUDA path for rationale:
        // pre-check the gpu_op's output_facts against the already-translated
        // target-side input shapes before wiring, so a stale Reshape (e.g.
        // after pulsification has changed an upstream axis size) falls back
        // to CPU rather than aborting the whole Metal transform.
        let target_inputs: TVec<TypedFact> = node
            .inputs
            .iter()
            .map(|i| target.outlet_fact(mapping[i]).cloned())
            .collect::<TractResult<_>>()?;
        // Mirror sync_inputs_if_required(ToDevice): wrap non-device facts as
        // device facts so the GPU op's `output_facts` sees uniform device
        // inputs, matching what it'll receive after sync nodes are wired.
        // Mixed inputs (e.g. host kv-cache + device current activation) make
        // `output_facts` bail with "Inconsistent facts", wrongly tripping CPU
        // fallback.
        let target_inputs_post_sync: TVec<TypedFact> = target_inputs
            .iter()
            .map(|f| -> TractResult<TypedFact> {
                if f.as_device_fact().is_some() {
                    Ok(f.clone())
                } else {
                    Ok(tract_gpu::fact::DeviceFact::from_host(f.clone())?.into_exotic_fact())
                }
            })
            .collect::<TractResult<_>>()?;
        let target_input_post_sync_refs: TVec<&TypedFact> =
            target_inputs_post_sync.iter().collect();
        if let Some(gpu_op) = try_make_metal_op(source, node)?
            && gpu_op.output_facts(&target_input_post_sync_refs).is_ok()
        {
            let device_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
            let outlet_ids = target.wire_node(node.name.clone(), gpu_op, &device_inputs)?;
            sync_model_outputs_if_required(source, node, target, outlet_ids)
        } else {
            let cpu_inputs =
                sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToHost)?;
            target.wire_node(&node.name, node.op.clone(), &cpu_inputs)
        }
    }
}

/// Sync after each MoE routed-matmul dispatch (see
/// `MetalRoutedQ40MatMul::sync_after_dispatch`). Defaults to on: without it,
/// GPT-OSS on Metal degenerates to constant <|endofprompt|> past ~1024 tokens
/// of context. `TRACT_METAL_MOE_UNSAFE_NOSYNC=1` disables it, for measuring
/// the eventual real fix against the fast broken baseline.
pub(crate) fn moe_sync_after_dispatch() -> bool {
    !env_flag("TRACT_METAL_MOE_UNSAFE_NOSYNC")
}

fn q40_moe_activation_supported(op: &MoeFfn) -> bool {
    matches!(op.activation.as_str(), "silu")
        || (op.has_w3 && matches!(op.activation.as_str(), "swiglu"))
}

fn env_flag(name: &str) -> bool {
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
    let experts_ok = num_experts.as_i64().is_some_and(|e| e <= 256);
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

/// Pre-translation repack: canonical-layout Q40 MoE experts (w1/w3 [E,D,H],
/// w2 [E,H,D]) are transposed once, per projection tensor, into the linear
/// layout ([E,H,D] / [E,D,H]) required by the Metal routed-Q40 kernels, and
/// the op is flipped to `ExpertLayout::Linear`. This makes canonical-layout
/// exports eligible for the fast MoE path without re-exporting. Linear-layout
/// models are untouched, as is any canonical op the lowering gate would
/// decline anyway (so its numerics stay exactly as today on the fallback
/// path; the transpose requantizes and is not bit-lossless).
///
/// Consts are rebuilt in place, one projection tensor at a time, so peak
/// memory stays at roughly one extra projection tensor.
fn repack_canonical_q40_moe_experts(model: &mut TypedModel) -> TractResult<()> {
    if env_flag("TRACT_METAL_DISABLE_Q40_MOE") || env_flag("TRACT_METAL_DISABLE_Q40_MOE_REPACK") {
        return Ok(());
    }
    let log_lowering = env_flag("TRACT_METAL_LOG_Q40_MOE");
    'nodes: for node_id in 0..model.nodes().len() {
        let Some(op) = model.node(node_id).op_as::<MoeFfn>() else { continue };
        if op.expert_layout != ExpertLayout::Canonical
            || !q40_moe_activation_supported(op)
            || (op.act_limit_bits.is_some() && !op.has_w3)
        {
            continue;
        }
        let op = op.clone();
        let facts = model.node_input_facts(node_id)?;
        if !q40_moe_facts_supported(&op, &facts, ExpertLayout::Canonical) {
            continue;
        }
        // Canonical expert shapes must be concrete and mutually consistent:
        // w1/w3 [E,D,H], w2 [E,H,D]. Both D and H must be quantizable along
        // the new innermost axis after the transpose.
        let indexes = moe_input_indexes(&op);
        let w1_shape = facts[2].shape.as_concrete().map(|s| s.to_vec());
        let w2_shape = facts[3].shape.as_concrete().map(|s| s.to_vec());
        let w3_shape = indexes.w3.map(|ix| facts[ix].shape.as_concrete().map(|s| s.to_vec()));
        let (Some(w1_shape), Some(w2_shape)) = (w1_shape, w2_shape) else { continue };
        let (e, d_model, d_hidden) = (w1_shape[0], w1_shape[1], w1_shape[2]);
        if w2_shape != [e, d_hidden, d_model]
            || w3_shape.is_some_and(|s| s.as_deref() != Some(&w1_shape[..]))
            || d_model % Q4_0.block_len() != 0
            || d_hidden % Q4_0.block_len() != 0
        {
            continue;
        }
        // `facts` borrows the model (and TVec's Drop keeps the borrow alive);
        // release it before the in-place const mutations below.
        drop(facts);
        // Every expert projection must be an exclusive block-quant Const.
        let node = model.node(node_id);
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
                continue 'nodes;
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
        for cid in const_ids {
            let old = model
                .node(cid)
                .op_as::<Const>()
                .context("expert weight node is not a Const")?
                .val()
                .clone();
            let repacked = transpose_block_quant_experts(&old)?;
            drop(old);
            let exotic_fact =
                repacked.exotic_fact()?.context("repacked expert tensor has no exotic fact")?;
            let konst = Const::new_with_exotic_fact(Arc::new(repacked), exotic_fact)?;
            let fact = konst.output_facts(&[])?.remove(0);
            let const_node = model.node_mut(cid);
            const_node.op = Box::new(konst);
            const_node.outputs[0].fact = fact;
        }
        let mut linear_op = op;
        linear_op.expert_layout = ExpertLayout::Linear;
        model.node_mut(node_id).op = Box::new(linear_op);
    }
    Ok(())
}

pub(crate) fn metal_cast_new(to: DatumType) -> Option<tract_gpu::ops::cast::GpuCast> {
    tract_gpu::ops::cast::GpuCast::new(
        to,
        "Metal",
        kernels::array::metal_cast_dispatch,
        kernels::array::Cast::is_supported_dt,
    )
}

fn check_matmul_in_dts(in_facts: &[TypedFact]) -> bool {
    MlxGemm.is_supported_dts(in_facts)
        || MfaGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(in_facts)
        || GgmlGemm.is_supported_dts(&[in_facts[1].clone(), in_facts[0].clone()])
}

fn is_input_broadcast(facts: TVec<&TypedFact>) -> bool {
    // Assume weights are in second postion
    let b_batch_dims: Vec<TDim> = if as_quant_fact(facts[1], &Q4_0).is_some() {
        facts[1].shape.dims().to_vec()
    } else {
        let rank = facts[1].rank();
        facts[1].shape.dims()[..rank - 2].to_vec()
    };

    let a_rank = facts[0].rank();
    let mut a_batch_dims = facts[0].shape[..(a_rank - 2)].to_vec();

    a_batch_dims.retain(|tdim| !matches!(tdim, TDim::Sym(_)) || b_batch_dims.contains(tdim));
    let symb_in_a = a_batch_dims != facts[0].shape[..(a_rank - 2)].to_vec();

    let a_batch_size = a_batch_dims.iter().product::<TDim>().gcd();
    let b_batch_size = b_batch_dims.iter().product::<TDim>().gcd();

    (a_batch_size % b_batch_size == 0) && ((a_batch_size != b_batch_size) || symb_in_a)
}

pub fn resolve_gemm_impl(
    gemm_impl: Option<MetalGemmImplKind>,
    input_facts: TVec<&TypedFact>,
) -> TractResult<MetalGemmImplKind> {
    if let Some(gemm) = gemm_impl {
        Ok(gemm)
    } else if as_quant_fact(input_facts[0], &Q4_0).is_some()
        || as_quant_fact(input_facts[1], &Q4_0).is_some()
        || input_facts[0].datum_type != input_facts[1].datum_type
        || is_input_broadcast(input_facts)
    {
        Ok(MetalGemmImplKind::Ggml)
    } else {
        Ok(MetalGemmImplKind::Mlx)
    }
}

fn convert_matmul_to_metal(
    model: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &mut [OutletId],
    op: &PrefixMatMul,
    gemm_impl: Option<MetalGemmImplKind>,
) -> TractResult<TVec<OutletId>> {
    let mut owned_facts: TVec<TypedFact> =
        model.node_input_facts(node.id)?.iter().map(|f| (*f).clone()).collect();

    // The metal GEMMs accumulate in their input dtype, so a matmul asking for a wider
    // `operating_dt` than its inputs must have them cast up: an SDPA scores matmul is
    // wired f32 precisely because its f16 products can leave f16 range, and a f16 GEMM
    // would saturate them to inf.
    if let Some(acc) = op.operating_dt {
        for i in 0..2 {
            let dt = owned_facts[i].datum_type;
            if dt.is_float() && acc.is_float() && dt.size_of() < acc.size_of() {
                inputs[i] = target.wire_node(
                    format!("{}.cast_acc_{i}", node.name),
                    metal_cast_new(acc).with_context(|| format!("No metal cast to {acc:?}"))?,
                    &[inputs[i]],
                )?[0];
                owned_facts[i].datum_type = acc;
            }
        }
    }
    let mut input_facts: TVec<&TypedFact> = owned_facts.iter().collect();

    let resolved_gemm_impl = resolve_gemm_impl(gemm_impl, input_facts.clone())?;
    if matches!(resolved_gemm_impl, MetalGemmImplKind::Mlx | MetalGemmImplKind::Mfa)
        && (input_facts[0].datum_type != input_facts[1].datum_type)
    {
        ensure!(
            input_facts[0].datum_type == DatumType::F16
                || input_facts[1].datum_type == DatumType::F16
        );
        let inp_to_cast = if input_facts[0].datum_type == DatumType::F16 {
            &mut inputs[0]
        } else {
            &mut inputs[1]
        };
        *inp_to_cast = target.wire_node(
            node.name.clone() + ".cast_input",
            metal_cast_new(DatumType::F32).unwrap(),
            &[*inp_to_cast],
        )?[0];
    }

    let mut matmul_output = match resolved_gemm_impl {
        MetalGemmImplKind::Mlx => {
            let op = ops::MetalGemm::<MlxGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Mfa => {
            let op = ops::MetalGemm::<MfaGemm>::new(op.transpose_a, op.transpose_b);
            target.wire_node(node.name.clone(), op, inputs)?
        }
        MetalGemmImplKind::Ggml => {
            let mut swap_inputs = false;
            if !GgmlGemm.is_supported_dts(&[input_facts[0].clone(), input_facts[1].clone()])
                && GgmlGemm.is_supported_dts(&[input_facts[1].clone(), input_facts[0].clone()])
            {
                input_facts.swap(0, 1);
                inputs.swap(0, 1);
                swap_inputs = true;
            }

            let a_pos = swap_inputs as usize;
            let b_pos = 1 - swap_inputs as usize;
            if op.transpose_a {
                ensure!(
                    as_quant_fact(input_facts[a_pos], &Q4_0).is_none(),
                    "Cannot transpose Q40 tensor"
                );

                let rank = input_facts[a_pos].rank();
                let perm_a_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                let perm_a_name = node.name.clone() + ".perm_a";
                inputs[a_pos] = target.wire_node(perm_a_name, perm_a_op, &[inputs[a_pos]])?[0];
            }

            // The GGML kernels now consume f16 activations directly (and emit
            // f16 output via output_dt), so no f16->f32 activation upcast is
            // inserted here anymore.

            if !op.transpose_b {
                ensure!(
                    as_quant_fact(input_facts[b_pos], &Q4_0).is_none(),
                    "Cannot transpose Q40 tensor"
                );

                let rank = input_facts[b_pos].rank();
                let perm_b_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                let perm_b_name = node.name.clone() + ".perm_b";
                inputs[b_pos] = target.wire_node(perm_b_name, perm_b_op, &[inputs[b_pos]])?[0];
            }
            let op = ops::MetalGemm::<GgmlGemm>::new(false, true);
            let mut matmul_output = target.wire_node(node.name.clone(), op, inputs)?;

            if swap_inputs {
                let out_fact = target.outlet_fact(matmul_output[0])?;
                let rank = &out_fact
                    .exotic_fact
                    .clone()
                    .map(|fact| fact.clarify_dt_shape().unwrap().1.len())
                    .unwrap();

                let perm_out_op =
                    tract_gpu::ops::change_axes::GpuAxisOp::new(AxisOp::Move(rank - 2, rank - 1));
                matmul_output = target.wire_node(
                    node.name.clone() + ".perm_out",
                    perm_out_op,
                    &matmul_output,
                )?;
            }
            matmul_output
        }
    };

    let out_fact = target.outlet_fact(matmul_output[0])?;
    let out_dt = out_fact.as_device_fact().map(|f| f.datum_type).unwrap_or(out_fact.datum_type);

    let expected_dt = model.node_output_facts(node.id)?[0].datum_type;

    if out_dt != expected_dt {
        ensure!(
            kernels::array::Cast::is_supported_dt(out_dt),
            "Matmul output type cannot be casted to expected type"
        );
        let cast_op = metal_cast_new(model.node_output_facts(node.id)?[0].datum_type).unwrap();
        matmul_output =
            target.wire_node(node.name.clone() + ".out_cast", cast_op, &matmul_output)?
    }
    Ok(matmul_output)
}

fn convert_const(op: &Const) -> TractResult<Const> {
    let typed_fact: TypedFact = Arc::clone(op.val()).try_into()?;
    let metal_fact = if let Some(of) = op.exotic_fact() {
        DeviceFact::from_host(typed_fact.with_exotic_fact(clone_box(of)))?
    } else {
        DeviceFact::from_host(typed_fact)?
    };

    let metal_const = op.val().clone().into_device()?.into_tensor().into_arc_tensor();
    Const::new_with_exotic_fact(metal_const, Box::new(metal_fact))
}

/// Rewrites a `Reduce` over several axes into a chain of single-axis reduces, which is
/// what `GpuReduce` accepts. Only reducers that compose associatively per axis qualify:
/// `MeanOfSquares` over a chain is not the multi-axis result.
fn split_multi_axis_reduce(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.axes.len() > 1);
    use tract_core::ops::nn::Reducer::*;
    rule_if!(matches!(op.reducer, Sum | Prod | Min | Max | Any | All));
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[0])?;
    let mut axes = op.axes.clone();
    axes.sort();
    for (i, &axis) in axes.iter().rev().enumerate() {
        let single = Reduce { axes: tvec![axis], reducer: op.reducer };
        wire = patch.wire_node(format!("{node_name}.axis_{i}"), single, &[wire])?[0];
    }
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

// NOTE(split-h): the full end-to-end lowering tests for this pass
// (`gpt_oss_moe_lowering_*`, `canonical_q40_moe_repack_lowers_and_matches`)
// live upstream on `feat/moe-ffn-operator` but exercise `MetalTransform`
// actually converting a `MoeFfn` node into `MetalRouteTopK` / the routed-Q40
// matmul chain via `convert_q40_moe_ffn_to_metal`. That translation function
// (and its handful of helpers: `add_routed_bias`, `sync_f32_input`,
// `sync_outlet_if_required`) is not part of this split -- it is the glue
// that stacks split/a's `MoeFfn` op, split/b's tuning knobs and this split's
// kernels together, and is deferred to the PR that merges all three. Only
// the pre-translation repack pass itself (`repack_canonical_q40_moe_experts`,
// which never touches Metal ops) is covered here.
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
        let quant = Q4_0.quant_f32(tensor.try_as_plain()?.as_slice::<f32>()?)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
        let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model
            .wire_node(name, tract_core::ops::konst::Const::new_with_exotic_fact(packed, Box::new(fact))?, &[])?[0])
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
            ExpertLayout::Canonical => {
                ([experts, d_model, d_hidden], [experts, d_hidden, d_model])
            }
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
        model
            .nodes()
            .iter()
            .find_map(|n| n.op_as::<MoeFfn>())
            .map(|op| op.expert_layout)
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
        let orig = orig.try_as_plain()?.as_slice::<f32>()?;
        let tr = tr.try_as_plain()?.as_slice::<f32>()?;
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
