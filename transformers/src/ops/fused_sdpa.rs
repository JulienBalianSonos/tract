//! Attention over caller-owned K/V history with optional normalization sinks.
//!
//! CPU evaluation is a stateless tensor computation. Metal may retain append-only
//! storage, but incoming history remains authoritative. Serialization expands to
//! portable tensor operations instead of introducing a device cache primitive.
//!
//! Recognition requires a single f16 QK product and scale multiply, f32 mask and
//! softmax, f16 probabilities and an AV product. Split or f32 score expressions
//! retain their original operations and precision boundaries. The mask defines
//! visible keys; unsupported graphs keep the standard backend fallback.

use tract_nnef::internal::*;

/// Sequence axis of `[B, H, S, D]`.
const SEQ_AXIS: usize = 2;

/// Semantic attention with caller-owned history and explicit rounding boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedSdpa {
    /// Softmax scale as f32 bits (f32 lacks Eq/Hash).
    pub scale_bits: u32,
    /// Sliding-attention window in keys (0 = full attention). Extracted by
    /// the fuse rule from the mask-building subgraph; runtimes may clamp
    /// their attention reads to the last `window + S - 1` keys, since the
    /// mask sends everything older to -inf anyway. The mask input remains
    /// the semantic source of truth, so 0 is always safe.
    pub window: u32,
    /// GPT-OSS-style attention sinks: one extra logit per q head in the
    /// softmax denominator, no value row. When set, the op takes a trailing
    /// `sinks` input of `Hq` f32 values.
    pub has_sinks: bool,
}

impl FusedSdpa {
    /// Build the portable tensor computation without mutable device cache state.
    pub fn expanded_model(&self, facts: &[&TypedFact]) -> TractResult<TypedModel> {
        self.output_facts(facts)?;
        let mut model = TypedModel::default();
        let inputs = facts
            .iter()
            .enumerate()
            .map(|(i, fact)| model.add_source(format!("input_{i}"), fact.without_value()))
            .collect::<TractResult<TVec<_>>>()?;
        let dt = facts[0].datum_type;
        let q_shape = facts[0].shape.to_tvec();
        let hq = q_shape[1].clone();
        let hkv = facts[1].shape[1].clone();
        let group = hq.clone() / hkv.to_usize()?;
        let k = model.wire_node("keys", TypedConcat { axis: 2 }, &[inputs[3], inputs[1]])?[0];
        let v = model.wire_node("values", TypedConcat { axis: 2 }, &[inputs[4], inputs[2]])?[0];
        let total = model.outlet_fact(k)?.shape[2].clone();
        let mut expanded = tvec![];
        for (name, wire) in [("keys", k), ("values", v)] {
            let wire = model.wire_node(format!("{name}.group"), AxisOp::Add(2), &[wire])?[0];
            let shape = tvec![
                q_shape[0].clone(),
                hkv.clone(),
                group.clone(),
                total.clone(),
                q_shape[3].clone()
            ];
            let wire = model.wire_node(
                format!("{name}.repeat"),
                MultiBroadcastTo { shape: shape.into() },
                &[wire],
            )?[0];
            let wire = model.wire_node(
                format!("{name}.heads"),
                AxisOp::Reshape(1, tvec![hkv.clone(), group.clone()], tvec![hq.clone()]),
                &[wire],
            )?[0];
            expanded.push(wire);
        }
        let scores = model.wire_node(
            "scores",
            EinSum::new("bhqd,bhkd->bhqk".parse()?, DatumType::F32),
            &[inputs[0], expanded[0]],
        )?[0];
        let scores = model.wire_node("scores.round", Cast { to: dt }, &[scores])?[0];
        let scale = model.add_const(
            "scale",
            tensor0(self.scale()).cast_to_dt(dt)?.into_owned().broadcast_into_rank(4)?,
        )?;
        let scores =
            model.wire_node("scaled", TypedBinOp(Box::new(Mul), None), &[scores, scale])?[0];
        let scores = model.wire_node("scaled.float", Cast { to: DatumType::F32 }, &[scores])?[0];
        let mask = model.wire_node("mask.float", Cast { to: DatumType::F32 }, &[inputs[5]])?[0];
        let mask = model.wire_node(
            "mask.shape",
            AxisOp::Reshape(
                0,
                facts[5].shape.to_tvec(),
                tvec![1.to_dim(), 1.to_dim(), q_shape[2].clone(), total.clone()],
            ),
            &[mask],
        )?[0];
        let mut logits = model.wire_node(
            "masked",
            TypedBinOp(Box::new(tract_nnef::tract_core::ops::math::Add), None),
            &[scores, mask],
        )?[0];
        if self.has_sinks {
            let sinks = model.wire_node(
                "sinks.shape",
                AxisOp::Reshape(
                    0,
                    facts[6].shape.to_tvec(),
                    tvec![1.to_dim(), hq, 1.to_dim(), 1.to_dim()],
                ),
                &[inputs[6]],
            )?[0];
            let shape =
                tvec![q_shape[0].clone(), q_shape[1].clone(), q_shape[2].clone(), 1.to_dim()];
            let sinks = model.wire_node(
                "sinks.repeat",
                MultiBroadcastTo { shape: shape.into() },
                &[sinks],
            )?[0];
            logits = model.wire_node("with-sinks", TypedConcat { axis: 3 }, &[logits, sinks])?[0];
        }
        let mut probs = model.wire_node(
            "probabilities",
            Softmax { axes: tvec![3], ..Softmax::default() },
            &[logits],
        )?[0];
        if self.has_sinks {
            probs = model.wire_node(
                "drop-sinks",
                Slice { axis: 3, start: 0.to_dim(), end: total },
                &[probs],
            )?[0];
        }
        let probs = model.wire_node("probabilities.round", Cast { to: dt }, &[probs])?[0];
        let output = model.wire_node(
            "attention",
            EinSum::new("bhqk,bhkd->bhqd".parse()?, DatumType::F32),
            &[probs, expanded[1]],
        )?[0];
        let output = model.wire_node("attention.round", Cast { to: dt }, &[output])?[0];
        model.select_output_outlets(&[output, k, v])?;
        Ok(model)
    }
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
    pub fn input_count(&self) -> usize {
        6 + self.has_sinks as usize
    }
}

impl Op for FusedSdpa {
    fn name(&self) -> StaticName {
        "FusedSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale={} window={} sinks={}", self.scale(), self.window, self.has_sinks)])
    }
    op_as_typed_op!();
}

impl EvalOp for FusedSdpa {
    op_out_of_plan!();
    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let facts = inputs
            .iter()
            .map(|i| TypedFact::try_from(i.clone().into_arc_tensor()))
            .collect::<TractResult<TVec<_>>>()?;
        TypedSimplePlan::new(self.expanded_model(&facts.iter().collect::<Vec<_>>())?)?.run(inputs)
    }
}

impl TypedOp for FusedSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == self.input_count(),
            "FusedSdpa expects [q, k_new, v_new, k_cache, v_cache, mask{}], got {} inputs",
            if self.has_sinks { ", sinks" } else { "" },
            inputs.len()
        );
        ensure!(self.scale().is_finite(), "attention scale must be finite");
        let (q, k_new, v_new, k_cache, v_cache, mask) =
            (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
        ensure!(inputs[..5].iter().all(|f| f.rank() == 4));
        ensure!(matches!(q.datum_type, DatumType::F16 | DatumType::F32));
        ensure!(inputs[..5].iter().all(|f| f.datum_type == q.datum_type));
        ensure!(k_new.shape == v_new.shape && k_cache.shape == v_cache.shape);
        ensure!(q.shape[0] == k_new.shape[0] && q.shape[0] == k_cache.shape[0]);
        ensure!(q.shape[2] == k_new.shape[2]);
        ensure!(q.shape[3] == k_new.shape[3] && q.shape[3] == k_cache.shape[3]);
        ensure!(k_new.shape[1] == k_cache.shape[1]);
        let hkv = k_new.shape[1].to_usize()?;
        let hq = q.shape[1].to_usize()?;
        ensure!(hq > 0 && hkv > 0 && hq.is_multiple_of(hkv));
        ensure!((2..=4).contains(&mask.rank()) && mask.datum_type.is_float());
        ensure!(mask.shape[..mask.rank() - 2].iter().all(|d| d == &1.to_dim()));
        ensure!(mask.shape[mask.rank() - 2] == q.shape[2]);
        ensure!(mask.shape[mask.rank() - 1] == k_cache.shape[2].clone() + k_new.shape[2].clone());
        if self.has_sinks {
            ensure!(inputs[6].datum_type == DatumType::F32);
            ensure!(inputs[6].shape.volume() == q.shape[1]);
        }
        let total = k_cache.shape[SEQ_AXIS].clone() + k_new.shape[SEQ_AXIS].clone();
        let mut k_out = k_cache.without_value();
        k_out.shape.set(SEQ_AXIS, total.clone());
        k_out.datum_type = q.datum_type;
        let mut v_out = v_cache.without_value();
        v_out.shape.set(SEQ_AXIS, total);
        v_out.datum_type = q.datum_type;
        Ok(tvec!(q.without_value(), k_out, v_out))
    }
    as_op!();
}

pub fn register(registry: &mut Registry) {
    registry.register_dumper(dump);
    // Primary, generic name: the trailing `sinks` tensor is optional
    // (defaults to false = absent), so one fragment covers both variants.
    // Serialization always emits this name.
    registry.register_primitive(
        "tract_transformers_fused_sdpa",
        &parameters(SinksParam::Optional),
        &outputs(),
        load,
    );
    // Historical name from when the op was GPT-OSS-specific (sinks always
    // present): kept as a deserialization alias so existing dumps keep
    // loading.
    registry.register_primitive(
        "tract_transformers_gpt_oss_sdpa",
        &parameters(SinksParam::Required),
        &outputs(),
        load_legacy_with_sinks,
    );
}

fn outputs() -> Vec<(&'static str, tract_nnef::ast::TypeSpec)> {
    vec![
        ("output", TypeName::Scalar.tensor()),
        ("k_cache_out", TypeName::Scalar.tensor()),
        ("v_cache_out", TypeName::Scalar.tensor()),
    ]
}

enum SinksParam {
    /// Generic fragment: `sinks` defaults to false (= absent).
    Optional,
    /// Legacy GPT-OSS fragment: `sinks` is a required input.
    Required,
}

fn parameters(sinks: SinksParam) -> Vec<Parameter> {
    let mut params = vec![
        TypeName::Scalar.tensor().named("q"),
        TypeName::Scalar.tensor().named("k_new"),
        TypeName::Scalar.tensor().named("v_new"),
        TypeName::Scalar.tensor().named("k_cache"),
        TypeName::Scalar.tensor().named("v_cache"),
        TypeName::Scalar.tensor().named("mask"),
    ];
    match sinks {
        SinksParam::Optional => {
            params.push(TypeName::Scalar.tensor().named("sinks").default(false))
        }
        SinksParam::Required => params.push(TypeName::Scalar.tensor().named("sinks")),
    }
    params.push(TypeName::Scalar.named("scale"));
    params.push(TypeName::Integer.named("window").default(0));
    params
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &FusedSdpa) -> TractResult<Option<Arc<RValue>>> {
    let facts = ast.model.node_input_facts(node.id)?;
    let mut semantic = op.expanded_model(&facts)?;
    semantic.symbols = ast.model.symbols.clone();
    let (mut fragment, mut tensors) = tract_nnef::ser::to_fragment_def(ast, &semantic)?;
    fragment.decl.id = Identifier(format!("attention_{}", ast.fragments.len()));
    let mut arguments = node.inputs.iter().map(|i| ast.mapping[i].clone()).collect::<Vec<_>>();
    tensors.sort_by_key(|tensor| tensor.label.clone());
    for tensor in tensors {
        arguments.push(ast.konst_variable(&tensor.label, &tensor.value)?);
    }
    let result = invocation(fragment.decl.id.clone(), &arguments, &[]);
    ast.fragments.insert(fragment.decl.id.clone(), fragment);
    Ok(Some(result))
}

fn load_common(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    sinks: Option<OutletId>,
) -> TractResult<Value> {
    let mut inputs: TVec<OutletId> = tvec!(
        invocation.named_arg_as(builder, "q")?,
        invocation.named_arg_as(builder, "k_new")?,
        invocation.named_arg_as(builder, "v_new")?,
        invocation.named_arg_as(builder, "k_cache")?,
        invocation.named_arg_as(builder, "v_cache")?,
        invocation.named_arg_as(builder, "mask")?,
    );
    let has_sinks = sinks.is_some();
    if let Some(sinks) = sinks {
        inputs.push(sinks);
    }
    let scale: f32 = invocation.named_arg_as(builder, "scale")?;
    let window: i64 = invocation.named_arg_as(builder, "window")?;
    builder
        .wire(FusedSdpa { scale_bits: scale.to_bits(), window: window as u32, has_sinks }, &inputs)
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let sinks: Option<OutletId> = invocation.optional_named_arg_as(builder, "sinks")?;
    load_common(builder, invocation, sinks)
}

fn load_legacy_with_sinks(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let sinks: OutletId = invocation.named_arg_as(builder, "sinks")?;
    load_common(builder, invocation, Some(sinks))
}

use tract_nnef::tract_core::ops::array::{MultiBroadcastTo, Slice, TypedConcat};
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::nn::{Reduce, Reducer, Softmax};
use tract_nnef::tract_core::ops::source::TypedSource;
use tract_nnef::tract_core::transform::ModelTransform;

#[derive(Debug)]
pub struct FusedSdpaTransform;

impl ModelTransform for FusedSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_sdpa_inplace_kv".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-sdpa-inplace-kv", fuse_sdpa_rule)
            .rewrite(&(), model)?;
        model.compact()
    }
}

fn prev<'m>(model: &'m TypedModel, node: &TypedNode, slot: usize) -> &'m TypedNode {
    model.node(node.inputs[slot].node)
}

fn is_bin(node: &TypedNode, name: &str) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.name() == name)
}

fn single_consumer<'m>(model: &'m TypedModel, node: &TypedNode) -> Option<&'m TypedNode> {
    let succ = &model.outlet_successors(node.id.into());
    if succ.len() == 1 { Some(model.node(succ[0].node)) } else { None }
}

/// Walk a GQA-expansion chain (AddAxis/Reshape/MultiBroadcastTo/Cast, in any
/// order) from `outlet` up to the KV cache concat. Returns the concat node id.
fn expand_chain_to_concat(model: &TypedModel, outlet: OutletId) -> Option<usize> {
    let mut id = outlet.node;
    for _ in 0..8 {
        let n = model.node(id);
        if n.op_as::<TypedConcat>().is_some() {
            return Some(id);
        }
        if n.op_is::<AxisOp>()
            || n.op_as::<MultiBroadcastTo>().is_some()
            || n.op_as::<Cast>().is_some()
        {
            id = n.inputs[0].node;
        } else {
            return None;
        }
    }
    None
}

/// Walk one QK einsum branch back to (q_part outlet, K concat node, head_dim
/// slice start). The branch is `[Reshape](EinSum([Cast][Reshape](q_part),
/// k_side))` where k_side is `Slice(expand(KConcat))` (rope-split models) or
/// directly `expand(KConcat)` (single-branch models, start = 0). The outer
/// reshape, the q-side f32 cast (einsum-in-f32 exports) and the q-side layout
/// reshape are each optional; the q-part shape checks in the caller guard the
/// tolerant walk.
fn qk_branch(model: &TypedModel, branch_root: &TypedNode) -> Option<(OutletId, usize, i64)> {
    let einsum =
        if branch_root.op_is::<AxisOp>() { prev(model, branch_root, 0) } else { branch_root };
    einsum.op_as::<EinSum>()?;
    if einsum.inputs.len() != 2 {
        return None;
    }
    let mut q_part = einsum.inputs[0];
    if model.node(q_part.node).op_as::<Cast>().is_some() {
        q_part = model.node(q_part.node).inputs[0];
    }
    if model.node(q_part.node).op_is::<AxisOp>() {
        q_part = model.node(q_part.node).inputs[0];
    }
    let k_side = prev(model, einsum, 1);
    let (start, k_expand_root) = if let Some(slice) = k_side.op_as::<Slice>() {
        let start = slice.start.to_i64().ok()?;
        (start, k_side.inputs[0])
    } else {
        (0, OutletId::new(k_side.id, 0))
    };
    let kconcat = expand_chain_to_concat(model, k_expand_root)?;
    Some((q_part, kconcat, start))
}

pub fn fuse_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _softmax: &Softmax,
) -> TractResult<Option<TypedModelPatch>> {
    let rank = model.outlet_fact(node.inputs[0])?.rank();
    rule_if!(rank > 0);
    rule_if!(_softmax.kind == tract_nnef::tract_core::ops::nn::SoftmaxKind::Softmax);
    rule_if!(_softmax.axes == tvec![rank - 1]);
    // GPT-OSS: Softmax(Sub(Concat([logits, sinks]), Reduce<Max>(same Concat)))
    // Plain:   Softmax(Add(logits, mask))
    let sm_in = prev(model, node, 0);
    let (mask_add, sinks_tensor) = if is_bin(sm_in, "Sub") {
        let cat = prev(model, sm_in, 0);
        let mx = prev(model, sm_in, 1);
        let Some(concat) = cat.op_as::<TypedConcat>() else { return Ok(None) };
        if cat.inputs.len() != 2 || concat.axis != cat.outputs[0].fact.rank() - 1 {
            return Ok(None);
        }
        if !mx
            .op_as::<Reduce>()
            .is_some_and(|r| r.reducer == Reducer::Max && r.axes == tvec![rank - 1])
            || mx.inputs[0] != sm_in.inputs[0]
        {
            return Ok(None);
        }
        let sinks_bcast = prev(model, cat, 1);
        if sinks_bcast.op_as::<MultiBroadcastTo>().is_none() {
            return Ok(None);
        }
        let Some(sinks_k) = prev(model, sinks_bcast, 0).op_as::<Const>() else {
            return Ok(None);
        };
        (prev(model, cat, 0), Some(sinks_k.val().clone()))
    } else if is_bin(sm_in, "Add") {
        (sm_in, None)
    } else {
        return Ok(None);
    };
    let has_sinks = sinks_tensor.is_some();

    if !is_bin(mask_add, "Add") {
        return Ok(None);
    }
    let mask_outlet = mask_add.inputs[1];
    let mut scale_mul = prev(model, mask_add, 0);
    if scale_mul.op_as::<Cast>().is_some() {
        scale_mul = prev(model, scale_mul, 0);
    }
    if !is_bin(scale_mul, "Mul") {
        return Ok(None);
    }
    // Scalar scale constant on either side of the Mul (exports are not
    // consistent about binop operand order).
    let Some(scale_slot) = (0..2)
        .find(|&s| prev(model, scale_mul, s).op_as::<Const>().is_some_and(|k| k.val().len() == 1))
    else {
        return Ok(None);
    };
    let scale_k = prev(model, scale_mul, scale_slot).op_as::<Const>().unwrap();
    let scale = scale_k.val().cast_to::<f32>()?.try_as_plain_ram()?.as_slice::<f32>()?[0];

    let qk_root = prev(model, scale_mul, 1 - scale_slot);
    let qk_f32 = model.outlet_fact(scale_mul.inputs[1 - scale_slot])?.datum_type == DatumType::F32;
    rule_if!(!qk_f32);
    let mut branches: Vec<(OutletId, usize, i64)> = vec![];
    if is_bin(qk_root, "Add") {
        for slot in 0..2 {
            let Some(b) = qk_branch(model, prev(model, qk_root, slot)) else {
                return Ok(None);
            };
            branches.push(b);
        }
    } else if let Some(b) = qk_branch(model, qk_root) {
        branches.push(b);
    } else {
        return Ok(None);
    }
    if !branches.iter().all(|(_, kc, _)| *kc == branches[0].1) {
        return Ok(None);
    }
    branches.sort_by_key(|(_, _, start)| *start);
    let kconcat = model.node(branches[0].1);
    if kconcat.inputs.len() != 2
        || kconcat.op_as::<TypedConcat>().is_none_or(|c| c.axis != SEQ_AXIS)
        || prev(model, kconcat, 0).op_as::<TypedSource>().is_none()
    {
        return Ok(None);
    }
    let k_cache = kconcat.inputs[0];
    let k_new = kconcat.inputs[1];
    let k_fact = model.outlet_fact(OutletId::new(kconcat.id, 0))?;
    if k_fact.rank() != 4 {
        return Ok(None);
    }
    let Ok(hkv) = k_fact.shape[1].to_usize() else { return Ok(None) };
    let Ok(d) = k_fact.shape[3].to_usize() else { return Ok(None) };

    let mut cursor = node;
    if has_sinks {
        let Some(drop_sink) = single_consumer(model, cursor) else { return Ok(None) };
        rule_if_some!(slice = drop_sink.op_as::<Slice>());
        rule_if!(slice.axis == rank - 1 && slice.start == 0.to_dim());
        rule_if!(slice.end == model.outlet_fact(mask_outlet)?.shape[rank - 1]);
        cursor = drop_sink;
    }
    let Some(cast_f16) = single_consumer(model, cursor) else { return Ok(None) };
    rule_if!(cast_f16.op_as::<Cast>().is_some_and(|cast| cast.to == DatumType::F16));
    let Some(after_cast) = single_consumer(model, cast_f16) else { return Ok(None) };
    let (group, av) = if after_cast.op_is::<AxisOp>() {
        // GQA: probs fold to [Hkv, G, S, T], the head split pins hkv/group.
        let probs_fact = &after_cast.outputs[0].fact;
        if probs_fact.rank() != 4 {
            return Ok(None);
        }
        let (Ok(p_hkv), Ok(group)) =
            (probs_fact.shape[0].to_usize(), probs_fact.shape[1].to_usize())
        else {
            return Ok(None);
        };
        if p_hkv != hkv {
            return Ok(None);
        }
        let Some(av) = single_consumer(model, after_cast) else { return Ok(None) };
        (group, av)
    } else {
        // MHA exports skip the fold: probs stay [Hq, S, T], group = 1.
        let probs_fact = &cast_f16.outputs[0].fact;
        if probs_fact.rank() != 3 || probs_fact.shape[0].to_usize().ok() != Some(hkv) {
            return Ok(None);
        }
        (1, after_cast)
    };
    let hq = hkv * group;
    rule_if!(branches.len() == 1);
    if av.op_as::<EinSum>().is_none() || av.inputs.len() != 2 {
        return Ok(None);
    }
    let Some(vconcat_id) = expand_chain_to_concat(model, av.inputs[1]) else {
        return Ok(None);
    };
    let vconcat = model.node(vconcat_id);
    if vconcat.inputs.len() != 2
        || vconcat.op_as::<TypedConcat>().is_none_or(|c| c.axis != SEQ_AXIS)
        || prev(model, vconcat, 0).op_as::<TypedSource>().is_none()
    {
        return Ok(None);
    }
    let v_cache = vconcat.inputs[0];
    let v_new = vconcat.inputs[1];

    // Cache concats must be model outputs (their slots get the op's views).
    let k_out_outlet = OutletId::new(kconcat.id, 0);
    let v_out_outlet = OutletId::new(vconcat.id, 0);
    if !model.outputs.contains(&k_out_outlet) || !model.outputs.contains(&v_out_outlet) {
        return Ok(None);
    }

    enum Layout {
        HeadMajor,
        SeqMajor,
    }
    let mut parts: Vec<(OutletId, Layout, usize)> = vec![];
    let mut covered = 0usize;
    for (q_part, _, start) in &branches {
        let f = model.outlet_fact(*q_part)?;
        if f.rank() != 4 || f.shape[0].to_usize().ok() != Some(1) {
            return Ok(None);
        }
        let Ok(d_part) = f.shape[3].to_usize() else { return Ok(None) };
        if *start != covered as i64 {
            return Ok(None);
        }
        let layout = if f.shape[1].to_usize().ok() == Some(hq) && f.shape[2].to_usize().is_err() {
            Layout::HeadMajor
        } else if f.shape[2].to_usize().ok() == Some(hq) {
            Layout::SeqMajor
        } else if f.shape[1].to_usize().ok() == Some(hq) {
            Layout::HeadMajor
        } else {
            return Ok(None);
        };
        covered += d_part;
        parts.push((*q_part, layout, d_part));
    }
    if covered != d {
        return Ok(None);
    }

    // AV output must be a known re-layout of the op's [1,Hq,S,D]: the GQA
    // foldings [Hkv,G,S,D] / [1,Hkv,G,S,D], or the MHA rank-3 forms [Hq,S,D]
    // (head-major) / [S,Hq,D] (seq-major). S is symbolic at fuse time, so
    // matching concrete head dims disambiguates the rank-3 layouts.
    #[derive(Clone, Copy, PartialEq)]
    enum AvLayout {
        Fold4,
        Fold5,
        HeadMajor3,
        SeqMajor3,
    }
    let av_fact = &av.outputs[0].fact;
    let av_layout = match av_fact.rank() {
        3 if group == 1
            && av_fact.shape[0].to_usize().ok() == Some(hq)
            && av_fact.shape[2].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::HeadMajor3)
        }
        3 if group == 1
            && av_fact.shape[1].to_usize().ok() == Some(hq)
            && av_fact.shape[2].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::SeqMajor3)
        }
        4 if av_fact.shape[0].to_usize().ok() == Some(hkv)
            && av_fact.shape[1].to_usize().ok() == Some(group)
            && av_fact.shape[3].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::Fold4)
        }
        5 if av_fact.shape[0].to_usize().ok() == Some(1)
            && av_fact.shape[1].to_usize().ok() == Some(hkv)
            && av_fact.shape[2].to_usize().ok() == Some(group)
            && av_fact.shape[4].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::Fold5)
        }
        _ => None,
    };
    let Some(av_layout) = av_layout else { return Ok(None) };

    let mut patch = TypedModelPatch::new(format!("fuse-sdpa-inplace-kv @ {node_name}"));
    let mut q_parts: Vec<OutletId> = vec![];
    for (i, (outlet, layout, _)) in parts.iter().enumerate() {
        let tapped = patch.tap_model(model, *outlet)?;
        let normalized = match layout {
            Layout::HeadMajor => tapped,
            Layout::SeqMajor => patch.wire_node(
                format!("{node_name}.q_part{i}_hsd"),
                AxisOp::Move(2, 1),
                &[tapped],
            )?[0],
        };
        q_parts.push(normalized);
    }
    let q = if q_parts.len() == 1 {
        q_parts[0]
    } else {
        patch.wire_node(format!("{node_name}.q"), TypedConcat { axis: 3 }, &q_parts)?[0]
    };

    let k_new_t = patch.tap_model(model, k_new)?;
    let v_new_t = patch.tap_model(model, v_new)?;
    let k_cache_t = patch.tap_model(model, k_cache)?;
    let v_cache_t = patch.tap_model(model, v_cache)?;
    let mask_t = patch.tap_model(model, mask_outlet)?;
    let window = 0;
    let mut op_inputs = tvec!(q, k_new_t, v_new_t, k_cache_t, v_cache_t, mask_t);
    if let Some(sinks) = sinks_tensor {
        op_inputs.push(patch.add_const(format!("{node_name}.sinks"), sinks)?);
    }

    let fused = patch.wire_node(
        format!("{node_name}.fused_sdpa"),
        FusedSdpa { scale_bits: scale.to_bits(), window, has_sinks },
        &op_inputs,
    )?;

    // Attention out [1, Hq, S, D] -> the AV einsum's output layout.
    let attn = match av_layout {
        AvLayout::Fold5 => patch.wire_node(
            format!("{node_name}.attn_reshape"),
            AxisOp::Reshape(1, tvec![hq.to_dim()], tvec![hkv.to_dim(), group.to_dim()]),
            &[fused[0]],
        )?[0],
        AvLayout::Fold4 => patch.wire_node(
            format!("{node_name}.attn_reshape"),
            AxisOp::Reshape(0, tvec![1.to_dim(), hq.to_dim()], tvec![hkv.to_dim(), group.to_dim()]),
            &[fused[0]],
        )?[0],
        AvLayout::HeadMajor3 | AvLayout::SeqMajor3 => {
            let squeezed = patch.wire_node(
                format!("{node_name}.attn_rm_batch"),
                AxisOp::Rm(0),
                &[fused[0]],
            )?[0];
            if av_layout == AvLayout::SeqMajor3 {
                patch.wire_node(
                    format!("{node_name}.attn_seq_major"),
                    AxisOp::Move(0, 1),
                    &[squeezed],
                )?[0]
            } else {
                squeezed
            }
        }
    };

    patch.shunt_outside(model, OutletId::new(av.id, 0), attn)?;
    patch.shunt_outside(model, k_out_outlet, fused[1])?;
    patch.shunt_outside(model, v_out_outlet, fused[2])?;
    Ok(Some(patch))
}
