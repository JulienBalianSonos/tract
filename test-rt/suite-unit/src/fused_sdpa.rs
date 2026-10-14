use infra::{Test, TestResult, TestSuite};
use tract_core::internal::*;
use tract_core::ops::array::TypedConcat;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::{Add, Mul};
use tract_core::ops::nn::Softmax;
use tract_core::transform::ModelTransform;
use tract_transformers::ops::fused_sdpa::{FusedSdpa, FusedSdpaTransform};

#[derive(Clone)]
struct SinksCase;

impl SinksCase {
    fn inputs() -> TractResult<TVec<Tensor>> {
        Ok(tvec![
            Tensor::from_shape(
                &[1, 2, 1, 2],
                &[f16::from_f32(0.5), f16::from_f32(-0.25), f16::from_f32(1.0), f16::from_f32(0.5)]
            )?,
            Tensor::from_shape(&[1, 1, 1, 2], &[f16::from_f32(0.25), f16::from_f32(0.75)])?,
            Tensor::from_shape(&[1, 1, 1, 2], &[f16::from_f32(3.0), f16::from_f32(-1.0)])?,
            Tensor::from_shape(&[1, 1, 1, 2], &[f16::from_f32(1.0), f16::from_f32(-0.5)])?,
            Tensor::from_shape(&[1, 1, 1, 2], &[f16::from_f32(-2.0), f16::from_f32(4.0)])?,
            Tensor::from_shape(&[1, 1, 1, 2], &[0.0f32, -0.25])?,
            Tensor::from_shape(&[2], &[0.125f32, -0.375])?,
        ])
    }

    fn reference(inputs: &[Tensor]) -> TractResult<TVec<Tensor>> {
        let q = inputs[0].try_as_plain_ram()?.as_slice::<f16>()?;
        let k = [
            inputs[3].try_as_plain_ram()?.as_slice::<f16>()?,
            inputs[1].try_as_plain_ram()?.as_slice::<f16>()?,
        ]
        .concat();
        let v = [
            inputs[4].try_as_plain_ram()?.as_slice::<f16>()?,
            inputs[2].try_as_plain_ram()?.as_slice::<f16>()?,
        ]
        .concat();
        let mask = inputs[5].try_as_plain_ram()?.as_slice::<f32>()?;
        let sinks = inputs[6].try_as_plain_ram()?.as_slice::<f32>()?;
        let scale = f16::from_f32(0.7).to_f32();
        let mut output = Vec::with_capacity(4);
        for head in 0..2 {
            let mut logits = Vec::with_capacity(3);
            for key in 0..2 {
                let q0 = q[head * 2].to_f32();
                let q1 = q[head * 2 + 1].to_f32();
                let score = q0 * k[key * 2].to_f32() + q1 * k[key * 2 + 1].to_f32();
                let rounded_score = f16::from_f32(score).to_f32();
                let scaled = f16::from_f32(rounded_score * scale).to_f32();
                logits.push(scaled + mask[key]);
            }
            logits.push(sinks[head]);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
            let denom = exps.iter().sum::<f32>();
            for dim in 0..2 {
                let value = (0..2)
                    .map(|key| {
                        f16::from_f32(exps[key] / denom).to_f32() * v[key * 2 + dim].to_f32()
                    })
                    .sum::<f32>();
                output.push(f16::from_f32(value));
            }
        }
        let expected_output = Tensor::from_shape(&[1, 2, 1, 2], &output)?;
        let expected_keys = Tensor::from_shape(&[1, 1, 2, 2], &k)?;
        let expected_values = Tensor::from_shape(&[1, 1, 2, 2], &v)?;
        Ok(tvec![expected_output, expected_keys, expected_values])
    }

    fn model(inputs: &[Tensor]) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let sources = inputs
            .iter()
            .enumerate()
            .map(|(ix, tensor)| {
                model.add_source(format!("input_{ix}"), TypedFact::shape_and_dt_of(tensor))
            })
            .collect::<TractResult<TVec<_>>>()?;
        let outputs = model.wire_node(
            "fused_sdpa",
            FusedSdpa { scale_bits: 0.7f32.to_bits(), window: 0, has_sinks: true },
            &sources,
        )?;
        model.select_output_outlets(&outputs)?;
        Ok(model)
    }
}

impl Test for SinksCase {
    fn run_with_approx(
        &self,
        id: &'static str,
        runtime: &dyn tract_core::runtime::Runtime,
        approx: Approximation,
    ) -> TestResult {
        let inputs = Self::inputs()?;
        let expected = Self::reference(&inputs)?;
        let model = Self::model(&inputs)?;
        let values = inputs.into_iter().map(Tensor::into_tvalue).collect::<TVec<_>>();
        let runnable = runtime.prepare(model)?;
        let output = runnable.run(values)?;
        ensure!(output.len() == expected.len());
        for (actual, expected) in output.into_iter().zip(expected) {
            actual.into_tensor().close_enough(&expected, approx)?;
        }
        let _ = id;
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add("f16_sinks_and_gqa", SinksCase);
    suite.add("scale_rounding", ScaleRounding);
    for (name, grouped, wrong_qk, wrong_av, transpose_k) in [
        ("mha_match", false, false, false, false),
        ("mha_wrong_qk", false, true, false, false),
        ("mha_wrong_av", false, false, true, false),
        ("mha_transposed_keys", false, false, false, true),
        ("gqa_match", true, false, false, false),
        ("gqa_wrong_qk", true, true, false, false),
        ("gqa_wrong_av", true, false, true, false),
        ("gqa_transposed_keys", true, false, false, true),
    ] {
        suite.add(name, FusionCase { grouped, wrong_qk, wrong_av, transpose_k });
    }
    Ok(suite)
}

#[derive(Clone)]
struct FusionCase {
    grouped: bool,
    wrong_qk: bool,
    wrong_av: bool,
    transpose_k: bool,
}

impl FusionCase {
    fn model(&self) -> TractResult<(TypedModel, TVec<TValue>)> {
        let (heads, group, seq, dim) = if self.grouped { (2, 2, 2, 3) } else { (1, 1, 1, 2) };
        let mut m = TypedModel::default();
        let shapes = [
            vec![1, heads * group, seq, dim],
            vec![1, heads, 1, dim],
            vec![1, heads, seq, dim],
            vec![1, heads, 1, dim],
            vec![1, heads, seq, dim],
        ];
        let mut sources = tvec![];
        let mut inputs = tvec![];
        for (ix, shape) in shapes.iter().enumerate() {
            sources.push(m.add_source(format!("input_{ix}"), DatumType::F16.fact(shape.clone()))?);
            let data: Vec<f16> = (0..shape.iter().product())
                .map(|n| f16::from_f32(((n * 13 + ix * 7) % 23) as f32 / 7. - 1.))
                .collect();
            inputs.push(Tensor::from_shape(shape, &data)?.into_tvalue());
        }
        let [q, kc, kn, vc, vn] = *sources.as_slice() else { unreachable!() };
        let k = m.wire_node("keys", TypedConcat { axis: 2 }, &[kc, kn])?[0];
        let v = m.wire_node("values", TypedConcat { axis: 2 }, &[vc, vn])?[0];
        let qr = if self.grouped {
            m.wire_node(
                "query_groups",
                AxisOp::Reshape(
                    0,
                    tvec![1.to_dim(), (heads * group).to_dim()],
                    tvec![heads.to_dim(), group.to_dim()],
                ),
                &[q],
            )?[0]
        } else {
            m.wire_node("query_heads", AxisOp::Rm(0), &[q])?[0]
        };
        let mut kr = m.wire_node("key_heads", AxisOp::Rm(0), &[k])?[0];
        if self.transpose_k {
            kr = m.wire_node("transpose_keys", AxisOp::Move(1, 2), &[kr])?[0];
        }
        let vr = m.wire_node("value_heads", AxisOp::Rm(0), &[v])?[0];
        let qk = match (self.grouped, self.wrong_qk) {
            (true, true) => "hgsd,hdt->hgst",
            (true, false) => "hgsd,htd->hgst",
            (false, true) => "hsd,hdt->hst",
            (false, false) => "hsd,htd->hst",
        };
        let mut score =
            m.wire_node("scores", EinSum::new(qk.parse()?, DatumType::F16), &[qr, kr])?[0];
        if self.grouped {
            score = m.wire_node(
                "score_heads",
                AxisOp::Reshape(
                    0,
                    tvec![heads.to_dim(), group.to_dim()],
                    tvec![(heads * group).to_dim()],
                ),
                &[score],
            )?[0];
        }
        let scale = m.add_const("scale", tensor0(f16::from_f32(0.7)).broadcast_into_rank(3)?)?;
        let score = m.wire_node("scaled", TypedBinOp(Box::new(Mul), None), &[score, scale])?[0];
        let score = m.wire_node("float", Cast { to: DatumType::F32 }, &[score])?[0];
        let mask = m.add_const("mask", Tensor::zero_dt(DatumType::F32, &[1, seq, seq + 1])?)?;
        let score = m.wire_node("masked", TypedBinOp(Box::new(Add), None), &[score, mask])?[0];
        let p =
            m.wire_node("softmax", Softmax { axes: tvec![2], ..Default::default() }, &[score])?[0];
        let mut p = m.wire_node("half", Cast { to: DatumType::F16 }, &[p])?[0];
        if self.grouped {
            p = m.wire_node(
                "probability_groups",
                AxisOp::Reshape(
                    0,
                    tvec![(heads * group).to_dim()],
                    tvec![heads.to_dim(), group.to_dim()],
                ),
                &[p],
            )?[0];
        }
        let av = match (self.grouped, self.wrong_av) {
            (true, true) => "hgst,hdt->hgsd",
            (true, false) => "hgst,htd->hgsd",
            (false, true) => "hst,hdt->hsd",
            (false, false) => "hst,htd->hsd",
        };
        let out = m.wire_node("attention", EinSum::new(av.parse()?, DatumType::F16), &[p, vr])?[0];
        m.select_output_outlets(&[out, k, v])?;
        Ok((m, inputs))
    }
}

impl Test for FusionCase {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (mut model, inputs) = self.model()?;
        let expected = DefaultRuntime.prepare(model.clone())?.run(inputs.clone())?;
        FusedSdpaTransform.transform(&mut model)?;
        let fused = model.nodes().iter().any(|node| node.op_is::<FusedSdpa>());
        ensure!(
            fused == !(self.wrong_qk || self.wrong_av || self.transpose_k),
            "unexpected fusion decision"
        );
        let actual = runtime.prepare(model)?.run(inputs)?;
        for (actual, expected) in actual.into_iter().zip(expected) {
            actual.into_tensor().close_enough(&expected.into_tensor(), approx)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ScaleRounding;

impl Test for ScaleRounding {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let inputs = tvec![
            Tensor::from_shape(&[1, 1, 1, 2], &[3f32, 0.])?.cast_to::<f16>()?.into_owned(),
            Tensor::zero::<f16>(&[1, 1, 1, 2])?,
            Tensor::from_shape(&[1, 1, 1, 2], &[0f32, 1.])?.cast_to::<f16>()?.into_owned(),
            Tensor::from_shape(&[1, 1, 1, 2], &[1f32, 0.])?.cast_to::<f16>()?.into_owned(),
            Tensor::from_shape(&[1, 1, 1, 2], &[1f32, 0.])?.cast_to::<f16>()?.into_owned(),
            Tensor::from_shape(&[1, 1, 1, 2], &[-2.1015625f32, 0.])?,
        ];
        let mut model = TypedModel::default();
        let sources = inputs
            .iter()
            .enumerate()
            .map(|(ix, value)| {
                model.add_source(format!("input_{ix}"), TypedFact::shape_and_dt_of(value))
            })
            .collect::<TractResult<TVec<_>>>()?;
        let outputs = model.wire_node(
            "attention",
            FusedSdpa { scale_bits: 0.7f32.to_bits(), window: 0, has_sinks: false },
            &sources,
        )?;
        model.select_output_outlets(&outputs[..1])?;
        let facts = inputs.iter().map(TypedFact::shape_and_dt_of).collect::<TVec<_>>();
        let expanded = FusedSdpa { scale_bits: 0.7f32.to_bits(), window: 0, has_sinks: false }
            .expanded_model(&facts.iter().collect::<TVec<_>>())?;
        let inputs = inputs.into_iter().map(Tensor::into_tvalue).collect::<TVec<_>>();
        for candidate in [model, expanded] {
            let output = runtime.prepare(candidate)?.run(inputs.clone())?;
            let output = output[0].clone().into_tensor();
            ensure!(
                output.try_as_plain_ram()?.as_slice::<f16>()? == [f16::from_f32(0.5); 2],
                "scale must round before multiplication: {output:?}"
            );
        }
        Ok(())
    }
}
