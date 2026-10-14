use infra::{Test, TestResult, TestSuite};
use tract_core::internal::*;
use tract_transformers::ops::fused_sdpa::FusedSdpa;

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
    Ok(suite)
}
