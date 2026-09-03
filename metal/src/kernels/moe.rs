use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

pub fn dispatch_kv_quantize_q8_0(
    stream: &MetalStream,
    src: &DeviceTensor,
    dst: &DeviceTensor,
    heads: usize,
    rows: usize,
    src_head_stride: usize,
    src_row_stride: usize,
    dst_head_stride_blocks: usize,
    dst_row_stride_blocks: usize,
    src_row_offset: usize,
    dst_row_offset: usize,
    b0: usize,
    n_blocks: usize,
    valid: usize,
) -> TractResult<()> {
    stream.retain_tensor(src);
    stream.retain_tensor(dst);
    ensure!(src.datum_type() == f16::datum_type());
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "kv_quantize_q8_0")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, src, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, dst, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, &[src_head_stride as u32]);
        encoder.set_slice(3, &[src_row_stride as u32]);
        encoder.set_slice(4, &[dst_head_stride_blocks as u32]);
        encoder.set_slice(5, &[dst_row_stride_blocks as u32]);
        encoder.set_slice(6, &[src_row_offset as u32]);
        encoder.set_slice(7, &[dst_row_offset as u32]);
        encoder.set_slice(8, &[b0 as u32]);
        encoder.set_slice(9, &[valid as u32]);
        let grid = MTLSize {
            width: heads as NSUInteger,
            height: rows as NSUInteger,
            depth: n_blocks as NSUInteger,
        };
        let group = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Sum split-k gemv partials over the chunk axis into the final output.
pub fn dispatch_sum_chunks_f16(
    stream: &MetalStream,
    partials: &DeviceTensor,
    out: &DeviceTensor,
    heads: usize,
    chunks: usize,
    plane: usize,
) -> TractResult<()> {
    stream.retain_tensor(partials);
    stream.retain_tensor(out);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "sum_chunks_f16")?;
    let total = heads * plane;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, partials, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, &[heads as u32]);
        encoder.set_slice(3, &[chunks as u32]);
        encoder.set_slice(4, &[plane as u32]);
        let group_width = (total as u64).min(256).max(1);
        let grid = MTLSize {
            width: (total as u64).div_ceil(group_width),
            height: 1,
            depth: 1,
        };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Fused flash-attention decode step: out = softmax(q.K^T*scale + mask).V
/// with the per-head sink logit in the denominator. Two dispatches: a
/// partial pass with one threadgroup per (kv head, key chunk, query pos)
/// sharing each K/V read across the whole GQA group, then a small merge.
/// `k`/`v` are seq-major capacity-buffer views (head strides passed
/// explicitly); `q`/`out` dense [1, Hq, S, D].
#[allow(clippy::too_many_arguments)]
/// Chunking rule shared with callers sizing the scratch buffer: enough
/// chunks to occupy the GPU (hkv * n_chunks * s_len threadgroups) without
/// shrinking chunks into pure per-dispatch overhead.
pub fn flash_attn_chunking(t_len: usize) -> (usize, usize) {
    let n_chunks = t_len.div_ceil(512).clamp(1, 16);
    // Chunk boundaries stay 32-aligned: the kernel's half4 loads require it,
    // and each simdgroup block covers 32 keys.
    let chunk = t_len.div_ceil(n_chunks).next_multiple_of(32);
    (t_len.div_ceil(chunk), chunk)
}

/// f32 elements of scratch the flash kernels need for a given geometry.
pub fn flash_attn_scratch_len(hq: usize, s_len: usize, t_len: usize, d: usize) -> usize {
    let (n_chunks, _) = flash_attn_chunking(t_len);
    hq * s_len * n_chunks * FLASH_SG * (2 + d)
}

const FLASH_SG: usize = 8;

pub fn dispatch_fused_sdpa_flash_attn_f16(
    stream: &MetalStream,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: &DeviceTensor,
    sinks: &DeviceTensor,
    out: &DeviceTensor,
    scratch: &DeviceTensor,
    dims: FlashAttnDims,
    scale: f32,
) -> TractResult<()> {
    let FlashAttnDims {
        hq,
        s_len,
        t_len,
        d,
        group,
        k_head_stride,
        v_head_stride,
        v_seq_stride,
    } = dims;
    let hkv = hq / group;
    for t in [q, k, v, mask, sinks] {
        stream.retain_tensor(t);
    }
    stream.retain_tensor(out);

    ensure!(q.datum_type() == f16::datum_type());
    ensure!(k.datum_type() == f16::datum_type() && v.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    ensure!(d <= 64, "flash attention kernel supports head dim <= 64, got {d}");
    ensure!(
        d % 4 == 0,
        "flash attention kernel QK dot is half4-vectorized (no scalar tail), head dim {d} not a multiple of 4"
    );
    ensure!(group <= 8, "flash attention kernel supports GQA group <= 8, got {group}");
    ensure!(mask.len() >= s_len * t_len, "mask too small");
    ensure!(sinks.len() == hq);

    let (n_chunks, chunk) = flash_attn_chunking(t_len);
    let rows = hq * s_len;
    let n_parts = n_chunks * FLASH_SG;
    ensure!(
        scratch.datum_type() == f32::datum_type() && scratch.len() >= rows * n_parts * (2 + d),
        "flash scratch too small: {} < {}",
        scratch.len(),
        rows * n_parts * (2 + d)
    );
    stream.retain_tensor(scratch);

    // group/dpl are function constants: compile-time loop bounds keep the
    // per-head register arrays out of stack memory.
    let fuse_merge = n_chunks == 1;
    let constants = crate::func_constants::ConstantValues::new(vec![
        (0, crate::func_constants::Value::USize(group)),
        (1, crate::func_constants::Value::USize(d.div_ceil(32))),
        (2, crate::func_constants::Value::Bool(fuse_merge)),
    ]);
    let part = stream.load_pipeline_with_constants(
        LibraryName::MoeOps,
        "fused_sdpa_flash_attn_part_f16",
        Some(constants),
    )?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&part);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, mask, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(4, scratch, metal::MTLResourceUsage::Write);
        encoder.set_slice(5, &[s_len as u32]);
        encoder.set_slice(6, &[t_len as u32]);
        encoder.set_slice(7, &[d as u32]);
        encoder.set_slice(8, &[k_head_stride as u32]);
        encoder.set_slice(9, &[v_head_stride as u32]);
        encoder.set_slice(10, &[v_seq_stride as u32]);
        encoder.set_slice(11, &[chunk as u32]);
        encoder.set_slice(12, &[n_chunks as u32]);
        encoder.set_slice(13, &[scale]);
        encoder.set_metal_tensor(14, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(15, out, metal::MTLResourceUsage::Write);
        let grid = MTLSize {
            width: hkv as NSUInteger,
            height: n_chunks as NSUInteger,
            depth: s_len as NSUInteger,
        };
        let group_size = MTLSize { width: (FLASH_SG * 32) as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group_size);
    });
    if fuse_merge {
        return Ok(());
    }
    let merge = stream.load_pipeline(LibraryName::MoeOps, "fused_sdpa_flash_attn_merge_f16")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&merge);
        encoder.set_metal_tensor(0, scratch, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &[s_len as u32]);
        encoder.set_slice(4, &[d as u32]);
        encoder.set_slice(5, &[n_parts as u32]);
        encoder.set_slice(6, &[scale]);
        let grid = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group_size);
    });
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct FlashAttnDims {
    pub hq: usize,
    pub s_len: usize,
    pub t_len: usize,
    pub d: usize,
    pub group: usize,
    /// K is seq-major: elements between consecutive heads.
    pub k_head_stride: usize,
    /// V is transposed ([Hkv, D, cap]): elements between consecutive heads.
    pub v_head_stride: usize,
    /// V transposed: elements between consecutive dims of one head (= cap).
    pub v_seq_stride: usize,
}

pub fn dispatch_sinks_softmax_f16(
    stream: &MetalStream,
    scores: &DeviceTensor,
    mask: &DeviceTensor,
    sinks: &DeviceTensor,
    probs: &DeviceTensor,
    s_len: usize,
    scale: f32,
    t_len: usize,
    mask_off: usize,
    mask_stride: usize,
) -> TractResult<()> {
    stream.retain_tensor(scores);
    stream.retain_tensor(mask);
    stream.retain_tensor(sinks);
    stream.retain_tensor(probs);

    ensure!(scores.datum_type() == f16::datum_type());
    ensure!(probs.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    // The physical row stride may exceed t_len (q8 block padding); the pad
    // columns of probs are zeroed by the kernel.
    let row_stride = *scores.shape().last().context("scores rank 0")?;
    ensure!(row_stride >= t_len);
    ensure!(mask.len() >= s_len * mask_stride);
    ensure!(mask_off + t_len <= mask_stride || s_len == 1 && mask_off + t_len <= mask.len());
    let rows = scores.len() / row_stride;
    ensure!(probs.len() == scores.len());
    ensure!(rows % s_len == 0, "rows {rows} not a multiple of s_len {s_len}");
    ensure!(mask.len() >= s_len * t_len, "mask too small");

    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "sinks_softmax_f16")?;
    let group_width = (pipeline.max_total_threads_per_threadgroup() as u64).min(256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, scores, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, probs, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[rows as u32]);
        encoder.set_slice(5, &[t_len as u32]);
        encoder.set_slice(6, &[s_len as u32]);
        encoder.set_slice(7, &[scale]);
        encoder.set_slice(8, &[row_stride as u32]);
        encoder.set_slice(9, &[mask_off as u32]);
        encoder.set_slice(10, &[mask_stride as u32]);
        let grid = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Block-wise prefill softmax: turn one key-block of QK scores into probs
/// while maintaining the per-row running max/denominator (m, l) f32 state,
/// emitting the accumulator rescale factor exp(m_old - m_new) per row.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sdpa_prefill_block_softmax_f16(
    stream: &MetalStream,
    scores: &DeviceTensor,
    mask: &DeviceTensor,
    probs: &DeviceTensor,
    m_state: &DeviceTensor,
    l_state: &DeviceTensor,
    rescale: &DeviceTensor,
    rows: usize,
    bt: usize,
    s_len: usize,
    scale: f32,
    mask_off: usize,
    mask_stride: usize,
    first_block: bool,
) -> TractResult<()> {
    for t in [scores, mask, probs, m_state, l_state, rescale] {
        stream.retain_tensor(t);
    }
    ensure!(scores.datum_type() == f16::datum_type());
    ensure!(probs.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(scores.len() >= rows * bt && probs.len() >= rows * bt);
    ensure!(m_state.len() >= rows && l_state.len() >= rows && rescale.len() >= rows);
    ensure!(mask.len() >= (s_len - 1) * mask_stride + mask_off + bt, "mask too small");

    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "sdpa_prefill_block_softmax_f16")?;
    let group_width = (pipeline.max_total_threads_per_threadgroup() as u64).min(256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, scores, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, probs, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, m_state, metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(4, l_state, metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(5, rescale, metal::MTLResourceUsage::Write);
        encoder.set_slice(6, &[rows as u32]);
        encoder.set_slice(7, &[bt as u32]);
        encoder.set_slice(8, &[s_len as u32]);
        encoder.set_slice(9, &[scale]);
        encoder.set_slice(10, &[mask_off as u32]);
        encoder.set_slice(11, &[mask_stride as u32]);
        encoder.set_slice(12, &[first_block as u32]);
        let grid = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// acc = acc * rescale[row] + partial (acc = partial on the first block).
pub fn dispatch_sdpa_prefill_rescale_acc_f32(
    stream: &MetalStream,
    partial: &DeviceTensor,
    rescale: &DeviceTensor,
    acc: &DeviceTensor,
    rows: usize,
    d: usize,
    first_block: bool,
) -> TractResult<()> {
    for t in [partial, rescale, acc] {
        stream.retain_tensor(t);
    }
    ensure!(partial.datum_type() == f16::datum_type());
    ensure!(acc.datum_type() == f32::datum_type());
    let total = rows * d;
    ensure!(partial.len() >= total && acc.len() >= total && rescale.len() >= rows);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "sdpa_prefill_rescale_acc_f32")?;
    let group_width = (total as u64).clamp(1, 256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, partial, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, rescale, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, acc, metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &[total as u32]);
        encoder.set_slice(4, &[d as u32]);
        encoder.set_slice(5, &[first_block as u32]);
        let grid = MTLSize { width: (total as u64).div_ceil(group_width), height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// out = acc / denominator with the per-head sink logit folded in (same
/// math as the decode merge; -inf sinks reproduce the plain softmax).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sdpa_prefill_finalize_f16(
    stream: &MetalStream,
    acc: &DeviceTensor,
    m_state: &DeviceTensor,
    l_state: &DeviceTensor,
    sinks: &DeviceTensor,
    out: &DeviceTensor,
    rows: usize,
    d: usize,
    s_len: usize,
) -> TractResult<()> {
    for t in [acc, m_state, l_state, sinks, out] {
        stream.retain_tensor(t);
    }
    ensure!(acc.datum_type() == f32::datum_type());
    ensure!(out.datum_type() == f16::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    let total = rows * d;
    ensure!(acc.len() >= total && out.len() >= total);
    ensure!(m_state.len() >= rows && l_state.len() >= rows);
    ensure!(sinks.len() >= rows / s_len);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "sdpa_prefill_finalize_f16")?;
    let group_width = (total as u64).clamp(1, 256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, acc, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, m_state, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, l_state, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(4, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(5, &[total as u32]);
        encoder.set_slice(6, &[d as u32]);
        encoder.set_slice(7, &[s_len as u32]);
        let grid = MTLSize { width: (total as u64).div_ceil(group_width), height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

#[cfg(test)]
mod sinks_softmax_tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn sinks_softmax_matches_cpu() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            let (hq, s_len, t_len) = (4usize, 3usize, 37usize);
            let rows = hq * s_len;
            let mut seed = 11u64;
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            };
            let scores: Vec<f32> = (0..rows * t_len).map(|_| next() * 3.0).collect();
            let mask: Vec<f32> =
                (0..s_len * t_len).map(|i| if i % 7 == 0 { -1e30 } else { 0.0 }).collect();
            let sinks: Vec<f32> = (0..hq).map(|_| next()).collect();
            let scale = 0.37f32;

            let scores_f16 =
                Tensor::from_shape(&[rows, t_len], &scores)?.cast_to::<f16>()?.into_owned();
            let scores_dev = scores_f16.clone().into_device()?;
            let mask_dev = Tensor::from_shape(&[s_len, t_len], &mask)?.into_device()?;
            let sinks_dev = Tensor::from_shape(&[hq], &sinks)?.into_device()?;
            let probs_dev = DeviceTensor::uninitialized_dt(f16::datum_type(), &[rows, t_len])?;

            dispatch_sinks_softmax_f16(
                stream, &scores_dev, &mask_dev, &sinks_dev, &probs_dev, s_len, scale, t_len, 0,
                t_len,
            )?;
            stream.wait_until_completed()?;
            let got = probs_dev.to_host()?.into_tensor().cast_to::<f32>()?.into_owned();
            let got = got.try_as_plain()?.as_slice::<f32>()?;

            // CPU reference from the f16-rounded scores.
            let s16 = scores_f16.cast_to::<f32>()?.into_owned();
            let s16 = s16.try_as_plain()?.as_slice::<f32>()?;
            for r in 0..rows {
                let head = r / s_len;
                let mrow = r % s_len;
                let logits: Vec<f32> = (0..t_len)
                    .map(|j| s16[r * t_len + j] * scale + mask[mrow * t_len + j])
                    .collect();
                let m = logits.iter().copied().fold(sinks[head], f32::max);
                let den: f32 =
                    logits.iter().map(|l| (l - m).exp()).sum::<f32>() + (sinks[head] - m).exp();
                for j in 0..t_len {
                    let want = (logits[j] - m).exp() / den;
                    let g = got[r * t_len + j];
                    ensure!(
                        (want - g).abs() <= 1e-3 + want.abs() * 2e-2,
                        "row {r} col {j}: want {want} got {g}"
                    );
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod flash_attn_bench {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    #[ignore]
    fn bench_flash_attn_gpt_oss_geometry() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            for t in [256usize, 1024, 2800, 8192] {
            let (hq, hkv, d) = (64usize, 8usize, 64usize);
            let group = hq / hkv;
            let cap = 8192;
            let q = Tensor::zero::<f16>(&[1, hq, 1, d])?.into_device()?;
            let k = Tensor::zero::<f16>(&[1, hkv, cap, d])?.into_device()?;
            let v = Tensor::zero::<f16>(&[1, hkv, d, cap])?.into_device()?;
            let mask = Tensor::zero::<f32>(&[1, t])?.into_device()?;
            let sinks = Tensor::zero::<f32>(&[hq])?.into_device()?;
            let out = Tensor::zero::<f16>(&[1, hq, 1, d])?.into_device()?;
            let scratch = unsafe {
                DeviceTensor::uninitialized_dt(
                    f32::datum_type(),
                    &[flash_attn_scratch_len(hq, 1, t, d)],
                )?
            };
            let dims = FlashAttnDims {
                hq,
                s_len: 1,
                t_len: t,
                d,
                group,
                k_head_stride: cap * d,
                v_head_stride: cap * d,
                v_seq_stride: cap,
            };
            // warmup
            for _ in 0..10 {
                dispatch_fused_sdpa_flash_attn_f16(
                    stream, &q, &k, &v, &mask, &sinks, &out, &scratch, dims, 0.125,
                )?;
            }
            stream.wait_until_completed()?;
            let start = std::time::Instant::now();
            const N: usize = 200;
            for _ in 0..N {
                dispatch_fused_sdpa_flash_attn_f16(
                    stream, &q, &k, &v, &mask, &sinks, &out, &scratch, dims, 0.125,
                )?;
            }
            stream.wait_until_completed()?;
            let per = start.elapsed().as_secs_f64() / N as f64;
            eprintln!("flash attn t={t}: {:.1} us/layer, {:.2} ms/token(24 layers)", per * 1e6, per * 24.0 * 1e3);
            }
            Ok(())
        })
    }
}
