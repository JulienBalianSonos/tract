#include <metal_stdlib>
using namespace metal;

// q8_0 block layout (must match ggml_mm_mv.metal).
typedef struct {
    half d;
    int8_t qs[32];
} moe_block_q8_0;

// Quantize rows of an f16 buffer into q8_0 blocks: KV-cache shadow
// maintenance for the fused attention (FusedSdpa). Grid (heads, rows, blocks
// from b0); one simdgroup per block; elements past `valid` (from row start)
// quantize to zero so gemvs over padded lengths read exact zeros.
[[kernel]] void kv_quantize_q8_0(
    device const half *src [[buffer(0)]],
    device char *dst [[buffer(1)]],
    constant uint &src_head_stride [[buffer(2)]],
    constant uint &src_row_stride [[buffer(3)]],
    constant uint &dst_head_stride_blocks [[buffer(4)]],
    constant uint &dst_row_stride_blocks [[buffer(5)]],
    constant uint &src_row_offset [[buffer(6)]],
    constant uint &dst_row_offset [[buffer(7)]],
    constant uint &b0 [[buffer(8)]],
    constant uint &valid [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint head = tgpig.x;
    const uint row = tgpig.y;
    const uint block = b0 + tgpig.z;

    device const half *srow =
        src + (uint64_t)head * src_head_stride
            + (uint64_t)(row + src_row_offset) * src_row_stride;
    device moe_block_q8_0 *brow = (device moe_block_q8_0 *)dst
        + (uint64_t)head * dst_head_stride_blocks
        + (uint64_t)(row + dst_row_offset) * dst_row_stride_blocks
        + block;

    const uint ix = block * 32 + lane;
    const float v = ix < valid ? (float)srow[ix] : 0.0f;
    const float amax = simd_max(fabs(v));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) {
        brow->d = (half)d;
    }
    brow->qs[lane] = (int8_t)rint(v * id);
}

[[kernel]] void sum_chunks_f16(
    device const half *partials [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant uint &heads [[buffer(2)]],
    constant uint &chunks [[buffer(3)]],
    constant uint &plane [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = heads * plane;
    if (gid >= total) {
        return;
    }
    const uint head = gid / plane;
    const uint i = gid - head * plane;
    float acc = 0.0f;
    for (uint c = 0; c < chunks; c++) {
        acc += (float)partials[(uint64_t)(head * chunks + c) * plane + i];
    }
    out[gid] = (half)acc;
}

// Fused flash-attention decode for FusedSdpa, two phases sharing K/V reads
// across the GQA group (each key is streamed once per KV head, serving all
// `group` q heads at once).
//
// Phase 1 (part): grid [Hkv, n_chunks, S]; each threadgroup runs an online
// f32 softmax over its chunk of keys for all q heads of its kv head, one
// simdgroup per key slice, and writes per-simdgroup partials (m, l, acc[D])
// to scratch. Phase 2 (merge): one threadgroup per output row combines the
// partials, folds the per-head SINK logit into the denominator, and writes
// the f16 output row. K/V are seq-major capacity buffers; q/out dense
// [Hq, S, D]. Requires D <= 64 and group <= 8.
constant constexpr uint FLASH_MAX_GROUP = 8;
constant constexpr uint FLASH_MAX_DPL = 2; // D <= 64
constant constexpr uint FLASH_SG = 8;      // simdgroups per threadgroup

// GQA group size and D-elements-per-lane specialized at PSO build time so
// every register array indexes with compile-time constants (a runtime bound
// would spill the accumulators to stack memory).
constant uint FC_GROUP [[function_constant(0)]];
constant uint FC_DPL [[function_constant(1)]];
// Single-chunk mode: merge the simdgroup partials in threadgroup memory and
// write the output row directly, skipping the merge dispatch entirely.
constant bool FC_FUSE_MERGE [[function_constant(2)]];

[[kernel]] void fused_sdpa_flash_attn_part_f16(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device const float *mask [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant uint &s_len [[buffer(5)]],
    constant uint &t_len [[buffer(6)]],
    constant uint &d [[buffer(7)]],
    constant uint &k_head_stride [[buffer(8)]],
    constant uint &v_head_stride [[buffer(9)]],
    constant uint &v_seq_stride [[buffer(10)]],
    constant uint &chunk [[buffer(11)]],
    constant uint &n_chunks [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    device const float *sinks [[buffer(14)]],
    device half *out [[buffer(15)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    // Lane-per-key: each lane owns one key of a 32-key block, so scores,
    // exps and mask adds all run 32-wide with a single simd_max/simd_sum
    // per block instead of one reduction per key. K rows are seq-major
    // (each lane streams its own row); V is transposed so the AV phase
    // reads each dim's row contiguously along the block.
    const uint kv_head = tgpig.x;
    const uint chunk_ix = tgpig.y;
    const uint qpos = tgpig.z;
    const uint simd_lane = lane % 32;
    const uint simd_ix = lane / 32;

    const uint j_lo = chunk_ix * chunk;
    const uint j_hi = min(t_len, j_lo + chunk);

    device const half *kh = k + (uint64_t)kv_head * k_head_stride;
    device const half *vh = v + (uint64_t)kv_head * v_head_stride;
    device const float *mrow = mask + (uint64_t)qpos * t_len;

    // q for the whole GQA group staged in threadgroup memory: the score
    // loop reads it dim by dim as a broadcast.
    threadgroup float q_tg[FLASH_MAX_GROUP * 64];
    for (uint i = lane; i < FC_GROUP * d; i += FLASH_SG * 32) {
        const uint g = i / d;
        const uint dim = i % d;
        q_tg[g * 64 + dim] =
            (float)q[(uint64_t)((kv_head * FC_GROUP + g) * s_len + qpos) * d + dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float p_tg[FLASH_SG * 32];

    float m[FLASH_MAX_GROUP];
    float l[FLASH_MAX_GROUP];
    float acc[FLASH_MAX_GROUP][FLASH_MAX_DPL];
    for (uint g = 0; g < FC_GROUP; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint c = 0; c < FC_DPL; c++) acc[g][c] = 0.0f;
    }

    for (uint j0 = j_lo + simd_ix * 32; j0 < j_hi; j0 += FLASH_SG * 32) {
        const uint blk = min(32u, j_hi - j0);
        const uint j = j0 + simd_lane;
        const bool live = simd_lane < blk;
        device const half4 *krow4 = (device const half4 *)(kh + (uint64_t)j * d);
        const float mj = live ? mrow[j] : -INFINITY;
        const uint d4 = d / 4;
        for (uint g = 0; g < FC_GROUP; g++) {
            // Scores: each lane dots its own K row against the shared q,
            // vectorized 4-wide to keep the load count down.
            float sc = 0.0f;
            if (live) {
                threadgroup const float4 *q4 =
                    (threadgroup const float4 *)(q_tg + g * 64);
                for (uint dim4 = 0; dim4 < d4; dim4++) {
                    sc += dot(float4(krow4[dim4]), q4[dim4]);
                }
            }
            sc = live ? sc * scale + mj : -INFINITY;
            const float m_new = max(m[g], simd_max(sc));
            const float corr = exp(m[g] - m_new);
            const float p = live ? exp(sc - m_new) : 0.0f;
            l[g] = l[g] * corr + simd_sum(p);
            m[g] = m_new;
            p_tg[simd_ix * 32 + simd_lane] = p;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // AV: lanes switch to dims; each streams its dim's transposed V
            // row contiguously across the block.
            for (uint c = 0; c < FC_DPL; c++) {
                const uint dim = simd_lane + 32 * c;
                float a = 0.0f;
                if (dim < d) {
                    device const half4 *vrow4 = (device const half4 *)(
                        vh + (uint64_t)dim * v_seq_stride + j0);
                    threadgroup const float4 *p4 =
                        (threadgroup const float4 *)(p_tg + simd_ix * 32);
                    const uint b4n = blk / 4;
                    for (uint b4 = 0; b4 < b4n; b4++) {
                        a += dot(float4(vrow4[b4]), p4[b4]);
                    }
                    for (uint b = b4n * 4; b < blk; b++) {
                        a += p_tg[simd_ix * 32 + b] * (float)(
                            (device const half *)vrow4)[b];
                    }
                }
                acc[g][c] = acc[g][c] * corr + a;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (FC_FUSE_MERGE) {
        // Merge the simdgroup partials here and write the output rows: one
        // dispatch per layer, no scratch round trip.
        threadgroup float tg_m[FLASH_SG * FLASH_MAX_GROUP];
        threadgroup float tg_l[FLASH_SG * FLASH_MAX_GROUP];
        threadgroup float tg_acc[FLASH_SG * 64];
        if (simd_lane == 0) {
            for (uint g = 0; g < FC_GROUP; g++) {
                tg_m[simd_ix * FLASH_MAX_GROUP + g] = m[g];
                tg_l[simd_ix * FLASH_MAX_GROUP + g] = l[g];
            }
        }
        for (uint g = 0; g < FC_GROUP; g++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < FC_DPL; c++) {
                const uint ix = simd_lane + 32 * c;
                if (ix < d) tg_acc[simd_ix * 64 + ix] = acc[g][c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_ix == 0) {
                float m_all = -INFINITY;
                for (uint sg = 0; sg < FLASH_SG; sg++) {
                    m_all = max(m_all, tg_m[sg * FLASH_MAX_GROUP + g]);
                }
                const float sink = sinks[kv_head * FC_GROUP + g];
                const float m_fin = max(m_all, sink);
                float l_fin = exp(sink - m_fin);
                float w[FLASH_SG];
                for (uint sg = 0; sg < FLASH_SG; sg++) {
                    const float mp = tg_m[sg * FLASH_MAX_GROUP + g];
                    w[sg] = mp == -INFINITY ? 0.0f : exp(mp - m_fin);
                    l_fin += tg_l[sg * FLASH_MAX_GROUP + g] * w[sg];
                }
                const uint row = (kv_head * FC_GROUP + g) * s_len + qpos;
                device half *orow = out + (uint64_t)row * d;
                for (uint ix = simd_lane; ix < d; ix += 32) {
                    float o = 0.0f;
                    for (uint sg = 0; sg < FLASH_SG; sg++) {
                        o += tg_acc[sg * 64 + ix] * w[sg];
                    }
                    orow[ix] = (half)(o / l_fin);
                }
            }
        }
        return;
    }

    // Per-simdgroup partial: [m, l, acc[d]] per (row, chunk, simdgroup).
    const uint stride = 2 + d;
    for (uint g = 0; g < FC_GROUP; g++) {
        const uint row = (kv_head * FC_GROUP + g) * s_len + qpos;
        device float *part = partials
            + (uint64_t)((row * n_chunks + chunk_ix) * FLASH_SG + simd_ix) * stride;
        if (simd_lane == 0) {
            part[0] = m[g];
            part[1] = l[g];
        }
        for (uint c = 0; c < FC_DPL; c++) {
            const uint ix = simd_lane + 32 * c;
            if (ix < d) part[2 + ix] = acc[g][c];
        }
    }
}

// Phase 2: one threadgroup (single simdgroup) per output row.
[[kernel]] void fused_sdpa_flash_attn_merge_f16(
    device const float *partials [[buffer(0)]],
    device const float *sinks [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &s_len [[buffer(3)]],
    constant uint &d [[buffer(4)]],
    constant uint &n_parts [[buffer(5)]],
    constant float &scale_unused [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint stride = 2 + d;
    device const float *parts = partials + (uint64_t)row * n_parts * stride;

    float m_all = -INFINITY;
    for (uint p = 0; p < n_parts; p++) {
        m_all = max(m_all, parts[p * stride]);
    }
    const float sink = sinks[row / s_len];
    const float m_fin = max(m_all, sink);
    float l_fin = exp(sink - m_fin);
    for (uint p = 0; p < n_parts; p++) {
        const float mp = parts[p * stride];
        l_fin += mp == -INFINITY ? 0.0f : parts[p * stride + 1] * exp(mp - m_fin);
    }
    device half *orow = out + (uint64_t)row * d;
    for (uint ix = lane; ix < d; ix += 32) {
        float o = 0.0f;
        for (uint p = 0; p < n_parts; p++) {
            const float mp = parts[p * stride];
            o += mp == -INFINITY ? 0.0f : parts[p * stride + 2 + ix] * exp(mp - m_fin);
        }
        orow[ix] = (half)(o / l_fin);
    }
}

// Row softmax for the fused attention: probs = softmax over T keys of
// (score*scale + mask[row % s_len]) with a per-head SINK logit participating
// in the denominator only. Rows are [num_q_heads, s_len] flattened; one
// threadgroup per row.
[[kernel]] void sinks_softmax_f16(
    device const half *scores [[buffer(0)]],
    device const float *mask [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device half *probs [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &t_len [[buffer(5)]],
    constant uint &s_len [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    constant uint &row_stride [[buffer(8)]],
    constant uint &mask_off [[buffer(9)]],
    constant uint &mask_stride [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float partials[32];
    if (row >= rows) {
        return;
    }
    const uint head = row / s_len;
    const uint mrow = row % s_len;
    device const half *srow = scores + (uint64_t)row * row_stride;
    device const float *mrow_p = mask + (uint64_t)mrow * mask_stride + mask_off;
    device half *prow = probs + (uint64_t)row * row_stride;
    const float sink = sinks[head];
    const uint simd_lane = lane % 32;
    const uint simd_ix = lane / 32;
    const uint n_simd = max(tptg / 32, 1u);

    // Pass 1: max of logits (sink included).
    float m = sink;
    for (uint j = lane; j < t_len; j += tptg) {
        m = max(m, (float)srow[j] * scale + mrow_p[j]);
    }
    m = simd_max(m);
    if (simd_lane == 0) partials[simd_ix] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partials[0];

    // Pass 2: denominator (sink seeds it).
    float den = 0.0f;
    for (uint j = lane; j < t_len; j += tptg) {
        den += exp((float)srow[j] * scale + mrow_p[j] - m);
    }
    den = simd_sum(den);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) partials[simd_ix] = den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    den = partials[0] + exp(sink - m);

    // Pass 3: write normalized probabilities (sink column dropped). The
    // padding columns (row_stride > t_len, q8 block alignment) are zeroed so
    // padded-length consumers read exact zeros.
    for (uint j = lane; j < t_len; j += tptg) {
        prow[j] = (half)(exp((float)srow[j] * scale + mrow_p[j] - m) / den);
    }
    for (uint j = t_len + lane; j < row_stride; j += tptg) {
        prow[j] = (half)0.0f;
    }
}

// ---------------------------------------------------------------------------
// Block-wise prefill attention (flash decomposed into existing gemms plus
// the three small kernels below). The prefill loop runs QK gemm per key
// block into a fixed scores buffer, then:
//   1. sdpa_prefill_block_softmax_f16 turns the block scores into block
//      probs while maintaining per-row running max `m` and denominator `l`
//      (f32), emitting the rescale factor exp(m_old - m_new) for the
//      accumulator.
//   2. (AV gemm of the block probs -> partial output)
//   3. sdpa_prefill_rescale_acc_f32 folds the partial into the f32
//      accumulator: acc = acc * rescale[row] + partial.
// After the last block sdpa_prefill_finalize_f16 folds the per-head sink
// logit into the denominator (exactly the decode merge math; -inf sinks
// reproduce the plain softmax) and writes out = acc / l.
// ---------------------------------------------------------------------------

// One threadgroup per scores row. Row layout matches the batched QK gemm
// output: rows = [hkv * group * s_len], mask row = row % s_len.
[[kernel]] void sdpa_prefill_block_softmax_f16(
    device const half *scores [[buffer(0)]],
    device const float *mask [[buffer(1)]],
    device half *probs [[buffer(2)]],
    device float *m_state [[buffer(3)]],
    device float *l_state [[buffer(4)]],
    device float *rescale [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &bt [[buffer(7)]],
    constant uint &s_len [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant uint &mask_off [[buffer(10)]],
    constant uint &mask_stride [[buffer(11)]],
    constant uint &first_block [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float partials[32];
    if (row >= rows) {
        return;
    }
    const uint mrow = row % s_len;
    device const half *srow = scores + (uint64_t)row * bt;
    device const float *mrow_p = mask + (uint64_t)mrow * mask_stride + mask_off;
    device half *prow = probs + (uint64_t)row * bt;
    const uint simd_lane = lane % 32;
    const uint simd_ix = lane / 32;
    const uint n_simd = max(tptg / 32, 1u);

    // Pass 1: block row max.
    float mb = -INFINITY;
    for (uint j = lane; j < bt; j += tptg) {
        mb = max(mb, (float)srow[j] * scale + mrow_p[j]);
    }
    mb = simd_max(mb);
    if (simd_lane == 0) partials[simd_ix] = mb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mb = partials[0];

    const float m_old = first_block ? -INFINITY : m_state[row];
    const float m_new = max(m_old, mb);

    // Fully masked so far (m_new == -inf): zero probs, keep state.
    if (m_new == -INFINITY) {
        for (uint j = lane; j < bt; j += tptg) {
            prow[j] = (half)0.0f;
        }
        if (lane == 0) {
            m_state[row] = -INFINITY;
            l_state[row] = 0.0f;
            rescale[row] = 0.0f;
        }
        return;
    }

    // Pass 2: block denominator + probs relative to m_new.
    float den = 0.0f;
    for (uint j = lane; j < bt; j += tptg) {
        const float p = exp((float)srow[j] * scale + mrow_p[j] - m_new);
        prow[j] = (half)p;
        den += p;
    }
    den = simd_sum(den);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) partials[simd_ix] = den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_ix == 0) {
        float v = simd_lane < n_simd ? partials[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partials[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    den = partials[0];

    if (lane == 0) {
        const float r = m_old == -INFINITY ? 0.0f : exp(m_old - m_new);
        l_state[row] = (first_block ? 0.0f : l_state[row]) * r + den;
        m_state[row] = m_new;
        rescale[row] = r;
    }
}

// acc[row, :] = acc[row, :] * rescale[row] + partial[row, :] (acc = partial
// on the first block so the accumulator never reads uninitialized memory).
[[kernel]] void sdpa_prefill_rescale_acc_f32(
    device const half *partial [[buffer(0)]],
    device const float *rescale [[buffer(1)]],
    device float *acc [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &d [[buffer(4)]],
    constant uint &first_block [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const uint ix = tgpig * tptg + lane;
    if (ix >= total) {
        return;
    }
    const float p = (float)partial[ix];
    acc[ix] = first_block ? p : acc[ix] * rescale[ix / d] + p;
}

// out[row, :] = acc[row, :] rescaled by the sink-folded denominator. Rows
// are [hq * s_len]; sink head = row / s_len (matches the decode merge).
[[kernel]] void sdpa_prefill_finalize_f16(
    device const float *acc [[buffer(0)]],
    device const float *m_state [[buffer(1)]],
    device const float *l_state [[buffer(2)]],
    device const float *sinks [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant uint &total [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    constant uint &s_len [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const uint ix = tgpig * tptg + lane;
    if (ix >= total) {
        return;
    }
    const uint row = ix / d;
    const float m = m_state[row];
    const float sink = sinks[row / s_len];
    const float m_fin = max(m, sink);
    if (m_fin == -INFINITY) {
        out[ix] = (half)0.0f;
        return;
    }
    const float w = m == -INFINITY ? 0.0f : exp(m - m_fin);
    const float l_fin = l_state[row] * w + exp(sink - m_fin);
    out[ix] = (half)(l_fin > 0.0f ? acc[ix] * w / l_fin : 0.0f);
}
