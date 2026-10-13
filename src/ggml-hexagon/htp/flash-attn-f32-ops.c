#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>

#include <float.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "flash-attn-ops.h"
#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hex-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-exp.h"
#include "hvx-utils.h"

// F32 flash attention for GGML_PREC_F32 requests. Logits and the softmax stay
// in F32 end to end: an F16 score tile (the HMX kernel) cannot hold the large,
// closely spaced logits of attention-sink heads. Each work item is one block
// of FA_F32_R * 32 queries of one head. The block is transposed in VTCM so a
// vector holds one dimension of 32 queries; key and value elements are read as
// scalars from cached memory and broadcast, four keys (or four value columns)
// at a time, so every broadcast feeds four vectors of queries.

#define FA_F32_R     HTP_FA_F32_QUERY_VECS
#define FA_F32_GROUP 4

struct fa_f32_context {
    struct htp_ops_context *            octx;
    struct hvx_fa_f32_vtcm_layout       L;
    const struct htp_fa_kernel_params * kparams;
    uint32_t                            DK;
    uint32_t                            DV;
    uint32_t                            n_qblocks;
    uint32_t                            n_items;
    uint32_t                            items_per_thread;
    HVX_Vector                          scale;
    HVX_Vector                          q_offsets;
    HVX_Vector                          o_offsets;
};

struct fa_f32_item {
    uint32_t q0;
    uint32_t n_q;
    uint32_t h;
    uint32_t b;
    uint32_t hk;
    uint32_t bk;
    uint32_t hv;
    uint32_t bv;
};

struct fa_f32_scratch {
    float *      qrows;
    HVX_Vector * qt;
    HVX_Vector * s;
    HVX_Vector * ot;
    float *      orow;
    HVX_Vector * m;
    HVX_Vector * l;
    HVX_Vector * alpha;
    HVX_Vector * tmp;
};

static struct fa_f32_scratch fa_f32_scratch_for(const struct fa_f32_context * c, unsigned int ith) {
    uint8_t * base = c->octx->ctx->vtcm_base + (size_t) ith * c->L.bytes_per_thread;
    HVX_Vector * ml = (HVX_Vector *) (base + c->L.off_ml);
    struct fa_f32_scratch s = {
        .qrows = (float *) (base + c->L.off_qrows),
        .qt    = (HVX_Vector *) (base + c->L.off_qt),
        .s     = (HVX_Vector *) (base + c->L.off_s),
        .ot    = (HVX_Vector *) (base + c->L.off_ot),
        .orow  = (float *) (base + c->L.off_orow),
        .m     = ml,
        .l     = ml + FA_F32_R,
        .alpha = ml + 2 * FA_F32_R,
        .tmp   = ml + 3 * FA_F32_R,
    };
    return s;
}

static struct fa_f32_item fa_f32_item_at(const struct fa_f32_context * c, uint32_t item) {
    const struct htp_tensor * q  = c->octx->src[0];
    const uint32_t            Bq = c->kparams->Br;
    const uint32_t            qb = item % c->n_qblocks;
    const uint32_t            bh = item / c->n_qblocks;
    struct fa_f32_item        it;
    it.h   = bh % q->ne[2];
    it.b   = bh / q->ne[2];
    it.q0  = qb * Bq;
    it.n_q = MIN(Bq, q->ne[1] - it.q0);
    it.hk  = fastdiv(it.h, &c->kparams->broadcast_rk2);
    it.bk  = fastdiv(it.b, &c->kparams->broadcast_rk3);
    it.hv  = fastdiv(it.h, &c->kparams->broadcast_rv2);
    it.bv  = fastdiv(it.b, &c->kparams->broadcast_rv3);
    return it;
}

static HVX_Vector fa_f32_lane_offsets(uint32_t stride_bytes) {
    int32_t lanes[VLEN_FP32] __attribute__((aligned(VLEN)));
    for (uint32_t i = 0; i < VLEN_FP32; i++) {
        lanes[i] = (int32_t) (i * stride_bytes);
    }
    return *(const HVX_Vector *) lanes;
}

static void fa_f32_zero_rows(float * rows, uint32_t from, uint32_t to, uint32_t width) {
    for (uint32_t i = from; i < to; i++) {
        hvx_splat_f32_a((uint8_t *) (rows + (size_t) i * width), 0.0f, width);
    }
}

static void fa_f32_load_q(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                          const struct fa_f32_item * it, dma_queue * dmaq) {
    const struct htp_tensor * q   = c->octx->src[0];
    const size_t              row = (size_t) c->DK * sizeof(float);
    const uint8_t *           src = (const uint8_t *) q->data + it->q0 * q->nb[1] + it->h * q->nb[2] + it->b * q->nb[3];
    fa_f32_zero_rows(sc->qrows, it->n_q, c->kparams->Br, c->DK);
    dma_queue_copy_rows(dmaq, dma_make_ptr(sc->qrows, src), row, q->nb[1], row, it->n_q);
}

// qt[d][r] holds dimension d of queries r * 32 .. r * 32 + 31, pre-scaled.
static void fa_f32_transpose_q(const struct fa_f32_context * c, const struct fa_f32_scratch * sc) {
    const uint32_t row_bytes = c->DK * sizeof(float);
    const uint32_t region    = (VLEN_FP32 - 1) * row_bytes + sizeof(float) - 1;
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        const float * block = sc->qrows + (size_t) r * VLEN_FP32 * c->DK;
        for (uint32_t d = 0; d < c->DK; d++) {
            Q6_vgather_ARMVw(sc->tmp, (size_t) (block + d), region, c->q_offsets);
            sc->qt[(size_t) d * FA_F32_R + r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*sc->tmp, c->scale));
        }
    }
}

static void fa_f32_reset_state(const struct fa_f32_context * c, const struct fa_f32_scratch * sc) {
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        sc->m[r] = hvx_vec_splat_f32(-FLT_MAX);
        sc->l[r] = Q6_V_vzero();
    }
    for (uint32_t i = 0; i < c->DV * FA_F32_R; i++) {
        sc->ot[i] = Q6_V_vzero();
    }
}

static inline HVX_Vector fa_f32_madd(HVX_Vector acc, HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(a, b));
}

static inline const float * fa_f32_row(const struct htp_tensor * t, uint32_t row, uint32_t h, uint32_t b) {
    return (const float *) ((const uint8_t *) t->data + row * t->nb[1] + h * t->nb[2] + b * t->nb[3]);
}

static void fa_f32_prefetch_rows(const struct htp_tensor * t, uint32_t row0, uint32_t n, uint32_t h, uint32_t b,
                                 uint32_t row_bytes) {
    if (n > 0) {
        hex_l2fetch(fa_f32_row(t, row0, h, b), row_bytes, t->nb[1], n);
    }
}

#define FA_F32_MADD_ROW(acc0, acc1, acc2, acc3, x, q0, q1, q2, q3) \
    do {                                                            \
        acc0 = fa_f32_madd(acc0, q0, x);                            \
        acc1 = fa_f32_madd(acc1, q1, x);                            \
        acc2 = fa_f32_madd(acc2, q2, x);                            \
        acc3 = fa_f32_madd(acc3, q3, x);                            \
    } while (0)

#define FA_F32_STORE_ROW(dst, acc0, acc1, acc2, acc3) \
    do {                                             \
        (dst)[0] = Q6_Vsf_equals_Vqf32(acc0);        \
        (dst)[1] = Q6_Vsf_equals_Vqf32(acc1);        \
        (dst)[2] = Q6_Vsf_equals_Vqf32(acc2);        \
        (dst)[3] = Q6_Vsf_equals_Vqf32(acc3);        \
    } while (0)

// Scores of keys j0 .. j0 + 3 for the whole query block, sixteen accumulators
// in registers.
static void fa_f32_scores_group(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                                const struct fa_f32_item * it, uint32_t key0, uint32_t j0) {
    const struct htp_tensor * k  = c->octx->src[1];
    const float *             k0 = fa_f32_row(k, key0 + j0 + 0, it->hk, it->bk);
    const float *             k1 = fa_f32_row(k, key0 + j0 + 1, it->hk, it->bk);
    const float *             k2 = fa_f32_row(k, key0 + j0 + 2, it->hk, it->bk);
    const float *             k3 = fa_f32_row(k, key0 + j0 + 3, it->hk, it->bk);
    HVX_Vector a00 = Q6_V_vzero(), a01 = Q6_V_vzero(), a02 = Q6_V_vzero(), a03 = Q6_V_vzero();
    HVX_Vector a10 = Q6_V_vzero(), a11 = Q6_V_vzero(), a12 = Q6_V_vzero(), a13 = Q6_V_vzero();
    HVX_Vector a20 = Q6_V_vzero(), a21 = Q6_V_vzero(), a22 = Q6_V_vzero(), a23 = Q6_V_vzero();
    HVX_Vector a30 = Q6_V_vzero(), a31 = Q6_V_vzero(), a32 = Q6_V_vzero(), a33 = Q6_V_vzero();
    const HVX_Vector * qt = sc->qt;
    for (uint32_t d = 0; d < c->DK; d++, qt += FA_F32_R) {
        const HVX_Vector q0 = qt[0], q1 = qt[1], q2 = qt[2], q3 = qt[3];
        FA_F32_MADD_ROW(a00, a01, a02, a03, hvx_vec_splat_f32(k0[d]), q0, q1, q2, q3);
        FA_F32_MADD_ROW(a10, a11, a12, a13, hvx_vec_splat_f32(k1[d]), q0, q1, q2, q3);
        FA_F32_MADD_ROW(a20, a21, a22, a23, hvx_vec_splat_f32(k2[d]), q0, q1, q2, q3);
        FA_F32_MADD_ROW(a30, a31, a32, a33, hvx_vec_splat_f32(k3[d]), q0, q1, q2, q3);
    }
    HVX_Vector * s = sc->s + (size_t) j0 * FA_F32_R;
    FA_F32_STORE_ROW(s + 0 * FA_F32_R, a00, a01, a02, a03);
    FA_F32_STORE_ROW(s + 1 * FA_F32_R, a10, a11, a12, a13);
    FA_F32_STORE_ROW(s + 2 * FA_F32_R, a20, a21, a22, a23);
    FA_F32_STORE_ROW(s + 3 * FA_F32_R, a30, a31, a32, a33);
}

static void fa_f32_scores_one(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                              const struct fa_f32_item * it, uint32_t key0, uint32_t j) {
    const float *      kr = fa_f32_row(c->octx->src[1], key0 + j, it->hk, it->bk);
    const HVX_Vector * qt = sc->qt;
    HVX_Vector         a[FA_F32_R];
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        a[r] = Q6_V_vzero();
    }
    for (uint32_t d = 0; d < c->DK; d++, qt += FA_F32_R) {
        const HVX_Vector kd = hvx_vec_splat_f32(kr[d]);
        for (uint32_t r = 0; r < FA_F32_R; r++) {
            a[r] = fa_f32_madd(a[r], qt[r], kd);
        }
    }
    HVX_Vector * s = sc->s + (size_t) j * FA_F32_R;
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        s[r] = Q6_Vsf_equals_Vqf32(a[r]);
    }
}

static void fa_f32_scores(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                          const struct fa_f32_item * it, uint32_t key0, uint32_t nk) {
    uint32_t j = 0;
    for (; j + FA_F32_GROUP <= nk; j += FA_F32_GROUP) {
        fa_f32_scores_group(c, sc, it, key0, j);
    }
    for (; j < nk; j++) {
        fa_f32_scores_one(c, sc, it, key0, j);
    }
}

static HVX_Vector fa_f32_tile_max(const struct fa_f32_scratch * sc, uint32_t nk, uint32_t r) {
    HVX_Vector mx = sc->s[r];
    for (uint32_t j = 1; j < nk; j++) {
        mx = Q6_Vsf_vmax_VsfVsf(mx, sc->s[(size_t) j * FA_F32_R + r]);
    }
    return mx;
}

static inline HVX_Vector fa_f32_exp_diff(HVX_Vector a, HVX_Vector b) {
    return hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b)));
}

// p = exp(s - m_new) in place; returns the sum of p over the tile's keys.
static HVX_Vector fa_f32_probabilities(const struct fa_f32_scratch * sc, uint32_t nk, uint32_t r, HVX_Vector m_new) {
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t j = 0; j < nk; j++) {
        HVX_Vector * s = sc->s + (size_t) j * FA_F32_R + r;
        *s             = fa_f32_exp_diff(*s, m_new);
        sum            = Q6_Vqf32_vadd_Vqf32Vsf(sum, *s);
    }
    return Q6_Vsf_equals_Vqf32(sum);
}

static void fa_f32_softmax_tile(const struct fa_f32_scratch * sc, uint32_t nk) {
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        const HVX_Vector m_new = Q6_Vsf_vmax_VsfVsf(sc->m[r], fa_f32_tile_max(sc, nk, r));
        const HVX_Vector alpha = fa_f32_exp_diff(sc->m[r], m_new);
        const HVX_Vector psum  = fa_f32_probabilities(sc, nk, r, m_new);
        sc->l[r]               = Q6_Vsf_equals_Vqf32(fa_f32_madd(Q6_Vqf32_vadd_VsfVsf(psum, Q6_V_vzero()), sc->l[r], alpha));
        sc->m[r]               = m_new;
        sc->alpha[r]           = alpha;
    }
}

#define FA_F32_RESCALE_ROW(o, acc0, acc1, acc2, acc3, alpha)                         \
    do {                                                                            \
        (o)[0] = Q6_Vsf_equals_Vqf32(fa_f32_madd(acc0, (o)[0], (alpha)[0]));         \
        (o)[1] = Q6_Vsf_equals_Vqf32(fa_f32_madd(acc1, (o)[1], (alpha)[1]));         \
        (o)[2] = Q6_Vsf_equals_Vqf32(fa_f32_madd(acc2, (o)[2], (alpha)[2]));         \
        (o)[3] = Q6_Vsf_equals_Vqf32(fa_f32_madd(acc3, (o)[3], (alpha)[3]));         \
    } while (0)

// ot[d] = ot[d] * alpha + sum_j p_j * v_j[d] for d in d0 .. d0 + 3, sixteen
// accumulators in registers.
static void fa_f32_pv_group(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                            const struct fa_f32_item * it, uint32_t key0, uint32_t nk, uint32_t d0) {
    const struct htp_tensor * v = c->octx->src[2];
    HVX_Vector a00 = Q6_V_vzero(), a01 = Q6_V_vzero(), a02 = Q6_V_vzero(), a03 = Q6_V_vzero();
    HVX_Vector a10 = Q6_V_vzero(), a11 = Q6_V_vzero(), a12 = Q6_V_vzero(), a13 = Q6_V_vzero();
    HVX_Vector a20 = Q6_V_vzero(), a21 = Q6_V_vzero(), a22 = Q6_V_vzero(), a23 = Q6_V_vzero();
    HVX_Vector a30 = Q6_V_vzero(), a31 = Q6_V_vzero(), a32 = Q6_V_vzero(), a33 = Q6_V_vzero();
    const HVX_Vector * p = sc->s;
    for (uint32_t j = 0; j < nk; j++, p += FA_F32_R) {
        const float *    vr = fa_f32_row(v, key0 + j, it->hv, it->bv) + d0;
        const HVX_Vector p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
        FA_F32_MADD_ROW(a00, a01, a02, a03, hvx_vec_splat_f32(vr[0]), p0, p1, p2, p3);
        FA_F32_MADD_ROW(a10, a11, a12, a13, hvx_vec_splat_f32(vr[1]), p0, p1, p2, p3);
        FA_F32_MADD_ROW(a20, a21, a22, a23, hvx_vec_splat_f32(vr[2]), p0, p1, p2, p3);
        FA_F32_MADD_ROW(a30, a31, a32, a33, hvx_vec_splat_f32(vr[3]), p0, p1, p2, p3);
    }
    HVX_Vector * o = sc->ot + (size_t) d0 * FA_F32_R;
    FA_F32_RESCALE_ROW(o + 0 * FA_F32_R, a00, a01, a02, a03, sc->alpha);
    FA_F32_RESCALE_ROW(o + 1 * FA_F32_R, a10, a11, a12, a13, sc->alpha);
    FA_F32_RESCALE_ROW(o + 2 * FA_F32_R, a20, a21, a22, a23, sc->alpha);
    FA_F32_RESCALE_ROW(o + 3 * FA_F32_R, a30, a31, a32, a33, sc->alpha);
}

static void fa_f32_accumulate_pv(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                                 const struct fa_f32_item * it, uint32_t key0, uint32_t nk) {
    for (uint32_t d0 = 0; d0 < c->DV; d0 += FA_F32_GROUP) {
        fa_f32_pv_group(c, sc, it, key0, nk, d0);
    }
}

static HVX_Vector fa_f32_reciprocal(HVX_Vector x) {
    float v[VLEN_FP32] __attribute__((aligned(VLEN)));
    *(HVX_Vector *) v = x;
    for (uint32_t i = 0; i < VLEN_FP32; i++) {
        v[i] = 1.0f / v[i];
    }
    return *(const HVX_Vector *) v;
}

static void fa_f32_normalize(const struct fa_f32_context * c, const struct fa_f32_scratch * sc) {
    HVX_Vector inv_l[FA_F32_R];
    for (uint32_t r = 0; r < FA_F32_R; r++) {
        inv_l[r] = fa_f32_reciprocal(sc->l[r]);
    }
    for (uint32_t d = 0; d < c->DV; d++) {
        HVX_Vector * o = sc->ot + (size_t) d * FA_F32_R;
        for (uint32_t r = 0; r < FA_F32_R; r++) {
            o[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(o[r], inv_l[r]));
        }
    }
}

// orow[i][d] = ot[d][i], gathered 32 output columns of one query at a time.
static void fa_f32_transpose_o(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                               const struct fa_f32_item * it) {
    const uint32_t Bq     = c->kparams->Br;
    const float *  ot     = (const float *) sc->ot;
    const uint32_t region = (VLEN_FP32 - 1) * Bq * sizeof(float) + sizeof(float) - 1;
    for (uint32_t i = 0; i < it->n_q; i++) {
        HVX_Vector * orow = (HVX_Vector *) (sc->orow + (size_t) i * c->DV);
        for (uint32_t d0 = 0; d0 < c->DV; d0 += VLEN_FP32) {
            Q6_vgather_ARMVw(sc->tmp, (size_t) (ot + (size_t) d0 * Bq + i), region, c->o_offsets);
            orow[d0 / VLEN_FP32] = *sc->tmp;
        }
    }
}

static void fa_f32_store(const struct fa_f32_context * c, const struct fa_f32_scratch * sc,
                         const struct fa_f32_item * it, dma_queue * dmaq) {
    const struct htp_tensor * dst = c->octx->dst;
    const size_t              row = (size_t) c->DV * sizeof(float);
    uint8_t * out = (uint8_t *) dst->data + it->h * dst->nb[1] + it->q0 * dst->nb[2] + it->b * dst->nb[3];
    fa_f32_transpose_o(c, sc, it);
    dma_queue_copy_rows(dmaq, dma_make_ptr(out, sc->orow), dst->nb[2], row, row, it->n_q);
}

static void fa_f32_item_run(const struct fa_f32_context * c, const struct fa_f32_scratch * sc, uint32_t item,
                            dma_queue * dmaq) {
    const struct fa_f32_item  it     = fa_f32_item_at(c, item);
    const struct htp_tensor * k      = c->octx->src[1];
    const struct htp_tensor * v      = c->octx->src[2];
    const uint32_t            n_keys = k->ne[1];
    const uint32_t            Bk     = c->kparams->Bc;

    fa_f32_prefetch_rows(k, 0, MIN(Bk, n_keys), it.hk, it.bk, c->DK * sizeof(float));
    fa_f32_prefetch_rows(v, 0, MIN(Bk, n_keys), it.hv, it.bv, c->DV * sizeof(float));
    fa_f32_load_q(c, sc, &it, dmaq);
    fa_f32_transpose_q(c, sc);
    fa_f32_reset_state(c, sc);

    for (uint32_t key0 = 0; key0 < n_keys; key0 += Bk) {
        const uint32_t nk   = MIN(Bk, n_keys - key0);
        const uint32_t next = key0 + nk;
        const uint32_t nn   = next < n_keys ? MIN(Bk, n_keys - next) : 0;
        fa_f32_prefetch_rows(k, next, nn, it.hk, it.bk, c->DK * sizeof(float));
        fa_f32_prefetch_rows(v, next, nn, it.hv, it.bv, c->DV * sizeof(float));
        fa_f32_scores(c, sc, &it, key0, nk);
        fa_f32_softmax_tile(sc, nk);
        fa_f32_accumulate_pv(c, sc, &it, key0, nk);
    }

    fa_f32_normalize(c, sc);
    fa_f32_store(c, sc, &it, dmaq);
}

static void fa_f32_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct fa_f32_context * c  = (const struct fa_f32_context *) data;
    const struct fa_f32_scratch   sc = fa_f32_scratch_for(c, ith);
    dma_queue *                   q  = c->octx->ctx->dma[ith];

    const uint32_t start = c->items_per_thread * ith;
    const uint32_t end   = MIN(start + c->items_per_thread, c->n_items);
    for (uint32_t item = start; item < end; item++) {
        fa_f32_item_run(c, &sc, item, q);
    }
}

// One query per head: vectorize over head dimensions rather than padding the
// query axis to 128. Scores and probabilities stay in per-worker VTCM; the
// value rows are read directly from the position-major cache. Online softmax
// retains F32 state across key tiles, so context length is not capped at 128.
struct fa_f32_decode_context {
    struct htp_ops_context * octx;
    struct hvx_fa_f32_vtcm_layout layout;
    const struct htp_fa_kernel_params * params;
    uint32_t items_per_thread;
    uint32_t n_items;
};

static inline float fa_f32_decode_lane0(HVX_Vector vector) {
    float lanes[VLEN_FP32] __attribute__((aligned(VLEN)));
    *(HVX_Vector *) lanes = vector;
    return lanes[0];
}

static void fa_f32_decode_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct fa_f32_decode_context * c = data;
    const struct htp_tensor * q = c->octx->src[0];
    const struct htp_tensor * k = c->octx->src[1];
    const struct htp_tensor * v = c->octx->src[2];
    const struct htp_tensor * dst = c->octx->dst;
    uint8_t * base = c->octx->ctx->vtcm_base + ith * c->layout.bytes_per_thread;
    float * query = (float *) (base + c->layout.off_qrows);
    float * scores = (float *) (base + c->layout.off_s);
    HVX_Vector * output = (HVX_Vector *) (base + c->layout.off_ot);
    const uint32_t begin = ith * c->items_per_thread;
    const uint32_t end = MIN(begin + c->items_per_thread, c->n_items);
    for (uint32_t item = begin; item < end; ++item) {
        const uint32_t h = item % q->ne[2];
        const uint32_t b = item / q->ne[2];
        const uint32_t hk = fastdiv(h, &c->params->broadcast_rk2);
        const uint32_t bk = fastdiv(b, &c->params->broadcast_rk3);
        const uint32_t hv = fastdiv(h, &c->params->broadcast_rv2);
        const uint32_t bv = fastdiv(b, &c->params->broadcast_rv3);
        memcpy(query, fa_f32_row(q, 0, h, b), q->ne[0] * sizeof(float));
        for (uint32_t d = 0; d < v->ne[0] / VLEN_FP32; ++d) output[d] = Q6_V_vzero();
        float maximum = -FLT_MAX;
        float denominator = 0.0f;
        for (uint32_t key0 = 0; key0 < k->ne[1]; key0 += HTP_FA_F32_KEY_TILE) {
            const uint32_t nk = MIN(HTP_FA_F32_KEY_TILE, k->ne[1] - key0);
            float next_max = maximum;
            fa_f32_prefetch_rows(k, key0, nk, hk, bk, q->ne[0] * sizeof(float));
            for (uint32_t j = 0; j < nk; ++j) {
                const float * kr = fa_f32_row(k, key0 + j, hk, bk);
                HVX_Vector sum = Q6_V_vzero();
                for (uint32_t d = 0; d < q->ne[0]; d += VLEN_FP32) {
                    sum = fa_f32_madd(sum, *(const HVX_Vector *) (query + d),
                                     *(const HVX_UVector *) (kr + d));
                }
                scores[j] = fa_f32_decode_lane0(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum))) * c->params->scale;
                next_max = fmaxf(next_max, scores[j]);
            }
            const float alpha = denominator > 0.0f ? expf(maximum - next_max) : 0.0f;
            const uint32_t padded = hex_align_up(nk, VLEN_FP32);
            for (uint32_t j = nk; j < padded; ++j) scores[j] = -INFINITY;
            HVX_Vector psum = Q6_V_vzero();
            for (uint32_t j = 0; j < padded; j += VLEN_FP32) {
                HVX_Vector s = *(HVX_Vector *) (scores + j);
                HVX_Vector p = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(
                    Q6_Vqf32_vsub_VsfVsf(s, hvx_vec_splat_f32(next_max))));
                // Invalid tail lanes and masked -infinity values must be zero.
                const HVX_VectorPred valid = Q6_Q_vcmp_gt_VsfVsf(s, hvx_vec_splat_f32(-INFINITY));
                p = Q6_V_vmux_QVV(valid, p, Q6_V_vzero());
                *(HVX_Vector *) (scores + j) = p;
                psum = Q6_Vqf32_vadd_Vqf32Vsf(psum, p);
            }
            denominator = denominator * alpha + fa_f32_decode_lane0(
                hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(psum)));
            fa_f32_prefetch_rows(v, key0, nk, hv, bv, v->ne[0] * sizeof(float));
            for (uint32_t d = 0; d < v->ne[0]; d += VLEN_FP32) {
                HVX_Vector acc = Q6_Vqf32_vmpy_VsfVsf(output[d / VLEN_FP32], hvx_vec_splat_f32(alpha));
                for (uint32_t j = 0; j < nk; ++j) {
                    acc = fa_f32_madd(acc, *(const HVX_UVector *) (fa_f32_row(v, key0 + j, hv, bv) + d),
                                      hvx_vec_splat_f32(scores[j]));
                }
                output[d / VLEN_FP32] = Q6_Vsf_equals_Vqf32(acc);
            }
            maximum = next_max;
        }
        const HVX_Vector inverse = hvx_vec_splat_f32(1.0f / denominator);
        float * out = (float *) ((uint8_t *) dst->data + h * dst->nb[1] + b * dst->nb[3]);
        for (uint32_t d = 0; d < v->ne[0]; d += VLEN_FP32) {
            *(HVX_UVector *) (out + d) = Q6_Vsf_equals_Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(output[d / VLEN_FP32], inverse));
        }
    }
}

static int fa_f32_decode(struct htp_ops_context * octx) {
    const struct htp_tensor * q = octx->src[0];
    const struct htp_tensor * k = octx->src[1];
    const struct htp_tensor * v = octx->src[2];
    struct fa_f32_decode_context c = {.octx = octx, .params = (const struct htp_fa_kernel_params *) octx->kernel_params};
    if (q->ne[1] != 1 || k->ne[1] == 0 || octx->src[3] || octx->src[4] ||
        q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        octx->dst->type != HTP_TYPE_F32 || q->ne[0] % VLEN_FP32 || v->ne[0] % VLEN_FP32 ||
        q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float)) return HTP_STATUS_NO_SUPPORT;
    hvx_fa_f32_decode_vtcm_layout_build(&c.layout, q->ne[0], v->ne[0], octx->n_threads);
    if (c.layout.total_bytes > octx->ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    c.n_items = q->ne[2] * q->ne[3];
    const uint32_t threads = MIN(octx->n_threads, c.n_items);
    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || threads == 0) return HTP_STATUS_OK;
    c.items_per_thread = (c.n_items + threads - 1) / threads;
    worker_pool_run_func(octx->ctx->worker_pool, fa_f32_decode_thread, &c, threads);
    return HTP_STATUS_OK;
}

int op_flash_attn_ext_f32(struct htp_ops_context * octx) {
    const struct htp_tensor *           q       = octx->src[0];
    const struct htp_tensor *           k       = octx->src[1];
    const struct htp_tensor *           v       = octx->src[2];
    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;

    if (kparams->kernel_type == HTP_FA_KERNEL_HVX_F32_DECODE) return fa_f32_decode(octx);

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        octx->dst->type != HTP_TYPE_F32 || kparams->Br != FA_F32_R * VLEN_FP32 || q->ne[0] % VLEN_FP32 != 0 ||
        v->ne[0] % VLEN_FP32 != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct fa_f32_context c = {
        .octx      = octx,
        .kparams   = kparams,
        .DK        = q->ne[0],
        .DV        = v->ne[0],
        .n_qblocks = (q->ne[1] + kparams->Br - 1) / kparams->Br,
        .scale     = hvx_vec_splat_f32(kparams->scale),
        .q_offsets = fa_f32_lane_offsets(q->ne[0] * sizeof(float)),
        .o_offsets = fa_f32_lane_offsets(kparams->Br * sizeof(float)),
    };
    hvx_fa_f32_vtcm_layout_build(&c.L, c.DK, c.DV, kparams->Br, kparams->Bc, octx->n_threads);
    if (c.L.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    c.n_items = c.n_qblocks * q->ne[2] * q->ne[3];
    const uint32_t threads = MIN(octx->n_threads, c.n_items);
    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || threads == 0) {
        return HTP_STATUS_OK;
    }
    c.items_per_thread = (c.n_items + threads - 1) / threads;
    worker_pool_run_func(octx->ctx->worker_pool, fa_f32_thread, &c, threads);
    return HTP_STATUS_OK;
}
