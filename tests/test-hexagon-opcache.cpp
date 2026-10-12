#include "ggml-impl.h"
// The included DSP layout helpers use __fp16 only for storage-size arithmetic.
// GCC on x86 has no such spelling; this test never executes DSP arithmetic.
#if !defined(__clang__) && !defined(__arm__) && !defined(__aarch64__)
#define __fp16 ggml_fp16_t
#define TEST_OPCACHE_FP16_ALIAS
#endif
#include "htp-opcache.h"
#ifdef TEST_OPCACHE_FP16_ALIAS
#undef __fp16
#undef TEST_OPCACHE_FP16_ALIAS
#endif

#include <cstdio>
#include <cstring>
#include <string>

// Exercise the same metadata snapshots read by response profiling and DSP
// error reporting, without loading a Hexagon backend or allocating device data.
static bool matches(const std::vector<htp_opnode> & actual, const std::vector<htp_opnode> & expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto & a = actual[i];
        const auto & e = expected[i];
        if (a.node != e.node || a.opcode != e.opcode || a.fused != e.fused || a.extra_dsts != e.extra_dsts ||
            std::memcmp(a.kernel_params, e.kernel_params, sizeof(a.kernel_params)) != 0 ||
            a.op_name() != e.op_name() || a.dst() != e.dst()) {
            return false;
        }
        const auto ao = a.get_outputs();
        const auto eo = e.get_outputs();
        if (ao.size() != eo.size() || !std::equal(ao.begin(), ao.end(), eo.begin())) {
            return false;
        }
        const htp_opformat af(a), ef(e);
        if (std::strcmp(af.names, ef.names) || std::strcmp(af.kparams, ef.kparams) ||
            std::strcmp(af.dims, ef.dims) || std::strcmp(af.types, ef.types)) {
            return false;
        }
    }
    return true;
}

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); ggml_free(ctx); return 1; } } while (0)

int main() {
    ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return 1;
    }
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 96, 8);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 96, 3);
    auto * mm = ggml_mul_mat(ctx, weights, input);
    auto * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    auto * add = ggml_add(ctx, mm, bias);
    auto * relu = ggml_relu(ctx, add);
    ggml_set_name(weights, "weights");
    ggml_set_name(input, "input");
    ggml_set_name(mm, "matmul");
    ggml_set_name(bias, "bias");
    ggml_set_name(add, "biased");
    ggml_set_name(relu, "activated");

    // Use real graph bookkeeping for the two graph guards, including a
    // consumer discovered after the candidate's immediately adjacent ADD.
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, relu);
    CHECK(ggml_graph_n_nodes(graph) == 3);
    CHECK(ggml_graph_node(graph, 0) == mm && ggml_graph_node(graph, 1) == add);
    CHECK(ggml_can_fuse(graph, 0, {GGML_OP_MUL_MAT, GGML_OP_ADD}));
    ggml_set_output(mm);
    CHECK(!ggml_can_fuse(graph, 0, {GGML_OP_MUL_MAT, GGML_OP_ADD}));
    mm->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    CHECK(ggml_can_fuse(graph, 0, {GGML_OP_MUL_MAT, GGML_OP_ADD}));
    auto * second_consumer = ggml_scale(ctx, mm, 2.0f);
    ggml_build_forward_expand(graph, second_consumer);
    CHECK(ggml_graph_n_nodes(graph) == 4 && ggml_graph_node(graph, 3) == second_consumer);
    CHECK(!ggml_can_fuse(graph, 0, {GGML_OP_MUL_MAT, GGML_OP_ADD}));

    htp_opnode fused(mm, {add, relu}, HTP_OP_MUL_MAT_ADD, {add, relu});
    htp_mm_kernel_params kernel = {};
    kernel.kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
    kernel.vtcm_size = 123456;
    static_assert(sizeof(kernel) <= sizeof(fused.kernel_params), "kernel metadata fits");
    std::memcpy(fused.kernel_params, &kernel, sizeof(kernel));
    htp_opnode plain(mm, {}, HTP_OP_MUL_MAT);
    std::memcpy(plain.kernel_params, &kernel, sizeof(kernel));

    // The blocked tag must remain visible to route guards in profile output.
    htp_opnode blocked = plain;
    htp_mm_kernel_params blocked_kernel = {};
    blocked_kernel.kernel_type = HTP_MM_KERNEL_HVX_F16_F32_BLOCKED;
    blocked_kernel.m_chunk = 16;
    blocked_kernel.n_chunk = 32;
    size_t worker_bytes = 0, total_bytes = 0;
    CHECK(htp_mm_f16_f32_blocked_scratch(96, 17, 16, 6, &worker_bytes, &total_bytes));
    CHECK(worker_bytes == 6144 && total_bytes == 36864 && worker_bytes % 128 == 0);
    blocked_kernel.vtcm_size = total_bytes;
    std::memcpy(blocked.kernel_params, &blocked_kernel, sizeof(blocked_kernel));
    CHECK(std::string(htp_opformat(blocked).kparams) == "hvx-blocked-f32 channels 16 rows 32 traversal channels vtcm 36864");
    CHECK(htp_mm_f16_f32_blocked_scratch(96, 7, 16, 6, &worker_bytes, &total_bytes));
    CHECK(worker_bytes == 2688 && total_bytes == 16128);
    CHECK(!htp_mm_f16_f32_blocked_scratch(97, 17, 16, 6, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch(96, 17, 12, 6, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch(96, 17, 16, 0, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch(SIZE_MAX - 31, 17, 16, 6, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch(1u << 25, 17, 16, 6, &worker_bytes, &total_bytes));

    blocked_kernel.pipeline = HTP_MM_F16_F32_BLOCKED_ROWS;
    std::memcpy(blocked.kernel_params, &blocked_kernel, sizeof(blocked_kernel));
    CHECK(std::string(htp_opformat(blocked).kparams) == "hvx-blocked-f32 channels 16 rows 32 traversal rows vtcm 36864");
    CHECK(htp_mm_f16_f32_blocked_scratch_for_traversal(384, 384, 16, 6,
        HTP_MM_F16_F32_BLOCKED_MATRIX, &worker_bytes, &total_bytes));
    CHECK(worker_bytes == 589824 && total_bytes == 3538944 && worker_bytes % 128 == 0);
    blocked_kernel.pipeline = HTP_MM_F16_F32_BLOCKED_MATRIX;
    blocked_kernel.vtcm_size = total_bytes;
    std::memcpy(blocked.kernel_params, &blocked_kernel, sizeof(blocked_kernel));
    CHECK(std::string(htp_opformat(blocked).kparams) == "hvx-blocked-f32 channels 16 rows 32 traversal matrix vtcm 3538944");
    blocked_kernel.kernel_type = HTP_MM_KERNEL_HVX_F16_F32_BLOCKED_ADD;
    std::memcpy(blocked.kernel_params, &blocked_kernel, sizeof(blocked_kernel));
    CHECK(std::string(htp_opformat(blocked).kparams) == "hvx-blocked-f32-add channels 16 rows 32 traversal matrix vtcm 3538944");
    htp_mm_kernel_params residual_kernel = {};
    residual_kernel.kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR_ADD;
    std::memcpy(blocked.kernel_params, &residual_kernel, sizeof(residual_kernel));
    CHECK(std::string(htp_opformat(blocked).kparams) == "hvx-panel-4x2-add vtcm 0");
    CHECK(htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 36, 1188));
    CHECK(htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 44, 1444));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 44, 1443));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 8, 36, 1188));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 35, 1188));
    CHECK(!htp_mm_f16_f32_rows_fit(SIZE_MAX, 33, 4, 4, 36, SIZE_MAX));
    CHECK(!htp_mm_f16_f32_rows_fit(9, SIZE_MAX, 4, 4, 36, SIZE_MAX));
    CHECK(htp_mm_f16_f32_rows_fit(9, 1, 4, 4, 44, 36));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 1, 4, 4, 44, 35));
    CHECK(!htp_mm_f16_f32_rows_fit(0, 33, 4, 4, 36, 1188));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 0, 4, 4, 36, 1188));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 0, 0, 36, 1188));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 37, 1221));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 4, size_t(UINT32_MAX) + 1, SIZE_MAX));
    CHECK(!htp_mm_f16_f32_rows_fit(9, 33, 4, 4, 36, size_t(UINT32_MAX) + 1));
    CHECK(!htp_mm_f16_f32_rows_fit(UINT32_MAX, UINT32_MAX, 4, 4, UINT32_MAX - 3, UINT32_MAX));
    CHECK(htp_mm_f16_f32_disjoint(128, 128, 256, 128));
    CHECK(htp_mm_f16_f32_disjoint(256, 128, 128, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 128, 252, 128));
    CHECK(!htp_mm_f16_f32_disjoint(252, 128, 128, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 256, 192, 32));
    CHECK(!htp_mm_f16_f32_disjoint(192, 32, 128, 256));
    CHECK(!htp_mm_f16_f32_disjoint(128, 128, 128, 128));
    CHECK(!htp_mm_f16_f32_disjoint(0, 128, 256, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 0, 256, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 128, 0, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 128, 256, 0));
    CHECK(!htp_mm_f16_f32_disjoint(UINTPTR_MAX - 16, 32, 128, 128));
    CHECK(!htp_mm_f16_f32_disjoint(128, 128, UINTPTR_MAX - 16, 32));
    // These are actual shared host/DSP eligibility predicates. Shifted ADD
    // aliases are undefined ggml graphs, so validate their rejected ranges
    // here; the device suite covers the valid exact-residual alias graph.
    CHECK(htp_mm_f16_f32_residual_alias_ok(128, 128, 128, 128));
    CHECK(htp_mm_f16_f32_residual_alias_ok(128, 128, 256, 128));
    CHECK(htp_mm_f16_f32_residual_alias_ok(256, 128, 128, 128));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(132, 128, 128, 128));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 128, 132, 128));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 128, 128, 124));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 124, 128, 128));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 256, 192, 32));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(0, 128, 0, 128));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 0, 128, 0));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(UINTPTR_MAX - 16, 32, UINTPTR_MAX - 16, 32));
    CHECK(!htp_mm_f16_f32_residual_alias_ok(128, 128, UINTPTR_MAX - 16, 32));
    CHECK(htp_mm_f16_f32_blocked_scratch_for_traversal(384, 384, 16, 6,
        HTP_MM_F16_F32_BLOCKED_ROWS, &worker_bytes, &total_bytes));
    CHECK(worker_bytes == 24576 && total_bytes == 147456);
    CHECK(!htp_mm_f16_f32_blocked_scratch_for_traversal(96, 17, 16, 6, 3, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch_for_traversal(96, 17, 16, 6, -1, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch_for_traversal(384, SIZE_MAX, 16, 6,
        HTP_MM_F16_F32_BLOCKED_MATRIX, &worker_bytes, &total_bytes));
    CHECK(!htp_mm_f16_f32_blocked_scratch_for_traversal(384, 1u << 20, 16, 6,
        HTP_MM_F16_F32_BLOCKED_MATRIX, &worker_bytes, &total_bytes));

    htp_op_cache cache;
    CHECK(cache.size() == 0);
    cache.resize(3);
    CHECK(cache.size() == 3);
    CHECK(cache[0].empty() && cache[1].empty() && cache[2].empty());
    std::vector<htp_opnode> source = {fused, plain, fused, plain};
    const std::vector<htp_opnode> first(source.begin(), source.begin() + 2);
    cache.capture(0, source, 2);
    CHECK(matches(cache[0], first));
    CHECK(cache[0][0].fused.data() != source[0].fused.data());
    CHECK(cache[0][0].extra_dsts.data() != source[0].extra_dsts.data());
    CHECK(std::string(htp_opformat(cache[0][0]).names).find("activated") != std::string::npos);
    CHECK(std::string(htp_opformat(cache[0][0]).kparams) == "hvx-flat vtcm 123456");

    // Mutate and then destroy the submission buffer while slot 0 is pending.
    source[0].fused.clear();
    source[0].extra_dsts.clear();
    source[0].node = relu;
    source[0].opcode = HTP_OP_INVALID;
    source[0].kernel_params[0] = -123;
    source.clear();
    source.shrink_to_fit();
    CHECK(matches(cache[0], first));

    source = {plain, fused, plain, fused};
    const auto second = source;
    cache.capture(1, source, source.size());
    cache.capture(2, source, 1);
    CHECK(matches(cache[1], second));
    CHECK(matches(cache[2], {plain}));
    CHECK(matches(cache[0], first));

    // Queue slot reuse: shrink, empty, then grow beyond its previous length.
    // Inactive source entries must never leak into profiler/error iteration.
    cache.capture(1, source, 1);
    CHECK(matches(cache[1], {plain}));
    cache.capture(1, source, 0);
    CHECK(cache[1].empty());
    source.insert(source.end(), {fused, plain, fused});
    cache.capture(1, source, source.size());
    CHECK(matches(cache[1], source));
    CHECK(matches(cache[0], first));
    CHECK(matches(cache[2], {plain}));
    source.clear();
    cache.capture(1, source, 0);
    CHECK(cache[1].empty());
    ggml_free(ctx);
    std::puts("PASS: live-prefix cache, nested ownership, slot reuse, profile/error metadata, graph guards, checked residual ranges");
    return 0;
}
