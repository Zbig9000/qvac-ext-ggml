#include "ggml-impl.h"
#if !defined(__clang__) && !defined(__arm__) && !defined(__aarch64__)
#define __fp16 ggml_fp16_t
#define TEST_RUNTIME_INFO_FP16_ALIAS
#endif
#include "htp-opcache.h"
#include "htp-runtime-info.h"
#ifdef TEST_RUNTIME_INFO_FP16_ALIAS
#undef __fp16
#undef TEST_RUNTIME_INFO_FP16_ALIAS
#endif

#include <cstdio>
#include <limits>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CHECK(htp_runtime_max_buffer_payload(1ull << 30, HTP_MMAP_MAX_VMEM, 4096) == (1ull << 30));
    CHECK(htp_runtime_max_buffer_payload(3ull << 30, HTP_MMAP_MAX_VMEM, 4096) == (2ull << 30) - 4096);
    CHECK(htp_runtime_max_buffer_payload(4096, 4096, 4096) == 0);
    CHECK(htp_runtime_max_buffer_payload(4096, 1, 4096) == 0);
    htp_memory_count count;
    count.add(max);
    CHECK(count.valid && count.bytes == max);
    count.add(1);
    CHECK(!count.valid);
    count = {};
    count.array(max, 2);
    CHECK(!count.valid);
    count = {};
    count.pages(max, 4096);
    CHECK(!count.valid);
    count = {};
    count.pages(1, 0);
    CHECK(!count.valid);
    count = {};
    count.pages(4096, 4096);
    count.pages(4097, 4096);
    CHECK(count.valid && count.bytes == 12288);
    // The shared queue estimate must preserve large trace areas and all
    // sessions, rather than assume an unprofiled single-session default.
    const uint64_t batch = 1024, depth = 16;
    const uint64_t block = sizeof(htp_buf_desc) * HTP_OP_MAX_BUFS +
            sizeof(htp_tensor) * batch * (HTP_OP_MAX_OUTPUTS + HTP_OP_MAX_INPUTS) +
            sizeof(htp_op_desc) * batch + sizeof(htp_prof_desc) * batch;
    CHECK(block * depth == 13637632);
    const uint64_t traces = uint64_t(HTP_MAX_NTHREADS + 1) * batch * 256 * sizeof(htp_trace_desc);
    htp_memory_count profiled, plain, sessions;
    plain.pages(block * depth, 16384);
    profiled.pages((block + traces) * depth, 16384);
    CHECK(profiled.valid && plain.valid && profiled.bytes > plain.bytes + 350 * 1024 * 1024);
    sessions.array(3, profiled.bytes);
    CHECK(sessions.valid && sessions.bytes == 3 * profiled.bytes);
    count = {};
    htp_runtime_future_metadata(count, max, 1024, 16, sizeof(htp_opnode), sizeof(void *));
    CHECK(!count.valid);
    count = {};
    htp_runtime_future_metadata(count, 0, 1024, 16, sizeof(htp_opnode), sizeof(void *));
    CHECK(count.valid && count.bytes == 0);

    // Exercise the allocation policy used by the prospective payload estimate
    // on the build's actual standard library. These are real htp_opnode vectors
    // and the same per-slot assign path used by the live queue.
    const size_t sizes[] = { 1, 2, 3, 7, 16, 255, 1024, 4097, 32768 };
    for (size_t n : sizes) {
        std::vector<htp_opnode> nodes;
        nodes.reserve(n);
        CHECK(nodes.capacity() <= 2 * n);
        for (size_t i = 0; i < n; ++i) {
            htp_opnode node;
            for (size_t j = 0; j < i % 16; ++j) {
                node.add_fused(nullptr);
            }
            CHECK(node.fused.capacity() <= 2 * node.fused.size());
            nodes.push_back(std::move(node));
        }
        htp_memory_count actual;
        htp_runtime_count_nodes(actual, nodes);
        htp_memory_count projected;
        htp_runtime_future_metadata(projected, n, 1024, 16, sizeof(htp_opnode), sizeof(void *));
        CHECK(actual.valid && projected.valid && projected.bytes >= actual.bytes);
        const uint64_t retained_nodes = actual.bytes;
        // A smaller next graph must not replace a larger retained graph in the
        // accounting. The live implementation adds both terms in this order.
        htp_runtime_future_metadata(actual, 1, 1024, 16, sizeof(htp_opnode), sizeof(void *));
        CHECK(actual.valid && actual.bytes > retained_nodes);
        actual = {};
        htp_runtime_count_nodes(actual, nodes);

        // Direct compute fuses at most two nodes, with at most two extra
        // outputs. Capture successively larger batches, then a smaller one:
        // the retained capacity must still be counted when size shrinks.
        const size_t batch = std::min(n, size_t(1024));
        std::vector<htp_opnode> ops(batch, htp_opnode(nullptr, { nullptr, nullptr }, HTP_OP_INVALID,
                                                             { nullptr, nullptr }));
        htp_op_cache cache;
        cache.resize(16);
        for (size_t slot = 0; slot < cache.size(); ++slot) {
            cache.capture(slot, ops, batch);
        }
        uint64_t retained_capacity = cache[0].capacity();
        cache.capture(0, ops, 1);
        CHECK(cache[0].size() == 1 && cache[0].capacity() == retained_capacity);
        htp_memory_count shrunk;
        htp_runtime_count_nodes(shrunk, cache[0]);
        CHECK(shrunk.valid && shrunk.bytes >= retained_capacity * sizeof(htp_opnode));
        for (const auto & slot : cache.entries()) {
            CHECK(slot.capacity() <= 2 * batch);
            htp_runtime_count_nodes(actual, slot);
        }
        CHECK(actual.valid && projected.bytes >= actual.bytes);
    }
    std::puts("Hexagon runtime-info checked sizing and retained-capacity policy: OK");
    return 0;
}
