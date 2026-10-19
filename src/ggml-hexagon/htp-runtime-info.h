#ifndef HTP_RUNTIME_INFO_H
#define HTP_RUNTIME_INFO_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

// Checked payload arithmetic, shared by runtime snapshots and host-only tests.
struct htp_memory_count {
    uint64_t bytes = 0;
    bool valid = true;

    void add(uint64_t value) {
        if (value > std::numeric_limits<uint64_t>::max() - bytes) {
            valid = false;
        } else {
            bytes += value;
        }
    }

    void array(uint64_t count, uint64_t size) {
        if (size && count > std::numeric_limits<uint64_t>::max() / size) {
            valid = false;
        } else {
            add(count * size);
        }
    }

    void pages(uint64_t value, uint64_t page) {
        if (!page || value > std::numeric_limits<uint64_t>::max() - (page - 1)) {
            valid = false;
        } else {
            add((value + page - 1) / page * page);
        }
    }
};

template <typename Node>
inline void htp_runtime_count_nodes(htp_memory_count & count, const std::vector<Node> & nodes) {
    count.array(nodes.capacity(), sizeof(Node));
    for (const auto & node : nodes) {
        count.array(node.fused.capacity(), sizeof(node.fused[0]));
        count.array(node.extra_dsts.capacity(), sizeof(node.extra_dsts[0]));
    }
}

inline uint64_t htp_runtime_max_buffer_payload(uint64_t configured, uint64_t mmap_limit, uint64_t guard) {
    return mmap_limit >= guard ? std::min(configured, mmap_limit - guard) : 0;
}

// Allocation payload allowance for one future graph and queue growth. Existing
// capacities are counted separately. libc++/libstdc++ grow these vectors by at
// most 2x; see the allocation-policy regression and accompanying documentation.
inline void htp_runtime_future_metadata(htp_memory_count & count, uint64_t nodes,
        uint64_t batch, uint64_t depth, uint64_t node_size, uint64_t pointer_size) {
    // Finalization and graph optimization: two node arrays, fusion lists,
    // reordered indices and the optimizer's used bitmap (one byte/node allowed).
    count.array(nodes, 4 * node_size + 8 * pointer_size + sizeof(int) + 1);
    // Batch-builder fusion lists and per-slot copies may retain their largest
    // batch. Compute fusion has at most two fused nodes and two extra outputs.
    const uint64_t ops = std::min(nodes, batch);
    htp_memory_count slot;
    slot.array(ops, 2 * node_size + 8 * pointer_size);
    if (!slot.valid) {
        count.valid = false;
        return;
    }
    count.array(depth, slot.bytes);
    count.array(ops, 8 * pointer_size);
}

#endif
