#ifndef HTP_OPCACHE_H
#define HTP_OPCACHE_H

#include "htp-opnode.h"

// Own the live host metadata until the corresponding queue response arrives.
// Profiling and DSP error reporting both use these snapshots; tensor storage
// itself remains owned by the graph, as it did in the batch builder.
class htp_op_cache {
    std::vector<std::vector<htp_opnode>> slots;

public:
    void resize(size_t depth) { slots.resize(depth); }
    size_t size() const { return slots.size(); }
    const std::vector<std::vector<htp_opnode>> & entries() const { return slots; }

    void capture(size_t slot, const std::vector<htp_opnode> & ops, size_t count) {
        GGML_ASSERT(slot < slots.size() && count <= ops.size());
        slots[slot].assign(ops.begin(), ops.begin() + count);
    }

    const std::vector<htp_opnode> & operator[](size_t slot) const {
        GGML_ASSERT(slot < slots.size());
        return slots[slot];
    }
};

#endif
