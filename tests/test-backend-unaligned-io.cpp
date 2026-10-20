#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Byte-level uploads, downloads and memsets through a view whose buffer
// offset is not a multiple of 4: backends that stage through 4-byte words
// (WebGPU) must neither reject them nor clobber the neighbouring bytes.

static const int64_t storage_bytes     = 64;
static const int64_t view_offset_bytes = 6;
static const int64_t view_bytes        = 25;
static const uint8_t memset_value      = 0xAB;

struct io_probe {
    ggml_context * ctx;
    ggml_tensor *  storage;
    ggml_tensor *  view;
};

static io_probe make_probe() {
    ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, storage_bytes);
    ggml_tensor *    view    = ggml_view_1d(ctx, storage, view_bytes, view_offset_bytes);
    return { ctx, storage, view };
}

static std::vector<uint8_t> pattern(size_t count, uint8_t seed) {
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; i++) {
        bytes[i] = (uint8_t) (seed + 7 * i);
    }
    return bytes;
}

static std::vector<uint8_t> expected_storage(const std::vector<uint8_t> & base,
                                             const std::vector<uint8_t> & patch,
                                             size_t                       patch_offset) {
    std::vector<uint8_t> expected = base;
    for (size_t i = 0; i < patch.size(); i++) {
        expected[patch_offset + i] = patch[i];
    }
    return expected;
}

static bool storage_matches(const io_probe & probe, const std::vector<uint8_t> & expected, const char * what) {
    std::vector<uint8_t> actual(storage_bytes);
    ggml_backend_tensor_get(probe.storage, actual.data(), 0, storage_bytes);
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "  %s: byte %zu is %02x, expected %02x\n", what, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool check_sync_upload(const io_probe & probe, const std::vector<uint8_t> & base) {
    const std::vector<uint8_t> patch = pattern(view_bytes, 0x40);
    ggml_backend_tensor_set(probe.view, patch.data(), 0, view_bytes);
    std::vector<uint8_t> readback(view_bytes);
    ggml_backend_tensor_get(probe.view, readback.data(), 0, view_bytes);
    const bool view_ok = readback == patch;
    if (!view_ok) {
        fprintf(stderr, "  sync upload: the view reads back differently\n");
    }
    return view_ok && storage_matches(probe, expected_storage(base, patch, view_offset_bytes), "sync upload");
}

static bool check_async_upload(ggml_backend_t backend, const io_probe & probe, const std::vector<uint8_t> & base) {
    const size_t               inner_offset = 1;
    const size_t               count        = view_bytes - 2;
    const std::vector<uint8_t> patch        = pattern(count, 0x80);
    ggml_backend_tensor_set_async(backend, probe.view, patch.data(), inner_offset, count);
    ggml_backend_synchronize(backend);
    return storage_matches(probe, expected_storage(base, patch, view_offset_bytes + inner_offset), "async upload");
}

static bool check_memset(const io_probe & probe, const std::vector<uint8_t> & base) {
    const size_t inner_offset = 3;
    const size_t count        = 10;
    ggml_backend_tensor_memset(probe.view, memset_value, inner_offset, count);
    const std::vector<uint8_t> patch(count, memset_value);
    return storage_matches(probe, expected_storage(base, patch, view_offset_bytes + inner_offset), "memset");
}

static bool check_backend(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "  %s: device failed to initialise, skipping\n", ggml_backend_dev_name(dev));
        return true;
    }
    io_probe               probe  = make_probe();
    ggml_backend_buffer_t  buffer = ggml_backend_alloc_ctx_tensors(probe.ctx, backend);
    bool                   ok     = buffer != nullptr;
    if (ok) {
        const std::vector<uint8_t> base = pattern(storage_bytes, 0x00);
        ggml_backend_tensor_set(probe.storage, base.data(), 0, storage_bytes);
        ok = check_sync_upload(probe, base);
        ggml_backend_tensor_set(probe.storage, base.data(), 0, storage_bytes);
        ok = check_async_upload(backend, probe, base) && ok;
        ggml_backend_tensor_set(probe.storage, base.data(), 0, storage_bytes);
        ok = check_memset(probe, base) && ok;
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(probe.ctx);
    ggml_backend_free(backend);
    fprintf(stderr, "  %s: %s\n", ggml_backend_dev_name(dev), ok ? "OK" : "FAIL");
    return ok;
}

static int count_failing_devices() {
    int failures = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (!check_backend(ggml_backend_dev_get(i))) {
            failures++;
        }
    }
    return failures;
}

int main() {
    ggml_backend_load_all();
    return count_failing_devices() == 0 ? 0 : 1;
}
