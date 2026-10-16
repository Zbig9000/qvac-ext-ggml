#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static constexpr int N_BUFFERS = 40;
static constexpr int64_t N_ELEMENTS = 4096;

struct Input {
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static float value(int buffer, int64_t i) {
    return float(buffer + 1) * 0.25f + float(i % 7);
}

static Input make_input(ggml_backend_t backend, int index) {
    Input in;
    in.ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    in.tensor = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, N_ELEMENTS);
    in.buffer = ggml_backend_alloc_ctx_tensors(in.ctx, backend);
    GGML_ASSERT(in.buffer);
    std::vector<float> data(N_ELEMENTS);
    for (int64_t i = 0; i < N_ELEMENTS; ++i) data[i] = value(index, i);
    ggml_backend_tensor_set(in.tensor, data.data(), 0, data.size() * sizeof(float));
    return in;
}

static float expected(int64_t i) {
    float sum = 0.0f;
    for (int b = 0; b < N_BUFFERS; ++b) sum += value(b, i);
    return sum;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) {
        std::puts("SKIP: Hexagon device unavailable");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    std::vector<Input> inputs;
    for (int b = 0; b < N_BUFFERS; ++b) inputs.push_back(make_input(backend, b));

    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * (N_BUFFERS + 8) + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * sum = inputs[0].tensor;
    for (int b = 1; b < N_BUFFERS; ++b) sum = ggml_add(ctx, sum, inputs[b].tensor);
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, graph));

    bool ok = true;
    for (int pass = 0; pass < 2 && ok; ++pass) {
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<float> out(N_ELEMENTS);
        ggml_backend_tensor_get(sum, out.data(), 0, out.size() * sizeof(float));
        for (int64_t i = 0; i < N_ELEMENTS && ok; ++i) {
            if (out[i] != expected(i)) {
                std::fprintf(stderr, "pass %d element %lld: %g != %g\n", pass, (long long) i, out[i], expected(i));
                ok = false;
            }
        }
    }
    std::printf("%s: %d buffers in one graph\n", ok ? "PASS" : "FAIL", N_BUFFERS);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    for (Input & in : inputs) {
        ggml_backend_buffer_free(in.buffer);
        ggml_free(in.ctx);
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
