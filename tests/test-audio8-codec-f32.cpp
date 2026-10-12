#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// CPU F16 MUL_MAT converts F32 activations to F16 even with GGML_PREC_F32.
// Instead, check explicitly dequantized F32 weights against a double-precision
// host dot product, then use that independent result for the Hexagon candidate.
// The cancellation inputs below lose their entire signal if narrowed to F16.
struct shape {
    const char * name;
    int k, channels, rows, batches;
    bool views, bias, panel;
    bool varied = false;
    int weight_padding = 13;
};

static const shape cases[] = {
    // Small inputs fit VTCM and must retain the existing activation narrowing.
    {"vtcm-k64-odd",       64,  7,  5, 1, false, false, false},
    {"vtcm-k96-odd",       96,  7,  5, 1, false, false, false},
    {"vtcm-k96-views",     96,  7,  5, 1, true,  false, false},
    {"vtcm-k192-odd",     192, 11,  9, 1, false, false, false},
    {"vtcm-k384-views",   384,  9,  7, 1, true,  false, false},
    // M>1 explicit-F32 MUL_MAT cannot fuse with ADD on the current backend;
    // the standalone VTCM product must still feed the normal bias operation.
    {"vtcm-separate-bias", 96,  7,  5, 1, false, true,  false},
    {"vtcm-channels96",   96, 96, 33, 1, false, false, false},
    {"vtcm-channels192", 192,192, 33, 1, false, false, false},
    {"vtcm-channels384", 384,384, 33, 1, false, false, false},
    // Independent values along every axis expose lane/row permutations that
    // the deliberately repeated cancellation pattern cannot distinguish.
    {"vtcm-varied-k96",   96,  9,  7, 1, true,  false, false, true},
    {"vtcm-varied-k192", 192,  7, 11, 1, false, false, false, true},
    {"vtcm-varied-k384", 384, 11,  9, 1, true,  false, false, true},
    // The target Snapdragon 8 Elite has 8 MiB VTCM. These odd row counts
    // exceed that budget in padded F16 activation staging alone (256, 384,
    // and 768 bytes/row); no test-only routing override is needed. Few output
    // channels bound CPU oracle work while retaining odd panel remainders.
    {"ddr-k96",          96,  7, 32769, 1, true,  false, true},
    {"ddr-k192",        192,  5, 21847, 1, false, false, true},
    {"ddr-k384",        384,  3, 10923, 1, true,  false, true},
    {"ddr-varied-k96",   96,  9, 32769, 1, true,  false, true, true},
    {"ddr-varied-k192", 192,  7, 21847, 1, false, false, true, true},
    {"ddr-varied-k384", 384,  5, 10923, 1, true,  false, true, true},
    // Exercise full four-channel panels and their two-channel remainder,
    // including K64, unaligned row views, and both even/odd activation rows.
    // These also remain valid oracle cases for the explicit 2x2 override.
    {"ddr-four-k96",      96,  4, 32770, 1, false, false, true},
    {"ddr-six-k64",       64,  6, 65537, 1, true,  false, true, true},
    {"ddr-eight-k192",   192,  8, 21849, 1, true,  false, true, true},
    {"ddr-ten-k384",     384, 10, 10924, 1, false, false, true, true},
    // Most codec-shaped cases split activation rows among workers. Here
    // padded weight-prefetch staging alone exceeds 8 MiB even with one
    // worker (16 * 524416 bytes), forcing natural DDR dispatch while
    // channels outnumber rows. The logical K96 dot product stays small;
    // NaN-filled padding checks bounded loads as well as channel ownership,
    // idle workers, half-vector tails and panel remainders.
    {"ddr-split-channels", 96, 22, 3, 1, true, false, true, false, 262112},
    {"ddr-split-varied",   96, 29, 3, 1, true, false, true, true,  262112},
    // Ineligible shapes retain the existing F16-activation arithmetic.
    {"guard-k32",     32,  7,  5, 1, false, false, false},
    {"guard-batched", 96,  7,  5, 2, false, false, false},
    {"guard-k97",     97,  7,  5, 1, true,  false, false},
    {"guard-one-col", 96,  1,  5, 1, false, false, false},
    {"guard-one-row", 96,  7,  1, 1, false, false, false},
};

struct inputs {
    std::vector<float> weights, activations;
    std::vector<double> expected, tolerance;
};

static float varied_value(uint32_t index, uint32_t seed) {
    uint32_t bits = index ^ seed;
    bits ^= bits >> 16;
    bits *= 0x7feb352du;
    bits ^= bits >> 15;
    bits *= 0x846ca68bu;
    bits ^= bits >> 16;
    return float(int32_t(bits & 0x00ffffffu) - 0x00800000) / 8388608.0f;
}

static inputs make_inputs(const shape & s, bool narrow) {
    inputs in;
    in.weights.resize(size_t(s.k) * s.channels * s.batches);
    in.activations.resize(size_t(s.k) * s.rows * s.batches);
    in.expected.resize(size_t(s.channels) * s.rows * s.batches);
    in.tolerance.resize(in.expected.size(), 2e-6);
    for (int batch = 0; batch < s.batches; ++batch) {
        for (int c = 0; c < s.channels; ++c) {
            for (int k = 0; k < s.k; ++k) {
                const float weight = s.varied
                    ? varied_value(uint32_t((batch * s.channels + c) * s.k + k), 0x735a2d91u)
                    : (k == s.k - 1 && s.k % 2 ? 0.0f
                       : float(1 + c % 4) * (k % 2 ? -1.0f : 1.0f));
                in.weights[(size_t(batch) * s.channels + c) * s.k + k] =
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(weight));
            }
        }
        for (int r = 0; r < s.rows; ++r) {
            for (int k = 0; k < s.k; ++k) {
                const float base = 1.0f + float((k / 2 + r) % 8) / 16.0f;
                const float delta = float(1 + (r + batch) % 3) / 8192.0f;
                float x = base + (k % 2 ? 0.0f : delta);
                if (s.varied) x = varied_value(uint32_t((batch * s.rows + r) * s.k + k), 0xd62f179bu);
                if (narrow) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
                in.activations[(size_t(batch) * s.rows + r) * s.k + k] = x;
            }
            for (int c = 0; c < s.channels; ++c) {
                double sum = 0, sum_abs = 0;
                for (int k = 0; k < s.k; ++k) {
                    const double product = double(in.weights[(size_t(batch) * s.channels + c) * s.k + k]) *
                                           in.activations[(size_t(batch) * s.rows + r) * s.k + k];
                    sum += product;
                    sum_abs += std::abs(product);
                }
                if (s.bias) sum += float(c + 1) / 16.0f;
                const size_t index = (size_t(batch) * s.rows + r) * s.channels + c;
                in.expected[index] = sum;
                if (s.varied) {
                    // Conservative forward-error bound for separate F32
                    // products and additions: gamma_(2K+1) * sum(abs(w*x)).
                    // This permits different reduction orders without hiding
                    // permutations; cancellation fixtures keep the strict bound.
                    const double roundoff = (2 * s.k + 1) * 0.5 * std::numeric_limits<float>::epsilon();
                    in.tolerance[index] = std::max(2e-6, roundoff / (1.0 - roundoff) * sum_abs);
                }
            }
        }
    }
    return in;
}

struct graph_storage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~graph_storage() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
};

// Fill row padding with NaNs: a partial-K vector must mask both operands.
static void upload(ggml_tensor * storage, const std::vector<float> & logical,
                   int k, int rows, int batches, int offset) {
    std::vector<float> data(ggml_nelements(storage), std::numeric_limits<float>::quiet_NaN());
    for (int b = 0; b < batches; ++b) {
        for (int r = 0; r < rows; ++r) {
            std::copy_n(logical.data() + (size_t(b) * rows + r) * k, k,
                        data.data() + (size_t(b) * rows + r) * storage->ne[0] + offset);
        }
    }
    if (storage->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(data.size());
        for (size_t i = 0; i < data.size(); ++i) half[i] = ggml_fp32_to_fp16(data[i]);
        ggml_backend_tensor_set(storage, half.data(), 0, half.size() * sizeof(half[0]));
    } else {
        ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(data[0]));
    }
}

static bool compute(ggml_backend_t backend, const shape & s, const inputs & in,
                    ggml_type weight_type, std::vector<float> & result) {
    graph_storage storage;
    storage.ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    if (!storage.ctx) return false;
    auto * ctx = storage.ctx;
    auto * graph = ggml_new_graph(ctx);
    auto * a = ggml_new_tensor_3d(ctx, weight_type, s.k + (s.views ? s.weight_padding : 0), s.channels, s.batches);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k + (s.views ? 11 : 0), s.rows, s.batches);
    auto * av = s.views ? ggml_view_3d(ctx, a, s.k, s.channels, s.batches, a->nb[1], a->nb[2], a->nb[0]) : a;
    auto * bv = s.views ? ggml_view_3d(ctx, b, s.k, s.rows, s.batches, b->nb[1], b->nb[2], b->nb[0]) : b;
    auto * out = ggml_mul_mat(ctx, av, bv);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_tensor * bias = nullptr;
    if (s.bias) {
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.channels);
        out = ggml_add(ctx, out, bias);
    }
    ggml_build_forward_expand(graph, out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            std::printf("%s: unsupported node on %s\n", s.name, ggml_backend_name(backend));
            return false;
        }
    }
    storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!storage.buffer) return false;
    upload(a, in.weights, s.k, s.channels, s.batches, s.views ? 1 : 0);
    upload(b, in.activations, s.k, s.rows, s.batches, s.views ? 1 : 0);
    if (bias) {
        std::vector<float> data(s.channels);
        for (int c = 0; c < s.channels; ++c) data[c] = float(c + 1) / 16.0f;
        ggml_backend_tensor_set(bias, data.data(), 0, data.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    result.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return true;
}

static bool compare(const char * backend, const shape & s, const std::vector<float> & actual,
                    const inputs & reference) {
    bool ok = actual.size() == reference.expected.size();
    double max_abs = 0, max_bound = 0;
    for (size_t i = 0; i < actual.size() && i < reference.expected.size(); ++i) {
        const double error = std::abs(double(actual[i]) - reference.expected[i]);
        ok = std::isfinite(actual[i]) && error <= reference.tolerance[i] && ok;
        max_abs = std::max(max_abs, error);
        max_bound = std::max(max_bound, reference.tolerance[i]);
    }
    std::printf("%s %s k=%d channels=%d rows=%d batches=%d max_abs=%.9g max_bound=%.9g %s\n",
                backend, s.name, s.k, s.channels, s.rows, s.batches, max_abs, max_bound, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    bool reference_only = false;
    std::string dump_prefix;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reference-only") == 0) {
            reference_only = true;
        } else if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
            if (dump_prefix.empty()) {
                std::fprintf(stderr, "--dump-prefix requires a nonempty path\n");
                return 1;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--reference-only] [--dump-prefix PATH]\n", argv[0]);
            return 1;
        }
    }
    if (reference_only && !dump_prefix.empty()) {
        std::fprintf(stderr, "--dump-prefix writes Hexagon results; incompatible with --reference-only\n");
        return 1;
    }
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!cpu) return 1;
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    using set_ref_fn = void (*)(ggml_backend_t, bool);
    const auto set_ref = reinterpret_cast<set_ref_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref"));
    if (!set_ref) { ggml_backend_free(cpu); return 1; }
    set_ref(cpu, true);
    auto * dev = reference_only ? nullptr : ggml_backend_dev_by_name("HTP0");
    auto hexagon = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    bool ok = !dev || hexagon;
    for (const auto & s : cases) {
        const auto full = make_inputs(s, false);
        const auto narrowed = make_inputs(s, true);
        const auto & expected = s.panel ? full : narrowed;
        std::vector<float> result;
        const bool ref_ok = compute(cpu, s, expected, GGML_TYPE_F32, result);
        ok = ref_ok && ok;
        if (ref_ok) ok = compare("CPU-dequantized-F32", s, result, expected) && ok;
        // Confirm this fixture would catch activation narrowing on every
        // eligible shape, rather than accidentally using F16-exact inputs.
        if (s.panel && full.expected == narrowed.expected) ok = false;
        if (hexagon) {
            const bool ran = compute(hexagon, s, full, GGML_TYPE_F16, result);
            ok = ran && ok;
            if (ran) ok = compare("Hexagon", s, result, expected) && ok;
            if (ran && !dump_prefix.empty()) {
                // Canonical binary32 little-endian output permits exact
                // cross-panel comparisons independent of printed tolerances.
                static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                              "result dumps require IEEE754 binary32");
                std::ofstream dump(dump_prefix + "." + s.name + ".f32", std::ios::binary);
                for (float value : result) {
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    for (int byte = 0; byte < 4; ++byte) dump.put(char((bits >> (byte * 8)) & 255));
                }
                dump.close();
                if (!dump) {
                    std::fprintf(stderr, "%s: cannot write result dump\n", s.name);
                    ok = false;
                }
            }
        }
    }
    ggml_backend_free(hexagon);
    ggml_backend_free(cpu);
    if (!ok) return 1;
    if (!reference_only && !dev) { std::puts("SKIP: Hexagon unavailable; independent F32 reference checks passed"); return 77; }
    return 0;
}
