#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

// Optional registry procedure "ggml_backend_hexagon_conv_taps_f32". Weights
// are contiguous F16 [in, out, taps], input is padded F32 [in, time]. Positive
// tap_step advances through a causal input; -1 reads delayed transpose groups.
// Returns NULL when the shape, scratch budget or existing DDR route is unsuitable.
typedef struct ggml_tensor * (*ggml_backend_hexagon_conv_taps_f32_t)(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor *, struct ggml_tensor *, int, int);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_hexagon_conv_taps_f32(
    struct ggml_context * ctx, ggml_backend_t backend, struct ggml_tensor * weights,
    struct ggml_tensor * input, int tap_step, int output_rows);

#ifdef  __cplusplus
}
#endif
