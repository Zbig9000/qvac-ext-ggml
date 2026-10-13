# F32 single-query attention and short softmax

The Hexagon backend selects `hvx-f32-decode` for single-query, maskless
`FLASH_ATTN_EXT` with F32 Q, K, V and output, and `GGML_PREC_F32`. Head
dimensions must be multiples of 32 and at most 256. Masks, sinks, ALiBi and
logit soft-capping retain their existing routes. Multi-query F32 attention
retains the existing `hvx-f32` kernel.

## Decode kernel

Each worker handles heads and batches. HVX vectors hold contiguous head
dimensions, avoiding the 128-query padding of the multi-query kernel.
Position-major, strided K/V caches are supported, including grouped query
heads and batch repetition. Per-worker VTCM contains a Q row, a 128-key
score/probability tile and the output row. For a 64-dimensional Q/V head,
this is 1024 bytes per worker.

Stable online softmax rescales the previous denominator and output when a
key tile raises the running maximum. QK reduction, probabilities and value
accumulation preserve F32; key prefixes can span multiple 128-key tiles.
Invalid tail probabilities are zeroed explicitly. Only valid output
head dimensions are written.

Audio8 can enable this route with `AUDIO8_LM_HEXAGON_FLASH_ATTN=1` in the
companion speech implementation, which selects a position-major V cache at
model load. Masked prefill materializes its channel-major value operand
without changing the persistent cache layout. The application option is
experimental and opt-in.

## Short-row softmax

For irregular F32 rows of at most 128 elements, the backend copies valid
input/mask bytes into aligned VTCM and vectorizes scale, mask addition,
maximum, exponential, sum and normalization. Padding uses negative infinity
and its probabilities are explicitly zero. Valid bytes alone are copied
back, including in-place operations. The existing aligned, longer-row and
F16-mask paths remain available. There is no new environment switch.

## Verification

The backend tests cover:

- Decode head dimensions 32, 64, 128 and 256; short prefixes and key-tile
  boundaries at 127, 128, 129 and 257; grouped heads, batched strided caches,
  unequal K/V head dimensions and large, closely spaced logits.
- Short softmax vector tails, masks broadcast over 14 heads, ALiBi,
  multiple batches and in-place output, with adjacent aligned lengths as
  controls.

The companion production-cache test covers masked prefill, decoding,
layer offsets, resetting a prefix and appending through the key-tile
boundary. Full Audio8 validation compares teacher-forced CPU histories
through EOS and fixed-code codec outputs. Kernel correctness does not
establish identical greedy LM trajectories: reduction order and existing
projection precision can change close top1 decisions. Keep numerical and
token disagreements in the report.

Use profiling to confirm `FLASH_ATTN_EXT` reports `hvx-f32-decode`. Keep
profiling disabled for timing comparisons and compare identical workloads.
