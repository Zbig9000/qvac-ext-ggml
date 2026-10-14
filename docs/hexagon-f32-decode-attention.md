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

Audio8 uses this route by default on supported Hexagon backends in the
companion speech implementation, which selects a position-major V cache at
model load. `AUDIO8_LM_HEXAGON_FLASH_ATTN=0` selects separate attention;
leaving it unset or setting it to `1` uses the default flash path. Masked
prefill materializes its channel-major value operand without changing the
persistent cache layout. Unsupported operations retain separate attention.

## Short-row softmax

For irregular F32 rows of at most 128 elements, the backend fuses scale,
mask addition and maximum while reading full input vectors. Partial vectors
alone use bounded copies into aligned VTCM. Exponential, sum, reciprocal and
normalization remain vectorized. Padding uses negative infinity and its
probabilities are explicitly zero. Predicated stores write only valid output
bytes, including in-place operations. The existing aligned, longer-row and
F16-mask paths remain available. The vectorized short-row route is selected
automatically for applicable shapes, with no new environment switch.

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

### Short-softmax measurements

On the OnePlus CPH2723 (Hexagon v79, six HVX workers), three alternating
baseline/candidate captures gave these median per-operation times. Each
backend performance probe warms up and measures repeated operations for at
least one second. Every capture started and ended at thermal status 0;
starting skin temperature was at most 38.5 C.

| Shape (14 heads) | Previous path | Vectorized path | Speedup |
| --- | ---: | ---: | ---: |
| 9 keys, 1 query, no mask | 4.16 us | 3.54 us | 1.18x |
| 30 keys, 30 queries, F32 mask | 78.53 us | 53.76 us | 1.46x |
| 95 keys, 1 query, no mask | 6.60 us | 4.15 us | 1.59x |

These are repeated-operation measurements, not full Audio8 inference
speedups. Decode flash attention also removes the standalone softmax
operation, so the remaining application benefit primarily concerns masked
prefill. An earlier implementation that copied whole rows and used a scalar
reciprocal regressed these probes and was replaced before final validation.

### Application measurements and limits

The final 66-frame Audio8 fixed-entry replay profile reduces attention core
event time from 301.860 ms to 75.868 ms, with 4,200 `hvx-f32-decode` calls.
Layout CONT calls decrease from 4,404 to 228. These are separate diagnostic
profiles; generated codes match each runtime's unprofiled replay. Q8 LM
projections account for 76.13% of remaining candidate leaf cycles, led by
the 896x9728 gated FFN and 4864x896 residual down-projection families.

The original full application suite completes 80 captures across S1-S5, with
three warmups and five timed runs per variant/prompt. Full inference results are
mixed: S1/S2 improve, S4 improves modestly, S3 regresses and S5 is roughly
unchanged. Native frame counts differ and timings vary, so these results
do not establish a universal end-to-end speedup.

The October 14 follow-ups retain S1/S2 from that suite and replace S3/S5 and
S4 with three timed observations per variant. Latest full inference medians
(baseline/candidate seconds) are S1 6.474/5.200, S2 4.130/3.539,
S3 6.717/4.931, S4 5.144/6.804 and S5 7.103/6.626. Independent LM medians
are lower for all five prompts. The S4 full-inference regression is in codec
synthesis; identical 81-frame candidate outputs have codec times spanning
1.743-5.019 s. Neither the timing variation nor its kernel cause is resolved.
The companion report retains original and follow-up measurements with their
sample counts and different generated workloads.

The final backend passes 84 softmax, 39 decode-attention and three large-logit
CPU-reference cases; 20 production cache comparisons also pass. Full
teacher-forced LM gates are not universally met: 1,026 numerical misses and
105 CPU top1 differences, versus 1,034 and 104 previously. All 25 fixed-code
codec boundaries pass and match preceding Hexagon tensors byte-for-byte.
The companion application now defaults to supported flash decode and retains
an explicit opt-out. Short softmax is already automatic for applicable shapes.

See the companion [Audio8 validation report](https://github.com/tetherto/qvac-fabric-speech.cpp/blob/feat/audio8-hexagon-attention/engines/tts/docs/audio8-hexagon-attention.md)
for stage medians, run ranges, generated frame counts, CPU comparisons,
output recognition transcripts, commands and runtime hashes.
