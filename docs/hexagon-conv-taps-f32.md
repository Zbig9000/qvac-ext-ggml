# Experimental F32 convolution tiles

The optional registry procedure `ggml_backend_hexagon_conv_taps_f32` creates
a custom operation with contiguous F16 weights `[input, output, taps]`,
padded F32 input `[input, time]`, and F32 output `[output, rows]`.
Positive tap steps select ordinary or dilated convolution windows. A step
of `-1` selects delayed transposed-convolution groups.

The creator returns null when the geometry is unsupported, the per-worker
scratch exceeds VTCM, or the ordinary product would use an existing VTCM
kernel. It currently selects only the ordinary DDR 4x2 panel route. Callers
retain their original graph when the procedure is unavailable or returns
null. No global ggml operation enum or base-library ABI changes are required.
The operation has a scalar F32 CPU callback for scheduler fallback; its
channel reduction order differs from HVX.

Six workers on the tested v79 device each keep up to 64 output rows in VTCM.
Tap products use the existing DDR panel reduction and product-rounding
barrier, followed by F32 addition. Only the completed tile is copied to DDR.
This reduces intermediate traffic but does not assert a performance gain.
Profiling reports `CONV_TAPS_F32` and `hvx-conv-taps-f32`; disable profiling
for performance measurements.

The focused `test-hexagon-conv-taps` regression uses dyadic data and natural
DDR shapes with four output channels. It compares the whole result byte for
byte with separate Hexagon matmul/add operations, then independently checks
sampled worker/tile boundaries against CPU arithmetic. The v79 run passed
ordinary, dilated, and transposed cases at K=96, 192, and 384. A fixed-code
81-frame Audio8 S4 decode also produced an identical WAV, with 11 tiled
operations confirmed in a separate profile. These checks cover the tested
shapes and waveform; they do not establish universal numerical equivalence.

Audio8 selects this experiment with `AUDIO8_CODEC_HEXAGON_FP16=0` and
`AUDIO8_CODEC_HEXAGON_CONV_TILE=1` before loading its decoder. Decoder
synthesis now selects the independently measured HMX mode by default, with
`AUDIO8_CODEC_HEXAGON_FP16=0` restoring F32 requests. The default mode
uses existing HMX kernels through explicit default-precision synthesis
products, without changing the F32 precision guard in the backend. HMX stages
activations and output tiles as F16 and expands graph output to F32. The LM,
latent transformer, and small channel tails retain their existing precision.
Existing HMX matmul/ADD fusion can also apply with normal backend fusion
enabled. This does not enable the separate experimental HVX blocked-reuse or
F32 residual-fusion switches.

Focused v79 FP16 waveform checks used identical fixed codes and unclipped raw
F32 PCM, without alignment, gain normalization, or lag search. A three-frame
comparison against CPU passed cosine 0.999998349 and NMSE 0.00000330349.
An 81-frame S4 comparison with preceding Hexagon passed cosine 0.999999634
and NMSE 0.000000731047. Both were finite and satisfied the existing cosine
0.9999 / NMSE 0.0002 limits. The S4 profile confirmed 117 HMX matmuls.
These are focused codec checks, not a new full LM CPU suite or perceptual
equivalence claim.

The optimized-only S1-S5 suite used one warmup and three timed fresh-process
runs per experiment. F32 tiles showed no codec gain against saved preceding
Hexagon measurements and increased reported graph scratch (S1: 207 to 233 MB).
The independent HMX option recorded codec medians of 0.728, 0.627, 0.948,
0.786 and 0.863 seconds for S1-S5: 2.39x-5.48x faster than the saved preceding
Hexagon codec medians. Saved OpenCL codec times were 1.221, 0.835, 1.163,
1.088 and 1.504 seconds. No new GPU or unoptimized performance series was run.
All runtime/model/script manifests and idle/thermal conditions pass; outputs
are repeatable, and all five HMX PCM16 comparisons with preceding Hexagon
pass the numerical gates. Separate batches, differing GPU trajectories and
unrecorded actual DSP/DDR clocks limit performance claims to the measured
prompt comparisons. Audio8 now makes the measured HMX mode the default;
F32 convolution tiles remain opt-in. This changes application selection,
with no new DSP arithmetic. A subsequent default-path S4 recheck used one
warmup and five timed Hexagon captures, all with identical codes/WAVs to the
explicit HMX path. The five times were 3.4484, 4.2411, 3.1218, 3.0297 and
3.7788 seconds; median 3.4484 seconds and RTF 0.916729601 for 3.761632653
seconds of generated audio. RTF is inference time divided by exact WAV
duration, with lower values better. All idle/thermal conditions and manifests
pass. Saved OpenCL S4 has 3.1466 seconds latency, RTF 1.011289208, and shorter
audio (67 frames versus 81); this remains a prompt-level comparison.

See the [complete stage and waveform report](https://github.com/tetherto/qvac-fabric-speech.cpp/blob/feat/audio8-hexagon-attention/engines/tts/docs/audio8-hexagon-codec-tracks.md)
for independent LM/codec medians, ranges, frame counts, reproduction and
frozen runtime hashes.
