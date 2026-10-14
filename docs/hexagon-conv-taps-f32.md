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

Audio8 selects this experiment with `AUDIO8_CODEC_HEXAGON_CONV_TILE=1` before
loading its decoder. Its independent `AUDIO8_CODEC_HEXAGON_FP16=1` experiment
uses existing HMX kernels through explicit default-precision synthesis
products, without changing the F32 precision guard in the backend. HMX stages
activations and output tiles as F16 and expands graph output to F32. The LM,
latent transformer, and small channel tails retain their existing precision.
