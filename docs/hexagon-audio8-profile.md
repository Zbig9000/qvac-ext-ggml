# Audio8 matmul profiling and correctness

QVAC-26715 originally proposed routing Audio8's large fused `MUL_MAT+ADD` to
HMX. An HMX fused implementation exists, but **current Audio8 codec matmuls
request F32 precision and stay on HVX**. Historical HMX dispatch does not
establish the route taken by the current application.

## Dispatch investigation (QVAC-26714)

The task's historical "58% framework overhead" estimate treated inclusive
OPBATCH timing as exclusive dispatch time. It is not a valid overhead estimate.
In the saved `hexagon-codec-build/results/profiles/profile-p1.log`, 12,895
batches contain 67,539 leaf operations. Batch time totals 13.537086 s; the
leaf sum is 13.272701 s, leaving 0.264385 s (1.953% of batch time) outside
those leaf intervals. Leaf intervals themselves include tensor binding and
cache coherence, not only arithmetic. The 6.461 s of inference outside
OPBATCH also includes CPU work, scheduling, transport and profiling output.
This trace predates the final DDR-only selector and is historical diagnostic
evidence, not the current performance baseline.

The host previously copied the complete 1,024-entry operation vector for
every submission, including unused entries. The historical trace would copy
13,204,480 entries for 67,539 live operations (195.5x excess); 7,769 batches
contained just ADD followed by RMS_NORM+MUL. The queue now snapshots only the
live prefix into its slot. It still owns fused-operation vectors and kernel
metadata until the response arrives, retaining both profiling and DSP-error
diagnostics when the submission buffer is reused. Tensor storage, descriptors,
cache flushes, synchronization, arithmetic and kernel routing are unchanged.

`PROFILE=3` adds these main-thread trace spans using the existing trace layout:

| Span | Meaning |
| --- | --- |
| `BUFF` | Batch buffer mapping/preparation (existing span) |
| `TENSOR_PREP` | Batch tensor descriptor/view preparation |
| `WORKER_WAKE` / `WORKER_SUSPEND` | Worker/HMX wake and completion handling |
| `OP_SETUP` | Per-op metadata, tensor binding and coherence flushes |
| `OP_EXECUTE` | Kernel execution, including its nested traces |
| `OP_RETIRE` | Dirty tracking and scratch-pointer reset |

For a normal greedy S1 invocation, enable `GGML_HEXAGON_PROFILE=3` and
`GGML_HEXAGON_OPTRACE=65536`, keep computation enabled (`OPSTAGE=3`), use
`--verbose`, and capture stdout plus stderr. Then run:

```sh
python3 scripts/hexagon-dispatch-profile.py audio8-s1-trace.log
```

Check `complete_trace`: the parser requires matched spans, exact per-operation
index coverage and batch preparation/worker spans, and detects trace saturation.
Only matched spans contribute to totals. Nested L2 flush and worker/kernel
spans are not additive wall time. Trace timestamps wrap at 32 bits; a single
span must be shorter than one counter period. Increase trace capacity if
necessary, accounting for shared memory per queue slot and thread.

For `PROFILE>0`, `profile-host` reports cumulative cache-copy, packing,
submission, waiting, response-pop and capture-to-response lifetime timings on
backend release (and queue destruction, if reached). The parser's `host_latest`
selects the latest snapshot per session; repeated snapshots must not be summed.
Wait includes DSP work/transport/polling; pop includes profile formatting and
logging; queued lifetimes overlap. These totals cannot be summed as exclusive
dispatch time. Profiling remains off by default. Use separate unprofiled runs
for performance claims, with all validated optimizations enabled by default.

The shared cache regression covers deep metadata ownership after source
mutation/destruction, independent queue slots, shrink/empty/grow reuse and
profiler/error formatting. It passes host CTest, Android compilation and
focused ASan/UBSan. LeakSanitizer was unavailable under sandbox ptrace.
Ten dispatch-parser regressions and five existing profile-parser tests pass.

### S1 validation of the live-operation cache

On October 8, QDC SM8750/v79 device `e8b7f0c8` compared baseline ggml
`866ad51f` (runtime defaults `1f334383`) with candidate runtime `7d75209c`
and cache tests `7da5ad5b`. The speech CLI has the same runtime source as
PR #303 `3b74ad91`; changes since its build are test/documentation only.
All five tuning variables were unset, exercising compiled defaults
`OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`, `HOSTBUF=1`, `F16_F32_PANEL=1`.
Profiling was disabled. S1 used greedy seed 42, four threads, max 70 frames,
three warmups and five timed runs per variant with alternating pair order.

| Variant | Five inference times (s) | Median (s) | Median codec synthesis (s) |
| --- | --- | ---: | ---: |
| Baseline | 13.8197, 14.2418, 13.9042, 13.4194, 14.1262 | 13.9042 | 7.0655 |
| Live-operation cache | 13.8654, 13.8926, 14.2358, 14.0464, 14.0297 | 14.0297 | 7.0730 |

The candidate median is **0.90% slower**, within the observed run spread;
this experiment demonstrates **no end-to-end speedup**. Every run began at
thermal status 0, but timed post-run states were `0,0,2,0,1` for baseline
and `1,0,0,1,0` for candidate. Cooling does not eliminate all thermal variation,
and five samples cannot establish a performance-equivalence bound.

All 16 runs produced identical 66-frame code and WAV files (3.065034 s audio):

- codes SHA-256: `3d7e3716ec6c6e6776e4955fafb2eec08c3bc05fdcd3c2ba20c93ab2f46fed17`
- WAV SHA-256: `cabfe8e928545d755e41749d86cda74b2f1e4cfe16e505e22a56f6ee9388cbd9`

The candidate passes all 23 independent device oracle cases and all five
fixed-66-frame codec boundaries vs CPU. PCM cosine is `0.9999966593`, NMSE
`6.721473961e-6`. Both separate PROFILE=1 and PROFILE=3 captures preserve the
same codes/WAV hashes. This validates these inputs; it does not establish
universal CPU/OpenCL/Hexagon trajectory equivalence.

Artifact SHA-256:

| Artifact | Baseline | Candidate |
| --- | --- | --- |
| Hexagon host library | `248af7d0db677e47a38c25e1a57bbdeb8c826423824f87b7a7af63b6d496f253` | `3be5cc86d15d306b0dfbd3b34eb0a22713b531806105c71e7fb9e7bc3fd931a7` |
| v79 DSP skeleton | `49b8e8d93f12b19494da69666c47b9f4b766bd08d62aba6dcc5c0b7485fbb041` | `faeb5abeb402142cf2ae5f0bbb8231ff03266ff3735b160a569f8f82b10b0732` |

The CLI SHA-256 is `da8ca10b5928159e83fa6e978c2570b6d5091338ae97b1ef4f2758e29a2993ee`.
Model hashes remain those listed below. Reproduction scripts and raw evidence
are in `hexagon-26714-build/run-s1.sh`, `run-profile.sh`, `results/device/`,
`results/s1-summary.json` and `results/cli-provenance.txt` beside the checkout.

### Current DSP dispatch breakdown

The candidate's separate PROFILE=1 capture contains 12,895 batches and 67,539
operations. Inclusive batch time is 8.217117 s, leaf time 7.948592 s, and the
residual is 268.525 ms. The residual is 2.8850% by cycles and 3.2679% by
microseconds; independent timer sampling and integer-microsecond conversion
affect these small differences.
This residual is batch work outside leaf intervals, not all dispatch cost.
The largest batch contains 495 operations, below the 1,024 limit. Its 7.062788 s
inclusive time contains only 0.623 ms residual; raising batch capacity is not
supported by this evidence.

The full PROFILE=3 capture passes `complete_trace` for every batch and
operation with zero saturation, missing spans, duplicate indices or unmatched
edges. Main-thread spans as shares of inclusive DSP batch cycles:

| Span | Count | Batch cycle share |
| --- | ---: | ---: |
| `OP_EXECUTE` | 67,539 | 95.5081% |
| `OP_SETUP` | 67,539 | 1.2811% |
| `OP_RETIRE` | 67,539 | 0.2435% |
| `BUFF` | 12,895 | 0.0999% |
| `TENSOR_PREP` | 12,895 | 0.1107% |
| `WORKER_WAKE` | 12,895 | 0.7558% |
| `WORKER_SUSPEND` | 12,895 | 0.3660% |

These disjoint spans cover 98.3651% of batch cycles. Execute includes kernel
setup, synchronization and nested work; it is not pure arithmetic. Separately,
L2FLUSH totals 1.7063% but mixes nested and batch-boundary spans, so it must
not be added to this table. The trace's 28.7024 s application inference time
is heavily perturbed by trace handling; inclusive DSP batch time remains
8.243389 s. Neither profiled run supplies a speedup claim.

This evidence does not establish a major DSP batch-dispatch bottleneck.
Preserve coherence and worker synchronization. The three largest codec matmul
families still account for 71.18% of PROFILE=1 leaf microseconds; investigate
kernel execution and host/CPU scheduling before speculative batching changes.
Raw captures and analysis are `hexagon-26714-build/results/profile{1,3}.log.gz`
and `profile{1,3}-analysis.json` beside the checkout.

Those captures also exposed a host-summary lifetime issue: the registry-owned
queue survives backend release, so destructor-only host logging did not run.
Commit `573a4508` emits gated cumulative snapshots at backend release without
flushing or changing default execution. The S1 timings above use the earlier
stub; this follow-up changes profiling output only. Its matching diagnostic
stub SHA-256 is `f718204047fcaf33c31ee7562ada8366001cfc710c5ab65a7731f3c0d1d6375a`;
the DSP skeleton is unchanged. A snapshot reports currently recorded counters,
not an implicit queue drain.

Fresh device validation of that follow-up passes: PROFILE=1 emits two
identical cumulative snapshots for 12,895 batches / 67,539 operations. Using
only the latest snapshot gives cache copy 1.567 ms, packing 1.388 ms,
submission 2.203 ms, wait 9,023.276 ms, and response pop 176.633 ms. Overlapping
capture-to-response lifetimes sum to 9,029.127 ms. Wait includes DSP execution
and transport, while pop includes profile output; these are not exclusive
host-overhead totals. Both the profiled run and a separate PROFILE=0 S1 run
produce the same codes/WAV hashes as above. Evidence is in
`hexagon-26714-build/results/diagnostics/` and `run-diagnostics.sh`.

### Hexagon-only S1–S5 result

The requested five-prompt sweep completed on the same QDC device `e8b7f0c8`,
using the unchanged benchmark host/DSP pair above, all five compiled tuning
defaults enabled, profiling off, greedy seed 42 and four threads. Each prompt
has three warmups and five timed runs. S1 reuses the candidate half of the
paired experiment; S2–S5 run the candidate only. The staged CLI has production
source identical to speech PR #303 `a90dc140` (later changes are docs/tests).

| Prompt | Frame cap | Generated frames | Median inference (s) | Timed range (s) | Median codec synthesis (s) | Inference RTF |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| S1: The quick brown fox jumps over the lazy dog. | 70 | 66 | 14.0297 | 13.8654–14.2358 | 7.0730 | 4.5773 |
| S2: She sells seashells by the seashore. | 60 | 55 | 11.4858 | 11.3312–11.7409 | 5.6623 | 4.4968 |
| S3: In a world where machines think, humans still dream. | 90 | 70 | 14.3848 | 14.2154–15.2066 | 7.2083 | 4.4250 |
| S4: Testing one, two, three, four, five. | 70 | 70 | 14.3936 | 13.9417–15.2274 | 7.2095 | 4.4277 |
| S5: The year was 2026, and the signal was finally clear. | 110 | 94 | 19.4038 | 19.0168–19.7377 | 9.6671 | 4.4450 |

Inference excludes model loading; RTF is inference time divided by generated
audio duration. S4 reaches its configured cap, so do not assume natural EOS.
All 40 candidate runs pass raw-artifact validation: correct HTP0 selection,
profiling off, ten finite timing fields, frames×10 token codes in range,
mono PCM16 at 44.1 kHz with exactly frames×2048 samples, and matching code/WAV
hashes across all eight runs of each prompt. Repeatability does not establish
universal CPU/OpenCL parity or listening quality.

Every run began at thermal status 0. Timed post-run states were S1 `1,0,0,1,0`,
S2 `0,2,0,0,0`, S3/S4 `0,1,0,1,0`, and S5 `0,2,0,2,0`. These are start-cooled
measurements, not continuously unthrottled runs. The tunnel dropped during S5;
reconnection recovered all eight completed runs and the suite completion marker
from the same device. No missing timing was inferred or substituted.

Reproduce with `hexagon-26714-build/run-s1.sh`, `run-hex-s2-s5.sh` and
`summarize-hex-suite.py` beside the checkout. The complete report is
`results/hex-suite-summary.json`; per-run logs, thermal records, codes and WAVs
are in `results/device/` and `results/hex-suite/`. Exact model/library/CLI hashes
are in `results/runtime-provenance.json`. This sweep has no contemporaneous
CPU/OpenCL timings and does not replace the separate panel-on/off experiment.

## F32 codec panels

The Hexagon backend enables a 4x2 HVX DDR output panel by default for
unbatched F16-weight/F32-activation matmuls. Four output channels share each
pair of activation rows, with weights widened to F32. The existing
`vec_dot_f16_f32_uu_1x1` narrows activations to F16 internally, so this panel
changes arithmetic as well as data reuse. F32 precision requested by a graph
does not by itself prove that the old DDR helper retains F32 activations.

Eligibility requires at least two weight and activation rows, K at least 64
and divisible by 32, scalar-contiguous inner axes, and no fused ADD. Padded row strides and
unaligned row bases are supported; K=96 and odd output dimensions have explicit
remainders. Batched, fused, and other layouts retain the existing routes. HMX
selection keeps its existing precedence; set `GGML_HEXAGON_MM_SELECT=2` when
testing the panel explicitly. The option applies only when the normal VTCM
path cannot be used. Eligible small shapes retain the existing VTCM kernel;
focused DDR tests use activation matrices larger than the device VTCM budget.

Set `GGML_HEXAGON_F16_F32_PANEL=0` to restore the previous DDR route; `=1`
explicitly enables the panel. Evaluate with `GGML_HEXAGON_OPSTAGE=3`, the
exact-F32 oracle, fixed-code codec/PCM comparisons, and end-to-end Audio8
timing against the same build with the panel explicitly disabled. Local
compilation alone establishes no device correctness or speedup. The measured
codec shapes motivating the
candidate are `[192,192] x [192,71680]`, `[96,96] x [96,143360]`, and
`[384,384] x [384,17920]` in ggml dimension order.

October 8 isolated-kernel checks on the renewed QDC SM8750/v79 device
(`57dd7911`) passed all 14 initial independent-oracle cases. Three repeated
comparisons after an initial trial gave these medians (each test executable
internally ran 30 iterations for K=96 and 15 for K=192/384):

| K / output channels | Activation rows | Panel off (ms) | Panel on (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 184320 | 211.786 | 93.626 | 2.262x |
| 192 | 92160 | 243.832 | 118.364 | 2.060x |
| 384 | 23040 | 154.834 | 104.358 | 1.484x |

These large-shape probes use the same channel counts as the current profile
and the longer activation sequences from the historical fixture. They are
isolated matmuls, not a measurement of full Audio8 inference. Both variants
used the same candidate build, `OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`,
`HOSTBUF=1`, and default matmul selection; only `F16_F32_PANEL` changed.
The repeated order was panel on, then off. Run with:

```sh
GGML_HEXAGON_F16_F32_PANEL=1 ./test-backend-ops perf -b HTP0 -p audio8_codec_f32=1
```

Raw evidence is in `hexagon-codec-build/results/device-oracle.log`,
`device-perf-repeat.log`, and `device-perf-summary.json` beside the checkout.
The DSP artifact SHA-256 was
`49b8e8d93f12b19494da69666c47b9f4b766bd08d62aba6dcc5c0b7485fbb041`.
The initial broad selector passed fixed-code CPU/OpenCL parity, but repeated
full S1 inference medians were 19.8494 s off versus 19.8996 s on. Paired profiles
matched all 67,539 leaf operations: 76 existing DDR matmuls saved 5,408.814 ms,
while replacing 41 existing VTCM matmuls added 5,341.744 ms. The selector now
preserves every normal VTCM choice and only replaces DDR fallbacks. The results below validate this narrower selector.

## Validated DDR-only panel result, October 8

On the renewed QDC SM8750/v79 device (`57dd7911`), the panel reduced
median S1 inference from **20.346 s to 14.603 s**: **1.3933x faster**, or
**28.23% less inference time**. The codec synthesis median fell from 12.4843 s
to 7.0899 s. These are same-device comparisons of the same binary, changing
only `GGML_HEXAGON_F16_F32_PANEL` between 0 and 1.

| Setting | Median inference (s) | Median codec synthesis (s) | Inference RTF |
| --- | ---: | ---: | ---: |
| Panel off | 20.346 | 12.4843 | 6.6381 |
| DDR-only panel on | 14.603 | 7.0899 | 4.7644 |

Each setting had three warmups and five timed runs, interleaved off/on.
The prompt was "The quick brown fox jumps over the lazy dog.", with greedy
sampling, default seed 42, four threads, and a 70-frame cap. Every run produced
the same 66-frame code sequence and 3.065034 s of audio. WAV hashes repeat
within each setting; cross-setting WAVs differ slightly and pass the numeric
gates (cosine `0.9999999404`, NMSE `1.19266e-7`). Inference uses the CLI's
`StageTimings.total_ms`, excluding model loading, not process wall time.

Both 3-frame and 66-frame fixed-code checks pass all five codec/PCM boundaries
against CPU reference. The 3-frame PCM is identical between panel settings.
For 66 frames, candidate PCM versus CPU has cosine `0.9999966593` and NMSE
`6.72147e-6`. Direct candidate versus OpenCL PCM has cosine `0.9999900834` and
NMSE `1.98341e-5`; all compared PCM is finite. Gates are cosine at least
0.9999 and NMSE at most 2e-4. This establishes bounded numerical agreement,
not bit-identical waveforms, universal prompt coverage, or subjective audio
quality. OpenCL comparison uses identical codes; differing free-running
CPU/OpenCL/Hexagon trajectories remain outside this optimization's claim.

Validation also passed 23 independent-oracle device cases, host reference
checks, host/Android/DSP v79 builds, and independent review. The cases cover
natural DDR routing, VTCM preservation, varied and precision-sensitive inputs,
odd dimensions, strides, offsets, bias, and excluded shapes. The existing
small VTCM arithmetic is intentionally retained. The panel is now enabled by
default, with the same DDR-only routing guards used for these measurements.

Polling is also enabled by default for the entire Hexagon backend, including
workloads other than Audio8. It reduces batch-completion wait latency by
actively checking for results instead of waiting with a blocking timeout.
This can increase host CPU activity and may affect power use; CPU utilization
and power were not measured in these experiments. Set `GGML_HEXAGON_OPPOLL=0`
to restore blocking waits, or `=1` to explicitly enable polling.

Use the corrected ggml host-buffer and speech key-cache fixes. The current
defaults below need no environment configuration; explicit exports remain
useful for reproducing the historical measurements:

```sh
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_OPPOLL=1
export GGML_HEXAGON_OPSTAGE=3
export GGML_HEXAGON_OPFUSION=1
export GGML_HEXAGON_F16_F32_PANEL=1
```

Do not use `OPSTAGE=1`: it omits computation. Performance runs used
`GGML_HEXAGON_PROFILE=0` and default matmul selection. The model identities are
those listed in the corrected baseline below. Runtime routing commit was
`f4ab5509`, tests `e4e6ac15`, and speech runtime `3778e63d` (later documented at `0f75ca40`). The DSP is unchanged
from the isolated-kernel checks. The staged stripped host library SHA-256 is
`bc39c3acdb013644fec7adc08b93a7ee2a9c25d3aa7c1a000082b777954aa768`.

Evidence beside the checkout is under `hexagon-codec-build/`:

- `results/model-ddr/`: all raw logs, code/WAV outputs, and fixed-code PCM exports.
- `results/model-ddr-summary.json` and `summarize-model.py`: timings, hashes,
  waveform metrics and completeness checks for all ten timed runs.
- `stage/validate-audio-ddr.sh` and `results/model-validation-ddr.log`: exact
  run order, model hash verification, flags and commands.
- `results/device-oracle-ddr.log` and `results/ci-validation.txt`: test/build evidence.
- `results/profiles/`: paired profiles explaining why VTCM routes must be retained.

The OpenCL reference run uses the earlier OpenCL-enabled registry with the
same CPU/base libraries; the candidate registry is built for CPU/Hexagon.
This registry setup does not change the timed Hexagon library set. The earlier
polling-only baseline below came from a different QDC device serial; do not
combine the two speedup ratios as a same-device measurement.

## Default-enabled verification and S1–S5 follow-up, October 8

Commit `1f334383` makes polling and the DDR panel the backend defaults. PR #116
was merged into PR #115's feature branch at `866ad51f`; #115 remains the single
combined PR targeting `speech`.

The rebuilt host library was checked on QDC device `57dd7911` with
`OPPOLL`, `OPSTAGE`, `OPFUSION`, `HOSTBUF` and `F16_F32_PANEL` environment
overrides unset. All 23 focused oracle cases and all five fixed-66-frame codec
boundaries passed. Full S1 codes and WAV were byte-identical to the earlier
explicit-enabled panel result. Setting `OPPOLL=0` and `F16_F32_PANEL=0`
reproduced the earlier panel-off codes and WAV byte for byte. Host reference,
buffer and profile-parser checks, host/Android/DSP builds, and independent
review passed. The DSP binary remains unchanged.

The stripped default-enabled Hexagon host library SHA-256 is
`248af7d0db677e47a38c25e1a57bbdeb8c826423824f87b7a7af63b6d496f253`.
Device validation output is retained at
`hexagon-codec-build/results/defaults-validation.log`; the remote artifact
directory is `/data/local/tmp/qvac-codec/results-defaults-20261008`.

A requested CPU/OpenCL/Hexagon S1–S5 sweep used greedy decoding, seed 42,
four threads, per-prompt frame caps 70/60/90/70/110, and three warmups plus five
timed runs per cell. QDC access ended during S2. S1's thermal snapshots also
showed a transition from status 0 to status 3, with OpenCL inference rising
from about 3.4 s to 7.3 s. These mixed-state, incomplete results are not the
final cross-backend benchmark. The user retained S1 and subsequently
cancelled the S2–S5 continuation on replacement device `e8b7f0c8`. The partial
sweep is preserved as diagnostics; it does not replace the validated same-device
S1 panel comparison above. Under QVAC-26714, the user subsequently requested
a new Hexagon-only S1–S5 sweep with all validated defaults enabled; it is a
separate candidate experiment, not a continuation of that three-backend sweep.

## Corrected S1 baseline, October 8

On the same QDC Snapdragon 8 Elite, enabling polling reduced median inference
time from **24.5906 s to 19.4765 s**, a **1.2626x speedup** (20.80% less time).
The ten timed Hexagon runs produced identical code files and WAV files across
both settings. This is evidence for polling preserving the measured S1 output.

| Variant | Median inference (s) | Median process wall time (s) | Frames | Audio duration (s) | Inference RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hexagon baseline, `OPPOLL=0` | 24.5906 | 24.9704 | 66 | 3.065034 | 8.022945 |
| Hexagon polling, `OPPOLL=1` | 19.4765 | 19.8224 | 66 | 3.065034 | 6.354416 |
| OpenCL | 3.3727 | 4.7182 | 70 | 3.250794 | 1.037500 |

Inference time is Audio8's `StageTimings.total_ms`, which covers generation
after model loading. Process wall time additionally includes CLI startup,
loading, output writing, and teardown. RTF is inference seconds divided by the
actual WAV duration. Each column reports its own median of five timed runs.

Protocol: S1 text was "The quick brown fox jumps over the lazy dog." on QDC
`sa880573`, device serial `f3b4a4c5`, with Q8_0 LM/decoder models, greedy
decoding, four CPU threads, default seed 42, and a 70-frame cap. Each variant
had three warmups followed by five timed CLI invocations. Variants ran in
interleaved order `hex-base`, `hex-poll`, `opencl` on every repetition.
`GGML_HEXAGON_OPSTAGE=3`, `HOSTBUF=1`, `OPFUSION=1`, and `PROFILE=0` were fixed;
only `OPPOLL` changed between the two Hexagon variants. No diagnostic tensor
callback was enabled.

Runtime source was ggml `9ef658d3` (later documentation commit `73b087b1`) and
speech `3778e63d`. All variants used the same staged CLI and device library set.
That set included the corrected Hexagon host library from the standalone
Hexagon build alongside the OpenCL libraries. Model SHA-256 values were:

| Model | SHA-256 |
| --- | --- |
| `audio8-lm-q8_0.gguf` | `f33c58b2fd46320c01544eec961112b4b9778106ce883e86914dd38854072d40` |
| `audio8-codec-decoder-q8_0.gguf` | `84d4b88fc5b2274c2cbfd95c4ce1bdae7820eb1e2464b7bd715d3fe370895432` |

All five runs within each variant had repeatable code and WAV hashes. However,
Hexagon and OpenCL generated different codes and lengths: OpenCL reached the
70-frame cap, while Hexagon produced 66 frames. These measurements therefore
do not establish CPU/OpenCL/Hexagon parity or equal generated workloads. This
is a single-prompt baseline, not the earlier five-prompt suite. The completed
scope is the buffer correctness fix and this baseline; broader parity work
and further kernel/routing optimization remain follow-ups.

Local evidence resides under
`/home/alokr/code/speech-qvac/hexagon-26715-build/`:

- `results/baseline-S1.json`: per-run timings, medians, output hashes.
- `results/benchmark-S1.log`: run order and process timestamps, including warmups.
- `results/benchmark-device/{hex-base,hex-poll,opencl}-S1-{warmup1..3,timed1..5}.{log,codes,wav}`:
  raw CLI logs and generated outputs.
- `results/model-sha256.txt`: checked model identities.
- `device-stage/benchmark.sh`: device runner; `single` reproduces the S1 protocol.
- `summarize-benchmark.py`: reconstructs the summary from timestamps, logs, and WAVs.

After staging the matching build and models in `/data/local/tmp/qvac-26715`,
run `sh /data/local/tmp/qvac-26715/benchmark.sh single` on the device and capture
its output plus the generated `results/` files. Recompute the existing local
summary from the ggml checkout with:

```sh
python3 ../hexagon-26715-build/summarize-benchmark.py \
  ../hexagon-26715-build/results/benchmark-S1.log \
  ../hexagon-26715-build/results/benchmark-device
```

## Current application routing

In the speech checkout, `engines/tts/src/audio8/codec_ops.cpp` sets
`GGML_PREC_F32` on both `sum_taps` and `causal_conv_transpose` matmuls. These
requests were present in the original Audio8 implementation (`09566de3`).
The current ggml HMX eligibility check rejects that precision because HMX
multiplies in F16 (`be3edbd6`). Fused parameter selection also preserves the
original MUL_MAT node's precision (`ccb67cb8`); older code passed the ADD node,
which could bypass that check. Both ggml fixes are in the merged `8c85fb0f` base.

Fusion requires either an HMX kernel or a single activation row. Consequently,
the large current codec products use separate HVX `MUL_MAT` and ADD operations.
The October 8 three-frame `hex-base-profile.log` confirms zero `MUL_MAT+ADD`
calls; its largest codec products (`192:192 x 192:3072`,
`96:96 x 96:6144`, and `384:384 x 384:768`) are labelled `hvx-tiled`.
This trace used `OPSTAGE=3`, but preceded the quantized host-buffer correction
below and produced invalid outputs, so it establishes dispatch, not valid
end-to-end performance.

Preserve the explicit F32 requirement. Relaxing the HMX row threshold alone
does not make these products eligible. Any future precision change would need
separate application-level correctness evidence and must not silently override
`GGML_PREC_F32` in the backend.

A fresh S3 trace after both correctness fixes, with `OPSTAGE=3`, `OPPOLL=1`,
and `PROFILE=1`, also records zero `MUL_MAT+ADD` calls. It generated 70 frames.
The three largest groups were:

| Weight shape K:N | Activation shape K:M | Kernel | Share of leaf cycles |
| --- | --- | --- | ---: |
| 192:192 | 192:71680 | `hvx-flat` | 34.22% |
| 96:96 | 96:143360 | `hvx-flat` | 28.00% |
| 384:384 | 384:17920 | `hvx-flat` | 19.77% |

These groups account for about 82% of the 29,765,246,194 leaf cycles. Inclusive
batch cycles were 30,285,894,686 and must not be added to leaf cycles. Codec
synthesis took 13,132.9 ms of the reported 21,078.9 ms inference time. Logging
affects these timings; use the unprofiled S1 measurements above for the baseline.
The trace identifies current F32 codec matmuls as follow-up profiling targets;
it does not establish cross-backend parity or justify changing precision.
Local evidence is `results/hex-corrected-S3-profile.log` and its parsed
`results/hex-corrected-S3-profile.json` under the build directory above.

## Historical October 7 profile

The older profile labels every `MUL_MAT+ADD` call `hmx-tiled`. This shows that
the fused implementation was exercised in that run. Its exact repository
commits are not recorded, so the discrepancy with current routing cannot be
attributed to a specific historical build.

The source log is `audio8-raw/hex-profile.log` (not checked into this repository).
It contains two labelled runs, including the warmup despite its label saying
"not captured". It does not record `OPSTAGE`; therefore it establishes the
logged dispatch choices, not a validated full-compute performance baseline
(see the `OPSTAGE` requirement below). The captured run contains:

| Measurement | Captured run |
| --- | ---: |
| Fused calls | 98 |
| Fused cycles | 829,590,819 |
| All individual operation cycles | 2,687,987,851 |
| Inclusive OPBATCH cycles | 3,701,900,582 |
| Fused share of individual operation cycles | 30.86% |
| Fused share of inclusive batch cycles | 22.41% |

`OPBATCH` times the entire DSP batch, including its operations and batch setup
and cleanup. Adding it to individual operation times double-counts work. The
original 196 calls also combined the warmup and captured run. Neither DSP
percentage represents the share of end-to-end inference latency.

The dominant historical captured fused shapes are F16 weights, F32 activations, and a
full-size F32 partial convolution as the ADD operand:

| Weight shape K:N | Activation shape K:M | Calls | Cycles |
| --- | --- | ---: | ---: |
| 96:96 | 96:184320 | 18 | 291,346,079 |
| 192:192 | 192:92160 | 19 | 271,181,453 |
| 384:384 | 384:23040 | 18 | 133,180,485 |
| 768:768 | 768:2880 | 18 | 41,459,353 |

These are vocoder products with many activation rows. They are distinct from
the AR batch-one projection. The latter appears as an unfused F32 `MUL_MAT`
(`896:4096 x 896:1`) in this log. Relaxing a batch-one HMX predicate would not
change the fused operations above.

## Reproduce the analysis

Run from the ggml checkout:

```sh
python3 tests/test-hexagon-profile.py
python3 scripts/hexagon-profile.py /path/to/hex-profile.log --section 'profile run (captured)'
python3 scripts/hexagon-profile.py /path/to/hex-profile.log --json > profile.json
```

The analyzer preserves sections, repeated run labels, and HTP sessions
separately. Unlabelled input remains unlabelled; it cannot infer warmups from
the records. Groups retain operation, shape, types, and the selected kernel.
Malformed profile records fail rather than silently dropping timed work.

## Quantized weight buffer correctness

Before timing Audio8, verify the Hexagon weight-buffer contract:

```sh
./test-hexagon-buffer
GGML_HEXAGON_HOSTBUF=0 ./test-hexagon-buffer
```

The default Hexagon buffer advertises host-accessible storage when `HOSTBUF=1`
(the default). CPU fallback and tensor copies may read its data pointer without
calling the backend's `get_tensor`. Such storage must retain canonical ggml
quantized blocks. Uploading tiled DSP blocks there corrupts CPU fallback inputs
even if backend `set_tensor`/`get_tensor` roundtrips appear correct. Audio8 loads
weights into this default buffer, making this contract relevant to its Q8
projections.

Host buffers now retain canonical bytes; explicit `HTP0-REPACK` buffers and the
legacy non-host `HOSTBUF=0` mode retain tiled storage with conversion on upload
and readback. The test covers Q4_0, Q4_1, Q8_0, IQ4_NL, and MXFP4, checking API
roundtrips, host-visible bytes, copies to CPU, and partial transfers on host
buffers. It also checks explicit repack buffers when available. Run this test
on the device; the Hexagon portion is skipped on hosts without that backend.

On the October 8 QDC Snapdragon 8 Elite run, the pre-fix default buffer passed
API roundtrips but failed direct host-byte and CPU-copy checks for all five
quantized types. After the fix those checks pass with both `HOSTBUF=1` and
`HOSTBUF=0`, including explicit repack buffers in the default mode. The default
Audio8 three-frame smoke now produces nonzero codes, but codes still diverge
from CPU/OpenCL in the first frame. This establishes the buffer correction,
not end-to-end parity. The matched S1 timings above establish polling's
performance and output stability for that prompt; waveform finiteness and
perceptual quality remain separate validation requirements.

## Build the focused tests

Host correctness (CPU optimized path compared to the CPU reference):

```sh
cmake -S . -B ../hexagon-26715-build/host/ggml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON \
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF
cmake --build ../hexagon-26715-build/host/ggml --target test-backend-ops -j 6
../hexagon-26715-build/host/ggml/bin/test-backend-ops test -b CPU -o AUDIO8_MUL_MAT_ADD
```

Android and the Snapdragon 8 Elite v79 DSP skeleton (adjust installation paths
for another machine):

```sh
cmake -S . -B ../hexagon-26715-build/android/ggml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/home/alokr/android-sdk/android-ndk-r28b/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DGGML_HEXAGON=ON -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=ON \
  -DGGML_BUILD_EXAMPLES=OFF -DPREBUILT_LIB_DIR=android_aarch64 \
  -DHEXAGON_SDK_ROOT=/home/alokr/code/speech-qvac/poc/toolchains/Hexagon_SDK/6.6.0.0 \
  -DHEXAGON_TOOLS_ROOT=/home/alokr/code/speech-qvac/poc/toolchains/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
cmake --build ../hexagon-26715-build/android/ggml --target test-backend-ops test-hexagon-buffer htp-v79 -j 6
```

`PREBUILT_LIB_DIR` is required by this SDK's CMake helpers even for the host
stub. The backend test binary, `src/libqvac-speech-ggml*.so`,
`src/ggml-hexagon/libqvac-speech-ggml-hexagon.so`,
`src/ggml-hexagon/libggml-htp-v79.so`, and the NDK's `libc++_shared.so` must be
staged together on the device. Preserve the matching ggml/backend/skel build.

## Device verification and trace

Run these commands in the staged directory on the device. Set `LD_LIBRARY_PATH`
and `ADSP_LIBRARY_PATH` to that directory using the device's existing FastRPC
setup. Record the device, model hashes, both repository commits, thread count,
thermal state, and environment alongside the logs.

`GGML_HEXAGON_OPSTAGE` is a bitmask: queueing is bit 0 (`1`), and computation
is bit 1 (`2`). Use `3` (the default) to execute both. The previously reported
"tuned" setting `OPSTAGE=1` sets `HTP_OPFLAGS_SKIP_COMPUTE`; it skips arithmetic
in kernels that honor that flag. Fused and unfused paths do not all handle it
identically. Timings obtained with `OPSTAGE=1` cannot establish a valid inference
speedup or correctness baseline. Repeat those measurements with `OPSTAGE=3`.

```sh
GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=0 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-profile.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE=262144 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD \
  -p 'm=96,n=184320,k=96,residual=1' > audio8-matmul-trace.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-perf.log 2>&1
```

The correctness cases cover full residual and broadcast bias epilogues,
activation row counts 1/4/5 across the current HVX/HMX threshold, incomplete
activation tiles, and partial output-column tiles. They use the existing
matmul NMSE tolerance of `5e-4`; they do not certify end-to-end speech parity.
Perf mode uses the four exact shapes above with each ADD operand type.
These synthetic graphs use default matmul precision to exercise the existing
fused implementation; they do not reproduce the current codec's explicit F32
precision requirement. Passing them does not authorize routing the real codec
through HMX.

`PROFILE=3` exposes `HVX_A_PREP`, `HVX_W_PREP`, `HVX_O_PROC`, `HMX_COMP`, and
DMA events. These distinguish conversion/output work from HMX multiplication.
Check each OPBATCH `evt-cnt` against the per-thread `OPTRACE` capacity before
using a trace: the logger clips events at that capacity. Pair events within
each batch and thread; timestamps are 32-bit cycle values and can wrap.
Concurrent thread intervals overlap and must not be summed as elapsed time.
The analyzer above summarizes operation timings, not these trace intervals.

Use the unprofiled run for timing claims; profiler logging changes the workload.
For complete Audio8 validation, compare matched Hexagon/OpenCL runs on the same
device, separate model loading from inference, use three warmups and five timed
runs, and check per-stage cosine >=0.9999 plus generated tokens, EOS, frame
count, and audio quality. Do not enable a routing change based only on these
synthetic backend tests.

## Default 4x2 codec panel (QVAC-26763)

The follow-up [QVAC-26763](https://app.asana.com/1/45238840754660/project/1214153063536860/task/1219313976732459)
adds the default 4x2 HVX panel for the existing F16-weight/F32-activation DDR
route. Four output channels share each pair of activation rows. Each output
retains the 2x2 panel's F32 low-then-high accumulation order and pairwise
reduction. VTCM selection, F32 precision guards and unsupported-shape fallbacks
remain unchanged. Device correctness and paired S1 performance evidence are
recorded below. All previously validated optimizations remain enabled by default.

Unset `GGML_HEXAGON_F16_F32_PANEL_SHAPE` or set it to `4x2` for the default;
set it to `2x2` to select the previous panel. Invalid values warn and select
the default 4x2 panel. The existing
`GGML_HEXAGON_F16_F32_PANEL=0` override still disables panel routing.
With profiling enabled, selected 4x2 operations report `hvx-panel-4x2`.
Capture that routing evidence separately from unprofiled timing runs.

The independent oracle now has 29 cases. New cases cover full four-channel
panels, two-channel remainders, padded views, K64/K96 tails and channel-axis
worker partitioning. The latter use legal weight-row padding to exceed the
8 MiB VTCM budget even with one worker, retaining small K96 dot products and
NaN-filled padding. On the device, run both panel shapes with 1, 2 and 6 HVX
workers and compare the canonical little-endian result dumps byte for byte:

```sh
GGML_HEXAGON_NHVX=6 GGML_HEXAGON_F16_F32_PANEL_SHAPE=2x2 \
  ./test-audio8-codec-f32 --dump-prefix oracle-2x2
GGML_HEXAGON_NHVX=6 GGML_HEXAGON_F16_F32_PANEL_SHAPE=4x2 \
  ./test-audio8-codec-f32 --dump-prefix oracle-4x2
```

`--dump-prefix` writes Hexagon results and cannot be combined with
`--reference-only`. Host-only reference checks do not validate DSP execution.

`test-backend-ops` also includes the three dominant shapes from the complete
QVAC-26714 S1 trace: K96/output96/rows135168,
K192/output192/rows67584 and K384/output384/rows16896. Select only those cases
for an unprofiled comparison, with the chosen panel shape set in the environment:

```sh
GGML_HEXAGON_PROFILE=0 ./test-backend-ops perf -b HTP0 -o MUL_MAT \
  -p 'audio8_codec_f32=1.*n=(135168|67584|16896),'
```

Local validation on October 8 passes host/Android/DSP v79 builds, all 29 host
reference cases, and cache/buffer/profile-parser regressions. Independent v79
assembly inspection finds no HVX vector spills or calls inside the K loop.
The 4x2 worker is larger (850 instruction words and a 144-byte scalar frame,
versus 517 words and 104 bytes for 2x2). This establishes build and static
readiness, not a speedup.

Device validation subsequently completed on October 8 on the OnePlus 13s
(CPH2723, SM8750/v79). All 29 oracle cases passed for both panel shapes with
1, 2 and 6 HVX workers, with byte-identical results between panels. All 25
fixed-code CPU/Hexagon/OpenCL codec boundary gates passed. Hexagon/OpenCL PCM
cosine was 0.9999900834 and NMSE was 1.9834e-5 at 66 fixed frames.

Paired S1 runs used the same build, models and device, greedy seed 42, four
threads, all validated optimizations enabled, profiling off, three warmups and
five timed runs per panel, alternating order, and thermal status 0 at every
start:

| Panel | Median inference (s) | Median codec synthesis (s) |
| --- | ---: | ---: |
| 2x2 | 14.5259 | 7.1467 |
| 4x2 | 13.4106 | 5.8482 |

The 4x2 panel reduced total time by 7.68% and codec time by 18.17%. All 16
outputs had identical 66-frame codes and WAV hashes. These are existing S1
measurements of explicitly selected panels, not a new run of the default
change. The complete S1–S5 capture was subsequently recovered from the same
phone and validated on October 12; see the table below. Performance tuning
across worker counts is not included in these results.
Evidence is recorded on QVAC-26763 and in
`hexagon-26763-build/results/phone-s1-summary.json` and
`hexagon-26763-build/results/phone-fixed-acceptance.json` beside the checkout.


### Recovered October 8 S1–S5 comparison

The phone retained the completed `phone-20261008-1434` suite even though the
local handoff stopped at S1. All five prompt captures pass the existing strict
artifact validator: three warmups and five timed runs per variant, profiling
off, thermal status 0 before every run, finite stage timings, valid frame/code
and WAV geometry, and byte-identical codes/WAVs across both panels and all
repeats for each prompt. The manifest is the same as the saved S1 evidence.

| Prompt | Frames | 2x2 median inference (s) | 4x2 median inference (s) | Less inference time |
| --- | ---: | ---: | ---: | ---: |
| S1 | 66 | 14.5259 | 13.4106 | 7.68% |
| S2 | 55 | 11.8784 | 10.8054 | 9.03% |
| S3 | 70 | 15.9245 | 14.6666 | 7.90% |
| S4 | 70 | 17.1649 | 14.6040 | 14.92% |
| S5 | 94 | 20.9144 | 19.9595 | 4.57% |

The sum of per-prompt inference medians is 8.66% lower with 4x2. This is an
aggregate of those five medians, not a new measured run or a universal gain.
S4 reaches its frame cap. Some runs finish at thermal status 1, and several
codec stage measurements have substantial variation; the S4/S5 codec medians
are higher with 4x2. The consistently lower full-inference medians do not
establish a uniform codec-stage improvement. The raw records and validation
summary are retained in
`hexagon-4x2-default-build/results/historical-suite/` and
`hexagon-4x2-default-build/results/historical-suite-summary.json` beside the
checkout.


### Compiled-default correctness, October 12

A fresh Hexagon host and DSP v79 build of runtime commit `74861e5e` was tested
on the OnePlus 13s/SM8750. The same October CLI and shared CPU/OpenCL runtime
were used for fixed-code comparisons; the OpenCL-enabled reference loader was
recovered from the phone and its SHA-256 matches the historical manifest.
The core/CPU/OpenCL sources have no diff between `5687b0e7` and `e282e85b`.
Host, Android and DSP builds pass. All 29 host CPU-dequantized F32 oracle cases
and the host operation-cache regression pass.

On the device, all 29 oracle cases pass with 1, 2 and 6 HVX workers for each
of explicit 2x2, explicit 4x2 and the shape setting unset. Every canonical
result matches the same-build explicit 2x2 baseline byte for byte. The unset
setting selects `hvx-panel-4x2` for eligible DDR cases in the separate routing
capture. F32 arithmetic, VTCM precedence and excluded-layout fallbacks retain
their existing behavior.

All 25 fixed-code CPU/Hexagon/OpenCL boundary gates pass for 3 and 66 frames.
The compiled default and explicit 2x2 produce byte-identical PCM at both
lengths. Direct default Hexagon/OpenCL PCM cosine is 0.9999900834, NMSE
1.9834e-5. The host buffer/CPU-copy/repack device regression also passes.

Raw evidence and the independently validated fixed-code summary are retained
in `hexagon-4x2-default-build/results/default4x2-20261012b/` and
`hexagon-4x2-default-build/results/default-fixed-summary.json` beside the
checkout. Runtime hashes and the shared-runtime provenance are in
`hexagon-4x2-default-build/results/runtime-provenance-v3.json`.


An invalid shape setting warns and uses 4x2; all 29 device oracle outputs with
that setting and one worker match the compiled default exactly. A fresh greedy
S1 synthesis with the shape setting unset produces the same 66-frame code/WAV
SHA-256 values recorded above. The attempted new repeated S1 timing run was
stopped before synthesis while waiting for thermal status 0: the service kept
reporting status 1 with a cached `shell_skin` sample absent from current HAL
readings. No thermal override was applied. The single synthesis is correctness
evidence only; performance claims use the validated October 8 suite.

## Codec weight reuse and F32 residual epilogues (QVAC-26841)

The new routes are experimental and disabled by default; the compiled DDR
panel remains 4x2. Each optimization can be measured independently.

| Control | Values | Default |
| --- | --- | --- |
| `GGML_HEXAGON_F16_F32_BLOCKED` | `0`, `1` | `0` |
| `GGML_HEXAGON_F16_F32_BLOCKED_CHANNELS` | `8`, `16`, `32` | `16` |
| `GGML_HEXAGON_F16_F32_BLOCKED_ROWS` | `32`, `64` | `32` |
| `GGML_HEXAGON_F16_F32_BLOCKED_TRAVERSAL` | `channels`, `rows`, `matrix` | `channels` |
| `GGML_HEXAGON_F16_F32_FUSE_ADD` | `0`, `1` | `0` |

Blocked execution widens weights into per-worker, per-operation VTCM scratch
and retains F32 activations. `channels` scans the assigned rows for each weight
tile; `rows` repacks inside each activation block; `matrix` packs assigned
weights once and visits bounded activation blocks. Matrix scratch at six
workers is 216 KiB for K96/output96, 864 KiB for K192/output192 and 3,456 KiB
for K384/output384. Budget failures retain the fallback. Existing efficient
VTCM kernels retain precedence; no persistent weight cache is introduced.

Fusion adds a full-size contiguous F32 residual after the completed matmul
reduction and before the final store. It supports DDR4x2 without packing and
the blocked routes. Broadcast bias and unsupported layouts retain separate
ADD. Exact destination/residual reuse is safe; shifted overlaps and unsafe
input/output aliases are rejected. Existing consumer, view and output guards
protect observable matmul intermediates. A register barrier preserves the
product's F32 boundary before ADD; assembly confirms SF32 plus SF32, no FMA,
vector stack spills or reduction-loop helper calls. The production DDR4x2
function's 850 instruction lines match immutable baseline after relocation
addresses are normalized. A rounding fixture passes both earlier and final
artifacts; no numerical failure-before claim is made.

### Correctness and review

Host, Android and DSP v79 builds pass on the OnePlus 13s/SM8750 setup. The
expanded independent oracle has 48 cases, including protected outputs,
additional consumers, layouts, aliases and exact rounding. Default,
fusion-only and combined at 1/2/4/6 workers match immutable 4x2 byte for byte:
576 case runs and 612 materialized result dumps. Earlier reuse checks cover
all 24 tile/row/worker combinations for each traversal. Review fixed host
selection of odd-byte row pitches; a 193-byte F16 pitch regression confirms
DDR4x2 fallback. Device buffer/copy/repack and operation-cache tests pass.

At fixed 3/66/94 frames, reuse-only, fusion-only and combined all pass CPU and
OpenCL gates and produce identical baseline F32 PCM and PCM16 WAV bytes.
Worst Hexagon/CPU cosine is 0.9999965358, NMSE 6.9298e-6; worst Hexagon/OpenCL
cosine is 0.9999862050, NMSE 2.7591e-5. Repeated PCM is stable for these captures.
These codec checks do not establish universal cross-backend generation
equivalence. The separate 66-frame profile confirms 56 fused codec operations
for both fusion-only and combined. Shape-derived traffic is logical bytes,
not measured DDR bandwidth.

The existing model-backed harness passes at 3 frames. With cap 66 its CPU
trajectory reaches EOS at 26 and reports seven language-model numeric
threshold failures. Baseline and reuse have identical failure records, zero
token/chain mismatches and passing codec stages. This pre-existing language
model difference is outside the codec task; fixed-code tests supply actual
66/94-frame checks.

### Measurement status

The initial channel traversal was slower in its completed 8-channel/32-row
comparison. Matrix 16/64 synthetic medians are mixed and variable: K96 is
28.41 ms baseline versus 37.46 ms candidate; K192 69.40 versus 35.96 ms;
K384 48.63 versus 42.05 ms. These probes do not establish a codec or application
improvement. The new routes remain opt-in.

The full comparison uses 4x2, reuse-only, fusion-only and combined on S1-S5:
greedy seed 42, four CPU threads, six HVX workers, three warmups plus five
timed observations with rotating order. Profiling is off, OPSTAGE 3 computes
the work, artifacts/models are pinned, thermal status 0 is required before
timing, and codes/WAVs must match across variants and repeats. This final
suite is running on the reconnected OnePlus. No full-suite gain is claimed
until all measurements and artifact checks are complete.

Evidence beside the checkout: `hexagon-26841-build/results/` (including
`matrix-kernels-full-a/summary.json` and `fusion-fixed-final-b/summary.json`),
`hexagon-26841-build/assembly/fusion/provenance.json`, and frozen `stage/`
artifacts. Device raw results: `/data/local/tmp/qvac-26841/results/`.
Task: https://app.asana.com/1/45238840754660/project/1214153063536860/task/1219396967417382.
