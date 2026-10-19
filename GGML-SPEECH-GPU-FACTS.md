# ggml-speech GPU structural facts ledger

Cross-task facts about GPU backends in this tree and the engines that consume it.
Check each fact against the current target before relying on it; append new facts
with evidence (file:line or measured output). Do not record hypotheses here — only
verified mechanisms.

## Engine dispatch (qvac-ext-lib-whisper.cpp/engines/audiogen)

- All AceSTEP stages run DIRECT `ggml_backend_graph_compute` on one backend
  (dit_ggml.cpp, lm_ggml.cpp, textenc_ggml.cpp, cond_ggml.cpp, detok_ggml.cpp).
  `supports_op` is consulted only by the flash-attention probe (dit_ggml.cpp)
  and by the VAE scheduler ([GPU,CPU], op_offload=false, vae_ggml.cpp) which
  exists for progress/cancel callbacks. Scheduler-based fallback reasoning
  applies ONLY to the VAE stage.
- Stage placement is an allowlist (stage_placement.h resolve_stage_placement):
  detok/DiT/VAE/encoders run on the GPU; the LM runs on the GPU on Metal, OpenCL,
  and per-device on Vulkan (Mesa RADV devices, validated on Strix Halo against the
  F32-dequant reference), and on the CPU on every other Vulkan device (Mali-G715
  collapse observation) and on CUDA. ROCm/MUSA are not validated backends
  (backend_registry.h); unit tests pin these rules (test_acestep_units.cpp).
- Default memory mode loads and frees stage weights per generation
  (engine.cpp:1114); ACESTEP_KEEP_STAGES=1 keeps them resident.
- Hexagon runs only on an explicit `backend = "hexagon"` request; the automatic
  GPU walk skips the HTP registry (backend_registry.h gpu_tier_for). On HTP the
  detokenizer and encoders stay on the NPU and the LM on CPU unless
  `lm_backend` names another device (stage_placement.h, engine_backends.h).

## ggml-hexagon on Snapdragon 8 Elite (Galaxy S25, Hexagon v79 HTP0)

- HTP computes only on its own session buffers: supports_op rejects any src in a
  foreign buffer (ggml-hexagon.cpp ggml_hexagon_supported_buffers), so AceSTEP
  stages upload weights instead of mapping the GGUF (dit_gguf.cpp
  dit_gguf_backend_maps_weights). Quantized MUL_MAT also requires src0 in the
  repack buffer type exposed through `ggml_backend_dev_get_extra_bufts`
  (ggml_hexagon_supported_mul_mat).
- set/get_tensor repack Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 whole-tensor only:
  GGML_ASSERT(offset == 0) (ggml-hexagon.cpp:944-962). Row-range reads or
  uploads of a quantized weight (fused q|k|v blocks, a tied-head row slice)
  abort.
- With GGML_HEXAGON_HOSTBUF=1 (default) the default buffer type reports is_host
  while still storing quantized tensors repacked, so ggml_backend_tensor_copy
  memcpy's repacked bytes to a CPU copy. test-backend-ops CPU references for
  quantized ops on HTP0 therefore read NaN; run it with GGML_HEXAGON_HOSTBUF=0.
  A Q8_0 set/get round trip itself is byte-exact up to [1024, 217204].
- MUL_MAT refuses src0->ne[1] > 32768 ("refuse the lm-head"), so the AceSTEP LM
  head (151669/217204 rows) cannot run on HTP0 and the LM stays on CPU or GPU.
  Lifting the cap passes test-backend-ops once but the perf loop aborts with
  `dspqueue_read failed: 0x2e`.
- HMX multiplies in F16 and used to ignore GGML_PREC_F32; a MUL_MAT that asks
  for F32 precision now stays on HVX (ggml_hexagon_matmul_is_hmx_eligible), which
  fixes the prec_f32=1 bias-epilogue cases. The quantized b_absmax=1e5 stress
  cases still return NaN (14 MUL_MAT failures, the same on the base tip).
- Cache maintenance between the ops of one batch (htp-tensor.c lazy dirty
  ranges) had two holes, both invisible to single-op test-backend-ops cases and
  hidden by GGML_HEXAGON_OPBATCH<=4: per-line Q6_dccleaninva did not make an
  op's output visible to the next op (Supertonic's duration REPEAT read stale
  data; hex_l2flush now uses the QuRT range clean), and only inputs were
  flushed, so dirty lines of an older tensor in reused memory were written back
  over a DMA-written output (Supertonic text encoder off by 2%; outputs are now
  flushed before each op too).
- From v75 a 2D DMA descriptor holds 24-bit strides and row sizes and a 16-bit
  row count, and dma_queue_push hands every transfer to one descriptor, so a
  larger value is truncated (v73's dma_queue_push splits instead). On v79 a
  stride of exactly 2^24 still lands (the field wraps to 0, which the engine
  treats as 2^24), 2^24 + 128 aborts the DSP queue (`dspqueue_read failed:
  0x2e`) and 24 or 32 MiB read or write the wrong rows. The VTCM fast paths copy
  through dma_queue_copy_rows, which falls back to one 1D descriptor per row;
  an ACE-Step VAE output row passes 16 MiB beyond about 87 s of 48 kHz audio.
- ggml_can_fuse requires every fused node to have the same shape, so a
  shape-changing fusion such as IM2COL+MUL_MAT (depthwise conv1d) checks the use
  count itself. The allocator may place that MUL_MAT's output over the IM2COL
  input, so the fused kernel stages every channel in VTCM before writing.
- F32 transposes (CONT of a transposed view, single-tap IM2COL) ran element by
  element at about 0.4 GB/s; they now go through VTCM tiles and word gathers
  (transpose-ops.c). HMX pads a partial 32-row tile for F32 weights as for F16;
  the padding rows come from the VTCM weight buffer, not from DDR.
- Measured Supertonic 3 q8_0 stage parity vs CPU (whole graphs, identical
  inputs): duration 0.9999999, text encoder 0.9999998, vector estimator 0.99995,
  vocoder 0.9999996; 2.0-2.2x faster than Adreno OpenCL end to end.
- Binary ops staged whole rows in VTCM and returned VTCM-TOO-SMALL once a row
  exceeded the per-thread budget (an Oobleck VAE row is up to 345600 floats);
  rows that do not fit now stream from DDR (binary-ops.c execute_op_binary_direct).
- The fused MUL_MAT+ADD matvec added the bias with an aligned HVX load at
  vtcm_src2 + src0_start_row; per-thread start rows are not 32-aligned
  (2048 / 6 threads), so threads 1-5 added a shifted bias (matmul-ops.c:494,
  :1269, regression test_matvec_bias). This was the AceSTEP detokenizer's
  0.987 cosine.
- Measured AceSTEP stage parity vs CPU (identical inputs): detok 0.99999,
  text encoder 0.99941, cond 0.99947, DiT 0.989, VAE 0.999992 (Adreno OpenCL on
  the same phone: 0.99991 / 0.99913 / 0.99924 / 0.958 / 0.999992).

## ggml-opencl on Adreno 830 (Galaxy S25)

- The F16 x F32 KQ/KQV image kernels (`ggml_cl_mul_mat_kq_kqv_adreno`, run for 32+
  query tokens) tile K by 16 and M by 64 with no tail, so a KV length or head size
  off those tiles gave wrong results: `test-backend-ops -o MUL_MAT -p kqv=` measured
  NMSE 0.020-0.029 for KQ and 1.47 for KQV at n_kv 181/437, and 0.38 for a KQV with
  head size 80. Shared (grouped-query) KV heads are not the cause; aligned cases
  pass with 16 query heads over 8 KV heads. The dispatch now declines
  `ne00 % 16 != 0` and `ne01 % 64 != 0`, which leaves those shapes on the generic
  GEMM. MOSS-Transcribe-Diarize prefill on the unfused graph went from cosine
  0.844 to 0.99988 against the Hugging Face logits.

## ggml-vulkan on AMD Strix Halo (Radeon 8060S, RADV GFX1151, Mesa 25.2.8)

- Runtime device line: `uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory:
  65536 | int dot: 0 | matrix cores: KHR_coopmat`. Reported device memory = GTT
  (~116 GiB); VRAM carve-out is 2 GiB.
- Arch detection maps GFX1151 to AMD_RDNA3 (ggml-vulkan.cpp:343-345); there is no
  RDNA3.5/RDNA4 distinction.
- RADV gets KHR coopmat unconditionally (ggml-vulkan.cpp:17317-17322); coopmat2 is
  VK_NV-only. Default subgroup size reported by RADV is 64 (wave64).
- gpu_pipeline_configs (:3400-3415) pins subgroup sizes only for RDNA1/RDNA2;
  AMD_RDNA3 is unpinned (get_subgroup_size returns 0).
- The RDNA occupancy-limiting shmem workaround (:3151-3165) is annotated "guessed,
  tested on RDNA2".
- UMA devices force prefer_host_memory=true AFTER the env read (:5487-5492); the
  GGML_VK_PREFER_HOST_MEMORY env is presence-checked, so "off" requires a code
  change. Buffer allocation order for prefer_host_memory: HostVisible|HostCoherent
  first, DeviceLocal fallback (:2963-2975). Measured only on Samsung Xclipse 920
  before this campaign.
- The DiT sets GGML_PREC_F32 only on flash attention (engines/audiogen
  dit_ggml.cpp:340, 356). On coopmat devices, FA_COOPMAT1 requires f32acc coopmat
  support or the FA path silently becomes FA_SCALAR (:3244-3252).
- ggml_vk_buffer_from_host_ptr is implemented (:17101-17124, requires
  VK_EXT_external_memory_host) but device caps advertise buffer_from_host_ptr=false
  (:16417).
- Ubuntu 25.10 system glslc does not support GL_EXT_integer_dot_product (cmake
  feature probe), so integer-dot Vulkan shaders are not built and the device line
  shows `int dot: 0` even though the hardware is RDNA3.
- Verified coopmat configuration table (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR,
  Mesa 25.2.8): 16x16x16 subgroup scope only; f16xf16 with C/R f16 AND C/R f32
  (f32acc IS supported, so GGML_PREC_F32 flash attention can use FA_COOPMAT1);
  full int8 set (u8/s8 x u8/s8 -> s32/u32, with saturating variants). No bf16
  coopmat. The int8 coopmat hardware capability is currently unused by
  ggml-vulkan shaders.
- FIXED on this branch: test-backend-ops IM2COL_3D used to crash with
  GGML_ASSERT (compute workgroup count exceeded maxComputeWorkGroupCount[1] =
  65535 on RADV for large-OW cases) because the elements switch clamped only
  the z dimension and the shader had no y grid-stride loop. ggml-cuda received
  grid-striding fixes for the analogous overflow (im2col/pad); ggml-vulkan's
  im2col_3d dispatch was not covered until now. Was not part of any AceSTEP
  graph (VAE is 1D).
- Per-op GPU timing: GGML_VK_PERF_LOGGER=1 (+_CONCURRENT, _FREQUENCY). Memory
  placement audit: GGML_VK_MEMORY_LOGGER=1. Persistent pipeline cache:
  GGML_VK_PIPELINE_CACHE_DIR.
- SNAKE and COL2IM_1D have implementations on CPU, CUDA (inherited by HIP), Vulkan,
  Metal, OpenCL. No ACE-Step op participates in Vulkan fusion rules.
- Vulkan matmul pipelines come in three device classes: coopmat (fp16 matrix
  cores), scalar with device fp16 (base shader variants stage tiles through fp16
  shared memory even for the f32acc pipelines), and scalar without fp16 (base
  pipelines built from the _fp32 shader variants). GGML_PREC_F32 therefore needs
  the separate pipeline_dequant_mul_mat_mat_fp32 family on the first TWO classes;
  gating it on coopmat alone left every non-coopmat fp16 device (e.g. AMD Raphael
  iGPU on RADV, `fp16: 1 | matrix cores: none`) NaN-ing on quantized MUL_MAT with
  activations past 65504 (fixed on this branch). test-backend-ops covers this via
  MUL_MAT b_absmax=1e5 cases; the n=1 cases pass regardless because the mmv path
  differs.
- RADV does not expose KHR coopmat on the Raphael iGPU (Mesa 26.0.8): the
  "RADV gets KHR coopmat unconditionally" fact from Strix Halo (Mesa 25.2.8,
  RDNA3.5) does not generalize to small RDNA2 iGPUs.

## Strix Halo memory-access facts (measured on the AceSTEP optimization campaign)

- Linear streaming (big elementwise ADD, 393 MB): ~208 GB/s. Small-segment
  strided access collapses: 128-256 B segments at multi-KiB stride run at
  2.6-22 GB/s — roughly 10-40x below linear streaming. Any kernel whose
  per-workgroup global accesses are narrow strided segments is
  pattern-bound, not bandwidth-bound, once data spills the ~32 MiB MALL.
- Fixes that worked: LDS-staged tiles with contiguous slab loads
  (col2im_1d_tiled.comp: 5-22 -> 88-146 GB/s; copy_transpose_large.comp:
  2.6-5 -> 150-195 GB/s). Thread count matters as much as segment width
  (256 -> 512 -> 1024 threads roughly doubled throughput twice).
- prefer_host_memory ON vs OFF made NO difference to these patterns on Strix
  (H3b measured); the Xclipse-motivated UMA default stays.
- A slow kernel batched by test-backend-ops perf can exceed the ~10 s
  watchdog and kill the GPU context ("context is lost... hard recovery") —
  a perf-harness artifact; distinguish it from a real hang before concluding
  anything from such a run.
- Vulkan per-graph submit overhead: ~4.5 ms host per ~1100-node LM decode
  graph (encode + fence) vs ~5.3 ms GPU compute; graph/gallocr REBUILD cost
  is minor by comparison (LMGraphCache in the audiogen engine removed it;
  wall barely moved). Command-buffer replay would be the next lever.
- IM2COL dst-binding trap: on devices with BDA + int64, ggml_vk_op_f32 binds
  the im2col dst descriptor as a 1-BYTE DUMMY (the stock shaders write via
  buffer device address). Any new non-BDA pipeline dispatched under
  GGML_OP_IM2COL/IM2COL_3D silently writes into the dummy and produces
  garbage with no error; the real dst must be bound for such pipelines
  (see the tiled_1d exception at the IM2COL dispatch branch).
- PRE-EXISTING switch-label hazard in ggml_vk_op_f32's elements switch:
  splitting an op out of a shared case-label list silently reroutes the
  remaining labels if the new case is appended at the list tail (cost: lost
  element overrides, e.g. cpy_transpose tiles). Verify the intended branch
  fires with a print after any such split.

## HIP/ROCm on this machine

- ROCm 7.2 (HIP 7.2.53211) targets gfx1151 natively; 40 CUs. ggml-hip globs the
  whole ggml-cuda source tree (src/ggml-hip/CMakeLists.txt:63) including the
  ACE-Step kernels. gfx1151 is GGML_CUDA_CC_RDNA3_5 (common.cuh:76);
  amd_wmma_available=true (:315), amd_mfma_available=false.
- LavaSR ops (GRU, ZERO_UPSAMPLE, CHANNEL_SHUFFLE, AFFINE_PRELU) and the five
  SUPERTONIC ops have no CUDA/HIP kernels -> CPU fallback on a HIP build. Not in
  ACE-Step graphs.

## Build/runtime environment traps

- Shared-lib install prefix (~/ggml-install/<flavor>/lib) must be on
  LD_LIBRARY_PATH for audiogen binaries and ctest; missing path fails with
  "libqvac-speech-ggml-cpu.so.0: cannot open shared object file".
