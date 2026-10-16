# Hexagon runtime information

Applications that perform memory preflight can resolve the optional registry
procedure `ggml_backend_hexagon_get_runtime_info_v1`, using the typedef in
`ggml-hexagon.h`. The header is installed even when building without Hexagon.
Dynamic-backend consumers should resolve the procedure from the selected
device's registry, without directly linking the Hexagon module.

The call takes a Hexagon backend, the maximum node count of future application
graphs, an output struct and its exact size. It returns false for invalid
arguments, an unsupported node count (greater than `INT32_MAX`) or arithmetic
overflow, without changing the output. A zero node count requests the current
retained metadata estimate without a future graph allowance.

The execution-stage mask is retained at registry initialization. Both bits must
be set for real computation: bit 0 submits batches to the DSP and bit 1 enables
DSP computation. Diagnostic modes with either bit absent can return successful
graph status without producing the requested computation. Reading the current
`GGML_HEXAGON_OPSTAGE` environment variable cannot establish the effective mode
after registry initialization.

`host_bytes_bound` is a conservative metadata payload **estimate**, not a bound
on process RSS. It includes the actual allocated vector capacities of all
successfully initialized, registry-owned sessions: the builder descriptors and
pointer indices, retained graph nodes and their fusion lists, every queue cache
slot and profiling timestamps. A previously retained graph larger than the
requested future graph is therefore still counted. Queue-container payload
headroom is included because `std::queue` does not expose allocated capacity.

The prospective allowance includes new graph finalization and optimization
arrays, reordered indices, fusion lists, queue cache growth and builder fusion
lists. It is added to existing retained capacities, so an old graph and its
replacement can overlap. This estimate uses the at-most-two-times vector
capacity growth policy of supported libc++ and libstdc++ builds; the portable
test exercises actual node and queue-cache allocations up to 32768 nodes.
Android validation uses NDK r28b's libc++ with Clang 19. An alternative STL with
a larger capacity growth policy requires updating and validating this estimate.
No allocator headers, fragmentation or thread-stack allowances are included.

`shared_bytes` counts the actual RPC op-queue buffer size, including any
`PROFILE=3` trace area, and the requested DSP request/response queue sizes.
Each is rounded up to `allocation_page_bytes`. All successfully initialized
registry sessions are counted, including sessions with no live backend handles.
Vendor transport bookkeeping and private DSP/driver allocations remain opaque;
applications must retain a separate margin for those and allocator overhead.
The reported shared queue bytes and host metadata overlap in the same system
RAM pool: add them when estimating memory demand, rather than treating them as
separate memory capacities.

Model weights, graph arenas and application caches are excluded. For each
Hexagon tensor-buffer allocation, applications must separately add
`tensor_guard_bytes`, then round for `allocation_page_bytes` as appropriate.
`buffer_alignment` describes ggml tensor packing. `max_buffer_bytes` is the
effective tensor payload cap: the smaller of the configured preferred buffer
size and the DSP's 2 GiB individual mapping limit minus the tensor guard. Setting
`GGML_HEXAGON_MBUF` above 2 GiB does not bypass the hardware mapping limit.
`max_buffer_bytes` and
`max_batch_vmem_bytes` are separate limits on an individual allocation and the
buffer mappings used by one op batch. VTCM is on-chip DSP scratch space; it is
not an additional system RAM pool. Device memory reported as zero remains
unknown and should not be interpreted as zero usable memory.

The query serializes with graph computation and synchronization on each
registry session, taking one session mutex at a time. Different backend
handles sharing a session therefore cannot race on its retained graph/queue
metadata. Registry unloading and unrelated tensor-buffer destruction require
the caller's normal lifecycle synchronization. A snapshot does not reserve
memory or prevent later work from increasing retained state. Applications that
need the result to remain applicable must serialize their own subsequent
allocation and inference work and use graph sizes within the requested limit.

Validation:

```sh
cmake --build build --target test-hexagon-runtime-info test-hexagon-runtime-info-device
ctest --test-dir build -R '^test-hexagon-runtime-info' --output-on-failure
```

The first test runs without the SDK or a DSP. The second returns CTest skip code
77 if no Hexagon device is available; on a device it verifies malformed inputs,
non-Hexagon rejection, nonzero limits, prospective graph accounting and retained
configuration after environment changes. Run it in separate processes with
different initial `GGML_HEXAGON_NDEV`, `OPSTAGE`, `OPBATCH`, `OPQUEUE`, `PROFILE`
and `OPTRACE` settings to validate those initialized configurations.
