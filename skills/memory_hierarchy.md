# Memory Hierarchy Probing

## Purpose

Measure latency and capacity properties of the GPU memory hierarchy using CUDA
microbenchmarks that defeat prefetching and expose cache boundaries.

## When to use

Use this skill for targets such as:

- `l1_latency_cycles`, `l1_latency_ns`
- `l2_latency_cycles`, `l2_latency_ns`
- `dram_latency_cycles`, `dram_latency_ns`
- `l2_cache_capacity_bytes`, `l2_cache_size_mb`
- requests that mention cache latency, DRAM latency, or cache capacity

## Primary CUDA strategy

Use pointer chasing for latency. Build a randomized linked list in device memory
and run one dependent load per iteration:

```cuda
idx = arr[idx];
```

Because each address depends on the previous load, the hardware cannot coalesce
the timed operation, hide it with memory-level parallelism, or rely on simple
stride prefetching. Time the chase with `clock64()` inside the kernel and print
a small, parseable stdout payload.

Recommended working sets:

- L1: 16 KB to 48 KB, one block, one timed lane.
- L2: several MB, below the expected L2 size when known; otherwise sweep.
- DRAM: at least 128 MB, and increase to 256 MB or 512 MB if latency is too low.

For L2 capacity, sweep working-set sizes and look for the first sustained
latency cliff. Use powers of two or a dense range around the suspected cliff.

## Output fields

Print one key per line:

```text
target=l2_latency_cycles
latency_cycles=184
array_bytes=8388608
iters=4096
```

For a capacity sweep:

```text
size_bytes=1048576 latency_cycles=130
size_bytes=2097152 latency_cycles=139
size_bytes=4194304 latency_cycles=151
size_bytes=8388608 latency_cycles=260
l2_cache_capacity_bytes=4194304
```

## Metric mapping

| stdout field | target key | unit |
| --- | --- | --- |
| `latency_cycles` from small working set | `l1_latency_cycles` | cycles |
| `latency_cycles` from L2-resident working set | `l2_latency_cycles` | cycles |
| `latency_cycles` from large working set | `dram_latency_cycles` | cycles |
| `l2_cache_capacity_bytes` | `l2_cache_capacity_bytes` | bytes |
| `l2_cache_capacity_bytes / 1048576` | `l2_cache_size_mb` | MB |

## Cross-verification

Use `profile_with_ncu` only after a successful `run_cuda_probe` and pass the
returned `binary_path`. Useful metrics include:

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `lts__t_sectors_srcunit_tex_op_read.sum`
- `dram__sectors_read.sum`
- `sm__cycles_elapsed.avg`

If ncu cannot profile the kernel or reports missing metrics, keep the
microbenchmark result but lower confidence and flag a `data_quality` event.

## Anomaly signals

- DRAM latency below 200 cycles usually means the array did not exceed L2.
- L1 latency greater than L2 latency indicates timing overhead or wrong working set.
- A capacity cliff with a single noisy point is not enough; repeat around the cliff.
- High variance across trials suggests clock instability; record confidence below 0.85.
- Zero cycles, missing writes, or `ncu_no_kernel_found` from a first-attempt kernel
  are most likely caused by a kernel that never launched or a mismatched GPU
  architecture. Before changing the timing approach or adding a fallback timer,
  add a `cudaGetLastError` / `cudaDeviceSynchronize` after every kernel launch
  and confirm the kernel ran. If the error message mentions an unsupported PTX
  instruction or an invalid device function, the architecture flag is wrong;
  check that no `-arch` is hardcoded (see gpu_profiling_overview preflight rule)
  and that the Executor-injected flag matches the physical GPU.

## Failure fallback

- `user_code`: fix compiler diagnostics and retry with a simpler kernel.
- `timeout`: reduce sweep count or iterations, then retry once.
- `infrastructure`: flag `no_cuda_toolchain` or `ncu_unavailable`; do not retry
  the same tool repeatedly.
- `data_quality`: fix kernel name or metric name before another ncu call.
