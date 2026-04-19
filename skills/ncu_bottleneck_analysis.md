# ncu_bottleneck_analysis — Roofline & Bottleneck Diagnosis via Nsight Compute

## When to use

Use this skill when:
- A target spec requests `bottleneck_type`, `compute_utilization_pct`, or
  `memory_utilization_pct`
- A prior measurement (bandwidth, latency, clock) shows unexpectedly low results
  and you need to understand why
- You need to verify the evaluation environment has not been tampered with
  (clock throttling, SM masking, API spoofing)

This skill requires **no new CUDA kernel**. It instructs you to run
`profile_with_ncu` on an existing kernel and interpret the counters.

## Step 1 — Roofline classification

Fetch these two throughput metrics first:

| ncu metric | Meaning | Compute-bound signal |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM compute utilization | > 80% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Combined memory throughput | > 80% |

**Classification rules:**

```
Compute > 80% and Memory < 50%  →  compute-bound
Memory  > 80% and Compute < 50% →  memory-bound (VRAM/DRAM bound)
Both > 70%                       →  roofline knee (balanced; near peak efficiency)
Both < 50%                       →  latency-bound (check occupancy and parallelism)
```

## Step 2 — Detailed bottleneck diagnosis

Fetch the full metric set to identify the specific sub-bottleneck:

```
profile_with_ncu(
    source_type = "cuda_source",
    source_or_path = <any measurement kernel, e.g. dram_bw_kernel>,
    kernel_name = <kernel_name>,
    metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__maximum_warps_per_active_cycle_pct",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "sm__sass_branch_targets_threads_diverged.sum",
        "sm__sass_branch_targets.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
        "sm__cycles_elapsed.avg.per_second",
    ]
)
```

**Bottleneck diagnosis table:**

| Bottleneck type | Key signals | Optimization direction |
|---|---|---|
| VRAM bound | `dram__throughput > 70%` | Reduce global memory traffic; use shared memory; improve data reuse |
| L2 bound | `l2__throughput > 80%`, low `dram__throughput` | Improve cache locality; tile access patterns |
| Compute bound (Tensor) | `sm__throughput > 80%`, `tensor_op_hmma > 50%` | Algorithmic improvements; reduce precision (FP32→FP16/BF16) |
| Compute bound (FP32) | `sm__throughput > 80%`, low `tensor_op` | Enable Tensor Core use; reduce arithmetic intensity |
| Uncoalesced access | sectors_per_request > 2 | Fix memory layout; ensure adjacent threads access adjacent addresses |
| Low occupancy | `warps_active < 25%` | Reduce register/shared memory usage per thread; adjust launch bounds |
| Warp divergence | `diverged/total_branches > 0.1` | Reduce conditional branches in kernel |
| Bank conflicts | `data_bank_conflicts > 0` | Add padding to shared memory arrays |

**Computing derived metrics:**
```
sectors_per_request = l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
                    / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  → > 2.0 indicates poor memory coalescing

divergence_ratio = sm__sass_branch_targets_threads_diverged.sum
                 / sm__sass_branch_targets.sum
  → > 0.1 indicates > 10% divergent branches
```

## Step 3 — Environment tamper detection

**Clock throttle detection:**
```
ncu_clock_mhz = sm__cycles_elapsed.avg.per_second / 1e6
if |ncu_clock_mhz - actual_boost_clock_mhz| / actual_boost_clock_mhz > 0.10:
    flag_event type="clock_throttled_during_ncu"
    Note: ncu results reflect throttled state; measurements are valid but
    represent throttled conditions, not full-boost performance.
```

**SM masking detection:**
```
If sm__warps_active.avg.pct_of_peak_sustained_active is unexpectedly low
(< 10% for a kernel that should achieve > 50% occupancy), suspect SM masking.
flag_event type="sm_masking_suspected", severity="warn"
record_measurement with confidence=0.7
```

**API spoofing cross-check:**
```
If ncu dram__throughput > 95% but the measured peak_dram_bandwidth_GBps
from the memory_bandwidth skill is < 50 GB/s:
    flag_event type="api_spoofed_throughput_mismatch"
    Trust the bytes/wall_time measurement over the ncu percentage.
    The ncu percentage may be relative to a spoofed "theoretical peak".
```

## Step 4 — Record results

After analysis, record the bottleneck determination:

```
record_measurement(
    metric = "bottleneck_type",
    value  = "<one of: compute_bound, memory_bound_dram, memory_bound_l2,
               latency_bound, roofline_knee>",
    unit   = null,
    confidence = 0.85,
    method = "Roofline: sm__throughput=<X>%, memory_throughput=<Y>%",
    evidence = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed: <X>",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: <Y>",
        "<other relevant metrics>"
    ]
)
```

## Metric mapping

| derived metric | target_spec key | unit |
|---|---|---|
| Roofline label | `bottleneck_type` | string |
| `sm__throughput` | `compute_utilization_pct` | % |
| `gpu__compute_memory_throughput` | `memory_utilization_pct` | % |
| `sectors_per_request` | `access_coalescing_factor` | sectors/request |
| `divergence_ratio` | `warp_divergence_ratio` | ratio |

## Environment fallback

If `profile_with_ncu` returns `{"error": "ncu_not_found"}`:
- `flag_event` type="ncu_not_available", severity="warn"
- Fall back to `run_cuda_probe` with a self-instrumented kernel that measures
  wall time and cycle counts directly to derive partial roofline data
- Record results with confidence=0.6 and note the fallback in method field
