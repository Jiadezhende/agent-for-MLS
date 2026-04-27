# bank_conflict — Shared Memory Bank Conflict Penalty

## When to use

Use this skill when a target spec requests any of:
- `bank_conflict_penalty_cycles`
- `bank_conflict_penalty_x`
- `shmem_bank_conflict_ratio`
- shared memory conflict penalty in cycles or as a ratio

## Principle

NVIDIA GPUs (compute capability ≥ 2.0) have 32 shared memory banks, each 4
bytes wide. When all 32 threads in a warp access addresses that map to the
same bank (stride = 32 elements for 4-byte words), the accesses are serialized
into a 32-way conflict, multiplying latency by up to 32×.

Measure conflict-free access (stride 1) versus maximally conflicted access
(stride 32) using `clock64()`. The **penalty ratio** is the primary metric:

```
penalty_x = cycles_stride32 / cycles_stride1
```

This ratio is **dimensionless and clock-frequency-invariant** — it is the
same whether the GPU is locked at 500 MHz or running at 2700 MHz.

## CUDA kernel template

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Bank conflict probe.
// 32 threads access smem[(acc * stride) & 1023] in a loop.
// stride=1  → each thread hits a different bank (conflict-free)
// stride=32 → all 32 threads hit bank 0 (32-way conflict)
__global__ void bank_conflict_probe(
    int stride,
    uint64_t* out_cycles,
    uint32_t* out_sink
) {
    __shared__ uint32_t smem[1024];

    // Initialize shared memory
    smem[threadIdx.x]       = threadIdx.x;
    smem[threadIdx.x + 32]  = threadIdx.x + 32;
    smem[threadIdx.x + 64]  = threadIdx.x + 64;
    smem[threadIdx.x + 96]  = threadIdx.x + 96;
    __syncthreads();

    const int ITERS = 10000;
    uint32_t acc = threadIdx.x;

    uint64_t t0 = clock64();
    for (int i = 0; i < ITERS; ++i) {
        acc = smem[(acc * stride) & 1023];
    }
    uint64_t t1 = clock64();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = (t1 - t0) / (uint64_t)ITERS;
        out_sink[0] = acc;
    }
}

int main() {
    uint64_t* d_cycles;
    uint32_t* d_sink;
    cudaMalloc(&d_cycles, sizeof(uint64_t));
    cudaMalloc(&d_sink, sizeof(uint32_t));

    int strides[] = {1, 2, 4, 8, 16, 32};
    int n_strides = 6;

    // Global warmup
    bank_conflict_probe<<<1, 32>>>(1, d_cycles, d_sink);
    cudaDeviceSynchronize();

    uint64_t cycles_by_stride[6] = {0};
    for (int s = 0; s < n_strides; ++s) {
        // Per-stride warmup
        bank_conflict_probe<<<1, 32>>>(strides[s], d_cycles, d_sink);
        cudaDeviceSynchronize();
        // Timed run
        bank_conflict_probe<<<1, 32>>>(strides[s], d_cycles, d_sink);
        cudaDeviceSynchronize();
        cudaMemcpy(&cycles_by_stride[s], d_cycles,
                   sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("cycles_stride%d=%llu\n",
               strides[s], (unsigned long long)cycles_by_stride[s]);
    }

    // Find peak conflict stride
    uint64_t max_cyc = cycles_by_stride[0];
    int peak_stride = strides[0];
    for (int s = 1; s < n_strides; ++s) {
        if (cycles_by_stride[s] > max_cyc) {
            max_cyc = cycles_by_stride[s];
            peak_stride = strides[s];
        }
    }

    uint64_t c1  = cycles_by_stride[0];  // stride=1 (conflict-free baseline)
    uint64_t c32 = cycles_by_stride[5];  // stride=32 (max conflict)
    double penalty_x = (c1 > 0) ? (double)c32 / (double)c1 : 0.0;
    uint64_t penalty_abs = (c32 > c1) ? c32 - c1 : 0;

    printf("bank_conflict_penalty_x=%.2f\n", penalty_x);
    printf("bank_conflict_cycles=%llu\n",    (unsigned long long)penalty_abs);
    printf("cycles_stride1_baseline=%llu\n", (unsigned long long)c1);
    printf("cycles_stride32=%llu\n",         (unsigned long long)c32);
    printf("peak_conflict_stride=%d\n",      peak_stride);

    cudaFree(d_cycles);
    cudaFree(d_sink);
    return 0;
}
```

## How to invoke

```
run_cuda_probe(
    source = <above kernel>,
    probe_name = "bank_conflict",
    compile_flags = ["-O3"],
)
```

## Interpreting results

Parse `bank_conflict_penalty_x=<N>` (the ratio) as the primary metric and
`bank_conflict_cycles=<N>` (absolute extra cycles per access) as secondary.

Expected values for a 32-bank GPU (Turing, Ampere, Ada, Hopper, Blackwell):
- `cycles_stride1` ≈ 4–8 cycles (one-cycle shared memory read, amortized)
- `bank_conflict_penalty_x` ≈ 28–32× for stride=32 (32-way serialization)
- Intermediate strides scale linearly: stride=16 → ~16×, stride=8 → ~8×

**If `bank_conflict_penalty_x < 2.0` for stride=32:**
- `flag_event` type="unexpected_low_bank_conflict", severity="warn"
- Possible causes: (a) the hardware handles 32-way conflicts via warp scheduler
  rather than serialization on Blackwell; (b) compiler optimized away accesses
- Still report the measured ratio; do not fabricate an expected value

**If `bank_conflict_penalty_x > 33.0`:**
- `flag_event` type="unusually_high_bank_conflict", severity="warn"
- Extra overhead beyond pure serialization; still report

**If `cycles_stride1 == 0` (division by zero):**
- Retry with `ITERS = 100000` — the granularity of clock64() may be too coarse

**If `peak_conflict_stride` is not 32 (e.g., it is 16):**
- The GPU may use 8-byte bank width instead of 4-byte; stride 16 causes 32-way
  conflicts for 8-byte banks
- `flag_event` type="non_standard_bank_width", severity="info"
- Report `peak_conflict_stride` alongside the penalty

## ncu cross-verification

```
profile_with_ncu(
    source_type = "cuda_source",
    source_or_path = <kernel source>,
    kernel_name = "bank_conflict_probe",
    metrics = [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ]
)
```

For stride=32: `data_bank_conflicts_op_ld.sum` should be much larger than for
stride=1. If ncu shows zero bank conflicts for stride=32:
- `flag_event` type="bank_conflict_mechanism_differs", severity="info"
- Hardware may serialize differently without the legacy bank conflict counter

## Metric mapping

| stdout field                 | target_spec key                 | unit   |
|------------------------------|---------------------------------|--------|
| `bank_conflict_penalty_x`    | `bank_conflict_penalty_cycles`  | ratio  |
| `bank_conflict_cycles`       | `bank_conflict_absolute_cycles` | cycles |
| `cycles_stride1_baseline`    | `shmem_access_latency_cycles`   | cycles |

## Environment fallback

If `run_cuda_probe` returns `{"error": "binary_not_found", "name": "nvcc"}`:
- `flag_event` type="no_cuda_toolchain", severity="error"
- `record_measurement` value="unavailable", confidence=0.0
- `submit_results` immediately
