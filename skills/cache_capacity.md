# cache_capacity — L2 Cache Capacity and Max Shared Memory Per Block

## When to use

Use this skill when a target spec requests any of:
- `l2_cache_capacity_bytes`
- `l2_cache_size_mb`
- `max_shmem_per_block_kb`
- `max_shared_memory_per_block_kb`

## Principle

**L2 capacity**: Run the pointer-chasing kernel (same as `memory_latency`) over
a sweep of array sizes doubling from 512 KB to 64 MB. Latency stays low while
the array fits in L2; it jumps sharply (>2×) once the array exceeds L2. The
cliff location is the L2 capacity. The ratio-based detection is
clock-frequency-invariant: both sizes at the cliff use the same (possibly
locked) clock, so the relative jump is unaffected.

**Max shared memory per block**: Use `cudaFuncSetAttribute` with
`cudaFuncAttributeMaxDynamicSharedMemorySize` to request increasing amounts of
dynamic shared memory, then actually launch a kernel that reads that memory.
The last successful size is `max_shmem_per_block_kb`. Do NOT use
`cudaGetDeviceProperties().sharedMemPerBlockOptin` — that API call may be
intercepted or report misleading values in the test environment.

## CUDA kernel template

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ── Pointer-chasing kernel (same principle as memory_latency skill) ───────

__global__ void pointer_chase(
    uint32_t* __restrict__ arr,
    uint32_t N,
    uint64_t* __restrict__ out_cycles,
    int iters
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t idx = 0;
    for (int w = 0; w < 16; ++w) idx = arr[idx];  // warmup

    uint64_t t0 = clock64();
    for (int i = 0; i < iters; ++i) idx = arr[idx];
    uint64_t t1 = clock64();

    out_cycles[0] = (t1 - t0) / (uint64_t)iters;
    if (idx == 0xDEADBEEF) out_cycles[0] = 0;
}

static void build_random_chain(uint32_t* arr, uint32_t N) {
    for (uint32_t i = 0; i < N; ++i) arr[i] = i;
    for (uint32_t i = N - 1; i > 0; --i) {
        uint32_t j = rand() % (i + 1);
        uint32_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

// ── Shared memory probe (dynamic allocation) ─────────────────────────────

__global__ void shmem_probe(float* out, size_t n_floats) {
    extern __shared__ float smem[];
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < n_floats; i += blockDim.x)
        acc += smem[i];
    if (threadIdx.x == 0) out[0] = acc;
}

int main() {
    // ── Part A: L2 capacity sweep ─────────────────────────────────────────
    size_t sweep_bytes[] = {
        512ULL  * 1024,
        1024ULL * 1024,
        2ULL    * 1024 * 1024,
        4ULL    * 1024 * 1024,
        8ULL    * 1024 * 1024,
        16ULL   * 1024 * 1024,
        32ULL   * 1024 * 1024,
        64ULL   * 1024 * 1024,
    };
    int n_sweep = 8;
    int iters = 512;

    uint64_t* d_out;
    cudaMalloc(&d_out, sizeof(uint64_t));

    uint64_t prev_cycles = 0;
    size_t l2_capacity_bytes = 0;
    for (int s = 0; s < n_sweep; ++s) {
        size_t bytes = sweep_bytes[s];
        uint32_t N = (uint32_t)(bytes / sizeof(uint32_t));
        uint32_t* h = (uint32_t*)malloc(bytes);
        build_random_chain(h, N);
        uint32_t* d_arr;
        cudaMalloc(&d_arr, bytes);
        cudaMemcpy(d_arr, h, bytes, cudaMemcpyHostToDevice);
        free(h);

        pointer_chase<<<1, 1>>>(d_arr, N, d_out, iters);
        cudaDeviceSynchronize();
        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("sweep_latency_cycles=%llu array_bytes=%zu\n",
               (unsigned long long)cycles, bytes);

        // Detect cliff: current cycles > 2x previous
        if (prev_cycles > 0 && cycles > prev_cycles * 2 && l2_capacity_bytes == 0) {
            l2_capacity_bytes = sweep_bytes[s - 1];  // last size still in L2
        }
        prev_cycles = cycles;
        cudaFree(d_arr);
    }
    // If no cliff found in sweep, report as undetected
    if (l2_capacity_bytes == 0) {
        printf("l2_capacity_bytes=0\n");
        printf("l2_cliff_not_found=1\n");
    } else {
        printf("l2_capacity_bytes=%zu\n", l2_capacity_bytes);
    }
    cudaFree(d_out);

    // ── Part B: Max shared memory per block ───────────────────────────────
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    size_t test_kb[] = {48, 64, 96, 100, 128, 160, 192, 228};
    int n_tests = 8;
    int max_ok_kb = 0;

    for (int t = 0; t < n_tests; ++t) {
        size_t kb    = test_kb[t];
        size_t bytes = kb * 1024;

        // Request extended carveout; failure means limit exceeded
        cudaError_t attr_err = cudaFuncSetAttribute(
            shmem_probe,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)bytes
        );
        if (attr_err != cudaSuccess) {
            cudaGetLastError();  // clear error
            printf("shmem_test_kb=%zu status=attr_fail\n", kb);
            break;
        }
        shmem_probe<<<1, 256, bytes>>>(d_result, bytes / sizeof(float));
        cudaError_t launch_err = cudaGetLastError();
        cudaDeviceSynchronize();
        if (launch_err != cudaSuccess) {
            cudaGetLastError();
            printf("shmem_test_kb=%zu status=launch_fail\n", kb);
            break;
        }
        printf("shmem_test_kb=%zu status=ok\n", kb);
        max_ok_kb = (int)kb;
    }
    printf("max_shmem_per_block_kb=%d\n", max_ok_kb);

    cudaFree(d_result);
    return 0;
}
```

## How to invoke

```
run_cuda_probe(
    source = <above kernel>,
    probe_name = "cache_capacity",
    compile_flags = ["-O3"],
)
```

## Interpreting results

**L2 capacity:** Parse the `sweep_latency_cycles` / `array_bytes` pairs and look
for the cliff printed as `l2_capacity_bytes=<N>`. If `l2_cliff_not_found=1`,
extend the sweep to 128 MB by modifying `sweep_bytes` and re-running.

Typical GPU L2 sizes: 4–80 MB (RTX 5060 is ~24 MB per SM chiplet).

**Max shmem per block:** Parse `max_shmem_per_block_kb=<N>`. Typical CUDA values:
- Default limit: 48 KB (applies when extended carveout is not requested)
- Extended (Turing+): up to 96 KB
- Ampere+: up to 164 KB
- Blackwell: up to 228 KB

If `max_shmem_per_block_kb < 48`:
- `flag_event` type="shmem_below_minimum", severity="error"
  (impossible on a CUDA-capable GPU; likely compilation or launch error)

If `l2_cliff_not_found=1`:
- `flag_event` type="l2_cliff_not_found", severity="warn"
- Retry with sweep extended to 128 MB

## ncu cross-verification (L2 cliff)

```
profile_with_ncu(
    source_type = "cuda_source",
    source_or_path = <kernel source>,
    kernel_name = "pointer_chase",
    metrics = [
        "l1tex__t_sector_hit_rate.pct",
        "l2__t_sector_hit_rate.pct",
    ],
)
```

Run at `array_bytes = l2_capacity_bytes` (cliff boundary). L2 hit rate should
be ~50% at the boundary; < 10% for arrays 2× larger than L2.

If ncu `l2__t_sector_hit_rate.pct > 90%` but measured latency is DRAM-level:
- `flag_event` type="l2_hit_rate_mismatch", severity="warn"
- SM masking may be limiting the effective cache

## Metric mapping

| stdout field               | target_spec key             | unit  |
|----------------------------|-----------------------------|-------|
| `l2_capacity_bytes`        | `l2_cache_capacity_bytes`   | bytes |
| `l2_capacity_bytes / 1048576` | `l2_cache_size_mb`       | MB    |
| `max_shmem_per_block_kb`   | `max_shmem_per_block_kb`    | KB    |

## Environment fallback

If `run_cuda_probe` returns `{"error": "binary_not_found", "name": "nvcc"}`:
- `flag_event` type="no_cuda_toolchain", severity="error"
- `record_measurement` value="unavailable", confidence=0.0
- `submit_results` immediately
