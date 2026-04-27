# memory_bandwidth — Peak DRAM and Shared Memory Bandwidth

## When to use

Use this skill when a target spec requests any of:
- `peak_dram_bandwidth_GBps`
- `peak_shmem_bandwidth_TBps`
- `dram_bandwidth_gb_s`
- memory throughput in GB/s or TB/s

## Principle

**DRAM bandwidth**: Drive all GPU threads with a large coalesced streaming
read+write over an array that exceeds all cache levels (>256 MB). Measure
wall time with `cudaEvent`; bytes transferred = 2 × N × sizeof(float).
Clock locking does not affect bandwidth because it is bytes/wall_time, not
cycles/wall_time. Run 5 iterations; take the median to avoid DVFS ramp-up
artifacts.

**Shared memory bandwidth**: Each block loads a tile into shared memory then
performs 256 repeated reads from it. Total operations = blocks × 256 × tile
bytes. Wall time from `cudaEvent` gives TB/s.

## CUDA kernel template

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ── DRAM streaming bandwidth ──────────────────────────────────────────────

__global__ void dram_bw_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    size_t N
) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (size_t i = idx; i < N; i += stride) acc += in[i];
    for (size_t i = idx; i < N; i += stride) out[i] = acc;
}

// ── Shared memory bandwidth ───────────────────────────────────────────────

__global__ void shmem_bw_kernel(
    float* __restrict__ global_buf,
    size_t N
) {
    __shared__ float smem[256];
    size_t base = (size_t)blockIdx.x * blockDim.x;
    if (base + threadIdx.x < N)
        smem[threadIdx.x] = global_buf[base + threadIdx.x];
    __syncthreads();
    float acc = smem[threadIdx.x];
    // 256 repeated reads from shared memory (prevents DCE; varies index)
    #pragma unroll 32
    for (int r = 0; r < 256; ++r)
        acc += smem[(threadIdx.x + r) & 255];
    if (base + threadIdx.x < N)
        global_buf[base + threadIdx.x] = acc;
}

// ── Helper: median of 5 floats ────────────────────────────────────────────

static float median5(float a[5]) {
    // simple insertion sort
    for (int i = 1; i < 5; ++i)
        for (int j = i; j > 0 && a[j-1] > a[j]; --j) {
            float t = a[j-1]; a[j-1] = a[j]; a[j] = t;
        }
    return a[2];
}

int main() {
    // ── DRAM bandwidth ────────────────────────────────────────────────────
    // Array must exceed L2 capacity.  512 MB is safe for all current GPUs.
    const size_t DRAM_BYTES = 512ULL * 1024 * 1024;
    const size_t N_FLOATS   = DRAM_BYTES / sizeof(float);
    const int    BLOCKS     = 1024;
    const int    THREADS    = 256;
    const int    RUNS       = 5;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  DRAM_BYTES);
    cudaMalloc(&d_out, DRAM_BYTES);
    cudaMemset(d_in,  1, DRAM_BYTES);
    cudaMemset(d_out, 0, DRAM_BYTES);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warmup
    dram_bw_kernel<<<BLOCKS, THREADS>>>(d_in, d_out, N_FLOATS);
    cudaDeviceSynchronize();

    float bw_runs[5];
    float wall_ms_last = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
        cudaEventRecord(ev_start);
        dram_bw_kernel<<<BLOCKS, THREADS>>>(d_in, d_out, N_FLOATS);
        cudaEventRecord(ev_stop);
        cudaDeviceSynchronize();
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        // bytes = 2 passes (read + write) × DRAM_BYTES
        double bytes = 2.0 * (double)DRAM_BYTES;
        bw_runs[r] = (float)(bytes / (ms * 1e-3) / 1e9);
        printf("dram_bw_run_%d_GBps=%.2f\n", r, bw_runs[r]);
        wall_ms_last = ms;
    }
    float bw_median = median5(bw_runs);
    printf("dram_bw_median_GBps=%.2f\n", bw_median);
    printf("dram_bytes_transferred=%zu\n", (size_t)(2 * DRAM_BYTES));
    printf("dram_wall_ms=%.3f\n", wall_ms_last);

    cudaFree(d_in);
    cudaFree(d_out);

    // ── Shared memory bandwidth ───────────────────────────────────────────
    // Tile size = 256 floats (1 KB).  Use many blocks so SMs stay occupied.
    const size_t SHMEM_BLOCKS   = 2048;
    const size_t SHMEM_N        = SHMEM_BLOCKS * 256;
    const size_t SHMEM_BUF_BYTES = SHMEM_N * sizeof(float);
    const int    SHMEM_RUNS     = 3;

    float *d_sbuf;
    cudaMalloc(&d_sbuf, SHMEM_BUF_BYTES);
    cudaMemset(d_sbuf, 1, SHMEM_BUF_BYTES);

    // Warmup
    shmem_bw_kernel<<<SHMEM_BLOCKS, 256>>>(d_sbuf, SHMEM_N);
    cudaDeviceSynchronize();

    float shmem_bw_runs[3];
    for (int r = 0; r < SHMEM_RUNS; ++r) {
        cudaEventRecord(ev_start);
        shmem_bw_kernel<<<SHMEM_BLOCKS, 256>>>(d_sbuf, SHMEM_N);
        cudaEventRecord(ev_stop);
        cudaDeviceSynchronize();
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        // Each block performs 256 reads of 256 floats = 256 * 256 * 4 bytes
        double shmem_bytes = (double)SHMEM_BLOCKS * 256.0 * 256.0 * sizeof(float);
        shmem_bw_runs[r] = (float)(shmem_bytes / (ms * 1e-3) / 1e12);
        printf("shmem_bw_run_%d_TBps=%.3f\n", r, shmem_bw_runs[r]);
    }
    // median of 3: take middle value after sort
    float sb[3] = {shmem_bw_runs[0], shmem_bw_runs[1], shmem_bw_runs[2]};
    if (sb[0] > sb[1]) { float t = sb[0]; sb[0] = sb[1]; sb[1] = t; }
    if (sb[1] > sb[2]) { float t = sb[1]; sb[1] = sb[2]; sb[2] = t; }
    if (sb[0] > sb[1]) { float t = sb[0]; sb[0] = sb[1]; sb[1] = t; }
    printf("shmem_bw_median_TBps=%.3f\n", sb[1]);

    cudaFree(d_sbuf);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
```

## How to invoke

```
run_cuda_probe(
    source = <above kernel>,
    probe_name = "memory_bandwidth",
    compile_flags = ["-O3"],
)
```

## Interpreting results

Parse `dram_bw_median_GBps=<N>` and `shmem_bw_median_TBps=<N>` from stdout.

Typical ranges (vary by GPU):
- DRAM bandwidth: 200–1000 GB/s depending on GPU tier and memory clock
- Shared memory bandwidth: 10–50 TB/s

**Clock throttle cross-check:** Compare `dram_bytes_transferred / (dram_wall_ms * 1e-3)` 
to `dram_bw_median_GBps`. They should match. If the median bandwidth is very
low despite `dram__throughput > 90%` in ncu, the memory clock (not SM clock)
may be throttled — flag "memory_clock_throttled".

If `dram_bw_median_GBps < 50`:
- `flag_event` type="suspiciously_low_dram_bandwidth", severity="warn"
- The array may still be partially in L2; try doubling DRAM_BYTES

If run-to-run variance > 15%:
- `flag_event` type="bandwidth_unstable", severity="warn"
- Retry 3 additional runs and use the new median

## ncu cross-verification

```
profile_with_ncu(
    source_type = "cuda_source",
    source_or_path = <kernel source>,
    kernel_name = "dram_bw_kernel",
    metrics = [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]
)
```

If ncu `dram__throughput > 95%` but measured `dram_bw_median_GBps < 50`:
- `flag_event` type="api_spoofed_throughput_mismatch", severity="warn"
- Trust the measured bytes/wall_time over the ncu percentage

## Metric mapping

| stdout field              | target_spec key              | unit  |
|---------------------------|------------------------------|-------|
| `dram_bw_median_GBps`     | `peak_dram_bandwidth_GBps`   | GB/s  |
| `shmem_bw_median_TBps`    | `peak_shmem_bandwidth_TBps`  | TB/s  |

## Environment fallback

If `run_cuda_probe` returns `{"error": "binary_not_found", "name": "nvcc"}`:
- `flag_event` type="no_cuda_toolchain", severity="error"
- `record_measurement` value="unavailable", confidence=0.0
- `submit_results` immediately
