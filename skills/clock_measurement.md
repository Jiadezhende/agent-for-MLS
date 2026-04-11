# clock_measurement — Actual GPU Boost Clock Frequency

## When to use

Use this skill when a target spec requests any of:
- `actual_boost_clock_mhz`
- `actual_clock_mhz`
- `gpu_clock_mhz`
- `sm_clock_hz`
- GPU operating frequency in MHz or Hz

## Principle

Run a compute-intensive kernel and record the elapsed wall-clock time and the
elapsed GPU cycle count (via `clock64()`). Their ratio gives the true SM
frequency:

```
frequency_hz = gpu_cycles / elapsed_wall_seconds
```

This reflects the actual boost clock the GPU is running at (not the advertised
base or boost spec, which may differ due to thermal throttling or power limits).

## CUDA kernel template

```cuda
#include <cstdio>
#include <cuda_runtime.h>

// Compute-intensive kernel: keeps SMs busy for a measured number of GPU cycles.
// We use clock64() at start and end, plus cudaEventElapsedTime for wall time.
__global__ void spin_kernel(
    volatile uint64_t* out_start_cycles,
    volatile uint64_t* out_end_cycles,
    long long target_iters
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_start_cycles[0] = clock64();
    }
    __syncthreads();

    // Busy-work: a chain of FMA operations to prevent optimization
    float acc = (float)(threadIdx.x + blockIdx.x + 1);
    for (long long i = 0; i < target_iters; ++i) {
        acc = acc * 1.00001f + 0.00001f;
    }
    // Prevent dead-code elimination
    if (acc < 0.0f) out_end_cycles[0] = (uint64_t)(acc);

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_end_cycles[0] = clock64();
    }
}

int main() {
    const int BLOCKS  = 128;
    const int THREADS = 256;
    const long long ITERS = 10000000LL;  // enough for > 100 ms

    uint64_t *d_start, *d_end;
    cudaMalloc(&d_start, sizeof(uint64_t));
    cudaMalloc(&d_end,   sizeof(uint64_t));

    // Warmup
    spin_kernel<<<BLOCKS, THREADS>>>(d_start, d_end, ITERS / 100);
    cudaDeviceSynchronize();

    // Timed run using CUDA events for wall clock
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    spin_kernel<<<BLOCKS, THREADS>>>(d_start, d_end, ITERS);
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();

    float wall_ms = 0.0f;
    cudaEventElapsedTime(&wall_ms, ev_start, ev_stop);

    uint64_t t_start = 0, t_end = 0;
    cudaMemcpy(&t_start, d_start, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t_end,   d_end,   sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t gpu_cycles = t_end - t_start;
    double wall_s = wall_ms / 1000.0;
    double freq_hz = (double)gpu_cycles / wall_s;
    double freq_mhz = freq_hz / 1e6;

    printf("gpu_cycles=%llu\n",     (unsigned long long)gpu_cycles);
    printf("wall_ms=%.3f\n",        wall_ms);
    printf("actual_clock_mhz=%.1f\n", freq_mhz);

    // Query device name (clockRate was removed in CUDA 13 for newer architectures)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device_name=%s\n", prop.name);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_start);
    cudaFree(d_end);
    return 0;
}
```

## How to invoke

```
run_cuda_probe(
    source = <above kernel>,
    probe_name = "clock_measurement",
    compile_flags = ["-O3"],
)
```

## Interpreting results

Parse `actual_clock_mhz=<N>` from stdout.

Also compare against `driver_clock_mhz=<M>`:
- If `|actual - driver| / driver > 0.10` (more than 10% difference):
  - `flag_event` with type="clock_throttled" and severity="warn"
  - Report the **measured** value (not the driver-reported one)
- If actual < driver: GPU is thermally throttled or power-limited
- If actual > driver * 1.05: unlikely; may indicate clock64() drift or
  multi-SM skew — repeat measurement

## Metric mapping

| stdout field           | target_spec key            | unit |
|------------------------|----------------------------|------|
| `actual_clock_mhz=N`   | `actual_boost_clock_mhz`   | MHz  |
| `actual_clock_mhz=N`   | `actual_clock_mhz`         | MHz  |
| `actual_clock_mhz=N`   | `gpu_clock_mhz`            | MHz  |

## Environment fallback

If `run_cuda_probe` returns `{"error": "binary_not_found", "name": "nvcc"}`:
- The host has no CUDA compiler.
- `flag_event` with type="no_cuda_toolchain", severity="error".
- `record_measurement` with value="unavailable", confidence=0.0,
  method="cuda_toolchain_absent", evidence=[error string from tool result].
- `submit_results` immediately — do not loop.
