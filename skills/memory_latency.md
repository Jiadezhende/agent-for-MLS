# memory_latency — Pointer-Chasing DRAM / Cache Latency Measurement

## When to use

Use this skill when a target spec requests any of:
- `dram_latency_cycles`
- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_ns`
- memory access latency in cycles or nanoseconds

## Principle

A pointer-chasing kernel follows a linked list stored in GPU memory. Because
each load address depends on the previous load's result, the memory subsystem
cannot overlap or prefetch accesses. The measured latency is the true round-trip
latency for one cache miss at the target tier.

Working set sizes for a typical modern GPU (A100, RTX 4090, etc.):
- L1 cache: ~32 KB (fits inside one SM)
- L2 cache: ~4 MB – 40 MB depending on GPU
- DRAM (L2 miss): > 128 MB

## CUDA kernel template

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Pointer-chasing kernel.  Each thread independently chases its slice of
// the chain; we measure only the first thread's latency via clock64().
__global__ void pointer_chase(
    uint32_t* __restrict__ arr,
    uint32_t N,
    uint64_t* __restrict__ out_cycles,
    int iters
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t idx = 0;
    // Warmup — not timed
    for (int w = 0; w < 16; ++w)
        idx = arr[idx];

    uint64_t t0 = clock64();
    for (int i = 0; i < iters; ++i)
        idx = arr[idx];
    uint64_t t1 = clock64();

    out_cycles[0] = (t1 - t0) / (uint64_t)iters;
    // prevent dead-code elimination
    if (idx == 0xDEADBEEF) out_cycles[0] = 0;
}

// Build a random permutation so arr[arr[arr[...]]] visits every element once
void build_random_chain(uint32_t* arr, uint32_t N) {
    for (uint32_t i = 0; i < N; ++i) arr[i] = i;
    // Fisher-Yates
    for (uint32_t i = N - 1; i > 0; --i) {
        uint32_t j = rand() % (i + 1);
        uint32_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int main(int argc, char* argv[]) {
    // Default: 256 MB to force DRAM misses
    // Pass "l1" or "l2" as argv[1] for smaller working sets
    size_t bytes = 256ULL * 1024 * 1024;
    if (argc > 1) {
        if (argv[1][0] == 'l' && argv[1][1] == '1') bytes = 32 * 1024;
        else if (argv[1][0] == 'l' && argv[1][1] == '2') bytes = 8ULL * 1024 * 1024;
    }

    uint32_t N = (uint32_t)(bytes / sizeof(uint32_t));
    int iters = 512;

    uint32_t* h_arr = (uint32_t*)malloc(bytes);
    build_random_chain(h_arr, N);

    uint32_t* d_arr;
    uint64_t* d_out;
    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_out, sizeof(uint64_t));
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    free(h_arr);

    pointer_chase<<<1, 1>>>(d_arr, N, d_out, iters);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("latency_cycles=%llu\n", (unsigned long long)cycles);
    printf("array_bytes=%zu\n", bytes);

    cudaFree(d_arr);
    cudaFree(d_out);
    return 0;
}
```

## How to invoke

```
run_cuda_probe(
    source = <above kernel>,
    probe_name = "pointer_chase_dram",
    compile_flags = ["-O3"],
)
```

To measure DRAM latency pass no args (or `args=[]`).
To measure L2 latency pass `args=["l2"]`.
To measure L1 latency pass `args=["l1"]`.

## Interpreting results

Parse `latency_cycles=<N>` from stdout.

Typical values (vary by GPU architecture):
- L1 latency: 20–35 cycles
- L2 latency: 150–300 cycles
- DRAM latency: 400–800 cycles

If DRAM latency is < 200 cycles, the working set may not be large enough
to exceed L2; use `flag_event` type="suspiciously_low_dram_latency" and
try a larger array (512 MB).

## Metric mapping

| stdout field        | target_spec key        | unit    |
|---------------------|------------------------|---------|
| `latency_cycles=N`  | `dram_latency_cycles`  | cycles  |
| `latency_cycles=N`  | `l2_latency_cycles`    | cycles  |
| `latency_cycles=N`  | `l1_latency_cycles`    | cycles  |

## Environment fallback

If `run_cuda_probe` returns `{"error": "binary_not_found", "name": "nvcc"}`:
- The host has no CUDA compiler.
- `flag_event` with type="no_cuda_toolchain", severity="error".
- `record_measurement` with value="unavailable", confidence=0.0,
  method="cuda_toolchain_absent", evidence=[error string from tool result].
- `submit_results` immediately — do not loop.
