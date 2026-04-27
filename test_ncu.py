import sys, json
from agents.core.config import ExecutorConfig
from agents.tools.cuda_executor import Executor, _autodetect_env

cfg = ExecutorConfig(cache_enabled=False)
cfg, notes = _autodetect_env(cfg)
for n in notes:
    print(n, file=sys.stderr)

src = r"""
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= 2.0f;
}
int main() {
    int n = 1024;
    float* d; cudaMalloc(&d, n*sizeof(float));
    k<<<4,256>>>(d, n);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
"""

executor = Executor(cfg)

print("Compiling and running probe...", file=sys.stderr)
probe = executor.run_cuda_probe(source=src, probe_name="ncu_test")
print(json.dumps(probe, indent=2), file=sys.stderr)

if probe.get("status") != "done":
    raise SystemExit(1)

print("Profiling returned binary_path with ncu...", file=sys.stderr)
ncu = executor.profile_with_ncu(
    binary_path=probe["binary_path"],
    kernel_name="k",
    metrics=["sm__cycles_elapsed.avg"],
    timeout_s=60,
)

print("=== NCU RESULT ===")
print(json.dumps(ncu, indent=2, default=str))
