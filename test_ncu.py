import argparse
import json
import sys

from agents.core.config import ExecutorConfig
from agents.tools.cuda_executor import Executor
from agents.tools.executor.subprocess_runner import _run_subprocess


NCU_METRIC = "sm__cycles_elapsed.avg"


PROBE_SOURCE = r"""
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::fprintf(stderr, "cuda_error=%s:%d:%s\n",             \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 2;                                                  \
    }                                                             \
} while (0)

extern "C" __global__ void probe_kernel(
    float* data,
    unsigned long long* cycle_out,
    int n,
    int iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = data[idx] + 1.0f;
    unsigned long long start = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000001f);
    }
    unsigned long long stop = clock64();

    data[idx] = x;
    if (idx == 0) {
        cycle_out[0] = stop - start;
    }
}

int main(int argc, char** argv) {
    int iters = 1024;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) iters = 1024;
    }

    const int n = 1 << 18;
    float* data = nullptr;
    unsigned long long* cycles_d = nullptr;
    unsigned long long cycles_h = 0;
    float sample_h = 0.0f;

    CHECK_CUDA(cudaMalloc(&data, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cycles_d, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(data, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(cycles_d, 0, sizeof(unsigned long long)));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    probe_kernel<<<grid, block>>>(data, cycles_d, n, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&cycles_h, cycles_d, sizeof(cycles_h), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&sample_h, data, sizeof(sample_h), cudaMemcpyDeviceToHost));

    std::printf("probe_ok=1\n");
    std::printf("kernel=probe_kernel\n");
    std::printf("iters=%d\n", iters);
    std::printf("cycles_thread0=%llu\n", cycles_h);
    std::printf("sample=%0.6f\n", sample_h);

    CHECK_CUDA(cudaFree(cycles_d));
    CHECK_CUDA(cudaFree(data));
    return 0;
}
"""


class CheckFailed(RuntimeError):
    pass


SUMMARY_KEYS = [
    "status",
    "error",
    "error_class",
    "phase",
    "hint",
    "elapsed_s",
    "binary_path",
    "stdout",
    "stdout_tail",
    "stdout_total_bytes",
    "metrics",
    "metric_units",
    "missing_metrics",
    "kernels_profiled",
    "kernel_names_seen",
    "ncu_version",
    "returncode",
    "timed_out",
]


def compact_result(result: dict) -> dict:
    compact = {key: result[key] for key in SUMMARY_KEYS if key in result}
    if isinstance(compact.get("stdout_tail"), str) and len(compact["stdout_tail"]) > 500:
        compact["stdout_tail"] = compact["stdout_tail"][-500:]
    if isinstance(compact.get("stdout"), str) and len(compact["stdout"]) > 500:
        compact["stdout"] = compact["stdout"][-500:]
    return compact


def dump(title: str, result, verbose: bool = False) -> None:
    print(f"\n=== {title} ===")
    payload = result if verbose else compact_result(result)
    print(json.dumps(payload, indent=2, default=str))


def require(condition: bool, message: str, result=None) -> None:
    if condition:
        print(f"[ok] {message}", file=sys.stderr)
        return
    if result is not None:
        dump("FAILED RESULT", result)
    raise CheckFailed(message)


def check_timeout_kill() -> None:
    print("Checking subprocess timeout/tree-kill...", file=sys.stderr)
    result = _run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=1,
        truncate_bytes=None,
    )
    require(result.timed_out, "_run_subprocess reports timed_out=True", result.__dict__)
    require(result.returncode == -1, "_run_subprocess normalizes timeout returncode=-1", result.__dict__)


def choose_kernel_candidates(names: list[str]) -> list[str]:
    candidates = ["probe_kernel"]
    for name in names:
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for CUDA Executor compile/run/ncu behavior."
    )
    parser.add_argument("--skip-ncu", action="store_true", help="Only verify compile/run and timeout behavior.")
    parser.add_argument("--probe-iters", type=int, default=1024, help="Loop iterations inside the CUDA kernel.")
    parser.add_argument("--ncu-timeout", type=int, default=120, help="Timeout in seconds for each ncu call.")
    parser.add_argument("--verbose", action="store_true", help="Print full executor JSON results, including raw evidence.")
    args = parser.parse_args()

    check_timeout_kill()

    cfg = ExecutorConfig(cache_enabled=False)
    executor = Executor(cfg)
    for note in executor.detect_notes:
        print(note, file=sys.stderr)

    print("Compiling and running CUDA probe...", file=sys.stderr)
    probe = executor.run_cuda_probe(
        source=PROBE_SOURCE,
        probe_name="executor_ncu_smoke",
        args=[str(args.probe_iters)],
        timeout_s=60,
    )
    dump("RUN_CUDA_PROBE", probe, verbose=args.verbose)

    require(probe.get("status") == "done", "run_cuda_probe status is done", probe)
    require(bool(probe.get("binary_path")), "run_cuda_probe returns binary_path", probe)
    require(probe.get("stdout_total_bytes", 0) > 0, "run_cuda_probe captures stdout evidence", probe)
    stdout_evidence = probe.get("stdout_tail") or probe.get("stdout") or ""
    require("probe_ok=1" in stdout_evidence, "CUDA program completed and printed probe_ok=1", probe)
    require("cycles_thread0=" in stdout_evidence, "CUDA program printed measured cycle evidence", probe)

    if args.skip_ncu:
        print("\nALL CHECKS PASSED (ncu skipped)")
        return 0

    print("Discovering kernels with ncu (no kernel filter)...", file=sys.stderr)
    discovery = executor.profile_with_ncu(
        binary_path=probe["binary_path"],
        kernel_name="",
        metrics=[NCU_METRIC],
        args=[str(args.probe_iters)],
        timeout_s=args.ncu_timeout,
    )
    dump("NCU_DISCOVERY", discovery, verbose=args.verbose)

    require(discovery.get("status") == "done", "profile_with_ncu discovery status is done", discovery)
    require(discovery.get("kernels_profiled", 0) > 0, "ncu observed at least one kernel invocation", discovery)
    kernel_names = discovery.get("kernel_names_seen") or []
    require(bool(kernel_names), "ncu returned kernel_names_seen", discovery)
    require(NCU_METRIC in discovery.get("metrics", {}), f"ncu parsed exact metric {NCU_METRIC}", discovery)
    require(not discovery.get("missing_metrics"), "ncu has no missing_metrics in discovery profile", discovery)

    print("Profiling with a kernel filter...", file=sys.stderr)
    targeted = None
    last_targeted = None
    for candidate in choose_kernel_candidates(kernel_names):
        print(f"Trying kernel filter: {candidate}", file=sys.stderr)
        last_targeted = executor.profile_with_ncu(
            binary_path=probe["binary_path"],
            kernel_name=candidate,
            metrics=[NCU_METRIC],
            args=[str(args.probe_iters)],
            timeout_s=args.ncu_timeout,
        )
        dump(f"NCU_TARGETED {candidate}", last_targeted, verbose=args.verbose)
        if (
            last_targeted.get("status") == "done"
            and NCU_METRIC in last_targeted.get("metrics", {})
            and not last_targeted.get("missing_metrics")
        ):
            targeted = last_targeted
            break

    require(targeted is not None, "profile_with_ncu works with a kernel filter", last_targeted)
    require(targeted.get("ncu_version") is not None, "profile_with_ncu returns ncu_version", targeted)

    print("Checking structured data_quality error for a wrong kernel name...", file=sys.stderr)
    bad_kernel = executor.profile_with_ncu(
        binary_path=probe["binary_path"],
        kernel_name="definitely_missing_kernel_name_for_executor_smoke",
        metrics=[NCU_METRIC],
        args=[str(args.probe_iters)],
        timeout_s=args.ncu_timeout,
    )
    dump("NCU_BAD_KERNEL", bad_kernel, verbose=args.verbose)
    require(bad_kernel.get("status") == "error", "wrong kernel name returns status=error", bad_kernel)
    require(bad_kernel.get("error") == "ncu_no_kernel_found", "wrong kernel name maps to ncu_no_kernel_found", bad_kernel)
    require(bad_kernel.get("error_class") == "data_quality", "wrong kernel name maps to data_quality", bad_kernel)
    require("stdout_tail" in bad_kernel or "stderr" in bad_kernel, "wrong-kernel error includes raw ncu evidence", bad_kernel)

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as exc:
        print(f"\nCHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
