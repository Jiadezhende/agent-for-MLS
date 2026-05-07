#!/bin/bash
# Phase-2 evaluation entry point. Runs the LoRA-kernel optimization pipeline
# under the official 30-minute budget and leaves ./optimized_lora.cu at the
# repository root for torch.utils.cpp_extension.load to pick up.
set -e

# Dependencies (idempotent; harmless if already installed).
# ninja is required by torch.utils.cpp_extension.load — without it every
# candidate fails to compile and the run produces no submittable kernel.
pip3 install openai python-dotenv jsonschema pydantic ninja \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --default-timeout 30 || true

# Allow ncu to access GPU hardware counters where possible (no-op if not root).
sysctl -w kernel.perf_event_paranoid=2 2>/dev/null || true

# 1800s = 30 minutes. The pipeline syncs ./optimized_lora.cu eagerly:
#   - after INITIAL_CANDIDATE (compile + correctness pass) → root file exists
#     even if we time out later;
#   - after every best update during TUNING_LOOP.
python3 main.py \
  --operator lora_matmul \
  --time-budget 1800 \
  --output ./optimized_lora.cu \
  --verbose
