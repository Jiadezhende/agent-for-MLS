#!/bin/bash
set -e

pip3 install openai python-dotenv jsonschema \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --default-timeout 30

# Allow ncu to access GPU hardware counters (no-op if not root or not Linux)
sysctl -w kernel.perf_event_paranoid=2 2>/dev/null || true

python3 /workspace/main.py \
  --spec /target/target_spec.json \
  --output /workspace/output.json \
  --verbose
