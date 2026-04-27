#!/usr/bin/env bash
set -euo pipefail

cd /workspace

pip3 install -r /workspace/requirements.txt \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --default-timeout 30

# Allow ncu to access GPU hardware counters (no-op if not root or not Linux)
sysctl -w kernel.perf_event_paranoid=2 2>/dev/null || true

export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-${BASE_MODEL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-}}"
export AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/workspace/workspace}"

python /workspace/main.py \
  --spec /target/target_spec.json \
  --output /workspace/output.json \
  --verbose
