#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-${BASE_MODEL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-}}"
export AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/workspace/workspace}"

python /workspace/main.py \
  --spec /target/target_spec.json \
  --output /workspace/output.json \
  --verbose
