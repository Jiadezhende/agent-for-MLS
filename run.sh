#!/bin/bash
set -e

pip3 install openai python-dotenv jsonschema \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --default-timeout 30

python3 /workspace/main.py \
  --spec /target/target_spec.json \
  --output /workspace/output.json \
  --verbose
