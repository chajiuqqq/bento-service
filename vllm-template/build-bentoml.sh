#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f docker/Dockerfile \
  -t vllm-bento:0.0.1 \
  .
