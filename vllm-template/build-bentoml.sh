#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f docker/Dockerfile \
  -t vllm-bentoml:latest \
  .
