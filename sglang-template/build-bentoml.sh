#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f docker/Dockerfile.blackwell.bentoml \
  -t sglang-bentoml:cu130 \
  .
