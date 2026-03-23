#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f docker/Dockerfile.blackwell.bentoml \
  -t sglang-blackwell:sm120a-bento \
  .
