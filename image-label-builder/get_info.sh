#!/usr/bin/env bash
# Extract image metadata, print key="value" lines.
# Usage:  ./get_info.sh
#         IMG=other/image:tag ./get_info.sh
set -euo pipefail

IMG=${IMG:-vllm/vllm-openai:minimax27}

docker run --rm --entrypoint bash "$IMG" -c '
  # OS: keep full PRETTY_NAME
  system_version="$(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d "\"")"

  # Python: major.minor only
  python_version="$(python3 -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")"

  # CUDA: try version.json (new), then version.txt (legacy), then nvcc
  if [ -f /usr/local/cuda/version.json ]; then
    cuda_version="$(python3 -c "import json; d=json.load(open(\"/usr/local/cuda/version.json\")); print(f\"{d[\"cuda_nvcc\"][\"major\"]}.{d[\"cuda_nvcc\"][\"minor\"]}\")" 2>/dev/null || echo Unknown)"
  elif [ -f /usr/local/cuda/version.txt ]; then
    cuda_version="$(awk "{print \$4}" /usr/local/cuda/version.txt | cut -d. -f1,2)"
  elif command -v nvcc >/dev/null 2>&1; then
    cuda_version="$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")"
  else
    cuda_version="Unknown"
  fi

  # Infer engine: vllm or sglang, formatted as "Name X.Y"
  infer_engine="Unknown"
  if pip show vllm >/dev/null 2>&1; then
    v="$(pip show vllm | awk "/^Version:/{print \$2}")"
    infer_engine="vLLM $(echo "$v" | cut -d. -f1,2)"
  elif pip show sglang >/dev/null 2>&1; then
    v="$(pip show sglang | awk "/^Version:/{print \$2}")"
    infer_engine="SGLang $(echo "$v" | cut -d. -f1,2)"
  fi

  echo "system_version=\"$system_version\""
  echo "python_version=\"$python_version\""
  echo "CUDA_version=\"$cuda_version\""
  echo "infer_engine=\"$infer_engine\""
'

