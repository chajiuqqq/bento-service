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

  # Infer engine: vllm or sglang, formatted as "Name X.Y" or "Name branch@hash"
  infer_engine="Unknown"
  if pip show vllm >/dev/null 2>&1; then
    v="$(pip show vllm | awk "/^Version:/{print \$2}")"
    infer_engine="vLLM $(echo "$v" | cut -d. -f1,2)"
  elif pip show sglang >/dev/null 2>&1; then
    v="$(pip show sglang | awk "/^Version:/{print \$2}")"
    if [ "$v" = "0.0.0" ]; then
      sglang_dir="$(pip show sglang | awk "/^Editable project location:/{print \$NF}")"
      # .git may be in parent directory (e.g., /opt/sglang/.git not /opt/sglang/python/.git)
      git_dir="$sglang_dir/.git"
      if [ ! -d "$git_dir" ]; then
        git_dir="$(dirname "$sglang_dir")/.git"
      fi
      if [ -d "$git_dir" ]; then
        # Try git tag first
        if command -v git >/dev/null 2>&1; then
          git_tag="$(git -C "$git_dir" describe --tags --long 2>/dev/null || true)"
          if [ -n "$git_tag" ]; then
            v="$(echo "$git_tag" | sed 's/^v//' | cut -d. -f1,2)"
          fi
        fi
        # Fallback: read branch name and commit hash from .git files
        if [ "$v" = "0.0.0" ]; then
          head_ref="$(cat "$git_dir/HEAD" 2>/dev/null || true)"
          # Handle both "ref: refs/heads/branch" and "ref:refs/heads/branch" formats
          if [[ "$head_ref" =~ ^ref:\ ?refs/heads/ ]]; then
            branch="${head_ref#ref: }"
            branch="${branch#refs/heads/}"
            hash_file="$git_dir/refs/heads/$branch"
            hash="$(cat "$hash_file" 2>/dev/null | cut -c1-7 || echo "unknown")"
            v="$branch@$hash"
          fi
        fi
      fi
    fi
    infer_engine="SGLang $v"
  fi

  echo "system_version=\"$system_version\""
  echo "python_version=\"$python_version\""
  echo "CUDA_version=\"$cuda_version\""
  echo "infer_engine=\"$infer_engine\""
'

