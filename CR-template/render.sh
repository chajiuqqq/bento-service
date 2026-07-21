#!/usr/bin/env bash
#
# render.sh — 读取 .env 中的变量，用 envsubst 渲染当前目录下的 *.yaml，
#             渲染结果输出到 ./${BENTO_NAME}/ 下。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE=".env"

# 1. 加载 .env：set -a 让后续赋值自动导出，供 envsubst 读取
if [[ ! -f "$ENV_FILE" ]]; then
  echo "错误：在 $SCRIPT_DIR 下未找到 $ENV_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# 2. 输出目录由 BENTO_NAME 决定
if [[ -z "${BENTO_NAME:-}" ]]; then
  echo "错误：$ENV_FILE 中未设置 BENTO_NAME" >&2
  exit 1
fi
OUTPUT_DIR="./${BENTO_NAME}"
mkdir -p "$OUTPUT_DIR"

# 3. 渲染所有 *.yaml
#    -no-unset：模板里出现「未设置且无默认值」的变量时直接报错，
#    避免悄悄生成空值；带 :- 默认值的变量不受影响。
shopt -s nullglob
templates=( *.yaml )
if (( ${#templates[@]} == 0 )); then
  echo "提示：当前目录没有 *.yaml 文件" >&2
  exit 0
fi

# 用临时文件 + mv，确保 envsubst 失败时不留下半截/空输出文件
trap 'rm -f "$OUTPUT_DIR"/*.tmp' EXIT
for tmpl in "${templates[@]}"; do
  out="$OUTPUT_DIR/$tmpl"
  tmp="${out}.tmp"
  envsubst -no-unset < "$tmpl" > "$tmp"
  mv "$tmp" "$out"
  echo "渲染：$tmpl -> $out"
done

echo "完成，共 ${#templates[@]} 个文件 -> $OUTPUT_DIR/"
