#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Yatai 更新脚本（基于笔记：yatai更新harbor和minio.md）
#
# 需在包含以下目录的工作目录下运行：
#   ├── Yatai
#   ├── yatai-deployment
#   └── yatai-image-builder
#
# 用法：
#   ./update_yatai.sh                                  # 更新 minio + harbor（默认）
#   ./update_yatai.sh minio                            # 仅更新 minio（S3 配置）
#   ./update_yatai.sh harbor                           # 仅更新 harbor（镜像仓库凭据）
#   ./update_yatai.sh [minio|harbor|all] --dry-run     # 仅渲染/比对，不实际生效
#   ./update_yatai.sh [minio|harbor|all] --no-restart  # 升级后跳过 rollout restart
#
# 默认升级完成后会自动对相关 Deployment 执行 rollout restart 并验证 Pod 就绪。
#
# 可选环境变量：
#   SERVICE_TYPE       yatai 前端 Service 类型（默认 NodePort）
#   SERVICE_NODE_PORT  yatai 前端 Service 节点端口（默认 30080）
#   说明: 线上 yatai Service 曾被 kubectl patch 改为 NodePort 暴露，默认值保持与
#   现状一致，避免 helm server-side apply 与 kubectl-patch 字段冲突。
#   POSTGRESQL_*       yatai 数据库连接参数（默认对应线上 my-postgres）
#
# 运行前需先 export 对应环境变量：
#   minio : S3_ENDPOINT S3_BUCKET_NAME S3_ACCESS_KEY S3_SECRET_KEY
#   harbor: DOCKER_SERVER DOCKER_USERNAME DOCKER_PASSWORD
# ============================================================

# 解析参数（支持 --dry-run / --no-restart，位置任意）
DRY_RUN=false
DO_RESTART=true
MODE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --no-restart) DO_RESTART=false ;;
    minio|harbor|all) MODE="$arg" ;;
    *)
      echo "未知参数: $arg" >&2
      exit 1
      ;;
  esac
done
MODE="${MODE:-all}"

# dry-run 时 helm 追加该参数，kubectl 操作被跳过
DRY_RUN_FLAG=""
[ "$DRY_RUN" = true ] && DRY_RUN_FLAG="--dry-run"

YATAI_CHART="./yatai/helm/yatai"
IMAGE_BUILDER_CHART="./yatai-image-builder/helm/yatai-image-builder"
DEPLOYMENT_CHART="./yatai-deployment/helm/yatai-deployment"

# 检查工作目录结构是否完整
check_charts() {
  local missing=0
  for d in "$YATAI_CHART" "$IMAGE_BUILDER_CHART" "$DEPLOYMENT_CHART"; do
    if [ ! -d "$d" ]; then
      echo "[错误] 缺少目录: $d" >&2
      missing=1
    fi
  done
  if [ "$missing" -eq 1 ]; then
    echo "[错误] 请在包含 Yatai / yatai-deployment / yatai-image-builder 的工作目录下运行本脚本" >&2
    exit 1
  fi
}

# 检查必需的环境变量
require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[错误] 未设置环境变量: $name" >&2
    exit 1
  fi
}

# ---------- minio / S3 更新 ----------
update_minio() {
  echo "==> [minio] 更新 yatai (S3)"
  require_env S3_ENDPOINT
  require_env S3_BUCKET_NAME
  require_env S3_ACCESS_KEY
  require_env S3_SECRET_KEY

  helm upgrade yatai "$YATAI_CHART" \
      -n yatai-system \
      --reuse-values \
      $DRY_RUN_FLAG \
      --set s3.endpoint="$S3_ENDPOINT" \
      --set s3.region=us-east-1 \
      --set s3.bucketName="$S3_BUCKET_NAME" \
      --set s3.secure=false \
      --set s3.accessKey="$S3_ACCESS_KEY" \
      --set s3.secretKey="$S3_SECRET_KEY" \
      --set service.type="${SERVICE_TYPE:-NodePort}" \
      --set service.nodePort="${SERVICE_NODE_PORT:-30080}" \
      --set postgresql.database="${POSTGRESQL_DATABASE:-yatai}" \
      --set postgresql.host="${POSTGRESQL_HOST:-my-postgres-postgresql}" \
      --set postgresql.password="${POSTGRESQL_PASSWORD:-admin}" \
      --set postgresql.port="${POSTGRESQL_PORT:-5432}" \
      --set postgresql.sslmode="${POSTGRESQL_SSLMODE:-disable}" \
      --set postgresql.user="${POSTGRESQL_USER:-admin}"

  echo "==> [minio] 更新 yatai-image-builder (S3)"
  helm upgrade yatai-image-builder "$IMAGE_BUILDER_CHART" -n yatai-image-builder \
    --reuse-values \
    $DRY_RUN_FLAG \
    --set global.s3.endpoint="$S3_ENDPOINT" \
    --set global.s3.bucketName="$S3_BUCKET_NAME" \
    --set global.s3.accessKeyId="$S3_ACCESS_KEY" \
    --set global.s3.secretAccessKey="$S3_SECRET_KEY" \
    --set global.s3.secure=false \
    --set global.s3.region=us-east-1
}

# ---------- harbor 更新 ----------
update_harbor() {
  echo "==> [harbor] 更新 yatai-image-builder 凭据"
  require_env DOCKER_SERVER
  require_env DOCKER_USERNAME
  require_env DOCKER_PASSWORD

  echo "==> 删除并重建 secret harbor-credentials (yatai-image-builder)"
  if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] 跳过 kubectl 操作，将执行:"
    echo "  kubectl delete secret harbor-credentials -n yatai-image-builder --ignore-not-found"
    echo "  kubectl create secret docker-registry harbor-credentials -n yatai-image-builder --docker-server=$DOCKER_SERVER --docker-username=$DOCKER_USERNAME --docker-password=<已设置>"
  else
    kubectl delete secret harbor-credentials -n yatai-image-builder --ignore-not-found
    kubectl create secret docker-registry harbor-credentials \
      -n yatai-image-builder \
      --docker-server="$DOCKER_SERVER" \
      --docker-username="$DOCKER_USERNAME" \
      --docker-password="$DOCKER_PASSWORD"
  fi

  echo "==> [harbor] 升级 yatai-image-builder"
  helm upgrade yatai-image-builder "$IMAGE_BUILDER_CHART" -n yatai-image-builder \
    --reuse-values \
    $DRY_RUN_FLAG \
    --set registry="$DOCKER_SERVER" \
    --set image.repository=cgllm/yatai-image-builder \
    --set image.tag=dev \
    --set image.pullPolicy=Always \
    --set "imagePullSecrets[0].name=harbor-credentials" \
    --set dockerRegistry.server="$DOCKER_SERVER" \
    --set dockerRegistry.secure=false \
    --set dockerRegistry.bentoRepositoryName=cgllm/yatai-bentos \
    --set dockerRegistry.username="$DOCKER_USERNAME" \
    --set dockerRegistry.password="$DOCKER_PASSWORD"

  echo "==> 删除并重建 secret harbor-credentials (yatai-deployment)"
  if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] 跳过 kubectl 操作，将执行:"
    echo "  kubectl delete secret harbor-credentials -n yatai-deployment --ignore-not-found"
    echo "  kubectl create secret docker-registry harbor-credentials -n yatai-deployment --docker-server=$DOCKER_SERVER --docker-username=$DOCKER_USERNAME --docker-password=<已设置>"
  else
    kubectl delete secret harbor-credentials -n yatai-deployment --ignore-not-found
    kubectl create secret docker-registry harbor-credentials \
      -n yatai-deployment \
      --docker-server="$DOCKER_SERVER" \
      --docker-username="$DOCKER_USERNAME" \
      --docker-password="$DOCKER_PASSWORD"
  fi

  echo "==> [harbor] 升级 yatai-deployment"
  helm upgrade yatai-deployment "$DEPLOYMENT_CHART" -n yatai-deployment \
    --reuse-values \
    $DRY_RUN_FLAG \
    --set registry="$DOCKER_SERVER" \
    --set image.repository=cgllm/yatai-deployment-controller \
    --set image.tag=latest \
    --set "imagePullSecrets[0].name=harbor-credentials" \
    --set bentoDeploymentAllNamespaces=true
}

# ---------- 更新后处理：rollout restart + 验证 ----------

# 对指定 deployment 强制滚动重启
rollout_restart() {
  local deploy="$1" ns="$2"
  echo "==> [rollout] 重启 deployment/$deploy ($ns)"
  kubectl rollout restart "deployment/$deploy" -n "$ns"
}

# 等待 rollout 完成并校验副本就绪数
verify_deploy() {
  local deploy="$1" ns="$2"
  echo "--- [验证] deployment/$deploy ($ns) ---"
  kubectl rollout status "deployment/$deploy" -n "$ns" --timeout=300s
  local desired available
  desired=$(kubectl get deployment "$deploy" -n "$ns" -o jsonpath='{.spec.replicas}')
  available=$(kubectl get deployment "$deploy" -n "$ns" -o jsonpath='{.status.availableReplicas}')
  available="${available:-0}"
  echo "  available=$available / desired=$desired"
  if [ "$available" != "$desired" ]; then
    echo "[错误] $deploy 副本未就绪 (available=$available, desired=$desired)" >&2
    return 1
  fi
}

# 升级完成后的统一处理：重启（可选）+ 验证
post_update() {
  local t deploy ns
  if [ "$DRY_RUN" = true ]; then
    echo "==> [dry-run] 跳过重启与验证，将执行:"
    for t in "${RESTART_TARGETS[@]:-}"; do
      deploy="${t%%:*}"; ns="${t##*:}"
      echo "  kubectl rollout restart deployment/$deploy -n $ns"
      echo "  kubectl rollout status deployment/$deploy -n $ns --timeout=300s"
    done
    return
  fi

  if [ "$DO_RESTART" = true ]; then
    for t in "${RESTART_TARGETS[@]:-}"; do
      rollout_restart "${t%%:*}" "${t##*:}"
    done
  else
    echo "==> 跳过 rollout restart（--no-restart）"
  fi

  echo "==> [验证] 等待 rollout 完成并检查副本就绪"
  local fail=0
  for t in "${RESTART_TARGETS[@]:-}"; do
    deploy="${t%%:*}"; ns="${t##*:}"
    if ! verify_deploy "$deploy" "$ns"; then
      fail=1
    fi
  done
  echo "==> [验证] 相关 Pod 状态:"
  for ns in "${VERIFY_NS[@]:-}"; do
    kubectl get pods -n "$ns" --no-headers
  done
  if [ "$fail" -eq 1 ]; then
    echo "[错误] 存在未通过验证的 Deployment" >&2
    exit 1
  fi
  echo "==> 验证通过"
}

check_charts

# 根据模式收集需要重启/验证的 deployment（格式: 名称:命名空间）
RESTART_TARGETS=()
VERIFY_NS=()
case "$MODE" in
  minio)
    update_minio
    RESTART_TARGETS=("yatai:yatai-system" "yatai-image-builder:yatai-image-builder")
    VERIFY_NS=("yatai-system" "yatai-image-builder")
    ;;
  harbor)
    update_harbor
    RESTART_TARGETS=("yatai-image-builder:yatai-image-builder" "yatai-deployment:yatai-deployment")
    VERIFY_NS=("yatai-image-builder" "yatai-deployment")
    ;;
  all)
    update_minio
    update_harbor
    RESTART_TARGETS=("yatai:yatai-system" "yatai-image-builder:yatai-image-builder" "yatai-deployment:yatai-deployment")
    VERIFY_NS=("yatai-system" "yatai-image-builder" "yatai-deployment")
    ;;
esac

post_update

echo "==> 全部完成"