# 0、创建yatai连接的configmap(集群中只执行一次)
```bash
./render.sh
cd qwen36
kubectl apply -f configmap.yaml
```

# 1、构建bento

先生成该模型的 ConfigMap：

```bash
export BENTO_NAMESPACE="bento-dev"
export BENTO_NAME="bento-test"
export BENTO_VERSION="v1.0.0"
kubectl -n ${BENTO_NAMESPACE} create configmap bento-args-${BENTO_NAME}-${BENTO_VERSION} \
  --from-file=bento_args.yaml=/path/to/generated/bento_args.yaml
```

再创建 Job
```bash
kubectl apply -f job.yaml
```

# 2、构建服务镜像

```bash
kubectl apply -f bento-request.yaml
```

# 3、部署服务镜像

```bash
kubectl apply -f deployment.yaml
```