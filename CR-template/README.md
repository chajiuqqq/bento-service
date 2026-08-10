# 0、集群初始化（只执行一次）

## 0.1 创建 yatai 连接 configmap

```bash
./render.sh
cd qwen36
kubectl apply -f configmap.yaml
```

## 0.2 创建内网 S3 根证书 configmap

构建 Job 需要从内网 S3（`https://s3.pintechs.com`）下载项目 zip，该 S3 使用 Pintechs 内网自签 CA，Job 容器默认不信任，会导致 `SSL CERTIFICATE_VERIFY_FAILED`。
因此需将根证书 `s3-ca.crt` 注入集群（该文件即 Pintechs 内网根 CA，与开发机 `/usr/local/share/ca-certificates/company-ca.crt` 相同）。

```bash
kubectl -n ${BENTO_NAMESPACE:-bento-dev} create configmap s3-ca-cert \
  --from-file=s3-ca.crt=./s3-ca.crt \
  --dry-run=client -o yaml | kubectl apply -f -
```

> `--dry-run=client -o yaml | kubectl apply -f -` 保证幂等，重复执行不会报错。
>
> 若需更新证书：替换 `s3-ca.crt` 内容后重新执行上述命令，然后重建 Job（configmap 变更不会自动热更新到已运行的 Pod）。

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