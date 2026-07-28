# 1、构建bento

先生成 ConfigMap：

```bash
kubectl -n yatai create configmap bento-args-${BENTO_NAME}-${BENTO_VERSION} \
  --from-file=bento_args.yaml=/path/to/generated/bento_args.yaml
```

再创建 Job（render.sh 渲染 + kubectl apply）

# 2、构建服务镜像

- render.sh 渲染 bento-request.yaml
- kubectl apply bento-request CR文件

# 3、部署服务镜像

- render.sh 渲染 deployment.yaml
- kubectl apply deployment CR文件