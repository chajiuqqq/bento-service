
# envsubst 环境变量替换

```
# deploy.yaml 里写变量
image: ${IMAGE_TAG}
```

```
export IMAGE_TAG=v1.25
envsubst < deploy.yaml | kubectl apply -f -
```