
# 构建bento-builder镜像
```
./build-bento-builder.sh
```

# 上传项目文件夹到S3
```
mc mirror --overwrite --exclude ".*" ./ pt/chaoguang/work/bento-service
```