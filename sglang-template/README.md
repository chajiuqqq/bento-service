# 构建镜像
只需要构建一次
```
./build-bentoml.sh
```
# 运行容器
会读取当前目录下的service.py 以及conf.yaml，使用bentoml serve启动服务
```
docker run --rm -it --gpus all -p 3000:3000 \
  -v /mnt/modules:/models:ro \
  -v ./:/bentoml-workspace:ro \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  sglang-blackwell:sm120a-bento
```

# 测试
```
# 测试
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "Qwen3.5-397B-A17B-NVFP4", 
     "messages": [{"role": "user", "content": "你好,介绍你自己"}], 
      "stream": false }' 
```
