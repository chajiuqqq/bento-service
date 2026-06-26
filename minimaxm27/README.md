# 构建 Bento 并容器化

## 构建

```bash
bentoml build
bentoml containerize qwen3-8b:latest -t vllm-bento:latest
```

## 运行方式

### 方案一：打包配置，只挂载模型

配置已打包在 bento 中，运行时只需挂载模型目录。

```bash
docker run --rm -it --gpus all -p 3000:3000 \
  -v /mnt/modules:/models:ro \
  --ipc=host \
  vllm-bento:latest
```

适用于生产部署：镜像包含完整配置，无需额外文件。

### 方案二：挂载项目路径（开发调试）

将整个项目目录挂载到容器，覆盖 bento 内的配置。

```bash
docker run --rm -it --gpus all -p 3000:3000 \
  -v /mnt/modules:/models:ro \
  -v ./:/bentoml-workspace:ro \
  --ipc=host \
  vllm-bento:latest
```

适用于开发：修改 `conf.yaml` 后重启容器即可生效，无需重建镜像。

## 设计说明

### 问题
基础镜像 `vllm/vllm-openai:latest` 已将 vllm、torch 2.10.0+cu129 等依赖安装在 `/usr/local/lib/python3.12/dist-packages`（全局 Python）。默认的 BentoML Dockerfile 会创建 `/app/.venv`，新 venv 默认不继承系统 site-packages，导致无法访问基镜像预装的包。

### 方案
使用自定义 `dockerfile_template`（`docker/Dockerfile.template.j2`）绕过默认的 venv 创建流程：

- **`SETUP_BENTO_USER`** — 跳过用户创建（基镜像已有）
- **`SETUP_BENTO_ENVARS`** — 设置环境变量和 WORKDIR
- **`SETUP_BENTO_COMPONENTS`** — 直接复制文件，`pip install` 到基镜像的全局 Python 环境
- **`SETUP_BENTO_ENTRYPOINT`** — `bentoml serve /app`

### 注意
- 此模板与基镜像强关联：假设 `/usr/bin/python3` 为全局 Python，site-packages 在 `/usr/local/lib/python3.12/dist-packages`
- `dockerfile_template` 仅支持 BentoML v1 API（v2 不支持此选项，构建时会自动 fallback 到 v1）
- `pyproject.toml` 中使用 `[tool.bentoml.build.docker]` 配置，而不是 v2 的 `bentoml.images.Image`

## 测试

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "MiniMaxAI/MiniMax-M2.5",
     "messages": [{"role": "user", "content": "你好,介绍你自己"}],
       "stream": false }'
```
