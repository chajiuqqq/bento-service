# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库用途

用 BentoML 把 LLM 推理服务（vLLM / SGLang）打包成 bento，再通过 yatai（BentoML 的 K8s operator，namespace `yatai`）构建镜像并部署到集群。

## 常用命令

### 单个模型服务（在模型文件夹内执行，如 qwen3-8b/）

```bash
# 打 bento（bento_args.yaml 是构建参数入口）
bentoml build --arg-file bento_args.yaml

# 本地开发模式（改 bento_args.yaml 后重启即生效，无需重新 build）
bentoml serve --arg-file bento_args.yaml --port 3002

# 容器化（需要代理时加 --opt build-arg）
bentoml containerize qwen3-8b:latest
bentoml containerize --opt build-arg=http_proxy=http://192.168.39.240:7890 --opt build-arg=https_proxy=http://192.168.39.240:7890 qwen3-8b:latest

# 本地跑容器
docker run -it --rm --gpus '"device=1"' --shm-size=16gb -v /models:/models -p 3300:3000 qwen3-8b:latest
```

每个模型文件夹的 `脚本.md` 是该模型的命令速查（含特定环境变量，如 `FLASHINFER_DISABLE_VERSION_CHECK=1`、`CUDA_VISIBLE_DEVICES`）。

### K8s 构建/部署流程（CR-template/）

```bash
cd CR-template && ./render.sh   # 读 .env，envsubst -no-unset 渲染所有 *.yaml 到 ./${BENTO_NAME}/

# 1. 构建 bento：必须先建 ConfigMap（job.yaml 通过 subPath 挂载它），再 apply job.yaml
kubectl -n yatai create configmap bento-args-${BENTO_NAME}-${BENTO_VERSION} \
  --from-file=bento_args.yaml=/path/to/bento_args.yaml
# 2. 构建服务镜像：apply 渲染出的 bento-request.yaml（yatai BentoRequest CR）
# 3. 部署：apply 渲染出的 deployment.yaml（yatai BentoDeployment CR）
```

注意 `render.sh` 用 `envsubst -no-unset`：模板里新增 `${VAR}` 时必须同步加到 `.env`，否则渲染直接报错。

### Lint

ruff 配置在 [vllm-template/pyproject.toml](vllm-template/pyproject.toml)：`line-length = 119`、2 空格缩进、单引号。

## 架构

### 模型服务文件夹（qwen3-8b/、qwen36/、minimaxm27/、Qwen3.5-397B-A17B-NVFP4/）

每个文件夹是一个自包含的 BentoML 项目，结构相同、参数不同：

- `service.py` — 核心模式：FastAPI 透明流式代理（`@bentoml.asgi_app(openai_api_app, path='/v1')`）挡在 `vllm serve` 子进程前面，负责 JSON 结构化日志和 trace header（`X-Oneapi-Request-Id`）透传；`vllm serve` 命令行由 `__command__` 协议生成。**注意：各文件夹的 service.py 已各自分化**（如 qwen3-8b 的 BentoArgs 只有 name/engine_port/cli_args，vllm-template 的有 tp/gpu_type/envs 等完整字段），改一处不会同步到其他文件夹。
- `bento_args.yaml` — 构建参数：模型名、`engine_port`、传给 vllm 的 `cli_args`（含 `--served-model-name`、`--reasoning-parser`、`--tool-call-parser` 等）。业务系统动态生成此文件走 K8s 构建（见上）。
- `pyproject.toml` — `[tool.bentoml.build]` 指向自定义 `Dockerfile.template.j2` 和基镜像（各模型基镜像不同，如 `vllm-openai:v0.25.1`、`vllm-openai:minimax27`）。
- `Dockerfile.template.j2` — 绕过 BentoML 默认 venv 创建，直接 pip install 到基镜像的全局 Python。

### vllm-template/ 与 sglang-template/

模型文件夹的上游模板。`vllm-template/` 另有 `conf*.yaml` 变体和 `templates/`（tool-call jinja 模板）；`sglang-template/` 有多个 service 实验变体（service.runner.py、service-logging.py 等）。

### llm-router/

另一种 bento：不含推理引擎，纯 FastAPI 路由——检测请求 messages 里是否有图片，转发到文本模型或视觉模型后端（`TEXT_MODEL_URL` / `VISION_MODEL_URL` 环境变量配置）。透明流式代理刻意不走 bentoml 的 pydantic IO 校验。

### 关键约束

- **BentoML 锁 v1 API**（`bentoml==1.4.33`）：`dockerfile_template` 仅 v1 支持，v2 构建会静默 fallback。不要升级到 bentoml v2 的 `bentoml.images.Image` 写法。
- 自定义 Dockerfile.template.j2 与基镜像强绑定：假设全局 Python 在 `/usr/bin/python3`、site-packages 在 `/usr/local/lib/python3.12/dist-packages`。换基镜像时先用 [image-label-builder/get_info.sh](image-label-builder/get_info.sh) 提取镜像元数据（OS/Python/CUDA/引擎版本）。
- pip 全部走阿里云镜像 `https://mirrors.aliyun.com/pypi/simple/`。
- 流式代理如需挂载原生 ASGI app，用 `mount_asgi_app` / `@bentoml.asgi_app`；`@bentoml.api` 无法接收 starlette Request。

### 其他

- `save_model.py` — 把本地模型目录导入 BentoML Model Store。
- `image-label-builder/` — 各基镜像的 Dockerfile 及上述元数据提取脚本。
