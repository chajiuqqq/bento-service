import json
import logging
import os

import bentoml
import httpx
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import StreamingResponse

# 用 bentoml 子 logger，继承 bentoml 的 INFO 级别与 handler（root 默认是 WARNING，普通 logger 的 INFO 会被丢弃）
logger = logging.getLogger("bentoml.llm_router")

# 定义后台两个引擎的内部地址及对应的 served-model-name
# 均可由环境变量覆盖，未设置时使用下面的默认值
TEXT_MODEL_URL = os.getenv("TEXT_MODEL_URL", "http://localhost:5001/v1/chat/completions")   # 纯文本模型  modelid=minimax-m27
VISION_MODEL_URL = os.getenv("VISION_MODEL_URL", "http://localhost:5002/v1/chat/completions") # 多模态模型  modelid=Qwen3.5
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "minimax-m27")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "Qwen36")

client = httpx.AsyncClient(timeout=300.0)


def _detect_image_in_messages(messages: list) -> bool:
    """检测请求中是否存在图片字段"""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        # 1. 如果 content 是列表，检查是否有 type == "image_url" 或 "image"
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["image_url", "image"]:
                    return True
        # 2. 检查某些特定 API 格式的顶级 image 字段
        if "images" in msg or "image" in msg:
            return True
    return False


# 透明代理 ASGI app：不经过 bentoml 的 pydantic IO 校验，
# 自行控制 body 注入、流式转发与首部透传。
router_app = FastAPI(title="llm-router")


@router_app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    # 检查是否包含图片输入，动态选择目标后端及其 served-model-name
    has_image = _detect_image_in_messages(messages)
    if has_image:
        target_url = VISION_MODEL_URL
        model_id = VISION_MODEL_ID
    else:
        target_url = TEXT_MODEL_URL
        model_id = TEXT_MODEL_ID

    logger.info(
        "路由决策: has_image=%s stream=%s -> 后端=%s model=%s",
        has_image,
        stream,
        target_url,
        model_id,
    )

    # 注入/覆盖 model 字段，使其与后端 served-model-name 一致
    body["model"] = model_id

    # 重新序列化 body，需移除与原始报文绑定的首部，交由 httpx 重新计算
    req_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "content-encoding")
    }
    forwarded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")

    # 发起异步流式转发
    upstream_req = client.build_request(
        method=request.method,
        url=target_url,
        headers=req_headers,
        content=forwarded_body,
    )
    upstream_resp = await client.send(upstream_req, stream=True)

    logger.info(
        "上游响应: status=%s model=%s stream=%s 后端=%s",
        upstream_resp.status_code,
        model_id,
        stream,
        target_url,
    )

    async def stream_body():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()

    # 透传上游状态码与首部；去掉 hop-by-hop 及 length 类首部，
    # Content-Type 由 media_type 设定，流式分块由框架处理。
    passthrough_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower()
        not in ("content-length", "transfer-encoding", "connection", "content-encoding")
    }

    return StreamingResponse(
        stream_body(),
        status_code=upstream_resp.status_code,
        headers=passthrough_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


@bentoml.service(
    name="llm_router",
    resources={"cpu": "2"},
)
class LLMRouterService:
    """请求分流服务：根据是否含图片，将 /v1/chat/completions 转发到文本或多模态后端。"""


# 将透明代理 ASGI app 挂载到 bentoml 服务上（路径与 bentoml 自带 /healthz 等不冲突）
LLMRouterService.mount_asgi_app(router_app, path="/")
