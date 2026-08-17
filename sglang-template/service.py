from __future__ import annotations

import json
import logging
import os
import typing

import bentoml
import httpx
import pydantic

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    Jsonable = list[str] | list[dict[str, str]] | None
else:
    Jsonable = typing.Any


class BentoArgs(pydantic.BaseModel):

    # ===== 基础配置 =====
    name: str = "qwen3-8b"
    engine_port: int = 8000

    # ===== vLLM 参数 =====
    cli_args: list[str] = pydantic.Field(default_factory=list)

    exclude: list[str] = pydantic.Field(
        default_factory=lambda: ["*.pth", "*.pt", "original/**/*"]
    )

    @property
    def additional_cli_args(self) -> list[str]:
        
        default = [
            *self.cli_args,
        ]
 
        return default

    @property
    def runtime_envs(self) -> list[dict[str, str]]:
        return [
            # {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
        ]



bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
    name=bento_args.name,
    envs=bento_args.runtime_envs,
    traffic={"timeout": 300},
    endpoints={"readyz": "/health"},
    workers=1,
)
class SGL:

    def __command__(self) -> list[str]:
        return [
          'python3',
          '-m',
          'sglang.launch_server',
          "--served-model-name",
          bento_args.name,
          "--host",
          "0.0.0.0",
          "--port",
          str(bento_args.engine_port),
          *bento_args.additional_cli_args,
        ]

    async def __metrics__(self, content: str) -> str:

        client = typing.cast(
            httpx.AsyncClient,
            LLM.context.state["client"]
        )

        try:
            response = await client.get(
                f"http://localhost:{bento_args.engine_port}/metrics",
                timeout=5.0,
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.RequestError) as e:
            logger.error("Failed to get metrics: %s", e)
            return content
        else:
            return content + "\n" + response.text
