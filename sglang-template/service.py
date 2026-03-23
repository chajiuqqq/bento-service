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
  tp: int = 4
  dp: int | None = None
  port: int = 8000
  host: str = '0.0.0.0'
  mem_fraction_static: float = 0.85
  max_session_len: int = 32 * 1024
  max_tokens: int = 16 * 1024
  reasoning_parser: str | None = 'qwen3'
  tool_parser: str | None = 'qwen25'
  trust_remote_code: bool = False

  name: str = 'bentosglang-service'
  gpu_type: str = 'nvidia-h100-80gb'
  model_id: str = 'Qwen/Qwen3-235B-A22B-FP8'
  local_model_path: str | None = None

  post: list[str] = pydantic.Field(default_factory=list)
  cli_args: list[str] = pydantic.Field(default_factory=list)
  envs: list[dict[str, str]] = pydantic.Field(default_factory=list)
  exclude: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*'])
  metadata: dict[str, typing.Any] = pydantic.Field(
    default_factory=lambda: {
      'description': 'SGLang OpenAI-compatible service',
      'provider': 'Custom',
      'gpu_recommendation': 'An NVIDIA GPU sized for the selected model.',
    }
  )

  @pydantic.field_validator('exclude', 'cli_args', 'post', 'envs', 'metadata', mode='before')
  @classmethod
  def _coerce_json_or_csv(cls, value: typing.Any) -> Jsonable:
    if value is None or isinstance(value, (list, dict)):
      return typing.cast(Jsonable, value)
    if isinstance(value, str):
      try:
        return typing.cast(Jsonable, json.loads(value))
      except json.JSONDecodeError:
        return [item.strip() for item in value.split(',') if item.strip()]
    return typing.cast(Jsonable, value)

  @property
  def additional_cli_args(self) -> list[str]:
    import torch

    auto_tp_device = str(os.environ.get('TP_AUTO_ALLOCATE', True)).lower() in {'1', 'true', 'yes', 'y'}
    if auto_tp_device:
      tp_rank = torch.cuda.device_count()
    else:
      tp_rank = self.tp

    default = [
      '--host',
      self.host,
      '--port',
      str(self.port),
      '--tp',
      str(tp_rank),
      '--context-length',
      str(self.max_session_len),
      '--mem-fraction-static',
      str(self.mem_fraction_static),
      '--served-model-name',
      self.served_model_name,
      *self.cli_args,
    ]
    if self.trust_remote_code or self.local_model_path:
      default.append('--trust-remote-code')
    return default

  @property
  def additional_labels(self) -> dict[str, str]:
    return {
      'reasoning': '1' if self.reasoning_parser else '0',
      'tool': self.tool_parser or '',
      'openai_model': self.served_model_name,
    }

  @property
  def model_source(self) -> str | bentoml.models.HuggingFaceModel:
    if self.local_model_path:
      return os.path.abspath(os.path.expanduser(self.local_model_path))
    return bentoml.models.HuggingFaceModel(self.model_id, exclude=self.exclude)

  @property
  def served_model_name(self) -> str:
    if self.local_model_path and self.model_id == BentoArgs.model_fields['model_id'].default:
      return os.path.basename(os.path.abspath(os.path.expanduser(self.local_model_path.rstrip('/'))))
    return self.model_id

  @property
  def runtime_envs(self) -> list[dict[str, str]]:
    envs = [*self.envs]
    envs.extend([
      {'name': 'MAX_SESSION_LEN', 'value': str(self.max_session_len)},
      {'name': 'MAX_TOKENS', 'value': str(self.max_tokens)},
      {'name': 'NUM_GPUS', 'value': str(self.tp)},
      {'name': 'UV_INDEX_STRATEGY', 'value': 'unsafe-best-match'},
      {'name': 'UV_NO_PROGRESS', 'value': '1'},
    ])
    return envs

  @property
  def image(self) -> bentoml.images.Image:
    image = (
      bentoml.images.Image(
        python_version='3.12',
        base_image='docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04',
        lock_python_packages=False,
      )
      .system_packages('git', 'python3', 'python3-pip', 'libopenmpi-dev')
      .run('ln -sf /usr/bin/pip3 /usr/local/bin/pip')
    )
    if self.post:
      for cmd in self.post:
        image = image.run(cmd)
    return image


bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
  name=bento_args.name,
  envs=bento_args.runtime_envs,
  image=bento_args.image,
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentosglang',
    'openai_endpoint': '/v1',
    **bento_args.additional_labels,
  },
  traffic={'timeout': 300},
  endpoints={'readyz': '/health'},
  resources={'gpu': bento_args.tp, 'gpu_type': bento_args.gpu_type},
  metrics={
    "enabled": True,
    "namespace": "bentoml",
  },
  workers=1,
)
class SGL:
  hf = bento_args.model_source

  def __command__(self) -> list[str]:
    return [
      'python3',
      '-m',
      'sglang.launch_server',
      '--model-path',
      self.hf,
      *bento_args.additional_cli_args,
    ]

  async def __metrics__(self, content: str) -> str:
    client = typing.cast(httpx.AsyncClient, SGL.context.state['client'])
    try:
      response = await client.get(f'http://localhost:{bento_args.port}/metrics', timeout=5.0)
      response.raise_for_status()
    except httpx.HTTPError as exc:
      logger.warning('Failed to get SGLang metrics: %s', exc)
      return content
    else:
      return content + '\n' + response.text
