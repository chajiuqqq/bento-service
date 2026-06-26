from __future__ import annotations

import json
import os
import typing as t

import bentoml
import fastapi
import pydantic

openai_api_app = fastapi.FastAPI()
Jsonable = t.Any


class BentoArgs(pydantic.BaseModel):
  name: str = 'bentosglang-runner-service'
  model_id: str = 'Qwen/Qwen3-235B-A22B-FP8'
  local_model_path: str | None = None
  gpu_type: str = 'nvidia-h100-80gb'
  tp: int = 4
  max_session_len: int = 32 * 1024
  mem_fraction_static: float = 0.85
  tool_parser: str | None = 'qwen25'
  reasoning_parser: str | None = 'qwen3'
  trust_remote_code: bool = False
  server_args: dict[str, t.Any] = pydantic.Field(default_factory=dict)
  exclude: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*'])
  envs: list[dict[str, str]] = pydantic.Field(default_factory=list)

  @pydantic.field_validator('exclude', 'envs', 'server_args', mode='before')
  @classmethod
  def _coerce_json_or_csv(cls, value: t.Any) -> Jsonable:
    if value is None or isinstance(value, (list, dict)):
      return value
    if isinstance(value, str):
      try:
        return json.loads(value)
      except json.JSONDecodeError:
        return [item.strip() for item in value.split(',') if item.strip()]
    return value

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
    return [
      {'name': 'MAX_SESSION_LEN', 'value': str(self.max_session_len)},
      {'name': 'NUM_GPUS', 'value': str(self.tp)},
      {'name': 'UV_INDEX_STRATEGY', 'value': 'unsafe-best-match'},
      {'name': 'UV_NO_PROGRESS', 'value': '1'},
      *self.envs,
    ]

  @property
  def image(self) -> bentoml.images.Image:
    return (
      bentoml.images.Image(
        python_version='3.12',
        base_image='docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04',
        lock_python_packages=False,
      )
      .system_packages('git', 'python3', 'python3-pip', 'libopenmpi-dev')
      .run('ln -sf /usr/bin/pip3 /usr/local/bin/pip')
      .requirements_file('requirements.txt')
    )


bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
  name=bento_args.name,
  image=bento_args.image,
  envs=bento_args.runtime_envs,
  traffic={'timeout': 300},
  resources={'gpu': bento_args.tp, 'gpu_type': bento_args.gpu_type},
)
class SGL:
  hf_model = bento_args.model_source

  def __init__(self) -> None:
    import sglang as sgl
    from fastapi import Request
    from fastapi.responses import ORJSONResponse
    from sglang.srt.openai_api.adapter import v1_chat_completions, v1_completions
    from sglang.srt.openai_api.protocol import ModelCard, ModelList
    from sglang.srt.server_args import ServerArgs

    server_args_kwargs: dict[str, t.Any] = {
      'model_path': self.hf_model,
      'served_model_name': bento_args.served_model_name,
      'context_length': bento_args.max_session_len,
      'mem_fraction_static': bento_args.mem_fraction_static,
      'tp_size': bento_args.tp,
      **bento_args.server_args,
    }
    if bento_args.tool_parser:
      server_args_kwargs.setdefault('tool_call_parser', bento_args.tool_parser)
    if bento_args.reasoning_parser:
      server_args_kwargs.setdefault('reasoning_parser', bento_args.reasoning_parser)
    if bento_args.trust_remote_code or bento_args.local_model_path:
      server_args_kwargs.setdefault('trust_remote_code', True)

    self.engine = sgl.Engine(server_args=ServerArgs(**server_args_kwargs))

    @openai_api_app.post('/completions')
    async def openai_v1_completions(raw_request: Request):
      return await v1_completions(self.engine.tokenizer_manager, raw_request)

    @openai_api_app.post('/chat/completions')
    async def openai_v1_chat_completions(raw_request: Request):
      return await v1_chat_completions(self.engine.tokenizer_manager, raw_request)

    @openai_api_app.get('/models', response_class=ORJSONResponse)
    def available_models():
      served_model_names = [self.engine.tokenizer_manager.served_model_name]
      model_cards = [ModelCard(id=name, root=name) for name in served_model_names]
      return ModelList(data=model_cards)

  @bentoml.on_shutdown
  def shutdown(self) -> None:
    self.engine.shutdown()
