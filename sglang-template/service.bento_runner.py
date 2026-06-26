from __future__ import annotations

import json
import os
import typing as t

import bentoml
import pydantic

Jsonable = t.Any


class BentoArgs(pydantic.BaseModel):
  name: str = 'bentosglang-bento-runner-service'
  model_id: str = 'Qwen/Qwen3-235B-A22B-FP8'
  local_model_path: str | None = None
  gpu_type: str = 'nvidia-h100-80gb'
  tp: int = 4
  max_session_len: int = 32 * 1024
  mem_fraction_static: float = 0.85
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


class SGLangRunnable(bentoml.Runnable):
  SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'gpu')
  SUPPORTS_CPU_MULTI_THREADING = False

  def __init__(
    self,
    model_path: str,
    served_model_name: str,
    tp: int,
    max_session_len: int,
    mem_fraction_static: float,
    trust_remote_code: bool,
    server_args: dict[str, t.Any],
  ) -> None:
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs

    server_args_kwargs: dict[str, t.Any] = {
      'model_path': model_path,
      'served_model_name': served_model_name,
      'context_length': max_session_len,
      'mem_fraction_static': mem_fraction_static,
      'tp_size': tp,
      **server_args,
    }
    if trust_remote_code:
      server_args_kwargs.setdefault('trust_remote_code', True)

    self.engine = sgl.Engine(server_args=ServerArgs(**server_args_kwargs))

  @bentoml.Runnable.method(batchable=False)
  async def generate(self, prompt: str, sampling_params: dict[str, t.Any] | None = None) -> str:
    if sampling_params is None:
      sampling_params = {}
    result = await self.engine.async_generate(prompt, sampling_params=sampling_params)
    if isinstance(result, dict):
      text = result.get('text', '')
      if isinstance(text, list):
        return ''.join(t.cast(list[str], text))
      return t.cast(str, text)
    return t.cast(str, result)

  def teardown(self) -> None:
    self.engine.shutdown()


sglang_runner = bentoml.Runner(
  SGLangRunnable,
  name='sglang_runner',
  runnable_init_params={
    'model_path': t.cast(str, bento_args.model_source),
    'served_model_name': bento_args.served_model_name,
    'tp': bento_args.tp,
    'max_session_len': bento_args.max_session_len,
    'mem_fraction_static': bento_args.mem_fraction_static,
    'trust_remote_code': bento_args.trust_remote_code or bool(bento_args.local_model_path),
    'server_args': bento_args.server_args,
  },
)

svc = bentoml.Service(
  name=bento_args.name,
  runners=[sglang_runner],
)


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.Text())
async def generate(input_json: dict[str, t.Any]) -> str:
  prompt = input_json['prompt']
  sampling_params = input_json.get('sampling_params')
  return await sglang_runner.generate.async_run(prompt, sampling_params)
