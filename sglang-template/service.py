from __future__ import annotations

import json
import logging
import os
import sys
import typing

import bentoml
import fastapi
import httpx
import pydantic

logger = logging.getLogger(__name__)
openai_api_app = fastapi.FastAPI()
body_logger = logging.getLogger('openai.body')

if typing.TYPE_CHECKING:
  Jsonable = list[str] | list[dict[str, str]] | None
else:
  Jsonable = typing.Any


def _ensure_body_logger() -> logging.Logger:
  if body_logger.handlers:
    return body_logger

  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
  body_logger.setLevel(logging.INFO)
  body_logger.addHandler(handler)
  body_logger.propagate = False
  return body_logger


def _decode_bytes(data: bytes) -> str:
  return data.decode('utf-8', errors='replace')


def _format_body_for_log(body: bytes, content_type: str | None) -> str:
  text = _decode_bytes(body)
  if not text:
    return ''

  if content_type and 'application/json' in content_type:
    try:
      return json.dumps(json.loads(text), ensure_ascii=False)
    except json.JSONDecodeError:
      return text
  return text


def _merge_stream_response_body(body: bytes) -> str:
  text = _decode_bytes(body)
  if not text:
    return ''

  merged: dict[str, typing.Any] = {'object': 'chat.completion.stream_aggregated', 'choices': []}
  choices: dict[int, dict[str, typing.Any]] = {}
  usage: typing.Any = None

  for line in text.splitlines():
    line = line.strip()
    if not line or not line.startswith('data:'):
      continue
    payload = line[5:].strip()
    if not payload or payload == '[DONE]':
      continue

    try:
      chunk = json.loads(payload)
    except json.JSONDecodeError:
      continue

    for key in ('id', 'created', 'model', 'system_fingerprint'):
      if key in chunk:
        merged[key] = chunk[key]

    if chunk.get('usage') is not None:
      usage = chunk['usage']

    for choice in chunk.get('choices', []):
      index = choice.get('index', 0)
      entry = choices.setdefault(
        index,
        {
          'index': index,
          'message': {'role': 'assistant', 'content': '', 'reasoning_content': ''},
          'finish_reason': None,
          'matched_stop': None,
        },
      )
      delta = choice.get('delta') or {}
      message = entry['message']

      if delta.get('role') is not None:
        message['role'] = delta['role']
      if delta.get('content'):
        message['content'] += delta['content']
      if delta.get('reasoning_content'):
        message['reasoning_content'] += delta['reasoning_content']
      if delta.get('tool_calls') is not None:
        existing = message.setdefault('tool_calls', [])
        existing.extend(delta['tool_calls'])

      if choice.get('finish_reason') is not None:
        entry['finish_reason'] = choice['finish_reason']
      if choice.get('matched_stop') is not None:
        entry['matched_stop'] = choice['matched_stop']

  merged['choices'] = [choices[index] for index in sorted(choices)]
  if usage is not None:
    merged['usage'] = usage
  return json.dumps(merged, ensure_ascii=False, default=str)


def _log_openai_request(request: fastapi.Request, body: bytes) -> None:
  body_text = _format_body_for_log(body, request.headers.get('content-type')) if bento_args.log_openai_bodies else '<omitted>'
  _ensure_body_logger().info(
    'openai_request method=%s path=%s query=%s headers=%s body=%s',
    request.method,
    request.url.path,
    request.url.query,
    json.dumps(dict(request.headers), ensure_ascii=False, default=str),
    body_text,
  )


def _log_openai_response(
  request: fastapi.Request,
  *,
  status_code: int,
  headers: dict[str, str],
  body: bytes,
) -> None:
  if bento_args.log_openai_bodies:
    if headers.get('content-type') and 'text/event-stream' in headers['content-type']:
      body_text = _merge_stream_response_body(body)
    else:
      body_text = _format_body_for_log(body, headers.get('content-type'))
  else:
    body_text = '<omitted>'
  _ensure_body_logger().info(
    'openai_response method=%s path=%s status=%s headers=%s body=%s',
    request.method,
    request.url.path,
    status_code,
    json.dumps(headers, ensure_ascii=False, default=str),
    body_text,
  )


class BentoArgs(pydantic.BaseModel):
  tp: int = 4
  dp: int | None = None
  port: int = 8000
  host: str = '0.0.0.0'
  log_openai_bodies: bool = True
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


@openai_api_app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
async def openai_proxy(path: str, request: fastapi.Request):
  upstream_url = httpx.URL(f'http://127.0.0.1:{bento_args.port}/v1/{path}').copy_with(
    query=request.url.query.encode('utf-8')
  )
  request_body = await request.body()
  request_headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
  _log_openai_request(request, request_body)

  client = httpx.AsyncClient(timeout=None)
  upstream_request = client.build_request(
    method=request.method,
    url=upstream_url,
    headers=request_headers,
    content=request_body,
  )
  upstream_response = await client.send(upstream_request, stream=True)
  response_headers = {k: v for k, v in upstream_response.headers.items() if k.lower() != 'content-length'}
  content_type = upstream_response.headers.get('content-type')

  if content_type and 'text/event-stream' in content_type:
    response_chunks: list[bytes] = []

    async def stream_response():
      try:
        async for chunk in upstream_response.aiter_bytes():
          response_chunks.append(chunk)
          yield chunk
      finally:
        _log_openai_response(
          request,
          status_code=upstream_response.status_code,
          headers=response_headers,
          body=b''.join(response_chunks),
        )
        await upstream_response.aclose()
        await client.aclose()

    return fastapi.responses.StreamingResponse(
      stream_response(),
      status_code=upstream_response.status_code,
      headers=response_headers,
      media_type=content_type,
    )

  try:
    response_body = await upstream_response.aread()
  finally:
    await upstream_response.aclose()
    await client.aclose()

  _log_openai_response(
    request,
    status_code=upstream_response.status_code,
    headers=response_headers,
    body=response_body,
  )
  return fastapi.Response(
    content=response_body,
    status_code=upstream_response.status_code,
    headers=response_headers,
    media_type=content_type,
  )


@bentoml.asgi_app(openai_api_app, path='/v1')
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
