from __future__ import annotations

import datetime
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

if typing.TYPE_CHECKING:
  Jsonable = list[str] | list[dict[str, str]] | None
else:
  Jsonable = typing.Any


class BentoArgs(pydantic.BaseModel):
  class BentoConfig(pydantic.BaseModel):
    tp: int = 1
    port: int = 8000
    attn_backend: str = 'FLASH_ATTN'
    nightly: bool = False

    name: str = 'llama3.1-8b-instruct'
    gpu_type: str = 'nvidia-h100-80gb'
    model_id: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    local_model_path: str | None = None

    post: list[str] = pydantic.Field(default_factory=list)
    envs: list[dict[str, str]] = pydantic.Field(default_factory=list)
    exclude: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*'])
    hf_generation_config: dict[str, float | int] = pydantic.Field(
      default_factory=lambda: {'repetition_penalty': 1.0, 'temperature': 0.6, 'top_p': 0.9}
    )
    metadata: dict[str, typing.Any] = pydantic.Field(
      default_factory=lambda: {
        'description': 'Llama 3.1 8B Instruct',
        'provider': 'Meta',
        'gpu_recommendation': 'an Nvidia GPU with at least 80GB VRAM (e.g about 1 H100 GPU).',
      }
    )
    use_sglang_router: bool = False
    log_body: bool = False
    log_level: str = 'INFO'
    trace_header: str = 'X-Oneapi-Request-Id'

  tp: int = 1
  port: int = 8000
  attn_backend: str = 'FLASH_ATTN'
  nightly: bool = False

  name: str = 'llama3.1-8b-instruct'
  gpu_type: str = 'nvidia-h100-80gb'
  model_id: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
  local_model_path: str | None = None

  post: list[str] = pydantic.Field(default_factory=list)
  cli_args: list[str] = pydantic.Field(default_factory=list)
  envs: list[dict[str, str]] = pydantic.Field(default_factory=list)
  exclude: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*'])
  hf_generation_config: dict[str, float | int] = pydantic.Field(
    default_factory=lambda: {'repetition_penalty': 1.0, 'temperature': 0.6, 'top_p': 0.9}
  )
  metadata: dict[str, typing.Any] = pydantic.Field(
    default_factory=lambda: {
      'description': 'Llama 3.1 8B Instruct',
      'provider': 'Meta',
      'gpu_recommendation': 'an Nvidia GPU with at least 80GB VRAM (e.g about 1 H100 GPU).',
    }
  )
  use_sglang_router: bool = False
  log_body: bool = False
  log_level: str = 'INFO'
  trace_header: str = 'X-Oneapi-Request-Id'

  @pydantic.field_validator('exclude', 'cli_args', 'post', 'envs', 'hf_generation_config', 'metadata', mode='before')
  @classmethod
  def _coerce_json_or_csv(cls, v: typing.Any) -> Jsonable:
    if v is None or isinstance(v, (list, dict)):
      return typing.cast(Jsonable, v)
    if isinstance(v, str):
      try:
        return typing.cast(Jsonable, json.loads(v))
      except json.JSONDecodeError:
        return [item.strip() for item in v.split(',') if item.strip()]
    return typing.cast(Jsonable, v)

  @property
  def bentoargs(self) -> BentoConfig:
    return self.BentoConfig.model_validate(self.model_dump(exclude={'cli_args'}))

  @staticmethod
  def _find_cli_arg(flag: str, cli_args: list[str]) -> str | None:
    for index, arg in enumerate(cli_args):
      if arg == flag and index + 1 < len(cli_args):
        return cli_args[index + 1]
      if arg.startswith(f'{flag}='):
        return arg.split('=', 1)[1]
    return None

  @property
  def additional_cli_args(self) -> list[str]:
    default = [*self.cli_args]
    if '--tensor-parallel-size' not in default and not any(arg.startswith('--tensor-parallel-size=') for arg in default):
      default.extend(['--tensor-parallel-size', str(self.bentoargs.tp)])
    if '--served-model-name' not in default and not any(arg.startswith('--served-model-name=') for arg in default):
      default.extend(['--served-model-name', self.served_model_name])
    if self.local_model_path and '--trust-remote-code' not in default:
      default.append('--trust-remote-code')
    return default

  @property
  def additional_labels(self) -> dict[str, str]:
    tool_parser = self._find_cli_arg('--tool-call-parser', self.cli_args) or ''
    reasoning_parser = self._find_cli_arg('--reasoning-parser', self.cli_args)
    default = {
      'hf_generation_config': json.dumps(self.hf_generation_config),
      'reasoning': '1' if reasoning_parser else '0',
      'tool': tool_parser,
      'openai_model': self.served_model_name,
    }
    return default

  @property
  def model_source(self) -> str | bentoml.models.HuggingFaceModel:
    bentoargs = self.bentoargs
    if bentoargs.local_model_path:
      return os.path.abspath(os.path.expanduser(bentoargs.local_model_path))
    return bentoml.models.HuggingFaceModel(bentoargs.model_id, exclude=bentoargs.exclude)

  @property
  def served_model_name(self) -> str:
    served_model_name = self._find_cli_arg('--served-model-name', self.cli_args)
    if served_model_name:
      return served_model_name
    bentoargs = self.bentoargs
    if bentoargs.local_model_path and bentoargs.model_id == BentoArgs.model_fields['model_id'].default:
      return os.path.basename(os.path.abspath(os.path.expanduser(bentoargs.local_model_path.rstrip('/'))))
    return bentoargs.model_id

  @property
  def runtime_envs(self) -> list[dict[str, str]]:
    bentoargs = self.bentoargs
    envs = [*bentoargs.envs]
    envs.extend([
      {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
      {'name': 'UV_NO_PROGRESS', 'value': '1'},
      {'name': 'UV_TORCH_BACKEND', 'value': 'cu128'},
    ])
    if not bentoargs.gpu_type.startswith('amd'):
      envs.extend([
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': bentoargs.attn_backend},
        {'name': 'TORCH_CUDA_ARCH_LIST', 'value': '7.5 8.0 8.9 9.0a 10.0a 12.0'},
      ])
    if os.getenv('YATAI_T_VERSION'):
      envs.extend([
        {'name': 'HF_HUB_CACHE', 'value': '/home/bentoml/bento/hf-models'},
        {'name': 'VLLM_CACHE_ROOT', 'value': '/home/bentoml/bento/vllm-models'},
      ])
    return envs

  @property
  def image(self) -> bentoml.images.Image:
    bentoargs = self.bentoargs
    image = (
      bentoml.images.Image(
        python_version='3.12',
        base_image="nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04",
        lock_python_packages=False,
      )
      .system_packages('curl', 'git', 'python3', 'python3-pip')
      .run('ln -sf /usr/bin/pip3 /usr/local/bin/pip')
      .requirements_file('requirements.txt')
    )
    if bentoargs.post:
      for cmd in bentoargs.post:
        image = image.run(cmd)

    if False:  # self.gpu_type.startswith('nvidia'):
      image = image.run('uv pip install flashinfer-python flashinfer-cubin --torch-backend=cu128')
      image = image.run('uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128')

    if bentoargs.gpu_type.startswith('amd'):
      image.base_image = 'rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909'
      # Disable locking of Python packages for AMD GPUs to exclude nvidia-* dependencies
      image.lock_python_packages = False
      # The GPU device is accessible by group 992
      image.run('groupadd -g 992 -o rocm && usermod -aG rocm bentoml && usermod -aG render bentoml')
      # Remove the vllm and torch deps to reuse the pre-installed ones in the base image
      image.run('uv pip uninstall vllm torch torchvision torchaudio triton')

    if bentoargs.nightly:
      image.run('uv pip uninstall vllm')
      image.run('uv pip install -U vllm --torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly')

    return image


bento_args = bentoml.use_arguments(BentoArgs)


# ── JSON 结构化日志 ──────────────────────────────────────────────

def _setup_json_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(bento_args.bentoargs.log_level.upper())

    class _JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log = {
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }
            trace_id = getattr(record, 'trace_id', None)
            if trace_id:
                log['trace_id'] = trace_id
            event = getattr(record, 'event', None)
            if event:
                log['event'] = event
            extra = getattr(record, 'extra', None)
            if extra:
                log.update(extra)
            return json.dumps(log, ensure_ascii=False)

    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(bento_args.bentoargs.log_level.upper())


def _log_event(event: str, trace_id: str = '', **extra) -> None:
    logger.info(event, extra={'trace_id': trace_id, 'event': event, 'extra': extra})


_setup_json_logging()

# ── FastAPI 代理层 ──────────────────────────────────────────────

TRACE_HEADER = bento_args.bentoargs.trace_header
VLLM_PORT = bento_args.bentoargs.port
LOG_BODY = bento_args.bentoargs.log_body

openai_api_app = fastapi.FastAPI()


@openai_api_app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
async def proxy(path: str, request: fastapi.Request):
    trace_id = request.headers.get(TRACE_HEADER, '')
    start = datetime.datetime.now(datetime.timezone.utc)

    body = await request.body()
    _log_event('request_received', trace_id,
               method=request.method, path=f'/v1/{path}',
               body=body.decode(errors='replace') if LOG_BODY else '',
               content_type=request.headers.get('content-type', ''))

    upstream_url = f'http://127.0.0.1:{VLLM_PORT}/v1/{path}'

    headers = dict(request.headers)
    headers.pop('host', None)
    headers[TRACE_HEADER] = trace_id

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_resp = await client.request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=body,
        )
    except Exception as e:
        _log_event('error', trace_id, error=str(e), elapsed_ms=round(
            (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds() * 1000, 1))
        await client.aclose()
        return fastapi.Response('upstream error', status_code=502)

    latency = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds() * 1000
    _log_event('upstream_response', trace_id,
               status=upstream_resp.status_code, latency_ms=round(latency, 1))

    resp_headers = dict(upstream_resp.headers)
    content_type = upstream_resp.headers.get('content-type', '')

    if 'text/event-stream' in content_type:
        _log_event('stream_start', trace_id)

        async def stream():
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            except Exception as e:
                _log_event('stream_error', trace_id, error=str(e))
            finally:
                total_ms = round((datetime.datetime.now(datetime.timezone.utc) - start).total_seconds() * 1000, 1)
                _log_event('stream_end', trace_id, duration_ms=total_ms)
                await upstream_resp.aclose()
                await client.aclose()

        return fastapi.responses.StreamingResponse(
            stream(), status_code=upstream_resp.status_code, headers=resp_headers,
            media_type=content_type)

    resp_body = await upstream_resp.aread()
    await upstream_resp.aclose()
    await client.aclose()
    return fastapi.Response(content=resp_body, status_code=upstream_resp.status_code,
                            headers=resp_headers, media_type=content_type)

if bento_args.bentoargs.use_sglang_router:
  from bento_sgl_router import service
else:
  service = bentoml.service


@bentoml.asgi_app(openai_api_app, path='/v1')
@service(
  name=bento_args.bentoargs.name,
  envs=bento_args.runtime_envs,
  image=bento_args.image,
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentovllm',
    'openai_endpoint': '/v1',
    **bento_args.additional_labels,
  },
  traffic={'timeout': 300},
  endpoints={'readyz': '/health'},
  resources={'gpu': bento_args.bentoargs.tp, 'gpu_type': bento_args.bentoargs.gpu_type},
)
class LLM:
  hf = bento_args.model_source

  def __command__(self) -> list[str]:
    return [
      'vllm',
      'serve',
      self.hf,
      '--port',
      str(bento_args.bentoargs.port),
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      *bento_args.additional_cli_args,
    ]

  async def __metrics__(self, content: str) -> str:
    client = typing.cast(httpx.AsyncClient, LLM.context.state['client'])
    try:
      response = await client.get(f'http://localhost:{bento_args.bentoargs.port}/metrics', timeout=5.0)
      response.raise_for_status()
    except (httpx.ConnectError, httpx.RequestError) as e:
      logger.error('Failed to get metrics: %s', e)
      return content
    else:
      return content + '\n' + response.text
