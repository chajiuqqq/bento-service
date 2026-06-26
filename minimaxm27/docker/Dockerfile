FROM vllm/vllm-openai:latest

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir

WORKDIR /bentoml-workspace
RUN chmod -R 777 /bentoml-workspace
ENTRYPOINT ["bentoml","serve"]
CMD ["--arg-file","./conf.yaml"]
