FROM voipmonitor/sglang:cu130

RUN python3 -m pip install --no-cache-dir bentoml

WORKDIR /bentoml-workspace
RUN chmod -R 777 /bentoml-workspace
ENTRYPOINT ["bentoml","serve"]
CMD ["--arg-file","./conf.yaml"]
