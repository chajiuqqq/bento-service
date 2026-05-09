#!/bin/bash
# 让 BentoML 的 venv 能访问原镜像 /opt/venv 里的包（sglang、torch 等）
echo "/opt/venv/lib/python3.12/site-packages" > /app/.venv/lib/python3.12/site-packages/opt_venv.pth
