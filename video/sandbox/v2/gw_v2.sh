#!/bin/bash
# (re)start vi-gateway-v2: DeepSeek official API (deepseek-flash = V4.1 Flash), 40 tool steps, 25-min wall budget.
# The API key is read from stdin (piped from the operator's machine) and never stored.
docker rm -f vi-gateway-v2 >/dev/null 2>&1
cd ~/videoindex/v2/sandbox
export VI_DATA=~/videoindex/data VI_RESEARCH=~/videoindex/research VI_VDMSCAN=~/videoindex/v2/vdmscan \
       VI_NET=vi-internal-v2 VI_GW=vi-gateway-v2 GATEWAY_UPSTREAM=api.deepseek.com GATEWAY_UPSTREAM_PREFIX=/anthropic \
       GATEWAY_UPSTREAM_MODEL=deepseek-flash GATEWAY_MAX_STEPS=40 GATEWAY_WARN_STEPS=8 GATEWAY_TIME_BUDGET_S=1500 \
       GATEWAY_MAX_INFLIGHT=32 VDM_UPSTREAM=http://172.17.0.1:18084 GATEWAY_LOG=/logs/usage_v2.jsonl
exec timeout 60 python3 run.py --start-gateway
