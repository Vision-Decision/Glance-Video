# Local replica of the Video-Index v1.1 agent protocol

One Docker container per question, the official Claude Code configuration, any model behind an
Anthropic-compatible endpoint. Built to run open planners (now: DeepSeek V4.1 Flash on Ollama cloud) and,
later, the same planner with and without `vdm`.

## What is replicated

| Official v1.1 (from `command.json` in the traces) | Here |
|---|---|
| `video.mp4` in the working dir, original file, read-only | same (`/work/video.mp4`, read-only bind mount) |
| ffmpeg/ffprobe, Python + OpenCV, Pillow, NumPy, faster-whisper; Whisper large-v3 and small on CPU | same, in `Dockerfile` (Debian bookworm, venv `/opt/eval`); models mounted read-only at `/opt/models/faster-whisper/{large-v3,small}` (CTranslate2 weights from `Systran/faster-whisper-*`) |
| Claude Code 2.1.300, `-p --effort max --output-format stream-json --tools Bash,Read --permission-mode dontAsk --json-schema … --system-prompt …`, empty settings and MCP | same flags; system prompt, user prompt and JSON schema copied verbatim from the official `claude-opus-5-5` trace of each item (only the Whisper path is substituted) |
| One attempt, 3600 s; timeout or error = wrong | same; the orchestrator kills the container at 3600 s; no retries |
| Answer = `structured_output.answer_letter` of the final `result` event | same; no answer → status `error` (wrong) |
| No web search/fetch/downloads | stronger: the container sits on an `--internal` Docker network whose only other host is the model gateway; the shell has no internet at all |
| Calls through the Kaggle model proxy | calls through `gateway.py` (see below) |
| Trace files `prompt.txt`, `command.json`, `stdout.jsonl`, `stderr.txt`, `result.json` | same names and same `result.json` fields, plus `schema.json` and the scratch dir `work/` |

## Where it runs

On the GCP host `gen-host` (Ubuntu 24.04, x86-64, 30 vCPU, 295 GB RAM, no GPU), in `~/videoindex/`:
`sandbox/` (this folder), `research/` (items, `scores.csv`, official opus-5-5 `prompt.txt`/`command.json`),
`data/v1_1/` (144 videos, 36.27 GB, all SHA-256 equal to `video_id`), `data/models/faster-whisper/`,
`data/runs/`. Nothing bulky stays on the laptop. The host is shared (a GUI-synthesis job uses most cores), so we
cap ourselves at 5 containers × 2 CPUs = 10 cores.

**Isolation check (2026-11-18, from a container on `vi-internal`):** DNS for huggingface.co fails, TCP to
8.8.8.8:53 and 1.1.1.1:443 "network unreachable", https://ollama.com unreachable; the gateway answers, and a
request for `claude-opus-4-5` is served as `deepseek-v4.1-flash` (allowlist rewrite).

## Deliberate differences

- **Model backend.** `ANTHROPIC_BASE_URL` points at the gateway, which forwards to Ollama cloud's
  Anthropic-compatible `/v1/messages`. Claude Code does not know the model ("unrecognized_model"), so its
  own `total_cost_usd` is meaningless; tokens are counted at the gateway instead.
- **Working directory** is `/work`, not `.../calls/<item_id>`: the official path shows the item id (and so the
  source benchmark) to the agent.
- **Hardware.** Official: 12 calls in parallel per model on a host whose load average was ~100 (failure
  analysis). Here: 5 calls × 2 CPUs on a shared host (load ~50 from another job). Whisper large-v3 on 2 cores is
  slow, which matters for long videos with audio. Time per question is only roughly comparable.
- **bwrap → Docker.** Same effect (read-only system, private /tmp, own HOME/config, PID namespace), plus no network.

## Gateway (`gateway.py`)

The agent container has only a dummy token. The gateway (a stdlib Python server, in its own container on the
internal network and the default bridge) reads the real Ollama key **from its stdin** at start-up (piped from
the laptop's `.env` through ssh; never in argv, container env, files or logs) and adds it upstream. It forwards
only `POST /c/<run>.<item>/v1/messages`, **rewrites any model name outside the allowlist to the configured
DeepSeek model** (so nothing else can be called, by accident or by the agent), streams responses back and
logs per-request usage to `data/video_index/runs/_gateway/usage.jsonl`. Telemetry and other paths get a local 404.
The key never appears on a command line, in a file or in a log.

## Use

```bash
# on the host (once): docker, image, data
sudo apt-get install -y docker.io && sudo usermod -aG docker $USER
curl -LO --output-dir sandbox/vendor https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.300.tgz
docker build -t vi-agent:cc2.1.300 sandbox/
VI_DATA=$PWD/data HF_HUB_DISABLE_XET=0 bash sandbox/fetch.sh          # whisper models + 144 videos
export VI_DATA=$PWD/data VI_RESEARCH=$PWD/research

# from the laptop: start the gateway, key only on the ssh stdin
grep '^OLLAMA_API_KEY=' .env | cut -d= -f2- | ssh gen-host 'cd ~/videoindex && VI_DATA=... VI_RESEARCH=... python3 sandbox/run.py --start-gateway'

# on the host
python3 sandbox/run.py --run pilot10 --sample 10            # stratified pilot (seed 0)
python3 sandbox/run.py --run full1                           # all 144; resumable
python3 sandbox/summarize.py --run full1                     # accuracy, time, tokens, cost-equivalent
python3 sandbox/export_traces.py $VI_DATA/runs full1 out/    # redacted traces (images -> sha256+size)
```

Outputs: `$VI_DATA/runs/<run>/<model>/<item_id>/` (trace), `<run>/per_question.csv`, `<run>/summary.json`.
Exported, redacted traces for the laptop/repo go to `video/results/<run>/`.
Adding `vdm` later = mount the tool into the image and append its description to the system prompt; nothing else.

## Token accounting note

Ollama's streamed `message_start` event reports a rough input-token pre-count that counts base64 images as text
(~100k+ "tokens" per frame); the final `message_delta` has the real usage (~200–600 tokens per image). The
gateway uses the final numbers and keeps the pre-count as `start_estimate_input_tokens`.
