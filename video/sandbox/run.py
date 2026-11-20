#!/usr/bin/env python3
"""Run the Video-Index v1.1 agent protocol locally, one docker container per question.

Replicates the official Claude Code configuration (same system prompt, user prompt, JSON schema, flags,
tools Bash+Read, one attempt, 3600 s, timeout/error = wrong), taken verbatim from the official
claude-opus-5-5 trace of each item, with the model swapped and the API pointed at our gateway.

Usage:
  python3 run.py --start-gateway < keyfile-or-pipe   # reads the Ollama key from stdin (once per host boot)
  python3 run.py --run pilot10 --sample 10
  python3 run.py --run full1                          # all 144; resumable
Paths default to the repo layout; on a remote host set VI_DATA (videos/models/runs) and VI_RESEARCH
(items + official opus-5-5 prompt/command files).
The key is never put on a command line, in an environment variable of a container, in a file or a log:
it is piped into the gateway container's stdin, which keeps it only in memory.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEARCH = os.environ.get("VI_RESEARCH", os.path.join(ROOT, "data/research/video_index_v1_1"))
DATA = os.environ.get("VI_DATA", os.path.join(ROOT, "data/video_index"))
OFFICIAL = os.path.join(RESEARCH, "traces/claude-opus-5-5/claude-opus-5-5")
ITEMS = os.path.join(RESEARCH, "data/test.jsonl")
VIDEOS = os.path.join(DATA, "v1_1")
MODELS = os.path.join(DATA, "models/faster-whisper")
RUNS = os.path.join(DATA, "runs")
NET = os.environ.get("VI_NET", "vi-internal")      # vdm runs use their own pair (vi-internal-vdm / vi-gateway-vdm)
GW = os.environ.get("VI_GW", "vi-gateway")
IMAGE = "vi-agent:cc2.1.300"
GEMINI_IMAGE = "vi-agent-gemini:0.64.0"
WHISPER_IN = "/opt/models/faster-whisper"
VDM_DIR = os.environ.get("VI_VDM", os.path.join(ROOT, "video/vdm"))
VDM_CLIENT, VDM_TOOL_MD = os.path.join(VDM_DIR, "vdm"), os.path.join(VDM_DIR, "TOOL.md")
SCAN_DIR = os.environ.get("VI_VDMSCAN", os.path.join(ROOT, "video/vdm/gv"))
SCAN_CLIENT, SCAN_TOOL_MD = os.path.join(SCAN_DIR, "vdmscan"), os.path.join(SCAN_DIR, "TOOL_scan.md")
_print = threading.Lock()


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


SCAN_URL = os.environ.get("VI_SCAN_URL", "http://127.0.0.1:18084")   # host-side address of the scan server (warm/release)
CACHE = os.environ.get("VI_SCAN_CACHE", os.path.join(DATA, "scan_cache"))  # per-video frame caches (host)
TRANSCRIPT_NOTE = ("A speech transcript of video.mp4 made with faster-whisper small is at transcript.tsv in the working "
                   "directory (one segment per line: start_s<TAB>end_s<TAB>text; empty if there is no speech; it may "
                   "contain recognition errors).")


def video_id(path):
    """same content id as vdmscan.video_id (size + sha1 of the first and last MiB)."""
    import hashlib
    st = os.stat(path)
    h = hashlib.sha1(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, st.st_size - (1 << 20)))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:20]


def gw_status():
    ip = sh(["docker", "inspect", "-f", "{{.NetworkSettings.Networks.bridge.IPAddress}}", GW]).stdout.strip()
    try:
        import urllib.request
        return json.load(urllib.request.urlopen(f"http://{ip}:8080/gw/status", timeout=10))
    except Exception:
        return {}


def scan_post(path, body, timeout=30):
    import urllib.request
    try:
        req = urllib.request.Request(SCAN_URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:
        return {"error": str(e)[:200]}


ABORT = threading.Event()
RL = {}   # latest Claude-subscription windows seen in any finished Opus run: {"five_hour": {...}, "seven_day": {...}}


def rl_gate(a):
    """Opus: before launching, wait out a nearly full 5-hour window; stop the run above the 7-day cap."""
    while a.backend == "opus" and RL and not ABORT.is_set():
        w7, w5 = RL.get("seven_day") or {}, RL.get("five_hour") or {}
        if w7.get("utilization", 0) >= a.max_week:
            print(f"ABORT: 7-day subscription utilisation {w7['utilization']:.2f} >= {a.max_week}", flush=True)
            ABORT.set()
            return
        if w5.get("utilization", 0) < a.max_5h:
            return
        wait = max(60, (w5.get("resetsAt") or time.time()) - time.time() + 60)
        print(f"5-hour window at {w5['utilization']:.2f}: waiting {wait/60:.0f} min for its reset", flush=True)
        RL.clear()   # re-learn from the next run after the reset
        time.sleep(wait)


def start_gateway(model):
    """Start the gateway container; the key comes from our stdin and is piped into the container's stdin."""
    if sh(["docker", "network", "inspect", NET]).returncode:
        sh(["docker", "network", "create", "--internal", NET])
    if sh(["docker", "inspect", "-f", "{{.State.Running}}", GW]).stdout.strip() == "true":
        print("gateway already running")
        return
    key = "" if os.environ.get("GATEWAY_CONNECT_ALLOW") else sys.stdin.readline().strip()
    if not key and not os.environ.get("GATEWAY_CONNECT_ALLOW"):
        sys.exit("no key on stdin")
    sh(["docker", "rm", "-f", GW])
    logdir = os.path.join(RUNS, "_gateway")
    os.makedirs(logdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    err = open(os.path.join(logdir, "gateway.stderr"), "a")
    p = subprocess.Popen(["docker", "run", "-i", "--name", GW, "--network", "bridge", "--user", f"{os.getuid()}:{os.getgid()}",
                          "-e", f"GATEWAY_MODEL={model}", "-e", f"GATEWAY_ALLOW={model}",
                          "-e", f"GATEWAY_DUMP_PREFIX={os.environ.get('GATEWAY_DUMP_PREFIX', '')}",
                          "-e", f"VDM_UPSTREAM={os.environ.get('VDM_UPSTREAM', '')}",
                          "-e", f"GATEWAY_CONNECT_ALLOW={os.environ.get('GATEWAY_CONNECT_ALLOW', '')}",
                          "-e", f"GATEWAY_MAX_INFLIGHT={os.environ.get('GATEWAY_MAX_INFLIGHT', '6')}",
                          "-e", f"GATEWAY_MODE={os.environ.get('GATEWAY_MODE', 'anthropic')}",
                          *[x for k in ("GATEWAY_UPSTREAM", "GATEWAY_UPSTREAM_PREFIX", "GATEWAY_UPSTREAM_MODEL", "GATEWAY_MAX_STEPS",
                                        "GATEWAY_WARN_STEPS", "GATEWAY_RETRY_S", "GATEWAY_LOG", "GATEWAY_TIME_BUDGET_S") if os.environ.get(k)
                            for x in ("-e", f"{k}={os.environ[k]}")],
                          "-v", f"{here}/gateway.py:/gateway.py:ro", "-v", f"{logdir}:/logs",
                          "--entrypoint", "python3", IMAGE, "/gateway.py"],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, text=True,
                         start_new_session=True)
    p.stdin.write(key + "\n")
    p.stdin.flush()
    del key
    for _ in range(30):
        time.sleep(1)
        if sh(["docker", "inspect", "-f", "{{.State.Running}}", GW]).stdout.strip() == "true":
            break
    sh(["docker", "network", "connect", NET, GW])
    print("gateway started")


def ensure_gateway(model, logdir):
    if sh(["docker", "inspect", "-f", "{{.State.Running}}", GW]).stdout.strip() != "true":
        sys.exit("gateway not running: start it with `run.py --start-gateway` (key on stdin)")


LETTERS = "ABCDEFGHIJ"


def permute(item, prompt):
    """Option-permutation control: reorder options with random.Random("42|<item_id>") (Harbor/blind convention).
    Returns (new_prompt, new_key)."""
    opts = item["options"]
    block = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    assert block in prompt, item["item_id"]
    order = list(range(len(opts)))
    random.Random(f"42|{item['item_id']}").shuffle(order)
    new_block = "\n".join(f"{LETTERS[j]}. {opts[i]}" for j, i in enumerate(order))
    return prompt.replace(block, new_block), LETTERS[order.index(item["answer_idx"])]


GEMINI_OFFICIAL = os.path.join(RESEARCH, "traces/gemini-3.8-flash/gemini-3.8-flash")


def official_gemini(item_id):
    d = os.path.join(GEMINI_OFFICIAL, item_id)
    if not os.path.isdir(d):
        d = os.path.join(GEMINI_OFFICIAL, item_id.replace("^", "_"))
    return open(os.path.join(d, "prompt.txt")).read().replace("<lab>/models/faster-whisper", WHISPER_IN)


def official(item_id):
    d = os.path.join(OFFICIAL, item_id)
    if not os.path.isdir(d):
        d = os.path.join(OFFICIAL, item_id.replace("^", "_"))  # trace dirs use "_" for "^"
    cmd = json.load(open(os.path.join(d, "command.json")))
    sp = cmd[cmd.index("--system-prompt") + 1].replace("<lab>/models/faster-whisper", WHISPER_IN)
    schema = cmd[cmd.index("--json-schema") + 1]
    prompt = open(os.path.join(d, "prompt.txt")).read()
    return sp, schema, prompt


def gemini_answer(cdir, nopt):
    """Like the official Gemini runs: scratch/answer.json first, else the JSON object on the last line of the
    final assistant message."""
    letters = LETTERS[:nopt]
    events = []
    for line in open(os.path.join(cdir, "stdout.jsonl"), errors="ignore"):
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    final = next((e for e in reversed(events) if e.get("type") == "result"), None)
    try:
        aj = json.load(open(os.path.join(cdir, "work/scratch/answer.json")))
        if aj.get("answer_letter") in letters:
            return aj["answer_letter"], aj.get("reason", ""), final, ""
    except Exception:
        pass
    text = "".join(e.get("content", "") for e in events if e.get("type") == "message" and e.get("role") == "assistant")
    for line in reversed(text.strip().splitlines()):
        try:
            aj = json.loads(line.strip().strip("`"))
            if aj.get("answer_letter") in letters:
                return aj["answer_letter"], aj.get("reason", ""), final, ""
        except Exception:
            continue
    err = (final or {}).get("error") or ""
    return "", "", final, str(err)[:500]


def run_item(item, a):
    rl_gate(a)
    if ABORT.is_set():
        return None
    iid = item["item_id"]
    cdir = os.path.join(RUNS, a.run, a.model.replace(":", "_"), iid)
    if os.path.exists(os.path.join(cdir, "result.json")) and not a.force:
        return json.load(open(os.path.join(cdir, "result.json")))
    os.makedirs(cdir, exist_ok=True)
    try:  # atomic claim, so several run.py processes can share one run
        os.close(os.open(os.path.join(cdir, "claimed"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if not a.force:
            return None
    os.makedirs(os.path.join(cdir, "work"), exist_ok=True)
    os.chmod(os.path.join(cdir, "work"), 0o777)
    sp, schema, prompt = official(iid)
    key = item["answer"]
    if a.backend == "gemini":  # Gemini CLI: one -p text = instructions + question (official prompt.txt)
        prompt, sp = official_gemini(iid), ""
        if a.vdm:
            head, _, rest = prompt.partition("\n\n")
            prompt = head + "\n\n" + open(VDM_TOOL_MD).read().strip() + "\n\n" + rest
    if a.permute:
        prompt, key = permute(item, prompt)
    if a.transcripts and a.backend != "gemini":
        sp = sp + "\n\n" + TRANSCRIPT_NOTE
    if a.time_note:   # for planners the gateway cannot budget (TLS via CONNECT): state the wall limit up front
        sp = sp + "\n\n" + (f"Time limit: {a.timeout // 60} minutes of wall-clock time for this question, starting "
                            f"{dt.datetime.utcnow().strftime('%H:%M')} UTC (check with `date -u`). A question not answered "
                            "by then is scored wrong, so keep commands short and answer in time.")
    if a.vdm and a.backend != "gemini":
        sp = sp + "\n\n" + open(VDM_TOOL_MD).read().strip()
    if a.vdmscan:
        if a.backend == "gemini":
            head, _, rest = prompt.partition("\n\n")
            prompt = head + "\n\n" + open(SCAN_TOOL_MD).read().strip() + "\n\n" + rest
        else:
            sp = sp + "\n\n" + open(SCAN_TOOL_MD).read().strip()
    open(os.path.join(cdir, "prompt.txt"), "w").write(prompt)
    open(os.path.join(cdir, "schema.json"), "w").write(schema)
    open(os.path.join(cdir, "system_prompt.txt"), "w").write(sp)
    video = os.path.join(VIDEOS, item["video"])
    name = f"vi_{a.run}_{iid}".replace("^", "_").replace(":", "_")
    owner = f"{a.run}.{iid}"
    extra = []
    if a.transcripts:
        tp = os.path.join(a.transcripts, os.path.basename(item["video"]) + ".tsv")
        for _ in range(720):          # transcripts may still be in preparation: wait (up to 6 h) instead of failing
            if os.path.exists(tp) or ABORT.is_set():
                break
            time.sleep(30)
        if not os.path.exists(tp):
            os.remove(os.path.join(cdir, "claimed"))
            return None
        extra += ["-v", f"{tp}:/work/transcript.tsv:ro"]
    warm = {}
    if a.vdmscan:
        vid = video_id(video)
        vc = os.path.join(CACHE, vid)
        os.makedirs(vc, exist_ok=True)
        extra += ["-v", f"{vc}:/cache/vdmscan/{vid}:ro", "-e", f"VDMSCAN_CACHE_RO=/cache/vdmscan/{vid}", "-e", f"VDMSCAN_OWNER={owner}"]
        if a.warm:  # build the clip index (whole video) before the agent's clock starts; timed separately
            tw = time.time()
            w = sh(["docker", "run", "--rm", "--network", NET, f"--cpus={a.warm_cpus}", "--memory=24g",
                    "-v", f"{video}:/work/video.mp4:ro", "-v", f"{vc}:/cache/vdmscan/{vid}",
                    "-v", f"{SCAN_CLIENT}:/usr/local/bin/vdmscan:ro", "-e", f"VDMSCAN_URL=http://{GW}:8080/vdm",
                    "-e", f"VDMSCAN_CACHE=/cache/vdmscan/{vid}", "-e", f"VDMSCAN_OWNER={owner}", "-e", "VDMSCAN_LOG=/dev/null",
                    "--entrypoint", "vdmscan", IMAGE, "warm", "/work/video.mp4"], timeout=1800)
            warm = {"sec": round(time.time() - tw, 1), "rc": w.returncode, "out": (w.stdout + w.stderr)[-300:]}
    claude = ["claude", "-p", "--model", a.model, "--effort", a.effort, "--output-format", "stream-json",
              "--verbose", "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config",
              "--mcp-config", '{"mcpServers":{}}', "--tools", "Bash,Read", "--allowedTools", "Bash,Read",
              "--permission-mode", "dontAsk", "--disable-slash-commands", "--json-schema", schema,
              "--system-prompt", sp]
    if a.backend == "gemini":
        # Gemini API through the gateway (key injected there; the container only has a dummy key)
        env = {"GOOGLE_GEMINI_BASE_URL": f"http://{GW}:8080/c/{a.run}.{iid}", "GEMINI_API_KEY": "sandbox-dummy",
               "GEMINI_CLI_TRUST_WORKSPACE": "true", "GEMINI_CLI_NO_RELAUNCH": "true", "NO_COLOR": "1",
               "GEMINI_TELEMETRY_ENABLED": "false", "GEMINI_DEFAULT_AUTH_TYPE": "gemini-api-key"}
    elif a.backend == "opus":
        # Claude subscription (OAuth token via --env-file only); no ANTHROPIC_* key/token/base-url variables at all.
        # HTTPS goes through the gateway's CONNECT allowlist (api.anthropic.com:443 only).
        env = {"HTTPS_PROXY": f"http://{GW}:8080", "https_proxy": f"http://{GW}:8080", "NO_PROXY": GW}
    else:
        env = {"ANTHROPIC_BASE_URL": f"http://{GW}:8080/c/{a.run}.{iid}", "ANTHROPIC_AUTH_TOKEN": "sandbox-dummy",
               "ANTHROPIC_MODEL": a.model, "ANTHROPIC_DEFAULT_OPUS_MODEL": a.model,
               "ANTHROPIC_DEFAULT_SONNET_MODEL": a.model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": a.model,
               "ANTHROPIC_SMALL_FAST_MODEL": a.model, "CLAUDE_CODE_SUBAGENT_MODEL": a.model}
    env.update({"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
           "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1",
           "HOME": "/home/agent", "CLAUDE_CONFIG_DIR": "/home/agent/.claude",
           "API_TIMEOUT_MS": "900000", "OMP_NUM_THREADS": str(max(1, int(a.cpus))), "TMPDIR": "/tmp"})
    docker = ["docker", "run", "--rm", "-i", "--name", name, "--network", NET,
              f"--cpus={a.cpus}", f"--memory={a.memory}", "--pids-limit=4096",
              "--tmpfs", "/tmp:rw,size=8g", "--tmpfs", "/home/agent:rw,uid=1001,size=2g",
              "-v", f"{cdir}/work:/work", "-v", f"{video}:/work/video.mp4:ro", "-v", f"{MODELS}:{WHISPER_IN}:ro",
              *(["-v", f"{VDM_CLIENT}:/usr/local/bin/vdm:ro", "-e", f"VDM_URL=http://{GW}:8080/vdm"] if a.vdm else []),
              *(["-v", f"{SCAN_CLIENT}:/usr/local/bin/vdmscan:ro", "-e", f"VDMSCAN_URL=http://{GW}:8080/vdm"] if a.vdmscan else []),
              *extra, "-w", "/work"]
    for k, v in env.items():
        docker += ["-e", f"{k}={v}"]
    if a.backend == "opus":
        docker += ["--env-file", os.path.expanduser(a.env_file)]
    if a.backend == "gemini":
        here = os.path.dirname(os.path.abspath(__file__))
        docker += ["-v", f"{here}/gemini_policy.toml:/opt/policy/benchcheck.toml:ro",
                   "-v", f"{here}/gemini_settings.json:/opt/policy/settings.json:ro", GEMINI_IMAGE,
                   "sh", "-c", 'mkdir -p "$HOME/.gemini" && cp /opt/policy/settings.json "$HOME/.gemini/" && exec "$@"', "sh",
                   "gemini", "-m", a.model, "--approval-mode", "yolo", "--policy", "/opt/policy/benchcheck.toml",
                   "-o", "stream-json", "-p", prompt]
    else:
        docker += [IMAGE] + claude
    json.dump(docker, open(os.path.join(cdir, "command.json"), "w"), indent=1)
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    status, rc = "succeeded", None
    with open(os.path.join(cdir, "stdout.jsonl"), "w") as out, open(os.path.join(cdir, "stderr.txt"), "w") as err:
        p = subprocess.Popen(docker, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True)
        try:
            if a.backend != "gemini":
                p.stdin.write(prompt)
            p.stdin.close()
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
            sh(["docker", "kill", name])
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    wall = time.time() - t0
    if a.vdmscan:
        scan_post("/release", {"owner": owner})
    if a.backend == "opus":  # never keep the OAuth token in traces (the agent's shell could print its env)
        secrets = [l.split("=", 1)[1].strip() for l in open(os.path.expanduser(a.env_file)) if "=" in l]
        for fn in ("stdout.jsonl", "stderr.txt"):
            fp = os.path.join(cdir, fn)
            txt = open(fp, errors="ignore").read()
            if any(x and x in txt for x in secrets):
                for x in secrets:
                    txt = txt.replace(x, "[REDACTED]") if x else txt
                open(fp, "w").write(txt)
                open(os.path.join(cdir, "redacted_secret"), "w").close()
        del secrets
    final, late = None, ""
    n_bash = n_read = 0
    for line in open(os.path.join(cdir, "stdout.jsonl"), errors="ignore"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content", []):
                if c.get("type") == "tool_use":
                    n_bash += c.get("name") == "Bash"
                    n_read += c.get("name") == "Read"
                    if c.get("name") == "StructuredOutput" and isinstance(c.get("input"), dict):
                        late = c["input"].get("answer_letter") or late
        if ev.get("type") == "result":
            final = ev
        if ev.get("type") == "rate_limit_event":
            w = (ev.get("rate_limit_info") or {}).get("unifiedWindows")
            if w:
                RL.update(w)
    pred, reason, error = "", "", ""
    if a.backend == "gemini":
        pred, reason, final, error = gemini_answer(cdir, len(item["options"]))
    elif final:
        so = final.get("structured_output") or {}
        if isinstance(so, dict):
            pred, reason = so.get("answer_letter") or "", so.get("reason") or ""
        if final.get("is_error"):
            error = str(final.get("result") or final.get("subtype"))[:500]
    if status == "timeout" and pred:  # README rule: a timeout counts as wrong even if a letter was emitted
        late, pred = pred, ""
    txt = (str((final or {}).get("result", "")) + open(os.path.join(cdir, "stderr.txt"), errors="ignore").read()[-4000:]).lower()
    rejected = any('"status":"rejected"' in l.replace(" ", "") for l in open(os.path.join(cdir, "stdout.jsonl"), errors="ignore")
                   if "rate_limit_event" in l)
    if a.backend == "opus" and not pred and (rejected or any(w in txt for w in (
            "usage limit", "session limit", "weekly limit", "rate limit", "limit reached", "hit your limit", "429"))):
        status = "rate_limited"  # not scored: requeue after the window resets
    if status not in ("timeout", "rate_limited") and not pred:
        status = "error"
        error = error or (open(os.path.join(cdir, "stderr.txt")).read()[-500:] if rc else "no structured answer")
    gs = gw_status() if a.backend == "ollama" else {}
    stall = round((gs.get("stall") or {}).get(f"{a.run}.{iid}", 0.0), 1)
    if gs.get("fatal"):
        ABORT.set()
    if a.backend == "ollama" and not (status == "succeeded" and pred) and (
            gs.get("fatal") or stall > a.max_stall or "upstream unreachable" in txt or "overloaded" in txt or "quota exhausted" in txt):
        status = "rate_limited"   # infrastructure failure (upstream quota/outage), not the agent's: requeue, never score
    res = {"item_id": iid, "benchmark": item["benchmark"], "capability": item["capability_group"],
           "protocol": "video", "perm": "", "perm_idx": "", "expected": key, "prediction": pred,
           "correct": pred == key, "permuted": bool(a.permute), "vdm": bool(a.vdm), "vdmscan": bool(a.vdmscan), "status": status, "model": a.model, "started": started,
           "duration_s": round(min(wall, a.timeout) if status == "timeout" else wall, 1),
           "cost_usd": None, "num_turns": (final or {}).get("num_turns"), "error": error,
           "n_tool_calls": n_bash + n_read, "n_bash": n_bash, "n_read": n_read, "reason": reason,
           "run": a.run, "usage": (final or {}).get("usage"), "claude_code_version": "2.1.300",
           "init_model": a.model, "auth": {"opus": "claude-subscription-oauth", "gemini": "sandbox-gateway(gemini-api)"}.get(a.backend, "sandbox-gateway(ollama-cloud)"),
           "harness": "gemini-cli 0.64.0" if a.backend == "gemini" else "claude-code 2.1.300", "returncode": rc,
           "cc_reported_cost_usd": (final or {}).get("total_cost_usd"),
           # letter emitted via StructuredOutput before a timeout/kill; NOT scored (README rule), kept for audit
           "prediction_unscored": late if status != "succeeded" else "",
           "stall_s": stall, "upstream": gs.get("upstream"), "warm": warm, "transcripts": bool(a.transcripts), "max_steps": a.max_steps}
    if status == "rate_limited":
        json.dump(res, open(os.path.join(cdir, "rate_limited.json"), "w"), indent=1)
        os.remove(os.path.join(cdir, "claimed"))  # requeue on the next invocation
        with _print:
            print(f"{iid:28s} RATE_LIMITED (requeue)", flush=True)
        return None
    json.dump(res, open(os.path.join(cdir, "result.json"), "w"), indent=1)
    with _print:
        print(f"{iid:28s} {status:9s} pred={pred or '-'} key={key} {'OK' if res['correct'] else '  '} "
              f"{wall/60:5.1f} min", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--start-gateway", action="store_true")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--sample", type=int, default=0, help="random stratified-ish sample size (seed 0)")
    ap.add_argument("--ids", default="", help="comma-separated item ids")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--cpus", type=float, default=2)
    ap.add_argument("--memory", default="24g")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--arms", default="", help="paired mode: 'base,scan' runs both arms over the same order at the same time "
                    "(runs <run>_base / <run>_scan, --concurrency each)")
    ap.add_argument("--transcripts", default="", help="dir of <video>.tsv whisper transcripts, mounted as /work/transcript.tsv (all arms)")
    ap.add_argument("--warm", action="store_true", help="scan arm: encode the whole video before the agent starts")
    ap.add_argument("--warm-cpus", type=float, default=6)
    ap.add_argument("--max-week", type=float, default=0.70, help="opus: stop launching above this 7-day utilisation")
    ap.add_argument("--max-5h", type=float, default=0.85, help="opus: wait for the 5-hour reset above this utilisation")
    ap.add_argument("--time-note", action="store_true", help="state the wall-clock limit and start time in the system prompt")
    ap.add_argument("--max-steps", type=int, default=0, help="recorded only; the budget is enforced by the gateway (GATEWAY_MAX_STEPS)")
    ap.add_argument("--max-stall", type=float, default=300, help="requeue (not score) a failed run that waited this long on upstream retries")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backend", choices=["ollama", "opus", "gemini"], default="ollama")
    ap.add_argument("--first", type=int, default=0, help="take the first N items of the shared seeded order")
    ap.add_argument("--env-file", default="~/.claude-bench/bench.env", help="opus backend: docker --env-file with CLAUDE_CODE_OAUTH_TOKEN")
    ap.add_argument("--set", default="", help="named item set: 'hard60' = perception-bound/timeout-heavy subset")
    ap.add_argument("--vdm", action="store_true", help="put the vdm client on PATH and append TOOL.md to the system prompt")
    ap.add_argument("--vdmscan", action="store_true", help="put the vdmscan client on PATH and append TOOL_scan.md")
    ap.add_argument("--permute", action="store_true", help="option-permutation control (seeded per item)")
    ap.add_argument("--exclude-run", default="", help="skip items that belong to these runs (e.g. a pilot still in progress)")
    ap.add_argument("--reuse", default="", help="comma-separated earlier runs (same config) whose finished items are copied in")
    ap.add_argument("--available", action="store_true", help="sample only items whose video is fully downloaded")
    a = ap.parse_args()
    if a.start_gateway:
        return start_gateway(a.model)
    items = [json.loads(l) for l in open(ITEMS)]
    if a.set == "hard60":
        items = [x for x in items if x["item_id"] in set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard60.json"))))]
    if a.available:
        items = [x for x in items if os.path.exists(os.path.join(VIDEOS, x["video"]))]
    if a.ids:
        want = set(a.ids.split(","))
        items = [x for x in items if x["item_id"] in want]
    elif a.sample:
        by = {}
        for x in items:
            by.setdefault(x["capability_group"], []).append(x)
        rng = random.Random(0)
        for g in by.values():
            rng.shuffle(g)
        items = [x for i in range(a.sample) for g in sorted(by) if (x := (by[g][i] if i < len(by[g]) else None))]
        items = items[: a.sample]
    missing = [x["item_id"] for x in items if not os.path.exists(os.path.join(VIDEOS, x["video"]))]
    if missing:
        sys.exit(f"{len(missing)} videos missing, e.g. {missing[:3]}")
    logdir = os.path.join(RUNS, "_gateway")
    os.makedirs(os.path.join(RUNS, a.run), exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    ensure_gateway(a.model, logdir)
    meta = {"backend": a.backend, "set": a.set, "run": a.run, "model": a.model, "vdm": a.vdm, "vdmscan": a.vdmscan, "permute": a.permute, "effort": a.effort, "n": len(items), "concurrency": a.concurrency,
            "cpus": a.cpus, "memory": a.memory, "timeout": a.timeout, "image": IMAGE,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "items": [x["item_id"] for x in items]}
    json.dump(meta, open(os.path.join(RUNS, a.run, "meta.json"), "w"), indent=1)
    import shutil
    for prev in filter(None, a.reuse.split(",")):
        for x in items:
            src = os.path.join(RUNS, prev, a.model.replace(":", "_"), x["item_id"])
            dst = os.path.join(RUNS, a.run, a.model.replace(":", "_"), x["item_id"])
            if os.path.exists(os.path.join(src, "result.json")) and not os.path.exists(dst):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("work"))
                open(os.path.join(dst, "claimed"), "w").close()
    # One seeded random order shared by all runs: any partial result is a random subset, and partials of
    # different configs overlap as much as possible (paired comparison).
    items.sort(key=lambda x: x["item_id"])
    random.Random(0).shuffle(items)
    if a.first:
        items = items[: a.first]
    # exclusion AFTER the seeded order/--first, so "--first N --exclude-run pilot" = items 6..N of the same order
    for prev in filter(None, a.exclude_run.split(",")):
        mp = os.path.join(RUNS, prev, "meta.json")
        skip = set(json.load(open(mp))["items"]) if os.path.exists(mp) else set()
        items = [x for x in items if x["item_id"] not in skip]
    if a.first or a.exclude_run:
        meta["items"] = [x["item_id"] for x in items]
        json.dump(meta, open(os.path.join(RUNS, a.run, "meta.json"), "w"), indent=1)
    if a.arms:
        import copy
        arms = []
        for arm in a.arms.split(","):
            b = copy.copy(a)
            b.run, b.vdmscan = f"{a.run}_{arm}", arm == "scan"
            os.makedirs(os.path.join(RUNS, b.run), exist_ok=True)
            json.dump({**meta, "run": b.run, "vdmscan": b.vdmscan, "paired_with": a.run, "transcripts": a.transcripts,
                       "max_steps": a.max_steps, "warm": a.warm}, open(os.path.join(RUNS, b.run, "meta.json"), "w"), indent=1)
            arms.append(b)
        for b in arms:   # drop banks left behind by an earlier, interrupted invocation of the same run
            if b.vdmscan:
                print("release leftovers:", scan_post("/release", {"owner": b.run}), flush=True)
        pools = [cf.ThreadPoolExecutor(a.concurrency) for _ in arms]
        futs = [p.submit(run_item, x, b) for x in items for p, b in zip(pools, arms)]
        res = [r for f in futs if (r := f.result())]
        for p in pools:
            p.shutdown()
    else:
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            res = [r for r in ex.map(lambda x: run_item(x, a), items) if r]
    if ABORT.is_set():
        print("ABORTED: upstream quota exhausted (gateway fatal flag); unfinished items were requeued", flush=True)
    n = len(res)
    print(f"done: {sum(r['correct'] for r in res)}/{n} correct, "
          f"{sum(r['status'] != 'succeeded' for r in res)} no answer")


if __name__ == "__main__":
    main()
