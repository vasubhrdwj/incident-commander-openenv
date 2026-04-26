"""Orchestrator for RFT-on-SFT-warm.

Spawns 3 env servers (one per task), waits for them to be healthy, then
invokes ``train_rft`` with the right --env-url list. Used by HF Jobs to
avoid the bash quoting hell that bit two consecutive job submissions.

Why Python and not a bash script
================================
``hf jobs run`` ships the inline ``bash -c '<script>'`` to a remote runner
which can re-parse / re-tokenize the string in surprising ways. Backslash
continuations get dropped, associative arrays interact badly with strict
mode, and a quoting bug at line 57 produced a "syntax error near
unexpected token 'fi'" on a script that's syntactically clean locally.

Python's subprocess + urllib have no such surprise. Run is now:

    git clone <space>; pip install -e .[training];
    huggingface-cli download <sft-adapter> --local-dir ./ic-sft-oracle;
    python -m incident_commander.training.run_rft_on_sft

Configurable via env vars (all optional, with the same defaults the previous
two attempts used):

    RFT_ITERATIONS         (default 4)
    RFT_ROLLOUTS_PER_ITER  (default 12)
    RFT_KEEP_TOP_K         (default 3)
    RFT_SFT_EPOCHS         (default 1)
    RFT_LR                 (default 5e-5)
    RFT_SCORE_FLOOR        (default 0.30)
    RFT_ROLLOUT_TEMPERATURE (default 0.9)
    RFT_EVAL_TEMPERATURE   (default 0.7)
    RFT_EVAL_EPISODES      (default 3)
    RFT_MIN_IMPROVEMENT    (default 0.02)
    RFT_BASE_ADAPTER       (default ./ic-sft-oracle)
    RFT_OUTPUT_DIR         (default ./ic-rft-on-sft)
    RFT_METRICS_JSON       (default rft_on_sft_metrics.json)

Exit code = train_rft's exit code (0 = gate pass, 2 = gate blocked, other = crash).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path("/tmp/logs")

TASKS = [
    "easy_canary_regression",
    "medium_third_party_attribution",
    "hard_silent_data_corruption",
]
PORTS = [8000, 8001, 8002]
HEALTH_TIMEOUT_S = 60
HEALTH_POLL_INTERVAL_S = 1.0


def _env(key: str, default: str) -> str:
    """Read env var with a default, trimming whitespace."""
    val = os.environ.get(key)
    return val.strip() if val else default


def spawn_server(task: str, port: int) -> tuple[subprocess.Popen, Path]:
    """Spawn one uvicorn server for ``task`` on ``port``. Returns (proc, log_path)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"uvicorn_{port}.log"
    log_fh = open(log_path, "w")
    server_env = os.environ.copy()
    server_env["IC_TASK_ID"] = task
    server_env["IC_SEED"] = "0"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "server.app:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "info",
        ],
        env=server_env,
        cwd=str(REPO_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    print(
        f"[launch] spawned {task} on :{port} (pid={proc.pid}, log={log_path})",
        flush=True,
    )
    return proc, log_path


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    """Block until ``http://127.0.0.1:<port>/health`` returns 200 or process dies."""
    url = f"http://127.0.0.1:{port}/health"
    for attempt in range(1, timeout_s + 1):
        time.sleep(HEALTH_POLL_INTERVAL_S)
        if proc.poll() is not None:
            print(
                f"[launch] :{port} pid {proc.pid} died at attempt {attempt} "
                f"(rc={proc.returncode})",
                flush=True,
            )
            return False
        try:
            with urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[launch] :{port} healthy after {attempt}s", flush=True)
                    return True
        except (URLError, ConnectionError, OSError):
            # Server still starting — keep polling.
            continue
    return False


def kill_all(procs: list[subprocess.Popen]) -> None:
    """Send SIGTERM to every process, escalate to SIGKILL after 2s."""
    for p in procs:
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def dump_log(log_path: Path, port: int) -> None:
    """Print a uvicorn log to stdout, prefixed for grep-ability."""
    if not log_path.exists():
        print(f"[launch] (no log file at {log_path})", flush=True)
        return
    print(f"================== :{port} log ==================", flush=True)
    for line in log_path.read_text(errors="replace").splitlines():
        print(f"SERVER[{port}]: {line}", flush=True)
    print(f"==================================================", flush=True)


def build_train_rft_cmd() -> list[str]:
    """Construct the ``python -m incident_commander.training.train_rft`` argv."""
    return [
        sys.executable, "-m", "incident_commander.training.train_rft",
        "--base-adapter",        _env("RFT_BASE_ADAPTER", "./ic-sft-oracle"),
        "--env-url",
            f"http://127.0.0.1:{PORTS[0]}",
            f"http://127.0.0.1:{PORTS[1]}",
            f"http://127.0.0.1:{PORTS[2]}",
        "--iterations",          _env("RFT_ITERATIONS", "2"),
        "--rollouts-per-iter",   _env("RFT_ROLLOUTS_PER_ITER", "12"),
        "--keep-top-k",          _env("RFT_KEEP_TOP_K", "2"),
        "--sft-epochs",          _env("RFT_SFT_EPOCHS", "1"),
        "--lr",                  _env("RFT_LR", "2e-5"),
        "--score-floor",         _env("RFT_SCORE_FLOOR", "0.55"),
        "--rollout-temperature", _env("RFT_ROLLOUT_TEMPERATURE", "0.9"),
        "--eval-temperature",    _env("RFT_EVAL_TEMPERATURE", "0.7"),
        "--eval-episodes",       _env("RFT_EVAL_EPISODES", "3"),
        "--output-dir",          _env("RFT_OUTPUT_DIR", "./ic-rft-on-sft"),
        "--metrics-json",        _env("RFT_METRICS_JSON", "rft_on_sft_metrics.json"),
        "--min-improvement",     _env("RFT_MIN_IMPROVEMENT", "0.01"),
        "--require-done",        _env("REQUIRE_DONE", "1"),
    ]


def main() -> int:
    procs: list[subprocess.Popen] = []
    log_paths: list[Path] = []
    try:
        for task, port in zip(TASKS, PORTS):
            proc, log_path = spawn_server(task, port)
            procs.append(proc)
            log_paths.append(log_path)

        all_healthy = True
        for task, port, proc, log_path in zip(TASKS, PORTS, procs, log_paths):
            if not wait_healthy(port, proc):
                print(
                    f"[launch] {task} :{port} did not become healthy in "
                    f"{HEALTH_TIMEOUT_S}s — dumping log:",
                    flush=True,
                )
                dump_log(log_path, port)
                all_healthy = False
                break

        if not all_healthy:
            return 1

        print("[launch] all three env servers healthy; starting RFT", flush=True)
        cmd = build_train_rft_cmd()
        # Print the exact command for reproducibility / debug.
        printable = " ".join(cmd)
        print(f"[launch] $ {printable}", flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        print(f"[launch] train_rft exit code: {rc}", flush=True)
        return rc
    finally:
        kill_all(procs)


if __name__ == "__main__":
    sys.exit(main())
