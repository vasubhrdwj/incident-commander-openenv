"""Launch the Incident Commander RFT run on Hugging Face Jobs.

This is the Colab workflow expressed as one HF Job:

1. Clone the repo.
2. Install training dependencies.
3. Start the OpenEnv server locally inside the job.
4. Run ``training.train_rft`` with the same defaults as the Colab.
5. Generate README plots.
6. Upload the LoRA adapter and training artifacts to Hugging Face Hub.

Usage from a logged-in local machine:

    HF_TOKEN=... python -m incident_commander.training.launch_hf_rft_job \
      --repo-url https://github.com/vasubhrdwj/incident-commander-openenv.git \
      --model-repo-id vasubhrdwj/incident-commander-llama3.2-3b-rft

The token must include Hugging Face Jobs permission (``job.write``) and write
access to the target model repo. It is passed as an encrypted HF Job secret and
is not printed.
"""

from __future__ import annotations

import argparse
import os
from textwrap import dedent

try:
    from huggingface_hub import run_job
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "launch_hf_rft_job requires huggingface_hub. Install with:\n"
        "  pip install huggingface_hub"
    ) from e


def _job_script() -> str:
    """Return the bash script executed inside the HF Job container."""

    return dedent(
        r"""
        set -euo pipefail

        echo "[job] GPU check"
        nvidia-smi || true

        echo "[job] install system deps"
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git curl
        rm -rf /var/lib/apt/lists/*

        echo "[job] clone repo"
        rm -rf /workspace/repo
        git clone "${OPENENV_REPO}" /workspace/repo
        cd /workspace/repo/incident_commander

        echo "[job] install deps"
        python -m pip install -U pip
        python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        python -m pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes matplotlib huggingface_hub
        # The PyTorch 2.6 job image plus latest torchao can trip Transformers'
        # optional torchao quantizer import before Unsloth loads. We do not use
        # torchao for the 4-bit LoRA path, so remove it from the environment.
        python -m pip uninstall -y torchao || true
        python -m pip install -e .

        echo "[job] start env server"
        export IC_TASK_ID=easy_canary_regression
        export IC_SEED=0
        export IC_STEP_BUDGET=25
        python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level warning >/tmp/ic-server.log 2>&1 &
        SERVER_PID=$!
        trap 'kill ${SERVER_PID} || true' EXIT

        python - <<'PY'
        import time
        import urllib.request

        for _ in range(90):
            try:
                with urllib.request.urlopen("http://localhost:8000/health", timeout=1) as r:
                    if r.status == 200:
                        print("[job] env server ready", flush=True)
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("env server failed to start")
        PY

        echo "[job] run RFT"
        python -m incident_commander.training.train_rft \
          --base-model unsloth/Llama-3.2-3B-Instruct \
          --env-url http://localhost:8000 \
          --iterations "${RFT_ITERATIONS}" \
          --rollouts-per-iter "${RFT_ROLLOUTS_PER_ITER}" \
          --keep-top-k "${RFT_KEEP_TOP_K}" \
          --sft-epochs "${RFT_SFT_EPOCHS}" \
          --rollout-temperature 0.9 \
          --eval-temperature 0.7 \
          --eval-episodes 3 \
          --score-floor 0.30 \
          --lr 2e-4 \
          --lora-rank 16 \
          --output-dir ./ic-rft-lora \
          --metrics-json ./rft_metrics.json

        echo "[job] generate plots"
        python -m incident_commander.training.plot_metrics \
          --metrics ./rft_metrics.json \
          --out ./assets \
          --task-id easy_canary_regression

        echo "[job] upload artifacts"
        python - <<'PY'
        import os
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        model_repo = os.environ["HF_MODEL_REPO_ID"]
        api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path="ic-rft-lora",
            repo_id=model_repo,
            repo_type="model",
            commit_message="RFT LoRA adapter for Incident Commander",
        )

        for path in [
            "rft_metrics.json",
            "assets/score_summary.png",
            "assets/training_reward.png",
            "assets/training_loss.png",
            "assets/component_comparison.png",
        ]:
            if os.path.exists(path):
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"training_artifacts/{path}",
                    repo_id=model_repo,
                    repo_type="model",
                    commit_message=f"Add {path}",
                )
        print(f"[job] uploaded adapter and artifacts to https://huggingface.co/{model_repo}")
        PY
        """
    ).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Incident Commander RFT on HF Jobs")
    parser.add_argument(
        "--repo-url",
        default=os.environ.get(
            "OPENENV_REPO",
            "https://github.com/vasubhrdwj/incident-commander-openenv.git",
        ),
        help="Git repo cloned inside the job.",
    )
    parser.add_argument(
        "--model-repo-id",
        default=os.environ.get(
            "HF_MODEL_REPO_ID",
            "vasubhrdwj/incident-commander-llama3.2-3b-rft",
        ),
        help="HF model repo that receives the LoRA adapter and training artifacts.",
    )
    parser.add_argument("--flavor", default=os.environ.get("HF_JOB_FLAVOR", "t4-medium"))
    parser.add_argument("--timeout", default=os.environ.get("HF_JOB_TIMEOUT", "2h"))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("RFT_ITERATIONS", "8")))
    parser.add_argument(
        "--rollouts-per-iter",
        type=int,
        default=int(os.environ.get("RFT_ROLLOUTS_PER_ITER", "6")),
    )
    parser.add_argument("--keep-top-k", type=int, default=int(os.environ.get("RFT_KEEP_TOP_K", "2")))
    parser.add_argument("--sft-epochs", type=int, default=int(os.environ.get("RFT_SFT_EPOCHS", "2")))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set locally so it can be passed as a job secret.")

    job = run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", _job_script()],
        flavor=args.flavor,
        timeout=args.timeout,
        env={
            "OPENENV_REPO": args.repo_url,
            "HF_MODEL_REPO_ID": args.model_repo_id,
            "RFT_ITERATIONS": str(args.iterations),
            "RFT_ROLLOUTS_PER_ITER": str(args.rollouts_per_iter),
            "RFT_KEEP_TOP_K": str(args.keep_top_k),
            "RFT_SFT_EPOCHS": str(args.sft_epochs),
        },
        secrets={"HF_TOKEN": token},
    )
    print(f"HF Job launched: {job.url}")
    print(f"Job id: {job.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
