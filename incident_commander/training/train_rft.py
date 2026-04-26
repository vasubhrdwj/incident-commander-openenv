"""Rejection-sampling fine-tuning (RFT) for the Incident Commander env.

Also known as "expert iteration" or "ReST". Each iteration:

    1. Sample N rollouts from the env at temperature > 0.
    2. Filter to the top K by episode total score (and above ``score_floor``).
    3. Fine-tune (SFT, one or more epochs) on those high-scoring trajectories.
    4. Repeat for M iterations.

Why RFT over GRPO for this submission
=====================================

Both connect to the env every iteration (the criteria's hard requirement —
"training loop should connect to your environment, not a static dataset").
RFT picks up two practical advantages on small models:

* **Stable loss**: SFT NLL on filtered "good" trajectories monotonically
  decreases. GRPO's group-relative advantage explodes when parse failures
  zero a whole group, which is what killed our first run on Llama-3.2-3B
  (pre=0.736 → post=0.380, archived in ``training/`` as a documented
  ablation).

* **Reliable improvement**: we're imitating the top quartile of the
  current policy's own behaviour. As long as best-of-N beats greedy
  decoding (we measured 0.872 max / 0.855 mean at N=3 in the README),
  RFT will close most of that gap into the trained weights.

The cost: RFT can plateau before reaching the reward ceiling because it
only learns from existing rollouts (no exploration bonus, no off-policy
updates). For a 25-hour hackathon, that ceiling tradeoff is the right
choice.

Usage
=====

::

    python -m incident_commander.training.train_rft \\
        --base-model unsloth/Llama-3.2-3B-Instruct \\
        --env-url http://localhost:8000 \\
        --iterations 8 --rollouts-per-iter 6 --keep-top-k 2 \\
        --output-dir ./ic-rft-lora --metrics-json ./rft_metrics.json

Then plot results with ``python -m incident_commander.training.plot_metrics``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

# Heavy imports kept at module scope. Sensible callers are in a GPU context.
try:
    import torch
    from unsloth import FastLanguageModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "train_rft requires the [training] extra. On Colab:\n"
        "  pip install unsloth trl transformers accelerate torch"
    ) from e

from .eval import EvalSummary, evaluate
from .rollout import EpisodeResult, GenerateFn, run_rollout


# --------------------------------------------------------------------------- #
# Generator factory (mirrors train_grpo.py — one source of truth would be nice
# but a small duplication beats a circular import in the Colab path)
# --------------------------------------------------------------------------- #


def make_unsloth_generator(
    model,
    tokenizer,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> GenerateFn:
    """Wrap an Unsloth model so it implements the :data:`GenerateFn` protocol."""

    def _generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_ids = output[0, inputs.shape[1]:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    return _generate


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def _completion_nll(model, tokenizer, messages: list[dict], completion: str) -> torch.Tensor:
    """Mean NLL of ``completion`` tokens given ``messages``.

    Masks prompt tokens to ``-100`` so cross-entropy averages over completion
    tokens only. Returns a scalar tensor with autograd enabled.
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    full_messages = messages + [{"role": "assistant", "content": completion}]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        labels = full_ids.clone()
    else:
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100

    out = model(input_ids=full_ids, labels=labels)
    return out.loss


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


@dataclass
class IterationMetrics:
    """One iteration's recorded metrics — serialised to ``metrics.json`` for plotting."""

    iteration: int
    rollouts_collected: int
    rollouts_kept: int
    mean_rollout_score: float
    mean_kept_score: float
    max_rollout_score: float
    mean_loss: float
    n_steps_trained: int
    wall_seconds: float


@dataclass
class RunSummary:
    """End-of-run summary written alongside per-iteration metrics."""

    base_model: str
    iterations: list[IterationMetrics] = field(default_factory=list)
    pre_eval: Optional[dict] = None
    post_eval: Optional[dict] = None
    config: Optional[dict] = None


def run_rft(
    *,
    model,
    tokenizer,
    env_urls: list[str],
    iterations: int = 8,
    rollouts_per_iter: int = 6,
    keep_top_k: int = 2,
    sft_epochs: int = 2,
    lr: float = 2e-4,
    max_steps_per_episode: int = 25,
    rollout_temperature: float = 0.9,
    score_floor: float = 0.30,
    require_done: bool = True,
) -> list[IterationMetrics]:
    """Run online RFT and return per-iteration metrics.

    Parameters
    ----------
    env_urls
        One or more env-server URLs. Each URL serves a single task (per the
        ``IC_TASK_ID`` env-var-at-startup contract). Rollouts round-robin
        across this list so a single iteration produces a balanced sample
        across all tasks. Pass ``[url]`` for the original single-task
        behaviour; pass three URLs to train on easy/medium/hard together.
    iterations
        Outer loop count. 8 is enough to see clear improvement on easy.
    rollouts_per_iter
        How many fresh rollouts to sample per iteration. Higher = better
        gradient quality but slower (each rollout is ~6-8 env steps).
        With multiple ``env_urls``, set this to a multiple of len(env_urls)
        so every task gets an equal share each iteration.
    keep_top_k
        How many top-scoring rollouts to keep for SFT. K=2 of N=6 keeps
        the top third — strict enough to filter noise, loose enough to
        always have some signal.
    sft_epochs
        Number of passes over the kept trajectories per iteration.
    score_floor
        Discard rollouts below this score outright. Stops the trainer from
        learning to imitate parse-failure / no-signal episodes.
    """
    if not env_urls:
        raise ValueError("run_rft requires at least one env URL")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    generator = make_unsloth_generator(
        model, tokenizer, temperature=rollout_temperature
    )

    metrics: list[IterationMetrics] = []
    wall_start = time.time()

    for it in range(1, iterations + 1):
        it_start = time.time()
        # 1. Collect rollouts (no_grad — generation only).
        FastLanguageModel.for_inference(model)
        rollouts: list[EpisodeResult] = []
        for g in range(rollouts_per_iter):
            # Round-robin across env URLs so every task gets an equal share
            # of each iteration's rollouts. With 12 rollouts and 3 URLs, that's
            # 4 rollouts per task per iteration.
            env_url = env_urls[g % len(env_urls)]
            traj = run_rollout(
                generate=generator,
                env_url=env_url,
                max_steps=max_steps_per_episode,
            )
            rollouts.append(traj)
            print(
                f"[iter {it}] rollout {g + 1}/{rollouts_per_iter} "
                f"env={env_url} steps={traj.steps} score={traj.score:.3f} "
                f"done={traj.success} err={traj.error}",
                flush=True,
            )

        # 2. Filter top-K above score floor.
        # require_done filters out "step=25, didn't finish" trajectories — these
        # are the "queried forever, scored 0.45 from partial credit, never
        # resolved" rollouts that drove our previous RFT-on-SFT collapse: their
        # 0.45 score passed score_floor=0.30, they got reinforced, the model
        # learned to imitate "do queries forever," and by iter 3 every rollout
        # was step=25 with score 0.0. Filtering on r.success (which is True
        # only when result.done was hit cleanly) eliminates that failure mode
        # at its source — partial-credit trajectories can't enter the
        # training set even if their score passes the floor.
        valid = [
            r for r in rollouts
            if r.error is None
            and r.score >= score_floor
            and (not require_done or r.success)
        ]
        valid.sort(key=lambda r: r.score, reverse=True)
        kept = valid[:keep_top_k]

        if not kept:
            n_done = sum(1 for r in rollouts if r.success)
            n_passing_floor = sum(
                1 for r in rollouts if r.error is None and r.score >= score_floor
            )
            print(
                f"[iter {it}] no rollouts kept "
                f"(score_floor={score_floor}, require_done={require_done}; "
                f"{n_done}/{len(rollouts)} completed, "
                f"{n_passing_floor}/{len(rollouts)} passed floor); "
                f"skipping update",
                flush=True,
            )
            continue

        # 3. SFT epochs over kept trajectories.
        FastLanguageModel.for_training(model)
        all_step_losses: list[float] = []
        for epoch in range(sft_epochs):
            for r in kept:
                for step in r.trajectory:
                    nll = _completion_nll(
                        model, tokenizer, step.messages, step.completion
                    )
                    optimizer.zero_grad()
                    nll.backward()
                    optimizer.step()
                    all_step_losses.append(float(nll.detach().item()))

        wall = time.time() - it_start
        m = IterationMetrics(
            iteration=it,
            rollouts_collected=len(rollouts),
            rollouts_kept=len(kept),
            mean_rollout_score=mean(r.score for r in rollouts),
            mean_kept_score=mean(r.score for r in kept),
            max_rollout_score=max(r.score for r in rollouts),
            mean_loss=mean(all_step_losses) if all_step_losses else float("nan"),
            n_steps_trained=len(all_step_losses),
            wall_seconds=wall,
        )
        metrics.append(m)
        print(
            f"[iter {it}] sampled={m.rollouts_collected} kept={m.rollouts_kept} "
            f"sample_mean={m.mean_rollout_score:.3f} kept_mean={m.mean_kept_score:.3f} "
            f"max={m.max_rollout_score:.3f} loss={m.mean_loss:.4f} "
            f"steps={m.n_steps_trained} wall={wall:.1f}s "
            f"total={(time.time() - wall_start) / 60:.1f}min",
            flush=True,
        )

    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online rejection-sampling fine-tuning for Incident Commander"
    )
    p.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct"),
    )
    p.add_argument(
        "--env-url",
        nargs="+",
        default=None,
        help=(
            "One or more env-server URLs. Each URL must serve a different "
            "IC_TASK_ID (one server per task — see CLAUDE.md 'Request path'). "
            "Rollouts round-robin across the list, and pre/post eval is run "
            "separately per URL with macro-mean used for the gate. "
            "Examples: --env-url http://127.0.0.1:8000   "
            "--env-url http://127.0.0.1:8000 http://127.0.0.1:8001 http://127.0.0.1:8002"
        ),
    )
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--rollouts-per-iter", type=int, default=6)
    p.add_argument("--keep-top-k", type=int, default=2)
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-steps-per-episode", type=int, default=25)
    p.add_argument("--rollout-temperature", type=float, default=0.9)
    p.add_argument("--score-floor", type=float, default=0.30)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "./ic-rft-lora"),
        help="Where to save the trained LoRA adapter",
    )
    p.add_argument(
        "--metrics-json",
        default=os.getenv("METRICS_JSON", "./rft_metrics.json"),
        help="Where to write per-iteration metrics + pre/post eval (for plotting)",
    )
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument(
        "--eval-temperature",
        type=float,
        default=0.7,
        help="Temperature used for pre/post eval rollouts. Higher than 0.1 "
        "to avoid reading a collapsed greedy mode (lesson from the GRPO ablation).",
    )
    p.add_argument(
        "--base-adapter",
        default=os.getenv("BASE_ADAPTER", ""),
        help="Optional path or HF repo ID of a LoRA adapter to warm-start RFT "
        "from. Use this to run RFT on top of an SFT-trained checkpoint.",
    )
    p.add_argument(
        "--require-done",
        type=int,
        default=int(os.getenv("REQUIRE_DONE", "1")),
        help="If 1 (default), only rollouts that completed cleanly "
        "(result.done=True at episode end, i.e. agent called resolve or "
        "postmortem within step budget) are eligible to be kept and trained "
        "on. Set 0 to disable the check (matches the original RFT-on-base "
        "behaviour but allowed our previous run to collapse onto step=25 "
        "partial-credit rollouts).",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=float(os.getenv("MIN_IMPROVEMENT", "0.02")),
        help="Required (post.mean - pre.mean) for the adapter to be saved. "
        "RFT polish on an already-SFT-warm checkpoint typically adds modest "
        "gains (+0.02 to +0.05); set lower than the SFT gate.",
    )
    return p.parse_args(argv)


def _evaluate_per_url(
    *,
    generate: "GenerateFn",
    env_urls: list[str],
    n_episodes: int,
    max_steps: int,
    label: str,
) -> tuple[dict, "EvalSummary"]:
    """Run ``evaluate`` against each URL, return (per_url_breakdown, macro_summary).

    ``per_url_breakdown`` is keyed by URL and stores the raw EvalSummary dict
    for each task. ``macro_summary`` is an EvalSummary whose ``mean`` is the
    macro-average across tasks (each task weighted equally) — that's what the
    gate compares against.
    """
    per_url: dict[str, dict] = {}
    all_means: list[float] = []
    all_scores: list[float] = []
    total_errors = 0
    total_n = 0
    for url in env_urls:
        per_url_label = f"{label}[{url}]"
        s = evaluate(
            generate=generate,
            env_url=url,
            n_episodes=n_episodes,
            max_steps=max_steps,
            label=per_url_label,
        )
        per_url[url] = _eval_to_dict(s)
        all_means.append(s.mean)
        all_scores.extend(s.scores)
        total_errors += s.errors
        total_n += s.n
    macro_mean = mean(all_means) if all_means else 0.0
    macro_std = stdev(all_means) if len(all_means) > 1 else 0.0
    macro = EvalSummary(
        label=f"{label}[macro]",
        n=total_n,
        scores=all_scores,
        mean=macro_mean,
        std=macro_std,
        errors=total_errors,
    )
    print(
        f"[{label}] macro-mean across {len(env_urls)} task(s) = "
        f"{macro_mean:.3f} (per-task means: "
        + ", ".join(f"{u.split(':')[-1]}={per_url[u]['mean']:.3f}" for u in env_urls)
        + ")",
        flush=True,
    )
    return per_url, macro


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Resolve env URLs: CLI list > $ENV_URL > localhost:8000 default.
    env_urls: list[str] = list(args.env_url) if args.env_url else [
        os.getenv("ENV_URL", "http://localhost:8000")
    ]
    print(
        f"[setup] training across {len(env_urls)} env URL(s): "
        + ", ".join(env_urls),
        flush=True,
    )

    if args.base_adapter:
        # Unsloth's preferred path for "continue training an existing LoRA":
        # pass the adapter path/repo to FastLanguageModel.from_pretrained — it
        # reads adapter_config.json, loads the recorded base model in 4-bit,
        # attaches the LoRA, and applies *all* of Unsloth's training-mode
        # monkeypatches (including the LlamaDecoderLayer._gradient_checkpointing_func
        # patch that PeftModel.from_pretrained quietly skips, causing the
        # AttributeError we hit on the first SFT step inside RFT iter 1).
        print(
            f"[setup] loading SFT-warm adapter {args.base_adapter} "
            "(Unsloth auto-detects + reattaches base in 4-bit)",
            flush=True,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_adapter,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        # Mark for training mode — installs gradient-checkpointing patches and
        # flips LoRA params to require_grad=True.
        FastLanguageModel.for_training(model)
    else:
        print(f"[setup] loading {args.base_model} (4-bit via Unsloth)", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    summary = RunSummary(
        base_model=args.base_model,
        config={
            "iterations": args.iterations,
            "rollouts_per_iter": args.rollouts_per_iter,
            "keep_top_k": args.keep_top_k,
            "sft_epochs": args.sft_epochs,
            "lr": args.lr,
            "rollout_temperature": args.rollout_temperature,
            "score_floor": args.score_floor,
            "lora_rank": args.lora_rank,
            "eval_episodes": args.eval_episodes,
            "eval_temperature": args.eval_temperature,
            "max_steps_per_episode": args.max_steps_per_episode,
            "base_adapter": args.base_adapter or None,
            "min_improvement": args.min_improvement,
            "env_urls": env_urls,
        },
    )

    print("[eval] pre-training baseline", flush=True)
    FastLanguageModel.for_inference(model)
    pre_per_url, pre = _evaluate_per_url(
        generate=make_unsloth_generator(
            model, tokenizer, temperature=args.eval_temperature
        ),
        env_urls=env_urls,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps_per_episode,
        label="pre_training",
    )
    summary.pre_eval = {**_eval_to_dict(pre), "per_task": pre_per_url}
    _save_metrics(summary, args.metrics_json)

    print("[train] starting RFT", flush=True)
    metrics = run_rft(
        model=model,
        tokenizer=tokenizer,
        env_urls=env_urls,
        iterations=args.iterations,
        rollouts_per_iter=args.rollouts_per_iter,
        keep_top_k=args.keep_top_k,
        sft_epochs=args.sft_epochs,
        lr=args.lr,
        max_steps_per_episode=args.max_steps_per_episode,
        rollout_temperature=args.rollout_temperature,
        score_floor=args.score_floor,
        require_done=bool(args.require_done),
    )
    summary.iterations = metrics
    _save_metrics(summary, args.metrics_json)

    print("[eval] post-training baseline", flush=True)
    FastLanguageModel.for_inference(model)
    post_per_url, post = _evaluate_per_url(
        generate=make_unsloth_generator(
            model, tokenizer, temperature=args.eval_temperature
        ),
        env_urls=env_urls,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps_per_episode,
        label="post_training",
    )
    summary.post_eval = {**_eval_to_dict(post), "per_task": post_per_url}

    delta = post.mean - pre.mean
    print(
        "\n=========================================\n"
        f"pre:   {pre}\n"
        f"post:  {post}\n"
        f"delta: {delta:+.3f}\n"
        f"iterations completed: {len(metrics)}\n"
        "=========================================",
        flush=True,
    )

    # Regression gate — same shape as train_sft.py. Refuse to overwrite an
    # SFT-warm adapter with worse RFT-trained weights; that's exactly the
    # bug the prior unguarded RFT runs hit (-0.31 regression saved silently).
    gate_pass = delta >= args.min_improvement
    summary.config["gate_pass"] = gate_pass
    summary.config["gate_reason"] = (
        f"delta {delta:+.3f} >= +{args.min_improvement:.2f}"
        if gate_pass
        else f"delta {delta:+.3f} < +{args.min_improvement:.2f} (regression gate blocked save)"
    )
    _save_metrics(summary, args.metrics_json)
    print(f"[save] metrics → {args.metrics_json}", flush=True)

    if not gate_pass:
        print(
            f"[gate] BLOCKED — adapter NOT saved.\n"
            f"        reason: delta {delta:+.3f} < required +{args.min_improvement:.2f} "
            f"(pre={pre.mean:.3f} post={post.mean:.3f})\n"
            f"        inspect {args.metrics_json} for full pre/post breakdown.",
            flush=True,
        )
        return 2

    print(f"[gate] PASS (delta {delta:+.3f} >= +{args.min_improvement:.2f})", flush=True)
    print(f"[save] LoRA adapter → {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


def _eval_to_dict(e: "EvalSummary") -> dict:
    """Coerce an EvalSummary to a plain dict (compatible with old/new field names)."""
    if hasattr(e, "model_dump"):
        return e.model_dump()
    if hasattr(e, "_asdict"):
        return e._asdict()
    return asdict(e)


def _save_metrics(summary: RunSummary, path: str) -> None:
    out = {
        "base_model": summary.base_model,
        "config": summary.config,
        "pre_eval": summary.pre_eval,
        "post_eval": summary.post_eval,
        "iterations": [asdict(m) for m in summary.iterations],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
