"""SFT-on-oracle: warm-start a small LLM on labeled oracle trajectories.

This is Phase 2 of the recovery plan: before any RL on the model's own
samples, teach it to emit valid ``ICAction`` JSON by imitating deterministic
scripted oracle policies (one per task). Both prior RL attempts (GRPO and
direct RFT) regressed because their filter-and-amplify update step had
nothing to amplify — the base model couldn't reliably emit parseable
actions in the first place. SFT bypasses that whole loop: pure labeled
imitation, no rollout filtering, no advantage estimation.

Pipeline
--------

1. Load ``--base-model`` 4-bit via Unsloth, attach a LoRA adapter.
2. **Pre-eval**: run :func:`eval_unsloth.eval_all_tasks` on all 3 tasks ×
   ``--eval-seeds`` seeds. The freshly-attached LoRA is near-identity, so
   this measures the base model's behaviour.
3. Train via :class:`trl.SFTTrainer` for ``--epochs`` epochs over the
   ``--dataset`` JSONL produced by ``build_sft_dataset.py``.
4. **Post-eval**: same harness, fresh seeds offset.
5. **Regression gate**: refuse to save the adapter unless

       post.mean − pre.mean ≥ ``--min-improvement``  (default +0.10)

   AND no individual rubric component (RCA, mitigation, postmortem)
   regressed by more than ``--component-regression-tolerance`` (default
   0.005). The component check matches the CLAUDE.md guidance: "reject any
   checkpoint where RCA / mitigation / post-mortem did not improve, even if
   total did". For SFT this is a belt-and-suspenders sanity check — SFT
   can't really game the rubric — but it's cheap insurance.

The metrics JSON is written **before** the gate decision so plotting can
inspect a failed run without needing the adapter.

Usage (Colab / HF Job)
----------------------

::

    python -m incident_commander.training.train_sft \\
        --dataset sft_oracle.jsonl \\
        --output-dir ./ic-sft-oracle \\
        --metrics-json sft_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Triton C-compiler bootstrap. Must run BEFORE any unsloth/torch import.
#
# Triton ≥3 lazily JIT-compiles helper modules (``CudaUtils`` etc.) on first
# use, which happens inside ``unsloth.FastLanguageModel.from_pretrained`` when
# Llama-3.2 is loaded with 4-bit kernels. The lookup is:
#     1. ``$CC`` env var
#     2. ``shutil.which('cc' | 'gcc' | 'clang')``
# Many minimal HF Job base images (incl. ``ghcr.io/meta-pytorch/openenv-base``)
# ship without a C compiler in $PATH and without ``$CC`` set, which crashes
# every model.generate() call at runtime with a confusing
# "Failed to find C compiler" error during the eval phase. We probe for a
# compiler ourselves and export ``$CC`` so Triton stops complaining.
# --------------------------------------------------------------------------- #
def _ensure_cc_for_triton() -> None:
    if os.environ.get("CC"):
        return
    for candidate in ("cc", "gcc", "clang"):
        path = shutil.which(candidate)
        if path:
            os.environ["CC"] = path
            print(
                f"[bootstrap] CC unset; using {path!r} for Triton kernel JIT",
                flush=True,
            )
            return
    print(
        "[bootstrap] WARNING: no C compiler on PATH (cc/gcc/clang). Triton-jit "
        "kernels will fail at runtime. Install build-essential (apt) or set "
        "CC=/path/to/gcc before launching this script.",
        file=sys.stderr,
        flush=True,
    )


_ensure_cc_for_triton()


# Heavy training-only imports — guarded so plotting / module imports don't
# fail on a CPU-only machine that hasn't installed ``[training]``.
try:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "train_sft requires the [training] extra. Install with:\n"
        "  pip install -e '.[training]'\n"
        f"(missing: {e.name})"
    ) from e

from incident_commander.training.eval_unsloth import eval_all_tasks


# Components whose regression we treat as load-bearing per CLAUDE.md.
GATED_COMPONENTS = ("rca", "mitigation", "postmortem")


def _resolve_precision(mode: str) -> tuple[bool, bool]:
    """Return ``(use_bf16, use_fp16)`` for the SFTConfig dtype flags.

    bf16 needs Ampere or later (compute capability ≥ 8.0): A100, H100, RTX 30xx+.
    Turing (T4, RTX 20xx, sm_75) and Volta (V100, sm_70) only support fp16 in
    hardware — passing ``bf16=True`` there raises ValueError in
    transformers.training_args. Auto-detection avoids the user having to
    remember which flavor of HF Job they booked.
    """
    if mode == "bf16":
        return True, False
    if mode == "fp16":
        return False, True
    # ``auto``: ask the runtime.
    try:
        import torch  # local import: only required when training, not for plotting.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False
        return False, True
    except Exception:
        # No CUDA at all — fall back to fp16; the trainer will then error out
        # cleanly on the missing GPU rather than on a dtype mismatch.
        return False, True


@dataclass
class TrainSummary:
    """Everything the plotter / handoff need to read after a run."""

    base_model: str
    pre_eval: dict = field(default_factory=dict)
    post_eval: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)
    train_seconds: float = 0.0
    saved_adapter: bool = False
    gate_pass: bool = False
    gate_reason: str = ""


# --------------------------------------------------------------------------- #
# Regression gate
# --------------------------------------------------------------------------- #


def _aggregate_mean(per_task: dict) -> float:
    """Macro-average of per-task means (each task weighed equally)."""
    if not per_task:
        return 0.0
    return sum(t["mean"] for t in per_task.values()) / len(per_task)


def _component_macro(per_task: dict, key: str) -> float:
    """Macro-average of one rubric component across tasks."""
    if not per_task:
        return 0.0
    return sum(t.get("components", {}).get(key, 0.0) for t in per_task.values()) / len(
        per_task
    )


def evaluate_gate(
    pre: dict,
    post: dict,
    *,
    min_improvement: float,
    component_tolerance: float,
) -> tuple[bool, str]:
    """Decide whether the post-train checkpoint clears the regression gate.

    Returns ``(passed, reason)``. Reason is empty iff ``passed``.
    """
    pre_mean = _aggregate_mean(pre)
    post_mean = _aggregate_mean(post)
    delta = post_mean - pre_mean
    if delta < min_improvement:
        return False, (
            f"total-mean delta {delta:+.3f} < required +{min_improvement:.2f} "
            f"(pre={pre_mean:.3f} post={post_mean:.3f})"
        )

    for component in GATED_COMPONENTS:
        pre_c = _component_macro(pre, component)
        post_c = _component_macro(post, component)
        if post_c < pre_c - component_tolerance:
            return False, (
                f"component {component!r} regressed: "
                f"pre={pre_c:.3f} post={post_c:.3f} "
                f"(tolerance {component_tolerance:.3f})"
            )

    return True, ""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _save_summary(summary: TrainSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, default=str))


def _build_lora(model: Any, *, r: int, alpha: int) -> Any:
    """Attach a LoRA adapter with the standard 7-projection target set.

    The wider target set (q/k/v/o + gate/up/down) matches ``train_rft.py`` and
    gives SFT enough capacity to fit the structured-output format on a 3B
    model. Smaller subsets (q/k/v/o only) under-fit the assistant template.
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--dataset", default="sft_oracle.jsonl")
    p.add_argument("--output-dir", default="./ic-sft-oracle")
    p.add_argument("--metrics-json", default="sft_metrics.json")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help=(
            "Training precision. 'auto' picks bf16 on Ampere+ GPUs (A100, H100) "
            "and fp16 on Turing/Volta (T4, V100). 'bf16' or 'fp16' force the "
            "respective dtype — useful when running on a known fixed flavor."
        ),
    )
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--eval-max-steps", type=int, default=25)
    p.add_argument("--eval-max-new-tokens", type=int, default=512)
    p.add_argument("--retry-temperature", type=float, default=0.4)
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.10,
        help="Required increase in macro-mean reward (post − pre) to save.",
    )
    p.add_argument(
        "--component-regression-tolerance",
        type=float,
        default=0.005,
        help="Per-component slop allowed before the gate trips.",
    )
    p.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="For tiny-budget smoke runs only — disables the regression gate.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"[load] base model {args.base_model} (4-bit)", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = _build_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    summary = TrainSummary(
        base_model=args.base_model,
        config={k: v for k, v in vars(args).items()},
    )
    metrics_path = Path(args.metrics_json)

    if not args.skip_pre_eval:
        print("[eval] pre-training (LoRA attached, untrained ≈ base)", flush=True)
        FastLanguageModel.for_inference(model)
        summary.pre_eval = eval_all_tasks(
            model, tokenizer,
            seeds=args.eval_seeds,
            max_steps=args.eval_max_steps,
            max_new_tokens=args.eval_max_new_tokens,
            retry_temperature=args.retry_temperature,
            label="pre",
        )
        _save_summary(summary, metrics_path)
        print(
            f"[eval pre] macro-mean = {_aggregate_mean(summary.pre_eval):.3f}",
            flush=True,
        )

    print(f"[train] loading dataset {args.dataset}", flush=True)
    raw_dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    print(f"[train] {len(raw_dataset)} raw examples", flush=True)

    # Pre-render conversational ``messages`` rows into a single ``text`` field via
    # the model's chat template. We do this ourselves rather than depend on
    # SFTTrainer's auto-conversion, which differs across TRL releases:
    # 0.13 silently fails with KeyError('text') when handed a messages-only
    # dataset, while ≥0.16 expects an entirely different column shape. Pre-
    # rendering pins the contract on our side.
    def _render_to_text(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,  # keep the assistant turn so we train on it
            ),
        }

    dataset = raw_dataset.map(
        _render_to_text,
        remove_columns=raw_dataset.column_names,  # drop messages/task/seed/step
        desc="rendering chat template → 'text' column",
    )
    print(
        f"[train] {len(dataset)} formatted examples; columns={dataset.column_names}",
        flush=True,
    )

    use_bf16, use_fp16 = _resolve_precision(args.precision)
    print(f"[train] precision: bf16={use_bf16} fp16={use_fp16}", flush=True)
    sft_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        # We pre-render to a 'text' column above; pin the field name so a future
        # TRL release that changes the default doesn't silently break training.
        dataset_text_field="text",
    )
    # ``max_seq_length`` lives on SFTConfig in current TRL; older releases
    # took it as a kw to SFTTrainer instead. Try the modern path first.
    try:
        cfg = SFTConfig(max_seq_length=args.max_seq_length, **sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=cfg,
            train_dataset=dataset,
        )
    except TypeError:
        cfg = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=cfg,
            train_dataset=dataset,
            max_seq_length=args.max_seq_length,
        )

    FastLanguageModel.for_training(model)
    train_t0 = time.time()
    trainer.train()
    summary.train_seconds = time.time() - train_t0
    summary.loss_history = [
        float(log["loss"])
        for log in trainer.state.log_history
        if "loss" in log and "eval_loss" not in log
    ]
    print(
        f"[train] done in {summary.train_seconds:.1f}s "
        f"({len(summary.loss_history)} logged loss steps)",
        flush=True,
    )

    print("[eval] post-training", flush=True)
    FastLanguageModel.for_inference(model)
    summary.post_eval = eval_all_tasks(
        model, tokenizer,
        seeds=args.eval_seeds,
        max_steps=args.eval_max_steps,
        max_new_tokens=args.eval_max_new_tokens,
        retry_temperature=args.retry_temperature,
        label="post",
    )
    _save_summary(summary, metrics_path)
    print(
        f"[eval post] macro-mean = {_aggregate_mean(summary.post_eval):.3f}",
        flush=True,
    )

    if args.skip_pre_eval:
        summary.gate_pass = False
        summary.gate_reason = "regression gate skipped (--skip-pre-eval)"
    else:
        summary.gate_pass, summary.gate_reason = evaluate_gate(
            summary.pre_eval,
            summary.post_eval,
            min_improvement=args.min_improvement,
            component_tolerance=args.component_regression_tolerance,
        )

    if summary.gate_pass:
        print(
            f"[gate] PASS — saving adapter to {args.output_dir}",
            flush=True,
        )
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        summary.saved_adapter = True
    else:
        print(
            f"[gate] BLOCKED — adapter NOT saved.\n"
            f"        reason: {summary.gate_reason}\n"
            f"        inspect {metrics_path} for full pre/post breakdown.",
            flush=True,
        )
        summary.saved_adapter = False

    _save_summary(summary, metrics_path)
    print(f"[save] metrics → {metrics_path}", flush=True)
    return 0 if summary.gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
