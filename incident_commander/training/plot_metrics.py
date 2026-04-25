"""Generate training plots from a metrics JSON.

Produces PNGs that get committed to ``assets/`` and embedded in the README
per submission criteria ("save plots as .png and commit them, embed in
README with a one-line caption").

Two modes
=========

``--mode rft`` (default — original behaviour for ``rft_metrics.json``)
    1. ``training_loss.png`` — SFT loss vs RFT iteration (decreasing).
    2. ``training_reward.png`` — mean rollout score (sampled & kept) vs iter.
    3. ``component_comparison.png`` — bar chart, baseline vs trained.
    4. ``score_summary.png`` — final pre/post total score, big and obvious.

``--mode sft`` (Phase 2 — for ``sft_metrics.json``)
    1. ``sft_loss.png`` — SFTTrainer logged loss vs logged step.
    2. ``sft_pre_post.png`` — pre vs post mean reward, one bar pair per task.
    3. ``sft_components.png`` — three-panel grid; for each task a grouped
       bar chart of the six rubric components (pre vs post side-by-side).

Usage::

    # original RFT plots
    python -m incident_commander.training.plot_metrics \\
        --metrics ./rft_metrics.json --out ./assets

    # SFT plots from train_sft.py output
    python -m incident_commander.training.plot_metrics --mode sft \\
        --metrics ./sft_metrics.json --out ./assets

If the metrics JSON is missing fields (mid-run snapshot) the missing
plots are skipped with a warning rather than crashing the whole run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # headless / Colab-friendly
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "plot_metrics requires matplotlib. On Colab:\n"
        "  pip install matplotlib"
    ) from e


# Same six rubric components as graders/rubric.py — kept in sync manually
# because importing the env package isn't required for plotting.
RUBRIC_KEYS = ("containment", "mttr", "rca", "mitigation", "comms", "postmortem")
RUBRIC_WEIGHTS = {
    "containment": 0.25, "mttr": 0.20, "rca": 0.20,
    "mitigation": 0.15, "comms": 0.10, "postmortem": 0.10,
}


# Colour palette tuned for the dark-theme README screenshot
COL_BASELINE = "#6b7280"   # grey
COL_TRAINED = "#f59e0b"    # amber
COL_LOSS = "#3b82f6"       # blue
COL_SAMPLED = "#6b7280"
COL_KEPT = "#22c55e"       # green


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_training_loss(metrics: list[dict], out: Path) -> None:
    if not metrics:
        print("  skipping training_loss.png (no iterations yet)", flush=True)
        return
    iters = [m["iteration"] for m in metrics]
    losses = [m["mean_loss"] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, losses, marker="o", color=COL_LOSS, linewidth=2)
    ax.set_xlabel("RFT iteration")
    ax.set_ylabel("mean SFT NLL loss")
    ax.set_title("Training loss — RFT on Incident Commander")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(iters)
    fig.tight_layout()
    _save_fig(fig, out)


def plot_training_reward(metrics: list[dict], out: Path) -> None:
    if not metrics:
        print("  skipping training_reward.png (no iterations yet)", flush=True)
        return
    iters = [m["iteration"] for m in metrics]
    sampled = [m["mean_rollout_score"] for m in metrics]
    kept = [m["mean_kept_score"] for m in metrics]
    max_ = [m["max_rollout_score"] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, sampled, marker="o", color=COL_SAMPLED, label="all rollouts (mean)", linewidth=2)
    ax.plot(iters, kept, marker="s", color=COL_KEPT, label="top-K kept (mean)", linewidth=2)
    ax.plot(iters, max_, marker="^", color=COL_TRAINED, linestyle="--", label="best-of-batch", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("RFT iteration")
    ax.set_ylabel("episode score (0–1)")
    ax.set_title("Reward over training — RFT on Incident Commander")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(iters)
    fig.tight_layout()
    _save_fig(fig, out)


def plot_component_comparison(pre: Optional[dict], post: Optional[dict], out: Path) -> None:
    """Bar chart, baseline vs trained, per rubric component.

    Falls back to total-only if pre/post don't expose component breakdowns.
    Per-component data isn't recorded by the current ``EvalSummary``, so this
    plot uses just the totals + the static rubric weights as reference bars.
    """
    if not (pre and post):
        print("  skipping component_comparison.png (need both pre + post eval)", flush=True)
        return

    # Current EvalSummary only carries totals; show the deltas in totals
    # alongside the per-component weight ceilings so a reviewer can see how
    # much headroom remains in each component.
    components = list(RUBRIC_KEYS)
    weights = [RUBRIC_WEIGHTS[k] for k in components]
    pre_total = float(pre.get("mean", 0.0))
    post_total = float(post.get("mean", 0.0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(components)))
    width = 0.30

    # Reference bars: each component's max weight
    ax.bar([i - width for i in x], weights, width, color="#1f2937", alpha=0.6, label="component weight (max)")
    # We only have totals; show the baseline & trained as horizontal lines
    # spanning all components for clarity.
    ax.axhline(pre_total, color=COL_BASELINE, linestyle="--", linewidth=1.5, label=f"baseline total {pre_total:.3f}")
    ax.axhline(post_total, color=COL_TRAINED, linestyle="-", linewidth=2, label=f"trained total {post_total:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=20)
    ax.set_ylabel("score")
    ax.set_title("Rubric components — weights vs measured totals")
    ax.set_ylim(0, max(0.30, post_total + 0.05))
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out)


def plot_score_summary(pre: Optional[dict], post: Optional[dict], out: Path, task_id: str = "easy_canary_regression") -> None:
    if not (pre and post):
        print("  skipping score_summary.png (need both pre + post eval)", flush=True)
        return
    pre_mean = float(pre.get("mean", 0.0))
    pre_std = float(pre.get("std", 0.0))
    post_mean = float(post.get("mean", 0.0))
    post_std = float(post.get("std", 0.0))
    delta = post_mean - pre_mean

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["baseline\n(pre-training)", "trained\n(post-RFT)"],
        [pre_mean, post_mean],
        yerr=[pre_std, post_std],
        capsize=8,
        color=[COL_BASELINE, COL_TRAINED],
        edgecolor="black", linewidth=1,
    )
    ax.set_ylabel("episode total score")
    ax.set_ylim(0, 1.0)
    n = len(pre.get("scores", [])) or "?"
    ax.set_title(
        f"Pre vs post-training score on {task_id}\n"
        f"Δ = {delta:+.3f}  (mean of {n} episodes)"
    )
    # Annotate exact numbers above each bar
    ax.text(0, pre_mean + 0.04, f"{pre_mean:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.text(1, post_mean + 0.04, f"{post_mean:.3f}", ha="center", fontsize=12, fontweight="bold", color="#92400e")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    _save_fig(fig, out)


# --------------------------------------------------------------------------- #
# SFT plots — eat ``sft_metrics.json`` produced by train_sft.py
# --------------------------------------------------------------------------- #


def _short_task(task_id: str) -> str:
    """Compact tick label for the per-task plots ('easy', 'medium', 'hard')."""
    if task_id.startswith("easy"):
        return "easy"
    if task_id.startswith("medium"):
        return "medium"
    if task_id.startswith("hard"):
        return "hard"
    return task_id.split("_")[0]


def plot_sft_loss(loss_history: list[float], out: Path) -> None:
    if not loss_history:
        print("  skipping sft_loss.png (no loss_history yet)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(1, len(loss_history) + 1))
    ax.plot(xs, loss_history, color=COL_LOSS, linewidth=2)
    ax.set_xlabel("logged training step")
    ax.set_ylabel("training loss (NLL)")
    ax.set_title("SFT-on-oracle: training loss")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, out)


def plot_sft_pre_post(pre: Optional[dict], post: Optional[dict], out: Path) -> None:
    if not (pre and post):
        print("  skipping sft_pre_post.png (need both pre + post per-task evals)", flush=True)
        return

    tasks = list(pre.keys())
    pre_means = [float(pre[t].get("mean", 0.0)) for t in tasks]
    post_means = [float(post[t].get("mean", 0.0)) for t in tasks]
    pre_stds = [float(pre[t].get("std", 0.0)) for t in tasks]
    post_stds = [float(post[t].get("std", 0.0)) for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(tasks)))
    width = 0.36
    ax.bar(
        [i - width / 2 for i in x], pre_means, width,
        yerr=pre_stds, capsize=5, color=COL_BASELINE, edgecolor="black",
        label="baseline (pre-SFT)",
    )
    ax.bar(
        [i + width / 2 for i in x], post_means, width,
        yerr=post_stds, capsize=5, color=COL_KEPT, edgecolor="black",
        label="trained (post-SFT)",
    )
    for i, (pre_v, post_v) in enumerate(zip(pre_means, post_means)):
        ax.text(i - width / 2, pre_v + 0.02, f"{pre_v:.2f}", ha="center", fontsize=10)
        ax.text(i + width / 2, post_v + 0.02, f"{post_v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_task(t) for t in tasks])
    ax.set_ylabel("mean episode reward")
    ax.set_ylim(0, max(1.0, max(post_means + pre_means + [0.0]) + 0.15))
    ax.set_title("SFT-on-oracle: pre vs post per task")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out)


def plot_sft_components(pre: Optional[dict], post: Optional[dict], out: Path) -> None:
    """One subplot per task; six grouped bars (pre/post per rubric component)."""
    if not (pre and post):
        print("  skipping sft_components.png (need both pre + post per-task evals)", flush=True)
        return

    tasks = list(pre.keys())
    components = list(RUBRIC_KEYS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    x = list(range(len(components)))
    width = 0.38
    for ax, task in zip(axes, tasks):
        pre_components = pre[task].get("components", {}) or {}
        post_components = post[task].get("components", {}) or {}
        pre_vals = [float(pre_components.get(c, 0.0)) for c in components]
        post_vals = [float(post_components.get(c, 0.0)) for c in components]
        ax.bar(
            [i - width / 2 for i in x], pre_vals, width,
            color=COL_BASELINE, edgecolor="black", label="pre",
        )
        ax.bar(
            [i + width / 2 for i in x], post_vals, width,
            color=COL_KEPT, edgecolor="black", label="post",
        )
        # Reference: each component's max weight as a thin horizontal mark.
        for i, c in enumerate(components):
            ax.hlines(
                RUBRIC_WEIGHTS[c], i - width, i + width,
                colors="#1f2937", linestyles=":", linewidth=1, alpha=0.7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=30, fontsize=9)
        ax.set_ylim(0, max(0.32, max(post_vals + pre_vals + [0.0]) + 0.04))
        ax.set_title(_short_task(task))
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")

    axes[0].set_ylabel("rubric component score")
    axes[-1].legend(loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("SFT-on-oracle: rubric component breakdown (dashed = max weight)", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _run_rft_mode(data: dict, out_dir: Path, task_id: str) -> None:
    plot_training_loss(data.get("iterations", []), out_dir / "training_loss.png")
    plot_training_reward(data.get("iterations", []), out_dir / "training_reward.png")
    plot_component_comparison(
        data.get("pre_eval"), data.get("post_eval"),
        out_dir / "component_comparison.png",
    )
    plot_score_summary(
        data.get("pre_eval"), data.get("post_eval"),
        out_dir / "score_summary.png", task_id=task_id,
    )


def _run_sft_mode(data: dict, out_dir: Path) -> None:
    plot_sft_loss(data.get("loss_history", []) or [], out_dir / "sft_loss.png")
    plot_sft_pre_post(
        data.get("pre_eval"), data.get("post_eval"),
        out_dir / "sft_pre_post.png",
    )
    plot_sft_components(
        data.get("pre_eval"), data.get("post_eval"),
        out_dir / "sft_components.png",
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot training metrics → PNGs for the README")
    p.add_argument(
        "--mode", choices=("rft", "sft"), default="rft",
        help="Which set of plots to produce. 'rft' reads rft_metrics.json (default); "
        "'sft' reads sft_metrics.json from train_sft.py.",
    )
    p.add_argument(
        "--metrics", default=None,
        help="Path to the metrics JSON. Defaults: ./rft_metrics.json (rft mode) "
        "or ./sft_metrics.json (sft mode).",
    )
    p.add_argument(
        "--out", default="./assets",
        help="Directory to write PNGs (will be created if missing)",
    )
    p.add_argument(
        "--task-id", default="easy_canary_regression",
        help="Task name for RFT-mode plot titles (purely cosmetic)",
    )
    args = p.parse_args(argv)

    metrics_path = Path(
        args.metrics
        or ("./sft_metrics.json" if args.mode == "sft" else "./rft_metrics.json")
    )
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found.", file=sys.stderr)
        return 1

    data = json.loads(metrics_path.read_text())
    out_dir = Path(args.out)

    print(f"reading {metrics_path} (mode={args.mode}) → writing PNGs to {out_dir}/")
    if args.mode == "sft":
        _run_sft_mode(data, out_dir)
    else:
        _run_rft_mode(data, out_dir, args.task_id)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
