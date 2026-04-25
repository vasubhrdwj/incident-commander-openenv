"""Generate training plots from ``rft_metrics.json``.

Produces four PNGs that get committed to ``assets/`` and embedded in the
README per submission criteria ("save plots as .png and commit them, embed
in README with a one-line caption").

Plots produced
==============

1. ``training_loss.png`` — SFT loss vs RFT iteration (decreasing).
2. ``training_reward.png`` — mean rollout score (sampled & kept) vs iter.
3. ``component_comparison.png`` — bar chart, baseline vs trained, per
   rubric component.
4. ``score_summary.png`` — final pre/post total score, big and obvious.

Usage::

    python -m incident_commander.training.plot_metrics \\
        --metrics ./rft_metrics.json \\
        --out ./assets

If ``rft_metrics.json`` only has ``pre_eval`` / ``iterations`` populated
(training still in progress) the missing-data plots are skipped with a
warning rather than crashing the whole run.
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


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot RFT metrics → PNGs for the README")
    p.add_argument(
        "--metrics", default="./rft_metrics.json",
        help="Path to the metrics JSON written by train_rft.py",
    )
    p.add_argument(
        "--out", default="./assets",
        help="Directory to write PNGs (will be created if missing)",
    )
    p.add_argument(
        "--task-id", default="easy_canary_regression",
        help="Task name for plot titles (purely cosmetic)",
    )
    args = p.parse_args(argv)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run train_rft.py first.", file=sys.stderr)
        return 1

    data = json.loads(metrics_path.read_text())
    out_dir = Path(args.out)

    print(f"reading {metrics_path} → writing PNGs to {out_dir}/")
    plot_training_loss(data.get("iterations", []), out_dir / "training_loss.png")
    plot_training_reward(data.get("iterations", []), out_dir / "training_reward.png")
    plot_component_comparison(data.get("pre_eval"), data.get("post_eval"), out_dir / "component_comparison.png")
    plot_score_summary(data.get("pre_eval"), data.get("post_eval"), out_dir / "score_summary.png", task_id=args.task_id)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
