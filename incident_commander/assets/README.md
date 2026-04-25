# `assets/` — training plots

This directory holds the four PNGs the README embeds. They are produced by
running `training/train_colab.ipynb` end-to-end; the plotting cell calls
`python -m incident_commander.training.plot_metrics` against the
`rft_metrics.json` written during training.

**Contents after a real training run:**

- `training_loss.png` — SFT NLL loss vs RFT iteration. Should trend down.
- `training_reward.png` — episode score vs RFT iteration, with three lines:
  all rollouts (mean), top-K kept (mean), best-of-batch. Should trend up.
- `component_comparison.png` — baseline-vs-trained against the six rubric
  component weight ceilings.
- `score_summary.png` — pre vs post total score with error bars. The
  headline number.

**Reproducing locally** (skip if running the Colab):

```bash
# After running train_rft.py (which writes rft_metrics.json):
python -m incident_commander.training.plot_metrics \
  --metrics ./rft_metrics.json --out ./assets \
  --task-id easy_canary_regression
```

Plots are kept out of git until a real training run produces them — the
README's `![](...)` references will resolve at that point. We deliberately
do not commit synthetic placeholders so a reader can never mistake demo
output for measured results.
