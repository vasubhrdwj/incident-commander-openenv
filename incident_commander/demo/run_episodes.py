"""Three-episode demo for the submission recording.

Run with::

    python -m incident_commander.demo.run_episodes
    # or, if PYTHONPATH isn't set:
    PYTHONPATH=/path/to/OpenEnv python incident_commander/demo/run_episodes.py

The script runs three deterministic, in-process episodes against
:class:`IncidentCommanderEnvironment` (no HTTP server needed) and prints a
recording-friendly trace plus the six-component rubric breakdown for each.
A combined JSON artifact lands at ``demo/episodes.json`` so reviewers can
verify the numbers without re-running.

Episodes (the headline triplet):

    1. Easy oracle  ───────────────────────────────────────  ceiling on easy
    2. Hard oracle  ──────────────────────────────────  same ceiling on hard
    3. Hard task with the easy-task playbook  ────  rubric anti-gaming bites

The third episode is the demo. Same agent shape, same six rubric components,
same level of effort — but the agent reuses the easy-task heuristics
(``status_page`` + full ``rollback``) on a silent-corruption incident. The
score collapses by ~0.4 even though the diagnose + post-mortem are perfect,
because two structural rubric guards (task-conditional comms; strict
``partial_rollback`` matcher) zero out two whole components. That's the
"rubric is structural, not heuristic" story manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from incident_commander.models import ICAction
from incident_commander.server.incident_commander_environment import (
    IncidentCommanderEnvironment,
)


# --- Scripted action sequences --------------------------------------------


def _easy_oracle() -> list[ICAction]:
    """Optimal IC playbook for ``easy_canary_regression`` — scores 0.872."""
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="delegate", role="sre", task="investigate payments error spike after canary deploy"),
        ICAction(op="communicate", channel="status_page", message="Investigating elevated errors on payments; updates every 15 min."),
        ICAction(op="diagnose", root_cause_service="payments", root_cause_tag="bad_deploy"),
        ICAction(op="mitigate", mitigation="rollback", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": "Canary deploy regressed payments error rate; rolled back to restore the canary tier.",
                "root_cause_service": "payments",
                "root_cause_tag": "bad_deploy",
                "timeline": [
                    "t=30s canary released to 5% traffic",
                    "t=30s error rate breached SLO",
                    "t=120s rollback dispatched",
                    "t=180s service recovered",
                ],
                "actions_taken": ["queried metrics", "delegated to SRE", "posted status_page", "executed rollback"],
            },
        ),
    ]


def _hard_oracle() -> list[ICAction]:
    """Optimal IC playbook for ``hard_silent_data_corruption`` — scores 0.855.

    Notice three things this episode does that the easy episode does not,
    and that an easy-task-trained agent would fail to do:

    * ``query_audit`` instead of ``query_metrics`` — the dashboard is green.
    * ``customer_email`` with ``cohort`` instead of ``status_page``.
    * ``partial_rollback`` instead of full ``rollback``.
    """
    return [
        ICAction(op="query_audit", service="orders", since_sec=0),
        ICAction(op="delegate", role="eng_lead", task="scope affected cohort and confirm migration is the cause"),
        ICAction(
            op="communicate",
            channel="customer_email",
            message="We've identified an issue affecting your account balance after a recent maintenance window; we're correcting it now and will confirm when complete.",
            cohort="affected_accounts",
        ),
        ICAction(op="diagnose", root_cause_service="orders", root_cause_tag="data_corruption"),
        ICAction(op="mitigate", mitigation="partial_rollback", target="orders"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": "Migration migration-2026-04-19 silently corrupted balance rows on orders; partial rollback restored the affected cohort.",
                "root_cause_service": "orders",
                "root_cause_tag": "data_corruption",
                "timeline": [
                    "t=5s migration migration-2026-04-19 applied",
                    "t=10s anomalous db.write cluster on balance rows begins",
                    "t=30s customer report received",
                    "t=150s partial_rollback dispatched on orders",
                ],
                "actions_taken": [
                    "queried audit log",
                    "delegated to eng_lead",
                    "emailed affected cohort",
                    "executed partial_rollback",
                ],
            },
        ),
    ]


def _hard_with_easy_playbook() -> list[ICAction]:
    """Naive-but-competent IC running the easy-task playbook on the hard task.

    What this episode demonstrates:

    * ``query_metrics`` finds nothing (dashboard is green) — the agent has
      no signal to work with because it didn't think to check audit.
    * ``status_page`` on a silent corruption earns 0 comms credit (the
      grader is task-conditional on ``correct_mitigation``).
    * ``rollback`` on data corruption is rejected by the fault matcher —
      ``partial_rollback`` is the only mitigation that fixes it. The fault
      stays active for the rest of the episode, MTTR component is zero,
      and blast radius keeps growing.
    * ``diagnose`` and ``postmortem`` still credit fully — the model knew
      *what* the problem was, it just didn't choose the right *response*.

    The combined effect is a ~0.4 score gap on the same task with the same
    agent shape. The gap is entirely structural: it lives in the rubric,
    not in the prompt.
    """
    return [
        ICAction(op="query_metrics", service="orders"),
        ICAction(
            op="communicate",
            channel="status_page",
            message="Investigating a reported data issue on orders; will update.",
        ),
        ICAction(op="delegate", role="sre", task="investigate orders data discrepancy"),
        ICAction(op="diagnose", root_cause_service="orders", root_cause_tag="data_corruption"),
        ICAction(op="mitigate", mitigation="rollback", target="orders"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": "Data discrepancy on orders traced to a recent migration; attempted full rollback.",
                "root_cause_service": "orders",
                "root_cause_tag": "data_corruption",
                "timeline": [
                    "t=30s customer report received",
                    "t=90s status_page update posted",
                    "t=150s rollback attempted on orders",
                ],
                "actions_taken": [
                    "queried metrics",
                    "posted status_page",
                    "delegated to SRE",
                    "attempted full rollback",
                ],
            },
        ),
    ]


# --- Runner ---------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """One demo episode's full trace + rubric breakdown."""

    title: str
    task_id: str
    seed: int
    actions: list[dict[str, Any]]
    step_rewards: list[float]
    components: dict[str, float]
    total: float
    mitigated: bool
    blast_radius_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "task_id": self.task_id,
            "seed": self.seed,
            "actions": self.actions,
            "step_rewards": self.step_rewards,
            "components": self.components,
            "total": self.total,
            "mitigated": self.mitigated,
            "blast_radius_pct": self.blast_radius_pct,
        }


def _action_summary(action: ICAction) -> str:
    """One-line action summary suitable for terminal output."""
    op = action.op
    parts = [op]
    if action.service:
        parts.append(f"service={action.service}")
    if action.role:
        parts.append(f"role={action.role}")
    if action.mitigation:
        parts.append(f"mitigation={action.mitigation}")
    if action.target and not (action.role or action.mitigation):
        parts.append(f"target={action.target}")
    elif action.target and action.mitigation:
        parts.append(f"target={action.target}")
    if action.channel:
        parts.append(f"channel={action.channel}")
    if action.cohort:
        parts.append(f"cohort={action.cohort}")
    if action.root_cause_service:
        parts.append(f"rcs={action.root_cause_service}")
    if action.root_cause_tag:
        parts.append(f"rct={action.root_cause_tag}")
    return " ".join(parts)


def _action_to_dict(action: ICAction) -> dict[str, Any]:
    return action.model_dump(exclude_none=True, exclude={"postmortem_json"})


def run_episode(
    *,
    title: str,
    task_id: str,
    seed: int,
    script: list[ICAction],
    verbose: bool = True,
) -> EpisodeRecord:
    env = IncidentCommanderEnvironment(task_id=task_id, seed=seed)
    env.reset()

    if verbose:
        bar = "═" * 70
        print(f"\n{bar}")
        print(f"  {title}")
        print(f"  task={task_id}  seed={seed}")
        print(bar)

    step_rewards: list[float] = []
    actions_log: list[dict[str, Any]] = []
    last_obs = None

    for i, action in enumerate(script, start=1):
        obs = env.step(action)
        last_obs = obs
        reward = float(obs.reward or 0.0)
        step_rewards.append(reward)
        actions_log.append(_action_to_dict(action))
        if verbose:
            print(f"  [STEP {i:>2}] {_action_summary(action)}")
            print(f"           → +{reward:.3f}   (cumulative {sum(step_rewards):.3f})")
        if obs.done:
            break

    score = env._grader.score
    components = {
        "containment": round(score.containment, 4),
        "mttr": round(score.mttr, 4),
        "rca": round(score.rca, 4),
        "mitigation": round(score.mitigation, 4),
        "comms": round(score.comms, 4),
        "postmortem": round(score.postmortem, 4),
    }
    total = round(score.total, 4)

    if verbose:
        weights = {"containment": 0.25, "mttr": 0.20, "rca": 0.20, "mitigation": 0.15, "comms": 0.10, "postmortem": 0.10}
        print(f"\n  Final score: {total:.4f}")
        print(f"  Components (earned / weight):")
        for name, val in components.items():
            print(f"    {name:<13}{val:.4f} / {weights[name]:.2f}")
        print(f"  Fault mitigated: {env._sim.fault.mitigated}")
        print(f"  Max blast radius: {env._sim.max_blast_radius_pct:.4f}")

    return EpisodeRecord(
        title=title,
        task_id=task_id,
        seed=seed,
        actions=actions_log,
        step_rewards=step_rewards,
        components=components,
        total=total,
        mitigated=env._sim.fault.mitigated,
        blast_radius_pct=round(env._sim.max_blast_radius_pct, 4),
    )


def run_all(*, output_path: Optional[Path] = None, verbose: bool = True) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []

    episodes.append(
        run_episode(
            title="EPISODE 1/3 — Easy oracle (calibrates the rubric ceiling)",
            task_id="easy_canary_regression",
            seed=0,
            script=_easy_oracle(),
            verbose=verbose,
        )
    )
    episodes.append(
        run_episode(
            title="EPISODE 2/3 — Hard oracle (same ceiling, qualitatively different reasoning)",
            task_id="hard_silent_data_corruption",
            seed=0,
            script=_hard_oracle(),
            verbose=verbose,
        )
    )
    episodes.append(
        run_episode(
            title="EPISODE 3/3 — Hard task, easy-task playbook (anti-gaming guards bite)",
            task_id="hard_silent_data_corruption",
            seed=0,
            script=_hard_with_easy_playbook(),
            verbose=verbose,
        )
    )

    if verbose:
        bar = "═" * 70
        print(f"\n{bar}")
        print("  HEADLINE")
        print(bar)
        ep2, ep3 = episodes[1], episodes[2]
        delta = ep2.total - ep3.total
        print(
            f"  Same task ({ep2.task_id}), same six rubric components, same level of effort.\n"
            f"  Oracle (right playbook):     {ep2.total:.4f}\n"
            f"  Easy-task playbook on hard:  {ep3.total:.4f}\n"
            f"  Δ = {delta:+.4f}  ── earned by recognising the task type, not by raw skill."
        )
        print(
            f"\n  Where the Δ lives (per component):\n"
            f"    * comms:       {ep2.components['comms']:.4f} vs {ep3.components['comms']:.4f}  "
            f"── status_page on hard earns 0 (task-conditional grader)\n"
            f"    * mitigation:  {ep2.components['mitigation']:.4f} vs {ep3.components['mitigation']:.4f}  "
            f"── full rollback rejected by fault matcher (partial_rollback only)\n"
            f"    * mttr:        {ep2.components['mttr']:.4f} vs {ep3.components['mttr']:.4f}  "
            f"── follows from above: an unfixed fault has no MTTR"
        )
        print(
            f"\n  diagnose + postmortem fully credit in BOTH episodes — the agent\n"
            f"  knew WHAT the problem was; it just didn't choose the right RESPONSE.\n"
            f"  The rubric punishes that without any LLM-as-judge.\n"
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(
                {"episodes": [e.to_dict() for e in episodes]},
                f,
                indent=2,
            )
        if verbose:
            print(f"  Wrote artifact → {output_path}\n")

    return episodes


def main() -> None:
    out = Path(__file__).parent / "episodes.json"
    run_all(output_path=out, verbose=True)


if __name__ == "__main__":
    main()
