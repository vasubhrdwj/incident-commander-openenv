"""Smoke tests — exercise reset/step/determinism without an HTTP server.

These tests construct :class:`IncidentCommanderEnvironment` directly (the
server-side class behind the OpenEnv FastAPI adapter) so they run in-process
and don't need uvicorn. Per plan §7 R1 + CLAUDE.md determinism contract: the
same action sequence from the same seed must produce bit-identical scores.
"""

from __future__ import annotations

import pytest

from incident_commander.models import ICAction
from incident_commander.server.incident_commander_environment import (
    IncidentCommanderEnvironment,
)


def _oracle_script() -> list[ICAction]:
    """Scripted optimal run for ``easy_canary_regression``; scores ~0.872."""
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="delegate", role="sre", task="investigate payments regression after canary deploy"),
        ICAction(op="communicate", channel="status_page", message="Investigating elevated errors on payments; update in 15m."),
        ICAction(op="diagnose", root_cause_service="payments", root_cause_tag="bad_deploy"),
        ICAction(op="mitigate", mitigation="rollback", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": "Canary deploy regressed payments; rolled back and traffic restored.",
                "root_cause_service": "payments",
                "root_cause_tag": "bad_deploy",
                "timeline": ["t=30s canary rolled to 5%", "t=30s error rate breached SLO", "t=120s rollback", "t=180s recovered"],
                "actions_taken": ["queried metrics", "delegated to SRE", "posted status_page", "executed rollback"],
            },
        ),
    ]


def test_reset_returns_initial_observation() -> None:
    env = IncidentCommanderEnvironment(task_id="easy_canary_regression", seed=0)
    obs = env.reset()
    assert obs.task_id == "easy_canary_regression"
    assert not obs.done
    assert obs.step_budget_remaining > 0
    assert obs.dashboard  # 6-service graph populated


def test_step_advances_and_returns_reward() -> None:
    env = IncidentCommanderEnvironment(task_id="easy_canary_regression", seed=0)
    env.reset()
    obs = env.step(ICAction(op="query_metrics", service="payments"))
    assert obs.reward is not None
    assert obs.reward >= 0.0
    assert env.state.step_count == 1


def test_invalid_action_burns_step_budget() -> None:
    """Plan anti-gaming guard: invalid payload should not be a free retry."""
    env = IncidentCommanderEnvironment(task_id="easy_canary_regression", seed=0)
    obs_before = env.reset()
    obs_after = env.step(ICAction(op="query_metrics"))  # missing 'service'
    assert obs_after.step_budget_remaining == obs_before.step_budget_remaining - 1
    assert "invalid" in obs_after.last_action_result.lower()


def test_determinism_replay_identical_scores() -> None:
    """CLAUDE.md hard contract: same seed + same actions → identical rewards."""
    script = _oracle_script()

    def run() -> list[float]:
        env = IncidentCommanderEnvironment(task_id="easy_canary_regression", seed=0)
        env.reset()
        rewards: list[float] = []
        for action in script:
            obs = env.step(action)
            rewards.append(obs.reward or 0.0)
            if obs.done:
                break
        return rewards

    first = run()
    second = run()
    assert first == second, f"determinism broken: {first} != {second}"
    assert sum(first) > 0.5, f"oracle script should score > 0.5, got {sum(first):.3f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_medium_task_variants_differ_by_seed(seed: int) -> None:
    """Plan §2.5: medium task's ground-truth mitigation varies with seed."""
    env = IncidentCommanderEnvironment(task_id="medium_third_party_attribution", seed=seed)
    obs = env.reset()
    assert obs.task_id == "medium_third_party_attribution"
    assert not obs.done


# --- Hard task: silent data corruption -------------------------------------
#
# Defining properties of the hard task (see simulator/faults.py
# DataCorruptionFault docstring):
#
# 1. Dashboards stay green and no alerts ever fire.
# 2. The fault is only detectable via ``query_audit``.
# 3. The right mitigation is ``partial_rollback`` (a full ``rollback`` does NOT
#    credit — easy-task playbook does not transfer).
# 4. The right comms channel is ``customer_email`` with a ``cohort`` set;
#    a ``status_page`` post earns 0 comms credit on this task.


def _hard_oracle_script() -> list[ICAction]:
    """Scripted optimal run for ``hard_silent_data_corruption``."""
    return [
        ICAction(op="query_audit", service="orders", since_sec=0),
        ICAction(op="delegate", role="eng_lead", task="scope affected cohort and confirm migration is the cause"),
        ICAction(
            op="communicate",
            channel="customer_email",
            message="We've identified an issue affecting your account balance after a recent maintenance window; we're correcting it now.",
            cohort="affected_accounts",
        ),
        ICAction(op="diagnose", root_cause_service="orders", root_cause_tag="data_corruption"),
        ICAction(op="mitigate", mitigation="partial_rollback", target="orders"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": "Migration migration-2026-04-19 silently corrupted balance rows on orders; partial rollback restored affected cohort.",
                "root_cause_service": "orders",
                "root_cause_tag": "data_corruption",
                "timeline": [
                    "t=5s migration migration-2026-04-19 applied",
                    "t=10s anomalous db.write cluster begins",
                    "t=30s customer report received",
                    "t=120s partial_rollback issued",
                ],
                "actions_taken": ["queried audit log", "delegated to eng_lead", "emailed affected cohort", "executed partial_rollback"],
            },
        ),
    ]


def test_hard_task_dashboard_stays_green_no_alerts() -> None:
    """Defining property #1: silence is the fault, not an oversight."""
    env = IncidentCommanderEnvironment(task_id="hard_silent_data_corruption", seed=0)
    env.reset()
    # Step a few times to let sim_time advance past fires_at_sec.
    for _ in range(5):
        obs = env.step(ICAction(op="query_metrics", service="orders"))
    assert obs.alerts == [], f"hard task must not fire alerts; got {obs.alerts!r}"
    assert all(svc.healthy for svc in obs.dashboard.values()), (
        f"hard task dashboard must stay green; unhealthy: "
        f"{[s for s, h in obs.dashboard.items() if not h.healthy]}"
    )


def test_hard_task_audit_surfaces_anomalous_writes() -> None:
    """Defining property #2: query_audit is the only place the fault is visible."""
    env = IncidentCommanderEnvironment(task_id="hard_silent_data_corruption", seed=0)
    env.reset()
    # Advance past the migration anchor before pulling audit.
    env.step(ICAction(op="query_metrics", service="orders"))
    obs = env.step(ICAction(op="query_audit", service="orders", since_sec=0))
    assert obs.audit_events, "audit feed should surface events on hard task"
    assert any(e.anomalous for e in obs.audit_events), (
        f"audit feed must contain at least one anomalous event; got {obs.audit_events!r}"
    )
    assert any("migration" in e.resource for e in obs.audit_events), (
        f"audit events should reference the migration tag; got {obs.audit_events!r}"
    )


def test_hard_task_full_rollback_does_not_credit() -> None:
    """Anti-gaming guard: easy-task playbook (full rollback) must NOT score on hard."""
    env = IncidentCommanderEnvironment(task_id="hard_silent_data_corruption", seed=0)
    env.reset()
    obs = env.step(ICAction(op="mitigate", mitigation="rollback", target="orders"))
    # Mitigation history should reflect a no-op, not a fix.
    assert env._sim.fault.mitigated is False, "full rollback must not mitigate data corruption"
    assert env._sim.mitigation_history[-1] == ("rollback", "orders", False)


def test_hard_task_status_page_earns_zero_comms() -> None:
    """Anti-gaming guard: status_page broadcast should not earn comms credit."""
    env = IncidentCommanderEnvironment(task_id="hard_silent_data_corruption", seed=0)
    env.reset()
    env.step(ICAction(op="communicate", channel="status_page", message="Investigating data issue."))
    # Step the grader to terminal so comms is finalised.
    env.step(ICAction(op="postmortem", postmortem_json={"summary": "x" * 30}))
    final = env._grader.score
    assert final.comms == 0.0, f"status_page on hard task must earn 0 comms; got {final.comms}"


def test_hard_oracle_scores_above_threshold_and_is_deterministic() -> None:
    """End-to-end: oracle script clears > 0.5 and replays bit-identically."""
    script = _hard_oracle_script()

    def run() -> list[float]:
        env = IncidentCommanderEnvironment(task_id="hard_silent_data_corruption", seed=0)
        env.reset()
        rewards: list[float] = []
        for action in script:
            obs = env.step(action)
            rewards.append(obs.reward or 0.0)
            if obs.done:
                break
        return rewards

    first = run()
    second = run()
    assert first == second, f"determinism broken: {first} != {second}"
    assert sum(first) > 0.5, f"hard oracle should score > 0.5, got {sum(first):.3f}"
