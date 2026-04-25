"""Scripted oracle policies for SFT data generation.

Each policy emits a canonical action sequence that scores well above the
``+0.10`` regression-gate margin used by ``train_sft.py``. Policies are
deterministic — same ``(task_id, seed)`` produces the same action sequence —
which is what lets us treat their output as labeled supervised data.

Three task scripts, plus a seed-aware dispatcher for the medium task whose
ground-truth mitigation differs by ``seed % 3``:

    seed % 3 == 0 → provider     (mitigate hold)
    seed % 3 == 1 → integration  (mitigate feature_flag → backup processor)
    seed % 3 == 2 → our_deploy   (mitigate rollback on payments)

Reused by:

* ``training/build_sft_dataset.py`` — labels the (obs, action) pairs.
* ``training/eval_unsloth.py`` — sanity-check the env from a fresh Python
  process before / after training the model.
* the smoke-test mirror in ``tests/test_smoke.py`` — kept in sync by hand;
  the smoke tests are the load-bearing assertion that these scripts score
  above 0.5.

Determinism contract: all policies are pure functions over their script
index; they read no wall-clock time and no unseeded RNG.
"""

from __future__ import annotations

from typing import Callable, List

from incident_commander.models import ICAction, ICObservation


PolicyCallable = Callable[[ICObservation, List[str]], ICAction]


# --------------------------------------------------------------------------- #
# Easy: canary regression on payments → rollback.
# Mirrors ``tests/test_smoke._oracle_script`` exactly (proven > 0.5 / ~0.872).
# --------------------------------------------------------------------------- #
def _canary_script() -> List[ICAction]:
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(
            op="delegate",
            role="sre",
            task="investigate payments regression after canary deploy",
        ),
        ICAction(
            op="communicate",
            channel="status_page",
            message="Investigating elevated errors on payments; update in 15m.",
        ),
        ICAction(
            op="diagnose",
            root_cause_service="payments",
            root_cause_tag="bad_deploy",
        ),
        ICAction(op="mitigate", mitigation="rollback", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Canary deploy canary-v2.4.1 on payments doubled error rate; "
                    "rolled back and traffic restored."
                ),
                "root_cause_service": "payments",
                "root_cause_tag": "bad_deploy",
                "timeline": [
                    "t=30s canary canary-v2.4.1 rolled out to 5% traffic",
                    "t=120s rollback issued, metrics restored",
                ],
                "actions_taken": [
                    "Queried metrics on payments",
                    "Executed rollback on payments",
                ],
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# Medium: third-party attribution. Three sub-scripts keyed off ``seed % 3``.
# Names + canonical mitigations come from simulator/faults.py:
#   provider     → mitigate hold (no target needed)
#   integration  → mitigate feature_flag, target='payments_backup_processor'
#   our_deploy   → mitigate rollback, target='payments'
# --------------------------------------------------------------------------- #
def _third_party_provider_script() -> List[ICAction]:
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="query_external_status", target="stripe"),
        ICAction(
            op="delegate",
            role="sre",
            task="confirm scope of upstream payment provider degradation",
        ),
        ICAction(
            op="communicate",
            channel="status_page",
            message=(
                "Upstream payment provider degraded; holding traffic; "
                "next update in 15m."
            ),
        ),
        ICAction(
            op="diagnose",
            root_cause_service="payments",
            root_cause_tag="upstream_third_party",
        ),
        ICAction(op="mitigate", mitigation="hold", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Upstream Stripe outage caused payments errors; held our "
                    "service traffic instead of rolling back, rode out the "
                    "provider's recovery."
                ),
                "root_cause_service": "payments",
                "root_cause_tag": "upstream_third_party",
                "timeline": [
                    "t=60s external_status confirmed stripe degraded",
                    "t=120s held traffic on payments; provider recovered",
                ],
                "actions_taken": [
                    "Queried external provider status",
                    "Held traffic on payments (no rollback)",
                ],
            },
        ),
    ]


def _third_party_integration_script() -> List[ICAction]:
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="query_logs", service="payments", since_sec=0),
        ICAction(
            op="delegate",
            role="eng_lead",
            task="confirm payments→stripe integration is misconfigured and identify backup processor",
        ),
        ICAction(
            op="communicate",
            channel="status_page",
            message=(
                "Payments integration degraded; cutting over to backup "
                "processor; update in 15m."
            ),
        ),
        ICAction(
            op="diagnose",
            root_cause_service="payments",
            root_cause_tag="bad_integration",
        ),
        ICAction(
            op="mitigate",
            mitigation="feature_flag",
            target="payments_backup_processor",
        ),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Misconfigured payments→stripe integration caused webhook "
                    "errors; flipped feature_flag to the backup processor and "
                    "service recovered."
                ),
                "root_cause_service": "payments",
                "root_cause_tag": "bad_integration",
                "timeline": [
                    "t=60s logs showed auth.refused on stripe integration",
                    "t=120s feature_flag cut traffic to backup processor",
                ],
                "actions_taken": [
                    "Queried payments logs",
                    "Flipped backup-processor feature flag",
                ],
            },
        ),
    ]


def _third_party_our_deploy_script() -> List[ICAction]:
    return [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="query_logs", service="payments", since_sec=0),
        ICAction(
            op="delegate",
            role="sre",
            task="correlate payments error rate with the most recent payments deploy",
        ),
        ICAction(
            op="communicate",
            channel="status_page",
            message=(
                "Recent payments deploy caused webhook errors; rolling back; "
                "update in 15m."
            ),
        ),
        ICAction(
            op="diagnose",
            root_cause_service="payments",
            root_cause_tag="bad_deploy",
        ),
        ICAction(op="mitigate", mitigation="rollback", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Recent payments deploy payments-v1.8.3 broke stripe "
                    "webhook handling; rolled back and traffic recovered."
                ),
                "root_cause_service": "payments",
                "root_cause_tag": "bad_deploy",
                "timeline": [
                    "t=60s logs referenced deploy tag payments-v1.8.3",
                    "t=120s rollback issued on payments",
                ],
                "actions_taken": [
                    "Queried payments metrics + logs",
                    "Executed rollback on payments",
                ],
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# Hard: silent data corruption on orders → partial_rollback + cohort email.
# Mirrors ``tests/test_smoke._hard_oracle_script`` (proven > 0.5).
# --------------------------------------------------------------------------- #
def _data_corruption_script() -> List[ICAction]:
    return [
        ICAction(op="query_audit", service="orders", since_sec=0),
        ICAction(
            op="delegate",
            role="eng_lead",
            task="scope affected cohort and confirm migration is the cause",
        ),
        ICAction(
            op="communicate",
            channel="customer_email",
            message=(
                "We've identified an issue affecting your account balance "
                "after a recent maintenance window; we're correcting it now."
            ),
            cohort="affected_accounts",
        ),
        ICAction(
            op="diagnose",
            root_cause_service="orders",
            root_cause_tag="data_corruption",
        ),
        ICAction(
            op="mitigate",
            mitigation="partial_rollback",
            target="orders",
        ),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Migration migration-2026-04-19 silently corrupted balance "
                    "rows on orders; partial rollback restored affected cohort."
                ),
                "root_cause_service": "orders",
                "root_cause_tag": "data_corruption",
                "timeline": [
                    "t=5s migration migration-2026-04-19 applied to orders",
                    "t=120s partial_rollback restored affected cohort",
                ],
                "actions_taken": [
                    "Queried audit log on orders",
                    "Executed partial_rollback on orders",
                ],
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _medium_script_for_seed(seed: int) -> List[ICAction]:
    """Pick the medium-task script whose ground truth matches ``seed``.

    Mirrors ``simulator/faults.py:pick_third_party_variant``: the env keys the
    third-party variant off ``seed % 3``, so the oracle has to do the same.
    Otherwise we'd label a ``feature_flag`` action as correct on a
    ``hold``-variant episode and SFT would teach the model the wrong thing.
    """
    bucket = seed % 3
    if bucket == 0:
        return _third_party_provider_script()
    if bucket == 1:
        return _third_party_integration_script()
    return _third_party_our_deploy_script()


def script_for(task_id: str, seed: int = 0) -> List[ICAction]:
    """Return the canonical action list for ``(task_id, seed)``."""
    if task_id == "easy_canary_regression":
        return _canary_script()
    if task_id == "medium_third_party_attribution":
        return _medium_script_for_seed(seed)
    if task_id == "hard_silent_data_corruption":
        return _data_corruption_script()
    raise ValueError(f"unknown task_id: {task_id!r}")


def make_oracle(task_id: str, seed: int = 0) -> PolicyCallable:
    """Return a stateful policy that walks ``script_for(task_id, seed)``.

    The returned callable takes ``(observation, history)`` to match the
    signature expected by ``inference.PolicyFn`` and ``rollout.run_rollout``.
    Past the script's last action, it sticks on the final action — the env
    will terminate the episode (``postmortem`` is a terminal op) before that
    fallback ever fires, but the bound prevents an ``IndexError`` in case of
    weirdness.
    """
    script = script_for(task_id, seed=seed)
    state = {"i": 0}

    def _call(_obs: ICObservation, _history: List[str]) -> ICAction:
        i = state["i"]
        state["i"] = min(i + 1, len(script) - 1)
        return script[i]

    return _call


__all__ = [
    "PolicyCallable",
    "make_oracle",
    "script_for",
]
