"""
Rubric grader for the Incident Commander environment.

Six weighted components sum to 1.0 and produce both a dense per-step signal
(for the agent's ``obs.reward`` feedback) and a bounded terminal episode
score:

    * Containment       0.25  max blast radius kept low
    * MTTR              0.20  exponential decay from fault fire to mitigation
    * Correct RCA       0.20  diagnose action matches ground-truth service+tag
    * Right mitigation  0.15  correct mitigation action fired
    * Comms SLA         0.10  status-page update within SLA window after fault
    * Post-mortem       0.10  structured, factual post-mortem JSON at terminal

Design contract:

* Every method is a deterministic pure function of the :class:`Simulator`
  state (nothing reads wall-clock time or touches the RNG).
* ``observe_step`` returns the *incremental* reward since its last call;
  total cumulative reward emitted across the episode equals the final
  :class:`RubricScore.total`.
* All sub-scores and the final total are clamped into ``[0.0, 1.0]`` — the
  submission checklist disqualifies graders that return values outside that
  range.

Anti-gaming notes:

* ``hold`` as a mitigation only earns the mitigation component when the
  current task's ground-truth mitigation is ``hold``; picking ``hold`` on the
  easy (canary) task earns nothing.
* Wrong mitigations (e.g. ``rollback`` when ground-truth is ``hold``) zero
  the mitigation component — correct-play only.
* A second ``status_page`` update too soon after the first zeros the comms
  component to discourage spamming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .postmortem_check import check_postmortem

if TYPE_CHECKING:
    from ..simulator.simulator import Simulator


COMMS_SPAM_MIN_GAP_SEC: int = 60


@dataclass(frozen=True)
class ComponentWeights:
    """Weights for the six rubric components (must sum to 1.0)."""

    containment: float = 0.25
    mttr: float = 0.20
    rca: float = 0.20
    mitigation: float = 0.15
    comms: float = 0.10
    postmortem: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "containment": self.containment,
            "mttr": self.mttr,
            "rca": self.rca,
            "mitigation": self.mitigation,
            "comms": self.comms,
            "postmortem": self.postmortem,
        }

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"ComponentWeights must sum to 1.0, got {total:.6f}")


DEFAULT_WEIGHTS: ComponentWeights = ComponentWeights()


@dataclass
class RubricScore:
    """Breakdown of earned rubric component scores (each in ``[0, weight]``)."""

    containment: float = 0.0
    mttr: float = 0.0
    rca: float = 0.0
    mitigation: float = 0.0
    comms: float = 0.0
    postmortem: float = 0.0

    @property
    def total(self) -> float:
        return round(
            min(
                1.0,
                self.containment
                + self.mttr
                + self.rca
                + self.mitigation
                + self.comms
                + self.postmortem,
            ),
            6,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "containment": round(self.containment, 6),
            "mttr": round(self.mttr, 6),
            "rca": round(self.rca, 6),
            "mitigation": round(self.mitigation, 6),
            "comms": round(self.comms, 6),
            "postmortem": round(self.postmortem, 6),
            "total": self.total,
        }


class RubricGrader:
    """Stateful grader that tracks cumulative earnings across an episode.

    Expected lifecycle::

        grader = RubricGrader(weights=...)
        reward = grader.observe_step(simulator)    # called once per env.step
        ...
        final = grader.observe_terminal(simulator) # called once when done=True
    """

    def __init__(self, weights: Optional[ComponentWeights] = None) -> None:
        self.weights: ComponentWeights = weights or DEFAULT_WEIGHTS
        self.weights.validate()
        self.score: RubricScore = RubricScore()
        self._last_total: float = 0.0
        self._terminal_applied: bool = False

    def reset(self) -> None:
        """Reset cumulative earnings to zero (mirrors an env ``reset``)."""
        self.score = RubricScore()
        self._last_total = 0.0
        self._terminal_applied = False

    def observe_step(self, sim: "Simulator") -> float:
        """Update the per-step components and return the incremental reward.

        Called once *after* :class:`Simulator.step` has updated the world.
        Only the components that can be credited before the episode ends
        (RCA, mitigation, comms) move here; terminal-only components
        (containment, MTTR, post-mortem) land in :meth:`observe_terminal`.
        """
        self._update_rca(sim)
        self._update_mitigation(sim)
        self._update_comms(sim)
        return self._delta()

    def observe_terminal(self, sim: "Simulator") -> float:
        """Apply terminal components (containment, MTTR, post-mortem).

        Idempotent: calling more than once is a no-op.
        """
        if self._terminal_applied:
            return 0.0
        self._update_containment(sim)
        self._update_mttr(sim)
        self._update_postmortem(sim)
        self._terminal_applied = True
        return self._delta()

    def _delta(self) -> float:
        current = self.score.total
        inc = current - self._last_total
        self._last_total = current
        return round(max(0.0, inc), 6)

    def _update_rca(self, sim: "Simulator") -> None:
        if not sim.diagnose_history:
            return
        best_sub = max(sub for _, _, sub in sim.diagnose_history)
        earned = best_sub * self.weights.rca
        if earned > self.score.rca:
            self.score.rca = min(self.weights.rca, earned)

    def _update_mitigation(self, sim: "Simulator") -> None:
        if not any(matched for _, _, matched in sim.mitigation_history):
            return
        self.score.mitigation = self.weights.mitigation

    def _update_comms(self, sim: "Simulator") -> None:
        """Award the comms component, branching on what the task actually needs.

        For most incidents the right channel is a public ``status_page``
        update within SLA, so that's the default. For the hard
        silent-corruption task, however, broadcasting "we have data
        corruption" on the status page is actively wrong — it spooks users
        who weren't affected. The right comms is a ``customer_email`` to
        the targeted ``cohort``. We dispatch on ``correct_mitigation``
        rather than ``task_id`` so this stays a property of the fault, not
        a special case.
        """
        sla_window = sim.task.comms_sla_sec
        fault_fires_at = sim.fault.fires_at_sec

        if sim.fault.correct_mitigation == "partial_rollback":
            self._update_comms_targeted_email(sim, fault_fires_at, sla_window)
            return

        self._update_comms_status_page(sim, fault_fires_at, sla_window)

    def _update_comms_status_page(
        self,
        sim: "Simulator",
        fault_fires_at: int,
        sla_window: int,
    ) -> None:
        """Default broadcast-style comms grader (easy + medium tasks)."""
        status_page_events = [
            msg for msg in sim.chat_feed if msg.channel == "status_page"
        ]
        if not status_page_events:
            return

        first = status_page_events[0]
        elapsed_since_fault = first.timestamp_sec - fault_fires_at
        if elapsed_since_fault < 0 or elapsed_since_fault > sla_window:
            return

        comms_score = self.weights.comms

        if len(status_page_events) >= 2:
            second = status_page_events[1]
            if (second.timestamp_sec - first.timestamp_sec) < COMMS_SPAM_MIN_GAP_SEC:
                comms_score = 0.0

        if comms_score > self.score.comms:
            self.score.comms = comms_score
        elif comms_score < self.score.comms:
            self.score.comms = comms_score

    def _update_comms_targeted_email(
        self,
        sim: "Simulator",
        fault_fires_at: int,
        sla_window: int,
    ) -> None:
        """Hard-task comms grader: targeted customer_email with a cohort.

        Requirements for credit:

        * Channel is ``customer_email`` (not ``status_page``).
        * ``cohort`` is set to a non-empty string (targeted, not blast).
        * Sent within ``sla_window`` of the fault firing.

        A ``status_page`` post on a silent-corruption incident earns 0 — by
        design, since broadcasting noise about a contained issue creates
        unnecessary panic. This is a structural anti-gaming guard: the
        easy-task playbook (post status_page → done) does not transfer
        here.
        """
        for msg in sim.chat_feed:
            if msg.channel != "customer_email":
                continue
            if not msg.cohort or not msg.cohort.strip():
                continue
            elapsed_since_fault = msg.timestamp_sec - fault_fires_at
            if elapsed_since_fault < 0 or elapsed_since_fault > sla_window:
                continue
            if self.weights.comms > self.score.comms:
                self.score.comms = self.weights.comms
            return

    def _update_containment(self, sim: "Simulator") -> None:
        max_allowed = max(sim.task.max_blast_radius, 1e-9)
        fraction_of_budget = sim.max_blast_radius_pct / max_allowed
        containment_sub = max(0.0, 1.0 - min(1.0, fraction_of_budget))
        self.score.containment = containment_sub * self.weights.containment

    def _update_mttr(self, sim: "Simulator") -> None:
        if not sim.fault.mitigated or sim.fault.mitigated_at_sec is None:
            return
        elapsed = max(0, sim.fault.mitigated_at_sec - sim.fault.fires_at_sec)
        target = max(1, sim.task.target_mttr_sec)
        mttr_sub = math.exp(-elapsed / target)
        self.score.mttr = min(self.weights.mttr, mttr_sub * self.weights.mttr)

    def _update_postmortem(self, sim: "Simulator") -> None:
        if not sim.postmortem_submitted:
            return
        pm_sub = check_postmortem(sim.fault, sim.postmortem_json)
        self.score.postmortem = pm_sub * self.weights.postmortem
