"""
Fault injector.

A :class:`Fault` is the ground-truth source of everything the IC needs to
detect, attribute, and mitigate. Each concrete fault:

* mutates the :class:`ServiceGraph` health numbers as time advances,
* exposes ``ground_truth_service`` / ``ground_truth_tag`` / ``correct_mitigation``
  so graders can score ``diagnose`` and ``mitigate`` actions,
* reacts to a matching mitigation by entering the ``mitigated`` state, after
  which it decays back toward baseline over a short recovery window.

Determinism: faults must never call ``random`` on their own — they accept the
simulator's seeded RNG when they need randomness (none currently do). Fault
behaviour is a pure function of ``sim_time_sec`` and the mitigation log.

For the H4–H12 scaffold we only implement :class:`BadDeployFault` end-to-end
(powers the canary-regression easy task). The other two fault types are
carried here as well-structured stubs so the architecture is ready for the
medium / hard tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

from .service_graph import ServiceGraph


RootCauseTag = Literal[
    "bad_deploy",
    "bad_integration",
    "upstream_third_party",
    "data_corruption",
]

ThirdPartyVariant = Literal["provider", "integration", "our_deploy"]
THIRD_PARTY_VARIANTS: tuple[ThirdPartyVariant, ...] = (
    "provider",
    "integration",
    "our_deploy",
)
MitigationKind = Literal[
    "restart",
    "rollback",
    "partial_rollback",
    "scale",
    "feature_flag",
    "hold",
]


@dataclass
class Fault(ABC):
    """Base class for fault scenarios.

    Subclasses encode the ground truth for one task scenario.
    """

    fault_id: str
    fires_at_sec: int

    mitigated: bool = field(default=False, init=False)
    mitigated_at_sec: Optional[int] = field(default=None, init=False)
    description: str = field(default="", init=False)

    @property
    @abstractmethod
    def ground_truth_service(self) -> str: ...

    @property
    @abstractmethod
    def ground_truth_tag(self) -> RootCauseTag: ...

    @property
    @abstractmethod
    def correct_mitigation(self) -> MitigationKind: ...

    @property
    def correct_mitigation_target(self) -> Optional[str]:
        """Expected ``target`` payload for the correct mitigation (``None`` for ``hold``)."""
        if self.correct_mitigation == "hold":
            return None
        return self.ground_truth_service

    def is_active(self, sim_time_sec: int) -> bool:
        """Whether the fault is currently degrading service health."""
        if sim_time_sec < self.fires_at_sec:
            return False
        if not self.mitigated:
            return True
        assert self.mitigated_at_sec is not None
        return sim_time_sec < self.mitigated_at_sec + self.recovery_window_sec

    @property
    def recovery_window_sec(self) -> int:
        """How long after mitigation before health fully returns to baseline."""
        return 60

    @abstractmethod
    def apply(self, graph: ServiceGraph, sim_time_sec: int) -> None:
        """Apply fault effects to ``graph`` given the current simulator time.

        Must be idempotent — the simulator calls this once per tick after
        resetting the graph to baseline.
        """
        ...

    def try_mitigate(
        self,
        *,
        mitigation: MitigationKind,
        target: Optional[str],
        sim_time_sec: int,
    ) -> bool:
        """Attempt to mitigate; returns ``True`` iff this exact action fixes the fault.

        Default implementation: mitigation matches if ``mitigation`` and ``target``
        equal the ground truth. Subclasses may override for more nuanced matching
        (e.g. accept either of two valid mitigations).
        """
        if self.mitigated:
            return False
        if mitigation != self.correct_mitigation:
            return False
        expected_target = self.correct_mitigation_target
        if expected_target is not None and target != expected_target:
            return False
        self.mitigated = True
        self.mitigated_at_sec = sim_time_sec
        return True


@dataclass
class BadDeployFault(Fault):
    """Canary-regression fault: a new deploy on one service spikes error rate + latency.

    Used by the easy (``canary_regression``) task. Correct mitigation is a
    ``rollback`` on the affected service; any other mitigation leaves the fault
    active.
    """

    service: str = "payments"
    deploy_tag: str = "canary-v2.4.1"
    latency_multiplier: float = 2.0
    error_rate_floor: float = 0.04

    def __post_init__(self) -> None:
        self.description = (
            f"Canary deploy {self.deploy_tag!r} on {self.service}: "
            f"error rate ~{self.error_rate_floor:.0%}, latency ~{self.latency_multiplier:.1f}x baseline."
        )

    @property
    def ground_truth_service(self) -> str:
        return self.service

    @property
    def ground_truth_tag(self) -> RootCauseTag:
        return "bad_deploy"

    @property
    def correct_mitigation(self) -> MitigationKind:
        return "rollback"

    def apply(self, graph: ServiceGraph, sim_time_sec: int) -> None:
        state = graph.get(self.service)
        if state is None:
            return

        if sim_time_sec < self.fires_at_sec:
            return

        if not self.mitigated:
            state.has_active_deploy = True
            state.active_deploy_tag = self.deploy_tag
            spec = state.spec
            state.latency_p99_ms = spec.baseline_latency_p99_ms * self.latency_multiplier
            state.error_rate = max(spec.baseline_error_rate, self.error_rate_floor)
            return

        assert self.mitigated_at_sec is not None
        elapsed = sim_time_sec - self.mitigated_at_sec
        if elapsed >= self.recovery_window_sec:
            state.is_rolled_back = True
            state.has_active_deploy = False
            state.active_deploy_tag = None
            return

        decay = 1.0 - (elapsed / self.recovery_window_sec)
        spec = state.spec
        state.latency_p99_ms = spec.baseline_latency_p99_ms * (
            1.0 + (self.latency_multiplier - 1.0) * decay
        )
        state.error_rate = spec.baseline_error_rate + (
            self.error_rate_floor - spec.baseline_error_rate
        ) * decay
        state.has_active_deploy = True
        state.active_deploy_tag = self.deploy_tag


@dataclass
class ThirdPartyOutageFault(Fault):
    """Payment-webhook degradation with three qualitatively different root causes.

    Drives the medium task (``third_party_attribution``). The IC sees the same
    surface symptom on the ``payments`` service in all three variants, but the
    correct mitigation differs:

    * ``provider``    — Stripe itself is down. Correct action: ``hold`` +
      status-page comms; the provider recovers on its own timescale.
    * ``integration`` — Our integration is broken (misconfigured auth / retry
      storm / config drift). Correct action: ``feature_flag`` to the backup
      processor (deterministic target name the Eng Lead NPC advertises).
    * ``our_deploy`` — A recent payments deploy broke webhook handling.
      Correct action: ``rollback`` of the affected service.

    The discriminating signals live in:

    * ``query_external_status`` — only the ``provider`` variant reports
      ``degraded``.
    * ``query_logs`` — variant-specific error signatures (gateway.timeout vs
      auth.refused vs a deploy tag in the log line).
    * ``delegate(sre)`` + ``delegate(eng_lead)`` — NPCs give variant-specific
      findings that are informative but not trivialising.

    Recovery is per-variant: ``feature_flag`` and ``rollback`` restore
    quickly; ``hold`` waits out a longer provider-side recovery window during
    which blast radius continues to grow — that asymmetry is the natural
    reason a patient agent scores lower on containment + MTTR when ``hold``
    is correct, which creates an appropriate easy→medium gap without any
    grader-specific tuning.
    """

    variant: ThirdPartyVariant = "provider"
    provider: str = "stripe"
    downstream_service: str = "payments"
    backup_feature_flag: str = "payments_backup_processor"
    deploy_tag: str = "payments-v1.8.3"
    base_error_rate_floor: float = 0.10

    @property
    def ground_truth_service(self) -> str:
        return self.downstream_service

    @property
    def ground_truth_tag(self) -> RootCauseTag:
        if self.variant == "provider":
            return "upstream_third_party"
        if self.variant == "integration":
            return "bad_integration"
        return "bad_deploy"

    @property
    def correct_mitigation(self) -> MitigationKind:
        if self.variant == "provider":
            return "hold"
        if self.variant == "integration":
            return "feature_flag"
        return "rollback"

    @property
    def correct_mitigation_target(self) -> Optional[str]:
        if self.variant == "provider":
            return None  # ``hold`` carries no target
        # Lenient match policy lives on ``try_mitigate`` — ``correct_mitigation_target``
        # still reports the canonical target for docs / NPC narratives.
        if self.variant == "integration":
            return self.backup_feature_flag
        return self.downstream_service

    @property
    def recovery_window_sec(self) -> int:
        if self.variant == "provider":
            return 300
        return 60

    def try_mitigate(
        self,
        *,
        mitigation: MitigationKind,
        target: Optional[str],
        sim_time_sec: int,
    ) -> bool:
        """Per-variant matcher.

        For the ``integration`` variant we accept either the canonical feature
        flag name or a target starting with the downstream service — LLMs
        tend to reuse the service name as the target slot, and the NPC still
        advertises the full flag name for flavour.
        """
        if self.mitigated:
            return False
        if mitigation != self.correct_mitigation:
            return False

        if self.variant == "provider":
            self.mitigated = True
            self.mitigated_at_sec = sim_time_sec
            return True

        if self.variant == "integration":
            if not target:
                return False
            t = target.strip().lower()
            if t == self.backup_feature_flag or t.startswith(self.downstream_service):
                self.mitigated = True
                self.mitigated_at_sec = sim_time_sec
                return True
            return False

        # our_deploy — standard service-scoped rollback
        if target != self.downstream_service:
            return False
        self.mitigated = True
        self.mitigated_at_sec = sim_time_sec
        return True

    def apply(self, graph: ServiceGraph, sim_time_sec: int) -> None:
        state = graph.get(self.downstream_service)
        if state is None or sim_time_sec < self.fires_at_sec:
            return

        peak_error_rate = self.base_error_rate_floor
        peak_latency = state.spec.baseline_latency_p99_ms * (
            1.8 if self.variant == "our_deploy" else 1.4
        )

        if not self.mitigated:
            state.error_rate = max(state.error_rate, peak_error_rate)
            state.latency_p99_ms = max(state.latency_p99_ms, peak_latency)
            if self.variant == "our_deploy":
                state.has_active_deploy = True
                state.active_deploy_tag = self.deploy_tag
            return

        assert self.mitigated_at_sec is not None
        elapsed = sim_time_sec - self.mitigated_at_sec
        window = self.recovery_window_sec
        if elapsed >= window:
            if self.variant == "our_deploy":
                state.is_rolled_back = True
                state.has_active_deploy = False
                state.active_deploy_tag = None
            return

        decay = 1.0 - (elapsed / window)
        spec = state.spec
        state.error_rate = spec.baseline_error_rate + (
            peak_error_rate - spec.baseline_error_rate
        ) * decay
        state.latency_p99_ms = spec.baseline_latency_p99_ms + (
            peak_latency - spec.baseline_latency_p99_ms
        ) * decay
        if self.variant == "our_deploy":
            state.has_active_deploy = True
            state.active_deploy_tag = self.deploy_tag


@dataclass
class DataCorruptionFault(Fault):
    """Stub: silent post-migration data corruption (hard task).

    Architecture placeholder; full behaviour implemented in the hard-task todo.
    Dashboards stay green by design — detection requires querying audit logs.
    """

    service: str = "orders"
    migration_tag: str = "migration-2026-04-19"

    @property
    def ground_truth_service(self) -> str:
        return self.service

    @property
    def ground_truth_tag(self) -> RootCauseTag:
        return "data_corruption"

    @property
    def correct_mitigation(self) -> MitigationKind:
        return "partial_rollback"

    def apply(self, graph: ServiceGraph, sim_time_sec: int) -> None:
        return


def make_fault(
    kind: str,
    *,
    fires_at_sec: int = 30,
    fault_id: str | None = None,
    variant: Optional[str] = None,
) -> Fault:
    """Factory used by task configs to instantiate the right fault.

    ``variant`` is only meaningful for ``upstream_third_party`` (selects the
    provider / integration / our_deploy sub-scenario). It is silently
    ignored for other fault kinds.
    """
    fid = fault_id or kind
    if kind == "bad_deploy":
        return BadDeployFault(fault_id=fid, fires_at_sec=fires_at_sec)
    if kind == "upstream_third_party":
        chosen: ThirdPartyVariant = _coerce_third_party_variant(variant)
        return ThirdPartyOutageFault(
            fault_id=fid,
            fires_at_sec=fires_at_sec,
            variant=chosen,
        )
    if kind == "data_corruption":
        return DataCorruptionFault(fault_id=fid, fires_at_sec=fires_at_sec)
    raise ValueError(f"unknown fault kind: {kind!r}")


def _coerce_third_party_variant(variant: Optional[str]) -> ThirdPartyVariant:
    if variant is None:
        return "provider"
    if variant not in THIRD_PARTY_VARIANTS:
        raise ValueError(
            f"unknown third-party variant: {variant!r}; expected one of {THIRD_PARTY_VARIANTS}"
        )
    return variant  # type: ignore[return-value]


def pick_third_party_variant(seed: int) -> ThirdPartyVariant:
    """Deterministic mapping from simulator seed to medium-task variant.

    Seeds are equivalence-classed mod 3, so seed=0 / 3 / 6 → provider,
    seed=1 / 4 / 7 → integration, seed=2 / 5 / 8 → our_deploy. This gives
    predictable coverage of all three variants across a 3-seed evaluation run
    while keeping each (task_id, seed) pair fully deterministic.
    """
    return THIRD_PARTY_VARIANTS[seed % len(THIRD_PARTY_VARIANTS)]
