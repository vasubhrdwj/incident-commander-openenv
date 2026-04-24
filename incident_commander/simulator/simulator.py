"""
Top-level Incident Commander simulator.

The :class:`Simulator` owns every deterministic piece of the world:
    * a monotonic sim clock (``SECONDS_PER_STEP`` per env step),
    * the service graph + its mutable per-service health,
    * the active :class:`~.faults.Fault` for the current task,
    * the observability layer,
    * the four specialist NPCs,
    * a running chat feed and NPC-report log the IC accumulates.

Every step:

1. advance the clock,
2. reset the graph to baseline and re-apply the fault for the new sim time,
3. dispatch the agent's action,
4. render an :class:`ICObservation` from the new state.

All randomness flows through a single seeded :class:`random.Random` instance
so that replaying an identical action sequence produces identical
observations — a disqualifying requirement per the submission checklist.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from ..models import (  # type: ignore[import-not-found]
        Alert,
        AuditEvent,
        ChatMessage,
        ExternalStatusReport,
        ICAction,
        ICObservation,
        LogLine,
        NPCReport,
        ServiceHealth,
        Span,
    )
except ImportError:
    from models import (  # type: ignore[import-not-found, no-redef]
        Alert,
        AuditEvent,
        ChatMessage,
        ExternalStatusReport,
        ICAction,
        ICObservation,
        LogLine,
        NPCReport,
        ServiceHealth,
        Span,
    )

from .faults import Fault, make_fault, pick_third_party_variant
from .npcs import NPCSpecialist, build_default_roster
from .observability import ObservabilityLayer
from .service_graph import ServiceGraph, build_default_topology


SECONDS_PER_STEP: int = 30
MAX_BLAST_RADIUS_BAD_DEPLOY: float = 0.05
MAX_BLAST_RADIUS_THIRD_PARTY: float = 0.20
MAX_BLAST_RADIUS_DATA_CORRUPTION: float = 0.15
REVENUE_LOSS_USD_PER_SEC: float = 250.0
BLAST_GROWTH_HORIZON_SEC: int = 600


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration for one task scenario."""

    task_id: str
    description: str
    fault_kind: str
    fault_fires_at_sec: int = 30
    max_blast_radius: float = MAX_BLAST_RADIUS_BAD_DEPLOY
    step_budget: int = 80
    seed: int = 0
    target_mttr_sec: int = 300
    comms_sla_sec: int = 600

    def build_fault(self, seed: int = 0) -> Fault:
        """Build the task's active fault. ``seed`` selects sub-variants where applicable.

        For ``upstream_third_party`` (medium task), the seed picks one of three
        variants (provider / integration / our_deploy). For other fault kinds
        the seed is ignored.
        """
        if self.fault_kind == "upstream_third_party":
            variant = pick_third_party_variant(seed)
            return make_fault(
                self.fault_kind,
                fires_at_sec=self.fault_fires_at_sec,
                variant=variant,
            )
        return make_fault(self.fault_kind, fires_at_sec=self.fault_fires_at_sec)


KNOWN_TASKS: dict[str, TaskConfig] = {
    "easy_canary_regression": TaskConfig(
        task_id="easy_canary_regression",
        description=(
            "A fresh payments canary at ~5% traffic shows 2x the control group's "
            "error rate. Goal: confirm the regression, rollback the canary, post a "
            "short status-page update, and write the post-mortem."
        ),
        fault_kind="bad_deploy",
        fault_fires_at_sec=30,
        max_blast_radius=MAX_BLAST_RADIUS_BAD_DEPLOY,
        step_budget=80,
    ),
    "medium_third_party_attribution": TaskConfig(
        task_id="medium_third_party_attribution",
        description=(
            "Payment-webhook ingestion is failing. Three scenario variants selected "
            "deterministically by seed: provider-side (Stripe degraded → `hold` + "
            "communicate); our integration (config drift / retry storm → `feature_flag` "
            "to the backup processor); our recent deploy (`rollback` payments). The IC "
            "must distinguish via `query_external_status`, `query_logs`, and SRE / Eng "
            "Lead delegation."
        ),
        fault_kind="upstream_third_party",
        fault_fires_at_sec=30,
        max_blast_radius=MAX_BLAST_RADIUS_THIRD_PARTY,
        step_budget=80,
        target_mttr_sec=240,
        comms_sla_sec=600,
    ),
    "hard_silent_data_corruption": TaskConfig(
        task_id="hard_silent_data_corruption",
        description=(
            "A customer reports wrong account balances after a recent migration. "
            "No alerts firing. Detection requires querying audit logs."
        ),
        fault_kind="data_corruption",
        fault_fires_at_sec=10,
        max_blast_radius=MAX_BLAST_RADIUS_DATA_CORRUPTION,
        step_budget=80,
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    """Return the static :class:`TaskConfig` for ``task_id`` (raises ``KeyError`` if unknown)."""
    if task_id not in KNOWN_TASKS:
        raise KeyError(
            f"unknown task_id {task_id!r}; expected one of: {sorted(KNOWN_TASKS)}"
        )
    return KNOWN_TASKS[task_id]


@dataclass
class _StepResult:
    """Internal container for per-step query output + status text."""

    status: str
    log_samples: list[LogLine] = field(default_factory=list)
    trace_spans: list[Span] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)
    external_status: Optional[ExternalStatusReport] = None
    new_npc_report: Optional[NPCReport] = None
    new_chat_message: Optional[ChatMessage] = None
    diagnose_score: Optional[float] = None
    mitigation_matched: Optional[bool] = None
    ended_by_postmortem: bool = False
    ended_by_resolve: bool = False


class Simulator:
    """Deterministic Incident Commander world simulator."""

    def __init__(
        self,
        *,
        task: TaskConfig,
        seed: Optional[int] = None,
        graph_factory: Callable[[], ServiceGraph] = build_default_topology,
    ) -> None:
        self.task = task
        self.seed = task.seed if seed is None else seed
        self._graph_factory = graph_factory

        self._rng: random.Random = random.Random(self.seed)
        self.graph: ServiceGraph = graph_factory()
        self.fault: Fault = task.build_fault(self.seed)
        self.observability: ObservabilityLayer = ObservabilityLayer()
        self.npcs: dict[str, NPCSpecialist] = build_default_roster()

        self.sim_time_sec: int = 0
        self.step_count: int = 0
        self.step_budget: int = task.step_budget

        self.npc_reports: list[NPCReport] = []
        self.chat_feed: list[ChatMessage] = []
        self.max_blast_radius_pct: float = 0.0
        self.revenue_loss_usd: float = 0.0

        self.resolved: bool = False
        self.postmortem_submitted: bool = False
        self.postmortem_json: Optional[dict] = None
        self.diagnose_history: list[tuple[str, str, float]] = []
        self.mitigation_history: list[tuple[str, Optional[str], bool]] = []

    def reset(self) -> None:
        """Reset simulator state while preserving task config."""
        self._rng = random.Random(self.seed)
        self.graph = self._graph_factory()
        self.fault = self.task.build_fault(self.seed)
        self.sim_time_sec = 0
        self.step_count = 0
        self.npc_reports = []
        self.chat_feed = []
        self.max_blast_radius_pct = 0.0
        self.revenue_loss_usd = 0.0
        self.resolved = False
        self.postmortem_submitted = False
        self.postmortem_json = None
        self.diagnose_history = []
        self.mitigation_history = []
        self._refresh_world(advance_clock=False)

    def step(self, action: ICAction) -> tuple[ICObservation, bool]:
        """Apply ``action``, advance time, return ``(observation, done)``."""
        self.step_count += 1
        self.sim_time_sec += SECONDS_PER_STEP
        self._refresh_world(advance_clock=True)

        step = self._dispatch(action)

        done = self._compute_done(step)

        obs = self._render_observation(
            status=step.status,
            query_logs=step.log_samples,
            query_traces=step.trace_spans,
            query_audit=step.audit_events,
            query_external=step.external_status,
            done=done,
        )
        return obs, done

    def initial_observation(self) -> ICObservation:
        """Return the observation the env serves from ``reset()`` (before any action)."""
        self._refresh_world(advance_clock=False)
        return self._render_observation(
            status=f"Ready. Task: {self.task.task_id}.",
            query_logs=[],
            query_traces=[],
            query_audit=[],
            query_external=None,
            done=False,
        )

    def advance_with_noop(self, *, status: str) -> ICObservation:
        """Advance the clock by one step without dispatching any action.

        Used by the env when an invalid-payload action is rejected: the agent
        still burns one step of sim time + step budget, so invalid ops carry
        an opportunity cost.
        """
        self.step_count += 1
        self.sim_time_sec += SECONDS_PER_STEP
        self._refresh_world(advance_clock=True)
        done = self.step_count >= self.step_budget
        return self._render_observation(
            status=status,
            query_logs=[],
            query_traces=[],
            query_audit=[],
            query_external=None,
            done=done,
        )

    def _refresh_world(self, *, advance_clock: bool) -> None:
        """Recompute the service graph for the current ``sim_time_sec``."""
        self.graph.reset_all()
        self.fault.apply(self.graph, self.sim_time_sec)
        self._accumulate_blast_radius()
        if advance_clock:
            self._accumulate_revenue_loss()

    def _accumulate_blast_radius(self) -> None:
        if not self.fault.is_active(self.sim_time_sec):
            return
        elapsed = max(0, self.sim_time_sec - self.fault.fires_at_sec)
        growth = min(1.0, elapsed / BLAST_GROWTH_HORIZON_SEC)
        current = growth * self.task.max_blast_radius
        if current > self.max_blast_radius_pct:
            self.max_blast_radius_pct = current

    def _accumulate_revenue_loss(self) -> None:
        if not self.fault.is_active(self.sim_time_sec):
            return
        delta = REVENUE_LOSS_USD_PER_SEC * SECONDS_PER_STEP * self.max_blast_radius_pct
        self.revenue_loss_usd += delta

    def _dispatch(self, action: ICAction) -> _StepResult:
        op = action.op
        if op == "query_logs":
            service = action.service or ""
            logs = self.observability.gather_logs(
                graph=self.graph,
                fault=self.fault,
                service=service,
                since_sec=action.since_sec or 0,
                sim_time_sec=self.sim_time_sec,
                rng=self._rng,
            )
            return _StepResult(
                status=f"fetched {len(logs)} log lines for service={service}",
                log_samples=logs,
            )
        if op == "query_metrics":
            service = action.service or ""
            state = self.graph.get(service)
            if state is None:
                return _StepResult(status=f"unknown service '{service}'")
            return _StepResult(
                status=(
                    f"metrics[{service}] latency_p99={state.latency_p99_ms:.0f}ms "
                    f"error_rate={state.error_rate:.2%} rps={state.requests_per_sec:.0f} "
                    f"healthy={state.is_healthy()}"
                )
            )
        if op == "query_trace":
            traces = self.observability.gather_traces(
                graph=self.graph,
                fault=self.fault,
                service=action.service,
                trace_id=action.trace_id,
                sim_time_sec=self.sim_time_sec,
                rng=self._rng,
            )
            return _StepResult(
                status=f"fetched {len(traces)} trace spans",
                trace_spans=traces,
            )
        if op == "query_audit":
            events = self.observability.gather_audit_events(
                fault=self.fault,
                service=action.service,
                since_sec=action.since_sec or 0,
                sim_time_sec=self.sim_time_sec,
                rng=self._rng,
            )
            return _StepResult(
                status=f"fetched {len(events)} audit events",
                audit_events=events,
            )
        if op == "query_external_status":
            provider = action.target or ""
            if not provider:
                return _StepResult(status="missing provider target")
            report = self.observability.gather_external_status(
                fault=self.fault,
                provider=provider,
                sim_time_sec=self.sim_time_sec,
            )
            return _StepResult(
                status=f"external_status[{provider}]: {report.status}",
                external_status=report,
            )
        if op == "delegate":
            role = action.role or ""
            task = action.task or ""
            npc = self.npcs.get(role)
            if npc is None:
                return _StepResult(status=f"unknown specialist role '{role}'")
            report = npc.respond(
                task=task,
                fault=self.fault,
                graph=self.graph,
                sim_time_sec=self.sim_time_sec,
            )
            self.npc_reports.append(report)
            return _StepResult(
                status=f"delegated to {role}: report received ({len(report.findings)} chars)",
                new_npc_report=report,
            )
        if op == "mitigate":
            mitigation = action.mitigation
            target = action.target
            if mitigation is None:
                return _StepResult(status="missing mitigation kind")
            matched = self.fault.try_mitigate(
                mitigation=mitigation,
                target=target,
                sim_time_sec=self.sim_time_sec,
            )
            self.mitigation_history.append((mitigation, target, matched))
            status = (
                f"mitigation applied: {mitigation} target={target or '-'} "
                f"({'fix' if matched else 'no-op'})"
            )
            return _StepResult(status=status, mitigation_matched=matched)
        if op == "communicate":
            channel = action.channel or ""
            body = action.message or ""
            cohort = action.cohort
            chat = ChatMessage(
                timestamp_sec=self.sim_time_sec,
                channel=channel,  # type: ignore[arg-type]
                body=body,
                cohort=cohort,
            )
            self.chat_feed.append(chat)
            return _StepResult(
                status=f"comms sent on {channel} ({len(body)} chars)",
                new_chat_message=chat,
            )
        if op == "diagnose":
            rcs = action.root_cause_service or ""
            rct = action.root_cause_tag or ""
            service_match = rcs == self.fault.ground_truth_service
            tag_match = rct == self.fault.ground_truth_tag
            if service_match and tag_match:
                score = 1.0
                verdict = "matches ground truth"
            elif service_match:
                score = 0.5
                verdict = "service matches, tag is wrong"
            else:
                score = 0.0
                verdict = "no match"
            self.diagnose_history.append((rcs, rct, score))
            status = f"diagnose recorded: service={rcs} tag={rct} ({verdict})"
            return _StepResult(status=status, diagnose_score=score)
        if op == "resolve":
            self.resolved = True
            return _StepResult(
                status="incident marked resolved",
                ended_by_resolve=True,
            )
        if op == "postmortem":
            self.postmortem_submitted = True
            self.postmortem_json = action.postmortem_json
            return _StepResult(
                status="post-mortem submitted",
                ended_by_postmortem=True,
            )
        return _StepResult(status=f"unknown op '{op}'")

    def _compute_done(self, step: _StepResult) -> bool:
        if step.ended_by_postmortem:
            return True
        if self.step_count >= self.step_budget:
            return True
        return False

    def _render_observation(
        self,
        *,
        status: str,
        query_logs: list[LogLine],
        query_traces: list[Span],
        query_audit: list[AuditEvent],
        query_external: Optional[ExternalStatusReport],
        done: bool,
    ) -> ICObservation:
        dashboard: dict[str, ServiceHealth] = self.observability.render_dashboard(self.graph)
        alerts: list[Alert] = self.observability.render_alerts(
            self.graph, self.fault, self.sim_time_sec
        )
        budget_remaining = max(0, self.step_budget - self.step_count)
        return ICObservation(
            alerts=alerts,
            dashboard=dashboard,
            log_samples=query_logs,
            trace_spans=query_traces,
            audit_events=query_audit,
            external_status=query_external,
            npc_reports=list(self.npc_reports),
            chat_feed=list(self.chat_feed),
            sim_time_sec=self.sim_time_sec,
            blast_radius_pct=round(self.max_blast_radius_pct, 4),
            revenue_loss_usd=round(self.revenue_loss_usd, 2),
            step_budget_remaining=budget_remaining,
            last_action_result=status,
            task_id=self.task.task_id,
            done=done,
            reward=0.0,
        )
