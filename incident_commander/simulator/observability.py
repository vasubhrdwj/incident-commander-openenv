"""
Observability layer — deterministic synthetic telemetry.

Given the current simulator state (clock, service graph, active fault), this
layer produces the alerts, logs, traces, audit events, and external-status
snapshots the IC can inspect.

Determinism contract:
    Every method takes the simulator's seeded :class:`random.Random` and
    advances it in a reproducible, stateless way (i.e. the same inputs produce
    the same outputs). No wall-clock time is read anywhere.

Scope for H4–H12:
    * Alerts + dashboard + logs + traces fully implemented for the canary
      scenario (bad deploy).
    * Audit events and external-status hooks are plumbed end-to-end but kept
      minimal — they only surface non-trivial data for the data-corruption and
      third-party fault subclasses, which are still architectural stubs.
"""

from __future__ import annotations

import random
from typing import Optional

try:
    from ..models import (  # type: ignore[import-not-found]
        Alert,
        AuditEvent,
        ExternalStatusReport,
        LogLine,
        ServiceHealth,
        Span,
    )
except ImportError:
    from models import (  # type: ignore[import-not-found, no-redef]
        Alert,
        AuditEvent,
        ExternalStatusReport,
        LogLine,
        ServiceHealth,
        Span,
    )

from .faults import BadDeployFault, DataCorruptionFault, Fault, ThirdPartyOutageFault
from .service_graph import ServiceGraph

ALERT_LATENCY_BREACH = 1.8
ALERT_ERROR_BREACH = 1.0


class ObservabilityLayer:
    """Deterministic synthetic telemetry for the IC environment.

    The layer is stateless: all state lives on the :class:`ServiceGraph` and
    :class:`Fault`. Each ``gather_*`` call is a pure function of those inputs
    plus ``sim_time_sec`` and an RNG used only for stable cosmetic variation
    (e.g. trace span ids).
    """

    def render_dashboard(self, graph: ServiceGraph) -> dict[str, ServiceHealth]:
        """Build the always-on per-service health snapshot."""
        return {
            name: ServiceHealth(
                service=name,
                latency_p99_ms=round(state.latency_p99_ms, 2),
                error_rate=round(state.error_rate, 4),
                requests_per_sec=round(state.requests_per_sec, 2),
                healthy=state.is_healthy(),
            )
            for name, state in graph.services.items()
        }

    def render_alerts(
        self,
        graph: ServiceGraph,
        fault: Optional[Fault],
        sim_time_sec: int,
    ) -> list[Alert]:
        """Fire alerts whenever a service crosses its SLOs while the fault is active."""
        alerts: list[Alert] = []
        if fault is None or sim_time_sec < fault.fires_at_sec:
            return alerts

        for name, state in graph.services.items():
            spec = state.spec
            latency_ratio = state.latency_p99_ms / max(1e-6, spec.slo_latency_p99_ms)
            error_ratio = state.error_rate / max(1e-6, spec.slo_error_rate)
            if latency_ratio >= ALERT_LATENCY_BREACH:
                alerts.append(
                    Alert(
                        id=f"latency-{name}-{fault.fault_id}",
                        severity="error" if latency_ratio >= 3.0 else "warning",
                        service=name,
                        message=(
                            f"p99 latency {state.latency_p99_ms:.0f}ms breaches SLO "
                            f"({spec.slo_latency_p99_ms:.0f}ms) on {name}"
                        ),
                        fired_at_sec=max(fault.fires_at_sec, sim_time_sec - 30),
                    )
                )
            if error_ratio >= ALERT_ERROR_BREACH:
                alerts.append(
                    Alert(
                        id=f"error-{name}-{fault.fault_id}",
                        severity="critical" if error_ratio >= 4.0 else "error",
                        service=name,
                        message=(
                            f"error rate {state.error_rate:.1%} breaches SLO "
                            f"({spec.slo_error_rate:.1%}) on {name}"
                        ),
                        fired_at_sec=max(fault.fires_at_sec, sim_time_sec - 30),
                    )
                )
        return alerts

    def gather_logs(
        self,
        *,
        graph: ServiceGraph,
        fault: Optional[Fault],
        service: str,
        since_sec: int,
        sim_time_sec: int,
        rng: random.Random,
        limit: int = 12,
    ) -> list[LogLine]:
        """Return a short window of logs for ``service``.

        When a fault is active on ``service`` the window is dominated by error
        lines that hint at the root cause (e.g. deploy tag for bad deploys).
        """
        state = graph.get(service)
        if state is None:
            return []

        window_start = max(0, min(since_sec, sim_time_sec))
        window_end = sim_time_sec
        logs: list[LogLine] = []

        if fault is not None and fault.is_active(sim_time_sec) and fault.ground_truth_service == service:
            if isinstance(fault, BadDeployFault):
                logs.extend(_bad_deploy_logs(fault, service, window_start, window_end, limit, rng))
            elif isinstance(fault, ThirdPartyOutageFault):
                logs.extend(_third_party_logs(fault, service, window_start, window_end, limit, rng))
            elif isinstance(fault, DataCorruptionFault):
                logs.extend(_data_corruption_logs(fault, service, window_start, window_end, rng))

        if len(logs) < limit:
            logs.extend(
                _baseline_logs(
                    service=service,
                    window_start=window_start,
                    window_end=window_end,
                    remaining=limit - len(logs),
                    rng=rng,
                )
            )

        logs.sort(key=lambda line: line.timestamp_sec)
        return logs[:limit]

    def gather_traces(
        self,
        *,
        graph: ServiceGraph,
        fault: Optional[Fault],
        service: Optional[str],
        trace_id: Optional[str],
        sim_time_sec: int,
        rng: random.Random,
        limit: int = 6,
    ) -> list[Span]:
        """Return a small set of trace spans hinting at where latency/errors live."""
        spans: list[Span] = []

        focus_service: Optional[str] = None
        if trace_id:
            focus_service = trace_id.split(":", 1)[0] if ":" in trace_id else None
        focus_service = focus_service or service
        if focus_service is None and fault is not None:
            focus_service = fault.ground_truth_service

        if focus_service is None:
            return []

        state = graph.get(focus_service)
        if state is None:
            return []

        for idx in range(limit):
            tid = trace_id or f"{focus_service}:{_det_hex(rng, 8)}"
            duration = state.latency_p99_ms * (0.6 + 0.1 * idx)
            error = (
                idx == 0
                and fault is not None
                and fault.is_active(sim_time_sec)
                and fault.ground_truth_service == focus_service
            )
            op = _operation_for_service(focus_service)
            spans.append(
                Span(
                    trace_id=tid,
                    service=focus_service,
                    operation=op,
                    duration_ms=round(duration, 1),
                    error=error,
                )
            )
        return spans

    def gather_audit_events(
        self,
        *,
        fault: Optional[Fault],
        service: Optional[str],
        since_sec: int,
        sim_time_sec: int,
        rng: random.Random,
        limit: int = 10,
    ) -> list[AuditEvent]:
        """Return audit events visible to the IC — mostly empty unless a data-corruption fault is active."""
        events: list[AuditEvent] = []
        window_start = max(0, since_sec)

        if fault is not None and isinstance(fault, DataCorruptionFault) and fault.is_active(sim_time_sec):
            # 1. Migration completion event (non-anomalous, anchors the timeline).
            if fault.migration_at_sec >= window_start and fault.migration_at_sec <= sim_time_sec:
                events.append(
                    AuditEvent(
                        timestamp_sec=fault.migration_at_sec,
                        actor=f"deploy:{fault.service}",
                        action="migration.complete",
                        resource=f"{fault.service}/{fault.migration_tag}",
                        anomalous=False,
                    )
                )
            # 2. Anomalous writes clustered after the migration. The audit
            # pipeline flags these because their write pattern (bulk update
            # to a balance column) doesn't match this service's typical
            # workload — that's the only signal there is.
            anchor = fault.migration_at_sec
            for i in range(min(limit - len(events), 4)):
                ts = max(window_start, anchor) + 10 * (i + 1)
                if ts > sim_time_sec:
                    break
                events.append(
                    AuditEvent(
                        timestamp_sec=ts,
                        actor=f"svc:{fault.service}",
                        action="db.write",
                        resource=f"{fault.service}/{fault.migration_tag}/balances",
                        anomalous=True,
                    )
                )
            return events

        for i in range(min(limit, 2)):
            ts = window_start + 5 * i
            if ts > sim_time_sec:
                break
            if service is None:
                actor = "user:baseline"
                resource = "misc/healthcheck"
            else:
                actor = f"svc:{service}"
                resource = f"{service}/healthcheck"
            events.append(
                AuditEvent(
                    timestamp_sec=ts,
                    actor=actor,
                    action="api.read",
                    resource=resource,
                    anomalous=False,
                )
            )
        return events

    def gather_external_status(
        self,
        *,
        fault: Optional[Fault],
        provider: str,
        sim_time_sec: int,
    ) -> ExternalStatusReport:
        """Return a third-party status snapshot for ``provider``.

        Only the ``provider``-variant ``ThirdPartyOutageFault`` reports a
        non-operational status; the ``integration`` and ``our_deploy``
        variants both leave the external page green, which is the intended
        discriminating signal for the medium task.
        """
        if (
            fault is not None
            and isinstance(fault, ThirdPartyOutageFault)
            and fault.provider == provider
            and fault.variant == "provider"
            and fault.is_active(sim_time_sec)
        ):
            return ExternalStatusReport(
                provider=provider,
                status="degraded",
                incident_started_at_sec=fault.fires_at_sec,
                message=(
                    f"{provider} reports elevated error rates on the payment intents API; "
                    f"ETA to resolution: investigating."
                ),
            )
        return ExternalStatusReport(
            provider=provider,
            status="operational",
            incident_started_at_sec=None,
            message=f"{provider} reports all systems normal.",
        )


def _bad_deploy_logs(
    fault: BadDeployFault,
    service: str,
    window_start: int,
    window_end: int,
    limit: int,
    rng: random.Random,
) -> list[LogLine]:
    count = min(limit, 6)
    logs: list[LogLine] = []
    for i in range(count):
        ts = max(window_start, fault.fires_at_sec) + 15 * i
        if ts > window_end:
            break
        code = 503 if i % 2 == 0 else 500
        logs.append(
            LogLine(
                timestamp_sec=ts,
                service=service,
                level="error",
                message=(
                    f"canary={fault.deploy_tag} req_id={_det_hex(rng, 6)} "
                    f"status={code} upstream_err=db.timeout"
                ),
            )
        )
    if window_start <= fault.fires_at_sec <= window_end:
        logs.append(
            LogLine(
                timestamp_sec=fault.fires_at_sec,
                service=service,
                level="info",
                message=f"deploy started release={fault.deploy_tag} channel=canary traffic=5%",
            )
        )
    return logs


def _third_party_logs(
    fault: ThirdPartyOutageFault,
    service: str,
    window_start: int,
    window_end: int,
    limit: int,
    rng: random.Random,
) -> list[LogLine]:
    count = min(limit, 5)
    logs: list[LogLine] = []

    if fault.variant == "provider":
        for i in range(count):
            ts = max(window_start, fault.fires_at_sec) + 20 * i
            if ts > window_end:
                break
            logs.append(
                LogLine(
                    timestamp_sec=ts,
                    service=service,
                    level="error",
                    message=(
                        f"req_id={_det_hex(rng, 6)} provider={fault.provider} "
                        f"status=503 upstream_err=provider.gateway_timeout"
                    ),
                )
            )
    elif fault.variant == "integration":
        patterns = [
            "integration_err=auth.token_refused rotated=false",
            "integration_err=retry_storm attempts=16 backoff=exponential",
            "integration_err=idempotency_key_collision",
            "integration_err=signature_verification_failed",
        ]
        for i in range(count):
            ts = max(window_start, fault.fires_at_sec) + 20 * i
            if ts > window_end:
                break
            logs.append(
                LogLine(
                    timestamp_sec=ts,
                    service=service,
                    level="error",
                    message=(
                        f"req_id={_det_hex(rng, 6)} provider={fault.provider} "
                        f"status=500 {patterns[i % len(patterns)]}"
                    ),
                )
            )
        if window_start <= fault.fires_at_sec <= window_end:
            logs.append(
                LogLine(
                    timestamp_sec=fault.fires_at_sec,
                    service=service,
                    level="info",
                    message=(
                        f"integration_config_drift detected_vs=staging "
                        f"flag_available={fault.backup_feature_flag}"
                    ),
                )
            )
    else:  # our_deploy
        for i in range(count):
            ts = max(window_start, fault.fires_at_sec) + 18 * i
            if ts > window_end:
                break
            logs.append(
                LogLine(
                    timestamp_sec=ts,
                    service=service,
                    level="error",
                    message=(
                        f"release={fault.deploy_tag} req_id={_det_hex(rng, 6)} "
                        f"status=500 internal_err=webhook.handler_regression"
                    ),
                )
            )
        if window_start <= fault.fires_at_sec <= window_end:
            logs.append(
                LogLine(
                    timestamp_sec=fault.fires_at_sec,
                    service=service,
                    level="info",
                    message=(
                        f"deploy started release={fault.deploy_tag} channel=prod traffic=100%"
                    ),
                )
            )

    return logs


def _data_corruption_logs(
    fault: DataCorruptionFault,
    service: str,
    window_start: int,
    window_end: int,
    rng: random.Random,
) -> list[LogLine]:
    """One info-level migration-complete line; nothing else looks wrong.

    The deliberate sparseness is the point: a model that scans logs hoping
    for an error pattern finds nothing actionable. The migration line is
    the one breadcrumb that, combined with the audit cluster, lets the
    agent correlate the timeline.
    """
    logs: list[LogLine] = []
    ts = fault.migration_at_sec
    if window_start <= ts <= window_end:
        logs.append(
            LogLine(
                timestamp_sec=ts,
                service=service,
                level="info",
                message=(
                    f"migration_apply release={fault.migration_tag} status=ok "
                    f"rows_touched=~127k duration_ms=412"
                ),
            )
        )
    return logs


def _baseline_logs(
    *,
    service: str,
    window_start: int,
    window_end: int,
    remaining: int,
    rng: random.Random,
) -> list[LogLine]:
    logs: list[LogLine] = []
    span = max(1, window_end - window_start)
    for i in range(remaining):
        ts = window_start + int(i * span / max(1, remaining))
        logs.append(
            LogLine(
                timestamp_sec=ts,
                service=service,
                level="info",
                message=f"req_id={_det_hex(rng, 6)} status=200 latency_ms={20 + (i % 5) * 4}",
            )
        )
    return logs


def _operation_for_service(service: str) -> str:
    return {
        "api_gw": "http.handle",
        "auth": "auth.validate_token",
        "payments": "payments.charge",
        "orders": "orders.create",
        "inventory": "inventory.reserve",
        "notifications": "notifications.send",
    }.get(service, "generic.handle")


def _det_hex(rng: random.Random, length: int) -> str:
    return "".join(f"{rng.randrange(16):x}" for _ in range(length))
