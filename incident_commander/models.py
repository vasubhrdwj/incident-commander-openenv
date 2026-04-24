"""
Data models for the Incident Commander OpenEnv environment.

The env simulates an on-call Incident Commander coordinating a microservices
outage: querying observability data, delegating to specialist NPCs, executing
mitigations, handling customer comms, and writing a post-mortem.

This module defines:
    * The rich typed ``ICObservation`` the environment surfaces every step.
    * The flat-union ``ICAction`` the agent submits every step (an ``op`` enum
      selects the action kind; only the relevant typed optional payload fields
      should be populated).
    * The nested value types (``Alert``, ``ServiceHealth``, ...).

Scaffold stage: the simulator itself is implemented incrementally in later
todos; this module only fixes the wire contract.
"""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "warning", "error", "critical"]
LogLevel = Literal["debug", "info", "warning", "error"]
SpecialistRole = Literal["sre", "security", "comms", "eng_lead"]
MitigationKind = Literal[
    "restart",
    "rollback",
    "partial_rollback",
    "scale",
    "feature_flag",
    "hold",
]
CommsChannel = Literal["status_page", "customer_email", "exec_update"]
QueryKind = Literal[
    "query_logs",
    "query_metrics",
    "query_trace",
    "query_audit",
    "query_external_status",
]
OpKind = Literal[
    "query_logs",
    "query_metrics",
    "query_trace",
    "query_audit",
    "query_external_status",
    "delegate",
    "mitigate",
    "communicate",
    "diagnose",
    "resolve",
    "postmortem",
]


class Alert(BaseModel):
    """A single firing alert surfaced on the IC dashboard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable alert identifier.")
    severity: Severity = Field(..., description="Alert severity.")
    service: str = Field(..., description="Service the alert is attributed to.")
    message: str = Field(..., description="Human-readable alert message.")
    fired_at_sec: int = Field(..., description="Simulator time the alert first fired.")


class ServiceHealth(BaseModel):
    """Current health snapshot for one service."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(..., description="Service name (e.g. 'payments').")
    latency_p99_ms: float = Field(..., description="p99 request latency in milliseconds.")
    error_rate: float = Field(..., description="Fraction of requests returning 5xx, in [0, 1].")
    requests_per_sec: float = Field(..., description="Current traffic level.")
    healthy: bool = Field(..., description="Derived health flag against per-service SLOs.")


class LogLine(BaseModel):
    """A synthetic log line returned only when explicitly queried."""

    model_config = ConfigDict(extra="forbid")

    timestamp_sec: int
    service: str
    level: LogLevel
    message: str


class Span(BaseModel):
    """A synthetic trace span returned only when ``query_trace`` is invoked."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    service: str
    operation: str
    duration_ms: float
    error: bool


class AuditEvent(BaseModel):
    """An audit-log event returned only when ``query_audit`` is invoked."""

    model_config = ConfigDict(extra="forbid")

    timestamp_sec: int
    actor: str = Field(..., description="Principal or service account that performed the action.")
    action: str = Field(..., description="Action kind (e.g. 'db.write', 'credential.use').")
    resource: str = Field(..., description="Resource the action targeted.")
    anomalous: bool = Field(default=False, description="Flagged as anomalous by the audit pipeline.")


class ExternalStatusReport(BaseModel):
    """Synthetic third-party provider status (e.g. payment processor)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    status: Literal["operational", "degraded", "major_outage"]
    incident_started_at_sec: Optional[int] = Field(
        default=None,
        description="Simulator time the provider reports the incident started, if any.",
    )
    message: str = Field(default="")


class NPCReport(BaseModel):
    """A report back from a specialist NPC after a ``delegate`` action resolves."""

    model_config = ConfigDict(extra="forbid")

    role: SpecialistRole
    task: str = Field(..., description="Original delegation task string, echoed for traceability.")
    findings: str = Field(..., description="Specialist's deterministic findings.")
    received_at_sec: int


class ChatMessage(BaseModel):
    """A record of a ``communicate`` action the IC emitted."""

    model_config = ConfigDict(extra="forbid")

    timestamp_sec: int
    channel: CommsChannel
    body: str
    cohort: Optional[str] = Field(
        default=None,
        description="Targeted cohort (only meaningful for customer_email).",
    )


class ICAction(Action):
    """Single flat action model; ``op`` selects the kind.

    Per-op expected payload fields (others must be ``None``):

    * ``query_logs`` / ``query_metrics`` / ``query_trace``: ``service`` (required),
      ``since_sec`` (optional); ``trace_id`` (only for ``query_trace``).
    * ``query_audit``: ``since_sec`` (optional), ``service`` (optional filter).
    * ``query_external_status``: ``target`` = provider name (required).
    * ``delegate``: ``role`` and ``task`` (both required).
    * ``mitigate``: ``mitigation`` (required), ``target`` (required for every
      kind except ``hold``).
    * ``communicate``: ``channel`` (required), ``message`` (required), ``cohort``
      (optional, only meaningful for ``customer_email``).
    * ``diagnose``: ``root_cause_service`` and ``root_cause_tag`` (both required).
    * ``resolve``: no payload.
    * ``postmortem``: ``postmortem_json`` (required).
    """

    op: OpKind = Field(..., description="Action kind selector.")

    service: Optional[str] = Field(
        default=None,
        description="Target service name for query_* ops or for service-scoped mitigations.",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace identifier for ``query_trace``.",
    )
    since_sec: Optional[int] = Field(
        default=None,
        description="Lower-bound simulator timestamp (inclusive) for range queries.",
    )

    role: Optional[SpecialistRole] = Field(
        default=None,
        description="Specialist to delegate to.",
    )
    task: Optional[str] = Field(
        default=None,
        description="Delegation task string (what to investigate / produce).",
    )

    mitigation: Optional[MitigationKind] = Field(
        default=None,
        description="Mitigation kind.",
    )
    target: Optional[str] = Field(
        default=None,
        description=(
            "Target for the op: service name for mitigations, provider name for "
            "``query_external_status``, feature-flag key for ``feature_flag``, etc."
        ),
    )

    channel: Optional[CommsChannel] = Field(
        default=None,
        description="Communication channel for ``communicate``.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Message body for ``communicate`` or free-form note for ``delegate``.",
    )
    cohort: Optional[str] = Field(
        default=None,
        description="Affected cohort id for targeted ``customer_email``.",
    )

    root_cause_service: Optional[str] = Field(
        default=None,
        description="Hypothesised root-cause service for ``diagnose``.",
    )
    root_cause_tag: Optional[str] = Field(
        default=None,
        description=(
            "Hypothesised root-cause tag for ``diagnose`` "
            "(e.g. 'bad_deploy', 'upstream_third_party', 'data_corruption')."
        ),
    )

    postmortem_json: Optional[dict] = Field(
        default=None,
        description=(
            "Structured post-mortem JSON required by the terminal ``postmortem`` op. "
            "Schema is checked by the postmortem grader."
        ),
    )


class ICObservation(Observation):
    """Observation surfaced every step of the Incident Commander environment."""

    alerts: list[Alert] = Field(
        default_factory=list,
        description="Currently firing alerts visible on the dashboard.",
    )
    dashboard: dict[str, ServiceHealth] = Field(
        default_factory=dict,
        description="Current per-service health snapshot.",
    )
    log_samples: list[LogLine] = Field(
        default_factory=list,
        description="Log lines surfaced in response to ``query_logs`` (empty otherwise).",
    )
    trace_spans: list[Span] = Field(
        default_factory=list,
        description="Trace spans surfaced in response to ``query_trace`` (empty otherwise).",
    )
    audit_events: list[AuditEvent] = Field(
        default_factory=list,
        description="Audit events surfaced in response to ``query_audit`` (empty otherwise).",
    )
    external_status: Optional[ExternalStatusReport] = Field(
        default=None,
        description="Third-party status snapshot returned by ``query_external_status``.",
    )
    npc_reports: list[NPCReport] = Field(
        default_factory=list,
        description="Reports returned by delegated specialist NPCs since the last reset.",
    )
    chat_feed: list[ChatMessage] = Field(
        default_factory=list,
        description="History of ``communicate`` messages the IC has emitted this episode.",
    )

    sim_time_sec: int = Field(
        default=0,
        description="Simulator time elapsed since reset, in seconds.",
    )
    blast_radius_pct: float = Field(
        default=0.0,
        description="Estimated fraction of users currently affected, in [0, 1].",
    )
    revenue_loss_usd: float = Field(
        default=0.0,
        description="Estimated cumulative revenue impact of the incident in USD.",
    )

    step_budget_remaining: int = Field(
        default=0,
        description="Remaining steps before the episode is force-terminated.",
    )
    last_action_result: str = Field(
        default="",
        description="Short human-readable summary of the most recent action's effect.",
    )
    task_id: str = Field(
        default="",
        description="Identifier of the currently active task (e.g. 'easy_canary_regression').",
    )


ICAction.model_rebuild()
ICObservation.model_rebuild()


__all__ = [
    "Alert",
    "AuditEvent",
    "ChatMessage",
    "CommsChannel",
    "ExternalStatusReport",
    "ICAction",
    "ICObservation",
    "LogLevel",
    "LogLine",
    "MitigationKind",
    "NPCReport",
    "OpKind",
    "QueryKind",
    "ServiceHealth",
    "Severity",
    "Span",
    "SpecialistRole",
]
