"""
Specialist NPCs the IC can delegate to.

Each NPC is a deterministic policy: a pure function of simulator state at the
time the delegation is processed. The IC picks one of four roles — ``sre``,
``security``, ``comms``, ``eng_lead`` — and passes a free-form ``task``
string. The NPC returns an :class:`NPCReport` the simulator appends to the
observation.

Design constraints (matched to the submission checklist):
    * No randomness — NPCs must be reproducible.
    * No LLM calls — they are scripted FSMs, not agents in the RL sense.
      (Dual-mode LLM NPCs are tracked in Section 8 of the plan.)
    * The NPC's ``findings`` must be informative enough that a capable agent
      can turn them into a correct ``diagnose`` / ``mitigate`` action, but not
      so specific that they trivialise the task (e.g. never literally say
      "call rollback on payments" — say "the canary deploy is bad").

For the H4–H12 scaffold, SRE and Eng Lead are fleshed out with canary-aware
findings; Security and Comms ship with useful-but-generic responses that we
specialise in later todos for the medium/hard tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

try:
    from ..models import NPCReport  # type: ignore[import-not-found]
except ImportError:
    from models import NPCReport  # type: ignore[import-not-found, no-redef]

from .faults import BadDeployFault, DataCorruptionFault, Fault, ThirdPartyOutageFault
from .service_graph import ServiceGraph


SpecialistRole = Literal["sre", "security", "comms", "eng_lead"]


class NPCSpecialist(ABC):
    """Base class for a delegated specialist."""

    role: SpecialistRole

    @abstractmethod
    def respond(
        self,
        *,
        task: str,
        fault: Optional[Fault],
        graph: ServiceGraph,
        sim_time_sec: int,
    ) -> NPCReport:
        """Produce a deterministic report for the current simulator state."""
        ...


class SRENPC(NPCSpecialist):
    """Finds service-level signals: recent deploys, error hot spots, dependency hints."""

    role: SpecialistRole = "sre"

    def respond(
        self,
        *,
        task: str,
        fault: Optional[Fault],
        graph: ServiceGraph,
        sim_time_sec: int,
    ) -> NPCReport:
        findings = _sre_findings(fault, graph, sim_time_sec)
        return NPCReport(role=self.role, task=task, findings=findings, received_at_sec=sim_time_sec)


class SecurityNPC(NPCSpecialist):
    """Inspects audit logs and anomalous actor activity."""

    role: SpecialistRole = "security"

    def respond(
        self,
        *,
        task: str,
        fault: Optional[Fault],
        graph: ServiceGraph,
        sim_time_sec: int,
    ) -> NPCReport:
        if fault is None or not fault.is_active(sim_time_sec):
            findings = "Audit feed is clean — no anomalous principals or write spikes observed."
        elif isinstance(fault, DataCorruptionFault):
            findings = (
                f"Audit feed shows an unusual cluster of writes against "
                f"'{fault.service}/orders_migration' shortly after the "
                f"{fault.migration_tag} deploy. No credential abuse — looks "
                f"like a data-layer bug, not an attack."
            )
        else:
            findings = (
                "Audit feed is clean — no anomalous principals or write spikes. "
                "This does not look like a security incident."
            )
        return NPCReport(role=self.role, task=task, findings=findings, received_at_sec=sim_time_sec)


class CommsNPC(NPCSpecialist):
    """Drafts customer-facing copy and flags comms SLA risk."""

    role: SpecialistRole = "comms"

    def respond(
        self,
        *,
        task: str,
        fault: Optional[Fault],
        graph: ServiceGraph,
        sim_time_sec: int,
    ) -> NPCReport:
        # Silent data corruption: dashboards are clean, so the impacted-services
        # heuristic would say "all good" — but the right comms move is a
        # targeted customer email to the affected cohort, not a status-page
        # broadcast that spooks users who weren't impacted.
        if fault is not None and isinstance(fault, DataCorruptionFault):
            findings = (
                f"Do NOT post a status_page update — public broadcast on a "
                f"contained data issue creates unnecessary panic. Right move "
                f"is a targeted customer_email to the affected cohort "
                f"(cohort='affected_accounts' is the cohort tag the data "
                f"team uses post-{fault.migration_tag}). Draft: 'We've "
                f"identified an issue affecting your account balance after "
                f"a recent maintenance window. We're fixing it now and will "
                f"send a confirmation when corrected.'"
            )
            return NPCReport(role=self.role, task=task, findings=findings, received_at_sec=sim_time_sec)

        impacted = _impacted_services(graph)
        if not impacted:
            findings = (
                "No services currently breaching SLO; recommend holding the status page "
                "and keeping an exec pager primed."
            )
        else:
            joined = ", ".join(impacted)
            findings = (
                f"Draft status page: 'We're investigating elevated error rates on "
                f"{joined}. A fix is being deployed; updates every 15 min.' "
                f"Recommend customer_email only if impact persists > 10 min."
            )
        return NPCReport(role=self.role, task=task, findings=findings, received_at_sec=sim_time_sec)


class EngLeadNPC(NPCSpecialist):
    """Sizes blast radius and coordinates recovery plan with the owning team."""

    role: SpecialistRole = "eng_lead"

    def respond(
        self,
        *,
        task: str,
        fault: Optional[Fault],
        graph: ServiceGraph,
        sim_time_sec: int,
    ) -> NPCReport:
        findings = _eng_lead_findings(fault, graph, sim_time_sec)
        return NPCReport(role=self.role, task=task, findings=findings, received_at_sec=sim_time_sec)


def _sre_findings(
    fault: Optional[Fault],
    graph: ServiceGraph,
    sim_time_sec: int,
) -> str:
    if fault is None or not fault.is_active(sim_time_sec):
        return (
            "All services within SLO. Nothing in the last 5 minutes of alerts or logs "
            "suggests an ongoing incident."
        )

    target = graph.get(fault.ground_truth_service)
    if isinstance(fault, BadDeployFault):
        deploy = fault.deploy_tag
        return (
            f"Hot spot on '{fault.ground_truth_service}'. Error rate "
            f"{(target.error_rate if target else 0):.1%}, p99 latency "
            f"{(target.latency_p99_ms if target else 0):.0f}ms. A canary deploy "
            f"({deploy}) landed at t={fault.fires_at_sec}s — timing lines up with "
            "the regression. Canary rollback is the typical move here."
        )
    if isinstance(fault, ThirdPartyOutageFault):
        if fault.variant == "provider":
            return (
                f"Errors on '{fault.downstream_service}' cluster around outbound calls "
                f"to '{fault.provider}'. Their public status page shows a degradation "
                f"starting ~{fault.fires_at_sec}s ago. Our recent deploys and integration "
                f"config both look nominal. Looks upstream — the usual play is to "
                f"hold on mitigations and lean on customer comms."
            )
        if fault.variant == "integration":
            return (
                f"Errors on '{fault.downstream_service}' — provider '{fault.provider}' "
                f"reports operational on its status page, and no recent deploy on "
                f"{fault.downstream_service}. Log pattern smells like integration config "
                f"drift (auth.token_refused / retry_storm). Eng Lead can confirm the "
                f"feature-flag route; usually the backup-processor flag is the fix."
            )
        return (
            f"Errors on '{fault.downstream_service}' — provider '{fault.provider}' is "
            f"fine, but a deploy ({fault.deploy_tag}) landed at t={fault.fires_at_sec}s "
            f"and timing lines up exactly with the regression onset. Webhook handler "
            f"code-path looks new. Rollback is the typical move here."
        )
    if isinstance(fault, DataCorruptionFault):
        return (
            "Dashboards look green across the board, but you flagged a customer report. "
            "Recommend pulling audit logs and correlating against recent migrations."
        )
    return "Unable to localise the issue with current signals."


def _eng_lead_findings(
    fault: Optional[Fault],
    graph: ServiceGraph,
    sim_time_sec: int,
) -> str:
    if fault is None or not fault.is_active(sim_time_sec):
        return "No active incident. Team is on standby."

    if isinstance(fault, BadDeployFault):
        return (
            f"I own '{fault.service}'. The canary ({fault.deploy_tag}) is on the "
            f"rollback branch — rollback will restore within one deploy window (~60s). "
            f"Current blast radius: canary tier only, ~5% of traffic."
        )
    if isinstance(fault, ThirdPartyOutageFault):
        if fault.variant == "provider":
            return (
                f"I own the {fault.provider} integration. Their platform is "
                f"degraded — we don't have a safe way to fail over quickly. "
                f"Recommend `hold` and lean on customer comms until they restore. "
                f"Blast radius grows ~3–5%/min while they're down; drop a status "
                f"page update within 10 min."
            )
        if fault.variant == "integration":
            return (
                f"Our {fault.provider} integration is degraded but the provider "
                f"itself is fine. Toggling the feature flag "
                f"`{fault.backup_feature_flag}` routes traffic to the backup "
                f"processor — recovery is usually ~1 min after the flag flips. "
                f"I'd skip rollback; this was not a deploy."
            )
        return (
            f"I own '{fault.downstream_service}'. The recent release "
            f"({fault.deploy_tag}) is the likely culprit — webhook handler looks "
            f"changed. Rollback the service; recovery typically inside one "
            f"deploy window (~60s). Blast radius was starting to accelerate, "
            f"so don't wait on this."
        )
    if isinstance(fault, DataCorruptionFault):
        return (
            f"Looking at '{fault.service}' post-{fault.migration_tag}. I can scope "
            f"the affected cohort from audit logs — expect 10–30k customer accounts. "
            f"Partial rollback of the migration is the safe play; forward-fix would "
            f"take hours."
        )
    return "Ready to coordinate recovery once the diagnosis lands."


def _impacted_services(graph: ServiceGraph) -> list[str]:
    return [name for name, state in graph.services.items() if not state.is_healthy()]


def build_default_roster() -> dict[str, NPCSpecialist]:
    """Return the default set of four specialists keyed by role."""
    return {
        "sre": SRENPC(),
        "security": SecurityNPC(),
        "comms": CommsNPC(),
        "eng_lead": EngLeadNPC(),
    }
