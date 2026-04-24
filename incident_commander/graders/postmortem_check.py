"""
Programmatic check for the post-mortem JSON field.

The post-mortem component is worth 0.10 of the overall rubric. We keep the
check purely structural + ground-truth-based — no LLM-as-judge — per the
submission checklist's requirement that primary graders be programmatic,
deterministic, and partial-credit.

Expected schema (all fields optional; score is partial credit):

    {
        "summary": "<non-empty one-paragraph executive summary>",
        "root_cause_service": "<service name>",
        "root_cause_tag": "<root-cause tag>",
        "timeline": ["<event 1>", "<event 2>", ...],
        "actions_taken": ["<action 1>", ...],
        "sub_agent_summary": "<brief recap of NPC contributions, optional>"
    }

Scoring (each sub-item worth 0.2 of the 1.0 sub-score, summing to 1.0):

* ``summary`` is a non-empty string longer than 20 chars.
* ``root_cause_service`` matches the fault's ground-truth service.
* ``root_cause_tag`` matches the fault's ground-truth tag.
* ``timeline`` is a list of at least 2 strings.
* ``actions_taken`` is a list of at least 1 string.

The returned sub-score is then multiplied by the rubric weight inside
:class:`RubricGrader`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from ..simulator.faults import Fault


MIN_SUMMARY_LENGTH: int = 20
MIN_TIMELINE_ENTRIES: int = 2
MIN_ACTION_ENTRIES: int = 1


def check_postmortem(fault: "Fault", postmortem_json: Optional[Mapping[str, Any]]) -> float:
    """Return a post-mortem quality sub-score in ``[0.0, 1.0]``.

    ``postmortem_json`` may be ``None`` (agent never submitted one) — in which
    case the sub-score is ``0.0``.
    """
    if not postmortem_json:
        return 0.0

    score: float = 0.0

    summary = postmortem_json.get("summary")
    if isinstance(summary, str) and len(summary.strip()) >= MIN_SUMMARY_LENGTH:
        score += 0.2

    rcs = postmortem_json.get("root_cause_service")
    if isinstance(rcs, str) and rcs == fault.ground_truth_service:
        score += 0.2

    rct = postmortem_json.get("root_cause_tag")
    if isinstance(rct, str) and rct == fault.ground_truth_tag:
        score += 0.2

    timeline = postmortem_json.get("timeline")
    if isinstance(timeline, list) and sum(1 for t in timeline if isinstance(t, str) and t.strip()) >= MIN_TIMELINE_ENTRIES:
        score += 0.2

    actions_taken = postmortem_json.get("actions_taken")
    if isinstance(actions_taken, list) and sum(1 for a in actions_taken if isinstance(a, str) and a.strip()) >= MIN_ACTION_ENTRIES:
        score += 0.2

    return min(1.0, score)
