"""Incident-event decision logic (pure functions, no I/O).

The duty-rota system subscribes to a change feed and pages the operator only
when the incident *state* changes. This module decides, for one finished
diagnosis batch, whether it produces an event:

* Only a **completed** batch that is **not superseded** (no younger batch
  existed when it finished) and whose evidence is **credible**
  (``healthy`` or ``root_cause_found``) may emit an event. A late batch can
  therefore never roll back an already-notified state.
* ``needs_retest`` is not credible evidence: it emits nothing and, in
  particular, never counts as a recovery.
* The notified state is the set of root-cause services. An unchanged set
  emits nothing, so a persistent fault is notified exactly once.

Event types:

* ``fault``               - notified set was empty, now non-empty (incident opened)
* ``root_cause_changed``  - notified set non-empty, changed to a different non-empty set
* ``recovered``           - notified set was non-empty, now empty (incident closed)
"""
from __future__ import annotations

from typing import Any, Sequence

EVENT_TYPES = ("fault", "root_cause_changed", "recovered")

# Outcomes whose evidence is trustworthy enough to notify on. ``needs_retest``
# is deliberately absent: an unsure batch must never open, move, or close an
# incident.
CREDIBLE_OUTCOMES = ("healthy", "root_cause_found")


def root_cause_services(result: dict[str, Any]) -> list[str]:
    """Sorted service names of the batch's root causes (the incident state)."""
    services = {
        str(rc.get("service"))
        for rc in result.get("root_causes") or []
        if isinstance(rc, dict) and rc.get("service")
    }
    return sorted(services)


def decide_event_type(
    notified: Sequence[str], current: Sequence[str]
) -> str | None:
    """Map (previously notified set, current set) to an event type or None."""
    previous_set = sorted(set(notified))
    current_set = sorted(set(current))
    if current_set == previous_set:
        return None
    if not previous_set and current_set:
        return "fault"
    if previous_set and not current_set:
        return "recovered"
    return "root_cause_changed"


def plan_event(
    *,
    status: str,
    superseded: bool,
    result: dict[str, Any] | None,
    notified: Sequence[str],
) -> tuple[dict[str, Any], list[str]] | None:
    """Decide whether a finished batch emits an incident event.

    Returns ``(event_payload, new_notified_state)`` or ``None``. The payload
    carries everything the event log stores except the ``event_id`` cursor and
    ``created_at`` timestamp, which the store assigns at insert time.
    """
    if status != "completed" or superseded or result is None:
        return None
    outcome = result.get("outcome")
    if outcome not in CREDIBLE_OUTCOMES:
        return None

    current = root_cause_services(result) if outcome == "root_cause_found" else []
    event_type = decide_event_type(notified, current)
    if event_type is None:
        return None

    previous = sorted(set(notified))
    state_version = int(result.get("state_version") or 0)
    batch_summary = str(result.get("summary") or "").strip()
    if event_type == "fault":
        summary = (
            f"Incident opened at state version {state_version}: root cause(s) "
            f"{', '.join(current)}."
        )
    elif event_type == "recovered":
        summary = (
            f"Incident closed at state version {state_version}: all services "
            f"healthy again (previous root cause(s): {', '.join(previous)})."
        )
    else:
        summary = (
            f"Incident root cause changed at state version {state_version}: "
            f"{', '.join(previous)} -> {', '.join(current)}."
        )
    if batch_summary:
        summary = f"{summary} {batch_summary}"

    payload = {
        "event_type": event_type,
        "batch_id": result.get("batch_id"),
        "batch_seq": int(result.get("seq") or 0),
        "state_version": state_version,
        "root_causes": list(result.get("root_causes") or []),
        "previous_root_causes": previous,
        "primary_root_cause": result.get("primary_root_cause"),
        "summary": summary,
        "evidence": list(result.get("evidence") or []),
    }
    return payload, current


__all__ = [
    "CREDIBLE_OUTCOMES",
    "EVENT_TYPES",
    "decide_event_type",
    "plan_event",
    "root_cause_services",
]
