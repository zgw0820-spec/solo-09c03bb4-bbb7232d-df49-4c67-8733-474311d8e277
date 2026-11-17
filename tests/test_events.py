"""Incident-event feed: pure decision logic, store atomicity, and end-to-end API."""
from __future__ import annotations

import time

import httpx

from app.events import decide_event_type, plan_event
from app.store import Store
from conftest import reset, set_fault

# ---------------------------------------------------------------------------
# Pure decision logic
# ---------------------------------------------------------------------------


def test_decide_event_type_transitions():
    assert decide_event_type([], []) is None
    assert decide_event_type(["cache"], ["cache"]) is None
    assert decide_event_type(["cache", "auth"], ["auth", "cache"]) is None  # set semantics
    assert decide_event_type([], ["cache"]) == "fault"
    assert decide_event_type(["cache"], []) == "recovered"
    assert decide_event_type(["cache"], ["auth"]) == "root_cause_changed"
    assert decide_event_type(["cache"], ["cache", "gateway"]) == "root_cause_changed"


def _result(outcome: str, roots: list[str], **overrides) -> dict:
    payload = {
        "batch_id": "batch-x",
        "seq": 3,
        "state_version": 7,
        "outcome": outcome,
        "root_causes": [
            {
                "service": name,
                "role": "shared_cache" if name == "cache" else name,
                "failure": "readiness",
                "observation": f"{name} failed",
            }
            for name in roots
        ],
        "primary_root_cause": roots[0] if len(roots) == 1 else None,
        "evidence": [f"{name} /health/ready: HTTP 503" for name in roots] or ["all ok"],
        "summary": "batch summary text",
    }
    payload.update(overrides)
    return payload


def test_plan_event_gates_on_status_superseded_and_credibility():
    result = _result("root_cause_found", ["cache"])
    # Only completed + not superseded + credible outcome may emit.
    assert plan_event(status="running", superseded=False, result=None, notified=[]) is None
    assert plan_event(status="interrupted", superseded=False, result=None, notified=[]) is None
    assert plan_event(status="failed", superseded=False, result=None, notified=[]) is None
    assert plan_event(status="completed", superseded=True, result=result, notified=[]) is None
    assert plan_event(status="completed", superseded=False, result=None, notified=[]) is None
    retest = _result("needs_retest", [])
    assert plan_event(status="completed", superseded=False, result=retest, notified=["cache"]) is None


def test_plan_event_payload_and_dedupe():
    result = _result("root_cause_found", ["cache"], batch_id="b1", seq=3)
    planned = plan_event(status="completed", superseded=False, result=result, notified=[])
    assert planned is not None
    payload, new_notified = planned
    assert new_notified == ["cache"]
    assert payload["event_type"] == "fault"
    assert payload["batch_id"] == "b1"
    assert payload["batch_seq"] == 3
    assert payload["state_version"] == 7
    assert payload["previous_root_causes"] == []
    assert [rc["service"] for rc in payload["root_causes"]] == ["cache"]
    assert payload["primary_root_cause"] == "cache"
    assert payload["evidence"]
    assert "cache" in payload["summary"]

    # Same root set again -> no event (no duplicate notification).
    assert plan_event(
        status="completed", superseded=False, result=result, notified=["cache"]
    ) is None

    # Healthy while an incident is notified -> recovered.
    healthy = _result("healthy", [], batch_id="b2", seq=4, state_version=8)
    planned = plan_event(
        status="completed", superseded=False, result=healthy, notified=["cache"]
    )
    assert planned is not None
    payload, new_notified = planned
    assert payload["event_type"] == "recovered"
    assert payload["root_causes"] == []
    assert payload["previous_root_causes"] == ["cache"]
    assert new_notified == []

    # Healthy while nothing notified -> no event (no spurious recovery).
    assert plan_event(
        status="completed", superseded=False, result=healthy, notified=[]
    ) is None


# ---------------------------------------------------------------------------
# Store: atomic batch+event commit, cursor reads, restart persistence
# ---------------------------------------------------------------------------


def _finish(store: Store, batch_id: str, seq: int, version: int, outcome: str,
            roots: list[str], superseded_by: int | None = None):
    result = _result(outcome, roots, batch_id=batch_id, seq=seq, state_version=version)
    return store.finish_batch(
        batch_id,
        state_version=version,
        status="completed",
        outcome=outcome,
        summary=result["summary"],
        result=result,
        superseded_by=superseded_by,
    )


def test_store_commits_event_with_batch_and_survives_reopen(tmp_path):
    store = Store(str(tmp_path / "events.db"))
    store.create_running_batch("b1", 1, 5)
    event = _finish(store, "b1", 1, 5, "root_cause_found", ["cache"])
    assert event is not None
    assert event["event_id"] == 1
    assert event["event_type"] == "fault"
    assert store.notified_root_causes() == ["cache"]

    # Same root set: batch stored, no event.
    store.create_running_batch("b2", 2, 5)
    assert _finish(store, "b2", 2, 5, "root_cause_found", ["cache"]) is None
    assert store.count_events() == 1

    # needs_retest: no event, notified state untouched (not a recovery).
    store.create_running_batch("b3", 3, 5)
    assert _finish(store, "b3", 3, 5, "needs_retest", []) is None
    assert store.notified_root_causes() == ["cache"]

    # Superseded late batch: no event, cannot roll back the notified state.
    store.create_running_batch("b4", 4, 5)
    assert _finish(store, "b4", 4, 5, "healthy", [], superseded_by=5) is None
    assert store.notified_root_causes() == ["cache"]

    # Genuine recovery from the newest batch.
    store.create_running_batch("b5", 5, 6)
    event = _finish(store, "b5", 5, 6, "healthy", [])
    assert event is not None
    assert event["event_type"] == "recovered"
    assert event["event_id"] == 2
    assert store.notified_root_causes() == []

    # "Restart": a new Store over the same file sees the same feed and state.
    reopened = Store(str(tmp_path / "events.db"))
    assert reopened.notified_root_causes() == []
    assert reopened.latest_event_id() == 2
    page = reopened.list_events(cursor=0, limit=10)
    assert [e["event_type"] for e in page] == ["fault", "recovered"]
    assert [e["event_id"] for e in page] == [1, 2]
    # Cursor resume: nothing new after event 2, and page 1 is stable.
    assert reopened.list_events(cursor=2, limit=10) == []
    assert reopened.list_events(cursor=1, limit=10)[0]["event_type"] == "recovered"
    again = reopened.list_events(cursor=0, limit=10)
    assert again == page  # immutable: a page once returned never changes


# ---------------------------------------------------------------------------
# End-to-end API behaviour
# ---------------------------------------------------------------------------


def diagnose(client: httpx.Client, **params) -> dict:
    response = client.post("/diagnosis", params={"wait": "true", **params})
    assert response.status_code == 201, response.text
    return response.json()


def wait_batch(client: httpx.Client, batch_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/diagnosis/{batch_id}").json()
        if record["status"] != "running":
            return record
        time.sleep(0.05)
    raise AssertionError(f"batch {batch_id} still running after {timeout}s")


def events(client: httpx.Client, cursor: int = 0, limit: int = 50) -> dict:
    response = client.get("/events", params={"cursor": cursor, "limit": limit})
    assert response.status_code == 200, response.text
    return response.json()


def test_fault_event_once_for_persistent_fault(client: httpx.Client):
    reset(client)
    first = diagnose(client)
    assert first["outcome"] == "healthy"
    assert events(client)["events"] == []  # healthy start notifies nothing

    assert set_fault(client, "cache", ready_ok=False).status_code == 200
    batch = diagnose(client)
    assert batch["outcome"] == "root_cause_found"

    feed = events(client)
    assert feed["latest_event_id"] == 1
    assert feed["next_cursor"] == 1
    assert len(feed["events"]) == 1
    event = feed["events"][0]
    assert event["event_id"] == 1
    assert event["event_type"] == "fault"
    assert event["batch_id"] == batch["batch_id"]
    assert event["batch_seq"] == batch["seq"]
    assert event["state_version"] == batch["state_version"]
    assert [rc["service"] for rc in event["root_causes"]] == ["cache"]
    assert event["previous_root_causes"] == []
    assert event["notified_root_causes"] == ["cache"]
    assert event["primary_root_cause"] == "cache"
    assert event["evidence"]  # evidence summary carried over
    assert event["created_at"]

    state = client.get("/state").json()
    assert state["notified_root_causes"] == ["cache"]
    assert state["events_emitted"] == 1

    # Same fault, more diagnoses: the root-cause set is unchanged, so the
    # duty officer is not paged again.
    diagnose(client)
    diagnose(client)
    assert events(client)["events"] == [event]
    assert client.get("/state").json()["events_emitted"] == 1


def test_root_cause_change_and_recovery_sequence(client: httpx.Client):
    reset(client)
    diagnose(client)

    assert set_fault(client, "cache", ready_ok=False).status_code == 200
    diagnose(client)  # event 1: fault [cache]
    reset(client)
    diagnose(client)  # event 2: recovered
    assert set_fault(client, "auth", ready_ok=False).status_code == 200
    diagnose(client)  # event 3: fault [auth]
    assert set_fault(client, "order", live_ok=False).status_code == 200
    diagnose(client)  # event 4: root_cause_changed [auth, order]
    reset(client)
    diagnose(client)  # event 5: recovered

    feed = events(client)
    assert [e["event_id"] for e in feed["events"]] == [1, 2, 3, 4, 5]
    assert [e["event_type"] for e in feed["events"]] == [
        "fault",
        "recovered",
        "fault",
        "root_cause_changed",
        "recovered",
    ]
    notified = [e["notified_root_causes"] for e in feed["events"]]
    assert notified == [["cache"], [], ["auth"], ["auth", "order"], []]
    change = feed["events"][3]
    assert change["previous_root_causes"] == ["auth"]
    assert sorted(rc["service"] for rc in change["root_causes"]) == ["auth", "order"]
    assert feed["events"][4]["previous_root_causes"] == ["auth", "order"]
    assert client.get("/state").json()["notified_root_causes"] == []


def test_needs_retest_is_not_a_recovery(client: httpx.Client):
    reset(client)
    diagnose(client)
    assert set_fault(client, "cache", ready_ok=False).status_code == 200
    diagnose(client)  # fault [cache]
    assert events(client)["events"][-1]["event_type"] == "fault"

    # Make the cache too slow to prove anything: the batch loses confidence.
    assert set_fault(client, "cache", probe_delay_ms=3500).status_code == 200
    unsure = diagnose(client, wait_seconds=20)
    assert unsure["outcome"] == "needs_retest"

    # No event, and the incident is still considered open.
    assert len(events(client)["events"]) == 1
    assert client.get("/state").json()["notified_root_causes"] == ["cache"]

    reset(client)
    diagnose(client)  # healthy again -> recovered
    feed = events(client)
    assert [e["event_type"] for e in feed["events"]] == ["fault", "recovered"]


def test_superseded_late_batch_cannot_emit_or_roll_back(client: httpx.Client):
    reset(client)
    diagnose(client)
    # Slow the cache so the first batch is still probing when the second starts.
    assert set_fault(client, "cache", ready_ok=False, probe_delay_ms=600).status_code == 200

    first = client.post("/diagnosis", params={"wait": "false"}).json()
    time.sleep(0.15)
    second = client.post("/diagnosis", params={"wait": "false"}).json()
    first_record = wait_batch(client, first["batch_id"])
    second_record = wait_batch(client, second["batch_id"])
    assert first_record["superseded"] is True
    assert second_record["status"] == "completed"

    # Exactly one event, sourced from the newest batch; the late batch's
    # identical verdict neither re-notifies nor rolls anything back.
    feed = events(client)
    assert len(feed["events"]) == 1
    event = feed["events"][0]
    assert event["event_type"] == "fault"
    assert event["batch_id"] == second["batch_id"]
    assert event["batch_seq"] == second["seq"]
    assert [rc["service"] for rc in event["root_causes"]] == ["cache"]


def test_concurrent_diagnoses_emit_single_fault_event(client: httpx.Client):
    reset(client)
    diagnose(client)
    assert set_fault(client, "cache", ready_ok=False).status_code == 200

    accepted = [
        client.post("/diagnosis", params={"wait": "false"}).json() for _ in range(3)
    ]
    records = [wait_batch(client, item["batch_id"]) for item in accepted]
    assert all(record["status"] == "completed" for record in records)

    feed = events(client)
    assert len(feed["events"]) == 1
    assert feed["events"][0]["event_type"] == "fault"
    assert [rc["service"] for rc in feed["events"][0]["root_causes"]] == ["cache"]


def test_cursor_pagination_and_immutability(client: httpx.Client):
    reset(client)
    diagnose(client)
    for service in ("cache", "auth"):
        assert set_fault(client, service, ready_ok=False).status_code == 200
        diagnose(client)
        reset(client)
        diagnose(client)
    assert events(client)["latest_event_id"] == 4

    # Page through the feed one event at a time: no gaps, no duplicates.
    seen: list[dict] = []
    cursor = 0
    while True:
        page = events(client, cursor=cursor, limit=1)
        if not page["events"]:
            break
        assert len(page["events"]) == 1
        seen.append(page["events"][0])
        assert page["next_cursor"] == page["events"][0]["event_id"]
        cursor = page["next_cursor"]
    assert [e["event_id"] for e in seen] == [1, 2, 3, 4]
    assert [e["event_type"] for e in seen] == ["fault", "recovered", "fault", "recovered"]

    # Caught-up subscribers get an empty page whose cursor does not move.
    tail = events(client, cursor=4)
    assert tail["events"] == []
    assert tail["next_cursor"] == 4
    assert tail["latest_event_id"] == 4

    # Re-reading from the start returns byte-identical events.
    assert events(client, cursor=0, limit=50)["events"] == seen


def test_events_survive_restart_and_cursor_resumes(client: httpx.Client, stack):
    reset(client)
    diagnose(client)
    assert set_fault(client, "cache", ready_ok=False).status_code == 200
    diagnose(client)  # fault event, id 1
    assert events(client)["latest_event_id"] == 1

    stack.restart_api(hard=True)
    with httpx.Client(base_url=stack.api_url, timeout=30.0) as api:
        # The feed and the notified state survived the restart.
        feed = events(api)
        assert [e["event_type"] for e in feed["events"]] == ["fault"]
        assert api.get("/state").json()["notified_root_causes"] == ["cache"]

        # The fault is still injected; re-diagnosing must not re-notify.
        diagnose(api)
        assert events(api)["latest_event_id"] == 1

        reset(api)
        diagnose(api)  # recovery -> event id 2
        feed = events(api, cursor=1)  # resume from the last seen cursor
        assert [e["event_type"] for e in feed["events"]] == ["recovered"]
        assert feed["events"][0]["event_id"] == 2
        assert feed["next_cursor"] == 2 == feed["latest_event_id"]
        assert events(api, cursor=2)["events"] == []


def test_events_endpoint_in_openapi(client: httpx.Client):
    schema = client.get("/openapi.json").json()
    assert "/events" in schema["paths"]
    get_op = schema["paths"]["/events"]["get"]
    assert "events" in get_op["tags"]
    param_names = {p["name"] for p in get_op["parameters"]}
    assert {"cursor", "limit"} <= param_names
    for model in ("IncidentEvent", "EventList"):
        assert model in schema["components"]["schemas"]
