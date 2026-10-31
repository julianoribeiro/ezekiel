"""Camada centrada em objetos (metodologia, "Estruturação do log centrado em
objetos", "Regras de associação multiobjeto", "Validação do log")."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ezekiel.data.eventlog import JIRA_PROFILE, EventLog  # noqa: E402
from ezekiel.ocel.ocel2 import ObjectCentricLog, synchronization_indicators  # noqa: E402


def _issue(i, key, status_changes, links=(), fix=None, team=None):
    f = {"created": f"2020-01-{i:02d}T00:00:00.000+0000", "issuetype": {"name": "Bug"},
         "project": {"key": "P"}, "status": {"name": "Open"}, "issuelinks": list(links),
         "fixVersions": [fix] if fix else [], "components": [], "subtasks": []}
    if team:
        f["customfield_team"] = {"value": team}
    hist = [{"created": t, "items": [{"field": "status", "fromString": a, "toString": b}]}
            for t, a, b in status_changes]
    return {"id": str(100 + i), "key": key, "fields": f, "changelog": {"histories": hist}}


def test_jira_multiobject_conversion(monkeypatch):
    import build_jira_ocel2 as b
    monkeypatch.setattr(b, "_field_ids", lambda repo: ([], ["customfield_team"]))
    blk = {"type": {"name": "Blocker"}, "inwardIssue": {"key": "P-2"}}
    issues = [
        _issue(1, "P-1", [("2020-01-02T00:00:00.000+0000", "Open", "Resolved")], [blk],
               {"id": "50", "name": "1.0", "released": True, "releaseDate": "2020-01-10"}, team="Query"),
        _issue(2, "P-2", [("2020-01-05T00:00:00.000+0000", "Open", "Resolved")]),
    ]
    ocel_json, meta = b.build(issues, "X")
    log = ObjectCentricLog.from_json(ocel_json, meta)
    assert log.validate()["valid"]
    assert set(log.objects["type"]) == {"WorkItem", "Version", "Team"}
    o2o = set(zip(log.o2o["source_id"], log.o2o["qualifier"], log.o2o["target_id"]))
    assert ("jira-wi-101", "is blocked by", "jira-wi-102") in o2o
    sync = synchronization_indicators(log, {"Resolved"})
    assert sync["items_with_blocking_link"] == 1
    assert sync["share_completed_before_blocker"] == 1.0           # P-1 resolvido antes do bloqueador
    assert sync["wait_to_release_days_median"] == pytest.approx(8.0)
    el = EventLog.from_ocel2(log, JIRA_PROFILE)
    assert len(el.cases) == 2


def test_validation_flags_problems():
    d = {"objectTypes": [{"name": "WorkItem", "attributes": []}], "eventTypes": [],
         "objects": [{"id": "a", "type": "WorkItem", "attributes": [],
                      "relationships": [{"objectId": "zz", "qualifier": "x"}]}],
         "events": [{"id": "e1", "type": "item-created", "time": "2020-01-01T00:00:00Z",
                     "attributes": [], "relationships": [{"objectId": "a", "qualifier": "item"}]}]}
    v = ObjectCentricLog.from_json(d, {"relation_evidence": {"item": "explicita", "x": "inferida"}}).validate()
    assert not v["valid"]
    text = " ".join(v["issues"])
    assert "não declarados" in text and "inexistente" in text and "inferidas no log principal" in text
