"""Grafo de restrições por evidências (metodologia, "Grafo de restrições")."""

import pandas as pd
import pytest

from ezekiel.data.eventlog import DONE, EventLog
from ezekiel.dcop.dependencies import DependencySettings, candidate_dependencies, interaction_strength
from ezekiel.dcop.organization import AgentSpec, Organization
from ezekiel.ocel.ocel2 import ObjectCentricLog
from ezekiel.simulation.engine import SimModel, StageModel


def test_interaction_zero_for_additive_costs():
    A, B = ["a0", "a1"], ["b0", "b1"]
    additive = {(x, y): i + 2 * j for i, x in enumerate(A) for j, y in enumerate(B)}
    assert interaction_strength(additive, "a0", "b0") == pytest.approx(0.0)
    joint = dict(additive)
    joint[("a1", "b1")] += 0.5
    assert interaction_strength(joint, "a0", "b0") == pytest.approx(0.5)


def _ocel_and_log():
    # dois itens: X (bloqueador) em Development enquanto Y (bloqueado) está em QA
    ev, e2o = [], []
    def add(i, item, t, status, actor):
        ev.append({"id": f"e{i}", "type": "item-status-changed" if i % 10 else "item-created",
                   "time": f"2020-01-{t:02d}T00:00:00Z",
                   "attributes": [{"name": "toStatus", "value": status}, {"name": "actor", "value": actor}],
                   "relationships": [{"objectId": item, "qualifier": "item"}]})
    add(10, "X", 1, "Open", "p1"); add(11, "X", 2, "In Progress", "p1"); add(12, "X", 9, "Patch Available", "p1")
    add(13, "X", 12, "Closed", "p2")
    add(20, "Y", 1, "Open", "p3"); add(21, "Y", 3, "Patch Available", "p2"); add(22, "Y", 8, "Closed", "p2")
    d = {"objectTypes": [{"name": "WorkItem", "attributes": []}],
         "eventTypes": [{"name": "item-created", "attributes": []}, {"name": "item-status-changed", "attributes": []}],
         "objects": [{"id": "X", "type": "WorkItem", "attributes": [], "relationships": []},
                     {"id": "Y", "type": "WorkItem", "attributes": [],
                      "relationships": [{"objectId": "X", "qualifier": "is blocked by"}]}],
         "events": ev}
    o = ObjectCentricLog.from_json(d, {"relation_evidence": {"item": "explicita", "is blocked by": "explicita"}})
    return EventLog.from_ocel2(o)


def test_candidate_evidence_from_objects_and_people():
    log = _ocel_and_log()
    st = [StageModel("Backlog", None, [1.0], [0.0]), StageModel("Development", 1, [1.0], [0.0]),
          StageModel("Code Review", 1, [1.0], [0.0])]
    m = SimModel(st, 1.0, {"Backlog": 1.0}, {"Backlog": {"Development": 1.0},
                                             "Development": {"Code Review": 1.0}, "Code Review": {DONE: 1.0}})
    org = Organization("t", "", [AgentSpec("Dev", ("Development",), "", ()),
                                 AgentSpec("Rev", ("Code Review",), "", ())]).bind(m)
    s = DependencySettings(min_sync_pairs=1, min_resource_jaccard=0.1)
    t0 = log.t_min - 1
    row = candidate_dependencies(log, org, t0, t0 + 60, s)[0]
    assert row["evidence"]["synchronization"]["pairs"] == 1     # Y em revisão enquanto X em desenvolvimento
    # p1 tira itens de Development; p2 tira itens de Code Review: ninguém atua nos dois
    assert row["evidence"]["shared_resource"]["shared"] == 0
    assert {"fluxo", "sincronização"} <= set(row["sources_passed"])
    assert row["candidate"]
