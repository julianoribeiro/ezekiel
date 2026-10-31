"""Descoberta de processos e conformidade (metodologia, "Caracterização empírica")."""

import pandas as pd

from ezekiel.data.eventlog import EventLog
from ezekiel.mining.discovery import END_ACTIVITY, discover, traces


def _log() -> EventLog:
    rows = []
    for i in range(40):
        t = float(i)
        path = ["Backlog", "Development", "Code Review", "Deploy"] if i % 4 else ["Backlog", "Development", "Deploy"]
        for k, st in enumerate(path):
            rows.append((f"I{i}", st, t + k))
    log = EventLog.from_transitions(pd.DataFrame(rows, columns=["item", "stage", "t"]))
    log.cases["item_type"] = ["Bug" if i % 2 else "Task" for i in range(len(log.cases))]
    return log


def test_traces_end_with_completion():
    ev = traces(_log(), 0.0, 100.0)
    last = ev.groupby("case")["activity"].last()
    assert (last == END_ACTIVITY).all() and ev["case"].nunique() == 40


def test_discover_reports_models_and_conformance():
    r = discover(_log(), 0.0, 100.0, ["Bug", "Task"])
    assert r["status"] == "ok" and r["variants"] == 2
    assert r["models"]["inductive"]["fitness_log"] > 0.99
    assert r["models"]["split_miner"]["status"].startswith("não executado")
    assert set(r["conformance"]["fitness_by_class"]) == {"Bug", "Task"}
