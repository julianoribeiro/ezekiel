"""Verificação do log de casos e do simulador (metodologia, "Verificação e
validação": conservação de entidades, casos extremos, pequenos exemplos com
cálculo manual)."""

import numpy as np
import pandas as pd
import pytest

from ezekiel.data.eventlog import DONE, EventLog
from ezekiel.simulation.calibration import kaplan_meier
from ezekiel.simulation.engine import SimModel, StageModel, run_replica


def _log() -> EventLog:
    rows = [
        # item A: Backlog → Dev → Deploy, reaberto → Backlog → Deploy
        ("A", "Backlog", 0.0), ("A", "Development", 2.0), ("A", "Deploy", 5.0),
        ("A", "Backlog", 10.0), ("A", "Deploy", 12.0),
        # item B: ainda aberto em Development
        ("B", "Backlog", 1.0), ("B", "Backlog", 1.5), ("B", "Development", 3.0),
    ]
    return EventLog.from_transitions(pd.DataFrame(rows, columns=["item", "stage", "t"]))


def test_cases_split_on_reopen_and_censoring():
    log = _log()
    c = log.cases.set_index("case")
    assert set(c.index) == {"A#0", "A#1", "B#0"}
    assert c.loc["A#0", "end"] - c.loc["A#0", "start"] == 5.0
    assert c.loc["A#1", "end"] - c.loc["A#1", "start"] == 2.0
    assert np.isnan(c.loc["B#0", "end"])               # censurado, nunca concluído
    v = log.visits.set_index(["case", "stage"])
    assert v.loc[("A#0", "Development"), "next"] == DONE
    assert v.loc[("B#0", "Backlog"), "exit"] == 3.0   # repetição consecutiva fundida


def test_state_at_reconstructs_ages():
    st = _log().state_at(4.0).set_index("case")
    assert st.loc["A#0", "stage"] == "Development" and st.loc["A#0", "stage_age"] == 2.0
    assert st.loc["B#0", "case_age"] == 3.0


def test_kaplan_meier_with_censoring():
    # 4 visitas: saídas em 1 e 2; duas censuradas (abertas) — S cai para 0,75 e depois 0,5.
    enter = np.array([0.0, 0.0, 0.0, 0.0])
    exit_ = np.array([1.0, 2.0, np.nan, np.nan])
    times, surv = kaplan_meier(enter, exit_, 0.0, 10.0)
    assert times.tolist() == [1.0, 2.0]
    assert surv.tolist() == pytest.approx([0.75, 0.5])


def _model(wip=None, policy="FIFO") -> SimModel:
    st = [
        StageModel("Backlog", None, [1.0, 2.0, 3.0], [0.6, 0.3, 0.0], counts_as_wait=True),
        StageModel("Development", 2, [0.5, 1.0, 2.0], [0.5, 0.2, 0.0], wip_limit=wip, policy=policy),
    ]
    return SimModel(st, arrival_rate=1.5, entry_probs={"Backlog": 1.0},
                    routing={"Backlog": {"Development": 0.8, DONE: 0.2},
                             "Development": {DONE: 0.9, "Backlog": 0.1}},
                    initial_cases=[("Backlog", 30.0, 5.0), ("Development", 10.0, 0.7)], horizon=60.0)


def test_conservation_and_crn():
    r1 = run_replica(_model(), seed=3)
    r2 = run_replica(_model(), seed=3)
    assert np.array_equal(r1.lead_times, r2.lead_times)          # reprodutível
    started = len(r1.lead_times) + r1.open_at_end
    r3 = run_replica(_model(wip=2, policy="urgency"), seed=3)
    assert len(r3.lead_times) + r3.open_at_end == started          # mesmas chegadas (CRN)
    assert all(lt > 0 for lt in r1.lead_times)


def test_parked_cases_never_complete():
    m = _model()
    m.stages[0].km_times, m.stages[0].km_surv = [1.0, 2.0], [0.7, 0.4]  # 40% nunca saem do Backlog
    r = run_replica(m, seed=1)
    assert r.parked > 0
    assert r.wip_avg["Backlog"] > 0


def test_kaplan_meier_with_zero_durations():
    # metade das visitas tem duração 0; a curva deve cair para 0,5, não para 0.
    from ezekiel.simulation.calibration import km_from_durations
    left = np.zeros(4)
    obs = np.array([0.0, 0.0, 2.0, 3.0])
    times, surv = km_from_durations(left, obs, np.ones(4, bool))
    assert times.tolist() == [0.0, 2.0, 3.0]
    assert surv.tolist() == pytest.approx([0.5, 0.25, 0.0])


def test_replica_records_routes_rework_and_utilization():
    r = run_replica(_model(), seed=5)
    assert sum(r.transitions["Development"].values()) > 0
    assert r.rework == r.transitions["Development"].get("Backlog", 0)   # só Dev → Backlog volta
    assert len(r.done_times) == len(r.lead_times)
    assert 0.0 <= r.utilization["Development"] <= 1.0


def test_extended_validation_matches_model_with_itself():
    from ezekiel.evaluation.validation import ValidationCriteria, extended_metrics
    reps = [run_replica(_model(), seed=s) for s in range(20)]
    ref = reps[0]
    observed = {"waits": ref.waits, "done_times": ref.done_times, "wip_avg": ref.wip_avg,
                "transitions": ref.transitions, "rework": ref.rework}
    ext = extended_metrics(observed, reps[1:], ValidationCriteria(), horizon=60.0,
                           observed_utilization=ref.utilization)
    m = ext["metrics"]
    assert m["route_tv_weighted"] < 0.15
    assert m["throughput_block_coverage"] >= 0.5
    assert ext["checks"]["route_frequencies"] is True


def test_observed_routes_and_rework():
    log = _log()
    m = log.observed_metrics(0.0, 20.0)
    assert m["transitions"]["Backlog"]["Development"] == 2
    assert m["rework"] == 0
    assert len(m["done_times"]) == m["completions"]


def test_service_classes_preserve_common_random_numbers():
    base = run_replica(_model(), seed=7)
    m = _model()
    m.class_probs = {"Bug": 0.6, "Task": 0.4}
    m.initial_classes = ["Bug", "Task"]
    labeled = run_replica(m, seed=7)
    assert np.array_equal(base.lead_times, labeled.lead_times)        # rótulo não muda a dinâmica
    by = labeled.metrics_by_class()
    assert set(by) <= {"Bug", "Task"}
    assert sum(v["completions"] for v in by.values()) == len(labeled.lead_times)


def test_class_specific_service_curve():
    m = _model()
    m.class_probs = {"Lento": 1.0}
    m.stages[1].class_km = {"Lento": ([20.0], [0.0])}             # Development leva 20 dias
    r = run_replica(m, seed=2)
    assert np.mean(r.lead_times_by_class["Lento"]) > np.mean(run_replica(_model(), seed=2).lead_times)


def test_observed_metrics_by_class():
    log = _log()
    log.cases["item_type"] = log.cases["item"].map({"A": "Bug", "B": "Task"})
    m = log.observed_metrics_by_class(0.0, 20.0, ["Bug"])
    assert m["Bug"]["completions"] == 2 and m["Outros"]["completions"] == 0
    assert m["Outros"]["wip"] > 0                                    # B (Task) segue aberto


def test_business_calendar_roundtrip_and_weekends():
    from ezekiel.data.eventlog import business_days, from_business_days
    # 1970-01-05 (t=4) foi segunda; sábado 10 (t=9) e domingo 11 (t=10) não contam.
    assert business_days(9.0) == business_days(11.0)          # fim de semana colapsa
    assert business_days(11.0) - business_days(4.0) == 5.0    # uma semana = 5 dias úteis
    for t in (4.0, 4.5, 8.9, 11.0, 30.25):
        if (t + 3) % 7 < 5:                                    # dia útil
            assert abs(from_business_days(business_days(t)) - t) < 1e-9


def test_log_in_business_time_shrinks_spans_over_weekends():
    log = _log()
    b = log.with_business_calendar()
    assert b.calendar == "business" and b.days_per_week == 5.0
    lt = (log.cases["end"] - log.cases["start"]).dropna()
    blt = (b.cases["end"] - b.cases["start"]).dropna()
    assert (blt <= lt + 1e-9).all()
    assert str(b.date(b.cases["start"].iloc[0]).date()) == str(log.date(log.cases["start"].iloc[0]).date())


def test_multitask_penalty_slows_service_when_overloaded():
    def model(kappa):
        st = [StageModel("Development", 1, [1.0], [0.0], multitask=kappa)]   # serviço de 1 dia
        return SimModel(st, arrival_rate=0.0, entry_probs={"Development": 1.0},
                        routing={"Development": {DONE: 1.0}},
                        initial_cases=[("Development", 0.0, 0.0)] * 3, horizon=20.0)
    plain = sorted(run_replica(model(0.0), seed=1).lead_times)
    slow = sorted(run_replica(model(1.0), seed=1).lead_times)
    assert np.allclose(plain, [1.0, 2.0, 3.0])
    # L/c = 3, 2, 1 no início de cada serviço → durações 3, 2, 1
    assert np.allclose(slow, [3.0, 5.0, 6.0])


def _two_stage(sync=None, pool=None):
    dev = StageModel("Development", 1, [5.0], [0.0])
    cr = StageModel("Code Review", 1, [1.0], [0.0])
    if pool:
        for s in (dev, cr):
            s.pool, s.pool_share = "pool-1", 1.0
    return SimModel([dev, cr], arrival_rate=0.0, entry_probs={"Development": 1.0},
                    routing={"Development": {DONE: 1.0}, "Code Review": {DONE: 1.0}},
                    initial_cases=[("Development", 0.0, 0.0), ("Code Review", 0.0, 0.0)],
                    horizon=30.0, sync=sync or {})


def test_synchronization_waits_for_real_blocker():
    free = sorted(run_replica(_two_stage(), seed=1).lead_times)
    blocked = run_replica(_two_stage(sync={"Code Review": [("Development", 1.0)]}), seed=1)
    assert np.allclose(free, [1.0, 5.0])
    assert np.allclose(sorted(blocked.lead_times), [5.0, 6.0])     # CR espera Dev sair (t=5)
    assert blocked.sync_blocks == 1 and abs(blocked.sync_wait_mean - 5.0) < 1e-9


def test_shared_pool_lends_idle_capacity():
    m = _two_stage(pool=True)
    m.initial_cases = [("Development", 0.0, 0.0)] * 2               # dois casos, um servidor dedicado
    m.stages[0].pool_share = 1.0
    r = run_replica(m, seed=1)
    assert np.allclose(sorted(r.lead_times), [5.0, 5.0])           # pool (2 servidores) atende os dois
    m2 = _two_stage()
    m2.initial_cases = [("Development", 0.0, 0.0)] * 2
    assert np.allclose(sorted(run_replica(m2, seed=1).lead_times), [5.0, 10.0])


def test_initial_case_keeps_wait_accrued_before_horizon():
    """A espera acumulada antes de t entra na espera do caso, como a idade
    entra no lead time (mesma definição do observado)."""
    log = EventLog.from_transitions(pd.DataFrame(
        [("A", "Backlog", 0.0, "Open"), ("A", "Development", 4.0, "In Progress"), ("A", "Deploy", 9.0, "Done")],
        columns=["item", "stage", "t", "column"]))
    assert log.case_wait_until(6.0).to_dict() == {"A#0": 4.0}
    assert log.cases.set_index("case").loc["A#0", "wait"] == 4.0
    st = [StageModel("Development", 1, [1.0], [0.0])]
    m = SimModel(st, arrival_rate=0.0, entry_probs={"Development": 1.0}, routing={"Development": {DONE: 1.0}},
                 initial_cases=[("Development", 6.0, 2.0, 0.0, False, None, 4.0)], horizon=10.0)
    r = run_replica(m, seed=1)
    assert np.allclose(r.lead_times, [7.0]) and np.allclose(r.waits, [4.0])
