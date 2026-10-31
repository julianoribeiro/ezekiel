"""Descoberta de processos e conformidade (metodologia, "Caracterização
empírica e mineração de processos").

  "A análise será iniciada por estatísticas descritivas, DFGs, variantes e
   projeções por tipo de objeto. Em seguida, serão aplicados algoritmos de
   descoberta [...], incluindo Inductive Miner e Split Miner. Os modelos
   candidatos serão comparados quanto a aderência ao log, precisão,
   generalização e simplicidade. Nenhuma métrica isolada determinará a
   seleção."

Sobre a projeção nos itens de trabalho, na janela de calibração (sem
olhar a validação):

  * traços: estágios visitados por cada caso iniciado e concluído na
    janela, em ordem, mais a atividade final "Concluído";
  * DFG com frequências e tempo médio entre estágios; variantes e
    cobertura das mais frequentes;
  * modelos candidatos: Inductive Miner (sem filtro e com filtro de ruído)
    e Heuristics Miner. O Split Miner não tem implementação em Python (é
    distribuído em Java); fica registrado como não executado;
  * métricas por modelo: aderência (token replay), precisão (ETC por
    token replay), generalização e simplicidade, sem agregá-las num escore
    único;
  * conformidade: fração de traços aderentes, atividades com mais
    problemas no replay e aderência por classe de serviço (tipo do item).

O simulador não executa o modelo descoberto: usa a cadeia de Markov do
roteamento (metodologia: o modelo descoberto fornece parte da estrutura,
mas não é diretamente executável). A comparação entre o DFG e o
roteamento calibrado fica na validação estendida (frequências de rota).
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ezekiel.data.eventlog import EventLog

logger = logging.getLogger(__name__)

END_ACTIVITY = "Concluído"
os.environ.setdefault("TQDM_DISABLE", "1")   # barras de progresso do PM4Py


@dataclass
class DiscoverySettings:
    max_cases: int = 3000            # amostra determinística dos traços (custo do replay)
    noise_threshold: float = 0.2     # Inductive Miner com filtro
    top_variants: int = 10
    min_fitness: float = 0.95
    seed: int = 0


def traces(log: EventLog, a: float, b: float, classes: list[str] | None = None) -> pd.DataFrame:
    """Eventos (case, activity, timestamp, cls) dos casos iniciados e
    concluídos em [a, b)."""
    c = log.cases
    done = c[(c["start"] >= a) & (c["start"] < b) & (c["end"] < b)]
    v = log.visits[log.visits["case"].isin(done["case"])]
    ev = pd.DataFrame({"case": v["case"].to_numpy(), "activity": v["stage"].to_numpy(),
                       "t": v["enter"].to_numpy()})
    fin = pd.DataFrame({"case": done["case"].to_numpy(), "activity": END_ACTIVITY, "t": done["end"].to_numpy()})
    ev = pd.concat([ev, fin], ignore_index=True).sort_values(["case", "t"], kind="mergesort")
    ev["timestamp"] = [log.date(x) for x in ev["t"]]
    cls = log.case_classes(classes)
    ev["cls"] = cls.reindex(ev["case"]).to_numpy()
    return ev


def _pm4py_df(ev: pd.DataFrame) -> pd.DataFrame:
    import pm4py
    df = ev.rename(columns={"case": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"})
    df = df[["case:concept:name", "concept:name", "time:timestamp", "cls"]].copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name",
                                  timestamp_key="time:timestamp")


def _quality(df: pd.DataFrame, net, im, fm) -> dict:
    import pm4py
    t0 = time.perf_counter()
    fit = pm4py.fitness_token_based_replay(df, net, im, fm)
    prec = pm4py.precision_token_based_replay(df, net, im, fm)
    gen = pm4py.generalization_tbr(df, net, im, fm)
    simp = pm4py.simplicity_petri_net(net, im, fm)
    return {"fitness_log": float(fit["log_fitness"]),
            "fitting_traces_pct": float(fit["perc_fit_traces"]),
            "precision": float(prec), "generalization": float(gen), "simplicity": float(simp),
            "places": len(net.places), "transitions": len(net.transitions),
            "silent_transitions": sum(t.label is None for t in net.transitions),
            "time_s": time.perf_counter() - t0}


def discover(log: EventLog, a: float, b: float, classes: list[str] | None = None,
             s: DiscoverySettings = DiscoverySettings()) -> dict:
    import pm4py
    ev = traces(log, a, b, classes)
    cases = ev["case"].unique()
    n_all = len(cases)
    if n_all < 10:
        return {"status": "traços insuficientes", "cases": int(n_all)}
    if n_all > s.max_cases:
        keep = np.random.default_rng(s.seed).choice(cases, s.max_cases, replace=False)
        ev = ev[ev["case"].isin(keep)]
    df = _pm4py_df(ev)

    seqs = ev.groupby("case", sort=False)["activity"].apply(tuple)
    var = Counter(seqs)
    top = var.most_common(s.top_variants)
    dfg, starts, ends = pm4py.discover_dfg(df)
    perf, _, _ = pm4py.discover_performance_dfg(df)
    dfg_rows = sorted(({"from": x, "to": y, "count": int(n),
                        "mean_days": float(perf[(x, y)]["mean"]) / 86400 if (x, y) in perf else None}
                       for (x, y), n in dfg.items()), key=lambda r: -r["count"])

    models = {}
    candidates = {
        "inductive": lambda: pm4py.discover_petri_net_inductive(df),
        f"inductive_ruido_{s.noise_threshold}": lambda: pm4py.discover_petri_net_inductive(
            df, noise_threshold=s.noise_threshold),
        "heuristics": lambda: pm4py.discover_petri_net_heuristics(df),
    }
    nets = {}
    for name, fn in candidates.items():
        try:
            net, im, fm = fn()
            nets[name] = (net, im, fm)
            models[name] = _quality(df, net, im, fm)
        except Exception as exc:  # noqa: BLE001
            models[name] = {"status": f"erro: {exc!r}"}
    models["split_miner"] = {"status": "não executado: sem implementação em Python (Java)"}

    # Conformidade: modelo de referência = maior precisão entre os de
    # aderência ≥ min_fitness (o de aderência máxima tende a ser
    # trivialmente aderente e impreciso). Só para o diagnóstico; a seleção
    # final não usa uma métrica isolada.
    conformance = {}
    ok = {k: v for k, v in models.items() if "fitness_log" in v}
    if ok:
        good = {k: v for k, v in ok.items() if v["fitness_log"] >= s.min_fitness} or ok
        ref = max(good, key=lambda k: (good[k]["precision"], good[k]["fitness_log"]))
        net, im, fm = nets[ref]
        diag = pm4py.conformance_diagnostics_token_based_replay(df, net, im, fm)
        problems = Counter()
        for d in diag:
            for t in d.get("transitions_with_problems", []):
                problems[t.label or "τ"] += 1
        case_ids = list(df.groupby("case:concept:name", sort=False).groups)
        cls_of = ev.drop_duplicates("case").set_index("case")["cls"]
        by_cls: dict[str, list[float]] = {}
        for cid, d in zip(case_ids, diag):
            by_cls.setdefault(str(cls_of.get(cid, "—")), []).append(float(d["trace_fitness"]))
        conformance = {
            "reference_model": ref,
            "fitting_traces_pct": 100.0 * float(np.mean([d["trace_is_fit"] for d in diag])),
            "problem_activities": dict(problems.most_common(10)),
            "fitness_by_class": {c: {"mean": float(np.mean(v)), "n": len(v)} for c, v in by_cls.items()},
        }

    return {
        "status": "ok",
        "window": [str(log.date(a).date()), str(log.date(b).date())],
        "cases_complete": int(n_all), "cases_used": int(ev["case"].nunique()),
        "variants": len(var),
        "top_variants": [{"trace": list(t), "count": n} for t, n in top],
        "top_variants_coverage": sum(n for _, n in top) / max(len(seqs), 1),
        "start_activities": {k: int(v) for k, v in starts.items()},
        "end_activities": {k: int(v) for k, v in ends.items()},
        "dfg": dfg_rows,
        "models": models,
        "conformance": conformance,
        "note": "comparação sem escore único (metodologia); Split Miner não disponível em Python",
    }
