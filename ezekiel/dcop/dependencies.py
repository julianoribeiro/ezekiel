"""Grafo de restrições a partir de evidências (metodologia, "Caracterização
empírica e mineração de processos" e "Grafo de restrições").

  "As dependências candidatas entre unidades decisórias serão obtidas pela
   combinação de quatro fontes: precedência e sincronização entre objetos;
   compartilhamento de recursos; correlação temporal de filas; e
   conhecimento do domínio. A presença de um arco no DFG não será
   suficiente para criar uma restrição DCOP. Cada dependência deverá
   indicar quais decisões interagem e por qual mecanismo."

  "Uma aresta será criada quando decisões controladas por agentes
   distintos participarem de uma mesma função de custo."

Procedimento, para cada par de agentes (a, b):

  1. EVIDÊNCIA (dados da janela de calibração):
       fluxo         fração das saídas dos estágios de um agente que seguem
                     diretamente para estágios do outro (precedência);
       sincronização pares de itens ligados por bloqueio/dependência (OCEL
                     O2O) em que o bloqueador está num estágio de um agente
                     enquanto o bloqueado está num estágio do outro;
       recurso       sobreposição (Jaccard) das pessoas que movem itens nos
                     estágios de cada agente (autor anonimizado das
                     transições) — capacidade humana compartilhada;
       filas         correlação de Pearson entre as variações semanais do
                     WIP dos dois agentes, máxima em defasagens de 0 a
                     `max_lag` semanas;
       domínio       interface declarada no modelo organizacional.
     O par é CANDIDATO se ao menos uma fonte supera seu limiar.

  2. MECANISMO (teste de interação na função de custo, por simulação):
       I(x_a, x_b) = f(x_a, x_b) − f(x_a, b̄) − f(ā, x_b) + f(ā, b̄)
     (ā, b̄ = valores vigentes). Se max |I| < τ, as decisões são
     SEPARÁVEIS: o custo é aditivo e não há aresta — os efeitos entram
     como custos unários. Caso contrário, cria-se a aresta.

A matriz resultante registra, por par: evidências de cada fonte, mecanismo
proposto, variáveis, intensidade da interação e decisão (aresta ou não).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ezekiel.data.eventlog import EventLog
from ezekiel.dcop.organization import BoundOrganization


@dataclass
class DependencySettings:
    # ATENÇÃO: limiares provisórios — fixar a priori com o orientador.
    min_flow_share: float = 0.05
    min_sync_pairs: int = 5
    min_resource_jaccard: float = 0.10
    min_queue_corr: float = 0.30
    max_lag_weeks: int = 3
    interaction_tau: float = 0.01      # |I| mínimo (unidades da função de custo normalizada)


def _flow(org: BoundOrganization, a: str, b: str) -> dict:
    flows = []
    for s in org.controlled[a]:
        for t in org.controlled[b]:
            for x, y in ((s, t), (t, s)):
                p = org.model.routing.get(x, {}).get(y, 0.0)
                if p > 0:
                    flows.append((x, y, p))
    return {"max_share": max((p for *_, p in flows), default=0.0),
            "transitions": [f"{x}→{y} {p:.0%}" for x, y, p in sorted(flows, key=lambda f: -f[2])]}


def _intervals(log: EventLog, stages, a: float, b: float) -> pd.DataFrame:
    v = log.visits[log.visits["stage"].isin(stages)]
    v = v[(v["enter"] < b) & (v["exit"].fillna(np.inf) >= a)]
    item = v["case"].str.rsplit("#", n=1).str[0]
    return pd.DataFrame({"item": item.to_numpy(), "enter": v["enter"].to_numpy(),
                         "exit": v["exit"].fillna(b).to_numpy()})


def _sync(log: EventLog, org: BoundOrganization, a: str, b: str, t0: float, t1: float) -> dict:
    """Pares bloqueador→bloqueado com sobreposição temporal entre os
    estágios de agentes diferentes (em qualquer direção)."""
    if log.ocel is None:
        return {"pairs": 0, "available": False}
    q = log.ocel.o2o
    blk = q[q["qualifier"].isin(["is blocked by", "depends on"])]
    if blk.empty:
        return {"pairs": 0, "available": True}
    def index(df: pd.DataFrame) -> dict:
        return {k: (g["enter"].to_numpy(), g["exit"].to_numpy()) for k, g in df.groupby("item")}

    ia = index(_intervals(log, org.controlled[a], t0, t1))
    ib = index(_intervals(log, org.controlled[b], t0, t1))
    n = 0
    for blocked, blocker in zip(blk["source_id"], blk["target_id"]):
        for x, y in ((ia, ib), (ib, ia)):
            if blocker not in x or blocked not in y:
                continue
            (ke, kx), (le, lx) = x[blocker], y[blocked]
            overlap = (ke[:, None] < lx[None, :]) & (le[None, :] < kx[:, None])
            if overlap.any():
                n += 1
                break
    return {"pairs": n, "available": True}


def _actors(log: EventLog, stages, t0: float, t1: float) -> set:
    """Pessoas (anonimizadas) que movem itens PARA FORA dos estágios dados."""
    if log.ocel is None:
        return set()
    tr = log.ocel.workitem_transitions()
    if "actor" not in tr or tr["actor"].isna().all():
        return set()
    t = pd.to_datetime(tr["time"], utc=True)
    from ezekiel.data.eventlog import to_days
    tr = tr.assign(t=to_days(t))
    v = log.visits[log.visits["stage"].isin(stages) & (log.visits["exit"] >= t0) & (log.visits["exit"] < t1)]
    v = v.assign(item=v["case"].str.rsplit("#", n=1).str[0])
    m = tr.merge(v[["item", "exit"]], left_on=["item", "t"], right_on=["item", "exit"])
    return set(m["actor"].dropna())


def _resource(log: EventLog, org: BoundOrganization, a: str, b: str, t0: float, t1: float) -> dict:
    pa, pb = _actors(log, org.controlled[a], t0, t1), _actors(log, org.controlled[b], t0, t1)
    if not pa and not pb:
        return {"jaccard": None, "available": False}
    union = pa | pb
    return {"jaccard": len(pa & pb) / len(union) if union else 0.0, "people_a": len(pa),
            "people_b": len(pb), "shared": len(pa & pb), "available": True}


def _queues(log: EventLog, org: BoundOrganization, a: str, b: str, t0: float, t1: float, max_lag: int) -> dict:
    wk = log.days_per_week
    if t1 - t0 > wk * 520:     # janelas de calibração têm meses, não décadas
        raise ValueError("janela longa demais para a correlação semanal de filas")
    weeks = np.arange(t0, t1 - wk + 1e-9, wk)
    if len(weeks) < 8:
        return {"corr": None, "lag_weeks": None}
    sa, sb = [], []
    for w in weeks:
        wip = log.time_avg_wip(w, w + wk)
        sa.append(sum(wip.get(s, 0.0) for s in org.controlled[a]))
        sb.append(sum(wip.get(s, 0.0) for s in org.controlled[b]))
    da, db = np.diff(sa), np.diff(sb)
    best, lag_best = 0.0, 0
    for lag in range(0, max_lag + 1):
        for x, y in ((da, db), (db, da)):
            xx, yy = x[: len(x) - lag], y[lag:]
            if len(xx) > 3 and xx.std() > 0 and yy.std() > 0:
                r = float(np.corrcoef(xx, yy)[0, 1])
                if abs(r) > abs(best):
                    best, lag_best = r, lag
    return {"corr": best, "lag_weeks": lag_best}


def _domain(org: BoundOrganization, a: str, b: str) -> str | None:
    for x, y in ((a, b), (b, a)):
        for it in org.specs[x].declared_interfaces:
            if isinstance(it, dict) and it.get("with") == y:
                return it.get("mechanism", "interface declarada")
    return None


def candidate_dependencies(log: EventLog, org: BoundOrganization, t0: float, t1: float,
                           s: DependencySettings = DependencySettings()) -> list[dict]:
    """Etapa 1: evidências por par de agentes e decisão de candidatura."""
    rows = []
    ags = org.agents
    for i, a in enumerate(ags):
        for b in ags[i + 1:]:
            ev = {
                "flow": _flow(org, a, b),
                "synchronization": _sync(log, org, a, b, t0, t1),
                "shared_resource": _resource(log, org, a, b, t0, t1),
                "queue_correlation": _queues(log, org, a, b, t0, t1, s.max_lag_weeks),
                "domain": _domain(org, a, b),
            }
            passed = []
            if ev["flow"]["max_share"] >= s.min_flow_share:
                passed.append("fluxo")
            if ev["synchronization"]["pairs"] >= s.min_sync_pairs:
                passed.append("sincronização")
            j = ev["shared_resource"].get("jaccard")
            if j is not None and j >= s.min_resource_jaccard:
                passed.append("recurso compartilhado")
            r = ev["queue_correlation"].get("corr")
            if r is not None and abs(r) >= s.min_queue_corr:
                passed.append("correlação de filas")
            if ev["domain"]:
                passed.append("domínio")
            rows.append({"agents": [a, b], "evidence": ev, "sources_passed": passed,
                         "candidate": bool(passed)})
    return rows


def interaction_strength(pair_f: dict, base_a, base_b) -> float:
    """max |f(xa,xb) − f(xa,b̄) − f(ā,xb) + f(ā,b̄)| sobre o domínio."""
    f0 = pair_f[(base_a, base_b)]
    best = 0.0
    for (xa, xb), v in pair_f.items():
        fa = pair_f.get((xa, base_b))
        fb = pair_f.get((base_a, xb))
        if fa is None or fb is None:
            continue
        best = max(best, abs(v - fa - fb + f0))
    return best
