"""Identificação operacional da restrição ativa (fundamentação, Eq.
`eq:constraintid_operaciotional`; metodologia, Seção
"Identificação operacional da restrição ativa").

    C_i(t) = β1·ρ̃_i(t) + β2·W̃T_i(t) + β3·L̃_i(t) + β4·B̃_i(t)

com indicadores normalizados (min-max) entre as unidades candidatas, na
mesma janela. Operacionalização adotada neste protótipo:

  ρ_i  utilização: WIP ativo médio / capacidade estimada do estágio;
  WT_i espera: em estágios com fila/serviço separados, o tempo médio em
       status de fila; caso contrário, a permanência média
       no estágio — o log do Jira não separa espera de serviço num status;
  L_i  WIP médio no tempo;
  B_i  impacto sobre dependentes: WIP médio × fração das saídas do estágio
       que seguem para outra unidade candidata (carga retida que outras
       unidades aguardam).

Pesos β: fixos (`weight_mode="fixed"`, justificativa teórica — padrão) ou
estimados na janela de calibração pelo método da entropia
(`weight_mode="entropy"`): indicadores que mais discriminam entre unidades
e janelas mensais recebem mais peso. Nenhum dado da janela de validação é
usado.

Persistência temporal: o escore é calculado em janelas mensais da janela de
calibração; a restrição ativa é a unidade que lidera no maior número de
janelas. Empates são resolvidos pelo escore da janela inteira. A
estabilidade (fração de janelas em que a escolhida lidera) é reportada.

As candidatas são os AGENTES do modelo organizacional (unidades decisórias),
com os indicadores agregados sobre os estágios que cada um controla — a
equação é definida para unidades a_i, não para estágios. A fila de entrada
(Backlog) é ambiente: demanda ainda não puxada, não um recurso (fundamentação,
Seção sobre ToC: alta ocupação não implica restrição sistêmica).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ezekiel.data.eventlog import EventLog
from ezekiel.dcop.organization import BoundOrganization
from ezekiel.simulation.engine import SimModel


@dataclass
class ConstraintSettings:
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # β1..β4
    window_days: float = 30.0
    weight_mode: str = "fixed"          # "fixed" | "entropy" (estimados na calibração)


KEYS = ("rho", "wt", "wip", "block")


def entropy_weights(rows: list[dict]) -> tuple[float, ...]:
    """Pesos pelo método da entropia sobre as observações (unidade × janela).
    Indicador constante (entropia máxima) recebe peso 0; se todos forem
    constantes, pesos iguais."""
    n = len(rows)
    d = []
    for k in KEYS:
        x = np.array([max(r[k], 0.0) for r in rows], float)
        tot = x.sum()
        if n < 2 or tot <= 0:
            d.append(0.0)
            continue
        p = x / tot
        e = -float(np.sum(p[p > 0] * np.log(p[p > 0]))) / np.log(n)
        d.append(max(1.0 - e, 0.0))
    s = sum(d)
    return tuple(v / s for v in d) if s > 0 else (0.25,) * 4


def _stage_indicators(log: EventLog, model: SimModel, stage: str, a: float, b: float,
                      wip: dict, vis) -> dict:
    st = model.stage(stage)
    sv = vis[vis["stage"] == stage]
    if st.split:
        wt_sum = float(sv["wait"].sum())
        cap = float(np.quantile(sv["service"], 0.95)) if len(sv) else np.inf
        active = log.work_concurrency(stage, a, b, max_age=cap).mean()
    else:
        wt_sum = float((sv["exit"] - sv["enter"]).sum())
        active = log.concurrency_samples(stage, a, b, max_age=_age_cap(log, stage, a, b)).mean()
    return {"active": float(active), "servers": st.servers or 0, "wt_sum": wt_sum,
            "visits": int(len(sv)), "wip": wip.get(stage, 0.0), "next": sv["next"]}


def _indicators(log: EventLog, org: BoundOrganization, a: float, b: float) -> dict[str, dict]:
    """Indicadores por AGENTE, agregados sobre os estágios que ele controla:
    ρ = casos ativos / capacidade do agente; WT = espera (ou permanência)
    média por visita; L = WIP; B = WIP × fração das saídas que seguem para
    estágios de OUTROS agentes."""
    model = org.model
    wip = log.time_avg_wip(a, b)
    vis = log.visits_ending_in(a, b)
    out = {}
    for ag in org.agents:
        parts = [_stage_indicators(log, model, s, a, b, wip, vis) for s in org.controlled[ag]]
        cap = sum(p["servers"] for p in parts)
        n = sum(p["visits"] for p in parts)
        others = {s for x in org.agents if x != ag for s in org.controlled[x]}
        nxt = [v for p in parts for v in p["next"].tolist()]
        fwd = float(np.mean([v in others for v in nxt])) if nxt else 0.0
        L = float(sum(p["wip"] for p in parts))
        out[ag] = {
            "rho": min(1.0, sum(p["active"] for p in parts) / cap) if cap else 0.0,
            "wt": sum(p["wt_sum"] for p in parts) / n if n else 0.0,
            "wip": L,
            "block": L * fwd,
            "visits": n,
        }
    return out


def agent_states(log: EventLog, org: BoundOrganization, t: float, window_days: float = 30.0) -> dict[str, dict]:
    """Estado de monitoramento de cada agente no instante t (fundamentação,
    Eq. do estado do agente): ⟨ρ_i(t), L_i(t), W̄T_i(t)⟩, medidos na janela
    [t − Δt, t). Não alteram as funções de custo durante o ADOPT."""
    ind = _indicators(log, org, t - window_days, t)
    return {ag: {"rho": v["rho"], "wip": v["wip"], "wt_mean": v["wt"] if v["visits"] else None,
                 "visits": v["visits"], "window_days": window_days}
            for ag, v in ind.items()}


def _age_cap(log: EventLog, s: str, a: float, b: float, min_visits: int = 20,
             max_lookback: float = 365.0) -> float:
    """P95 das permanências encerradas até b. Com menos de `min_visits` na
    janela, amplia-se para trás (até `max_lookback` dias antes de b) — só
    passado, sem vazamento. Janelas curtas (monitoramento mensal) davam
    cortes instáveis e ρ oscilando entre 0 e 1."""
    lo = a
    while True:
        sv = log.visits_ending_in(lo, b)
        d = (sv.loc[sv["stage"] == s, "exit"] - sv.loc[sv["stage"] == s, "enter"]).to_numpy()
        if len(d) >= min_visits or b - lo >= max_lookback:
            return float(np.quantile(d, 0.95)) if len(d) else np.inf
        lo = max(b - max_lookback, lo - (b - a))


def _scores(ind: dict[str, dict], w: tuple[float, ...]) -> dict[str, float]:
    keys = ["rho", "wt", "wip", "block"]
    norm = {}
    for k in keys:
        vals = np.array([ind[s][k] for s in ind])
        lo, hi = vals.min(), vals.max()
        norm[k] = {s: (ind[s][k] - lo) / (hi - lo) if hi > lo else 0.0 for s in ind}
    return {s: float(sum(wi * norm[k][s] for wi, k in zip(w, keys))) for s in ind}


def identify_constraint(
    log: EventLog, org: BoundOrganization, a: float, b: float,
    settings: ConstraintSettings = ConstraintSettings(),
) -> dict:
    """Restrição ativa a* entre os AGENTES (unidades decisórias)."""
    full_ind = _indicators(log, org, a, b)
    windows = []
    t = a
    while t + settings.window_days <= b + 1e-9:
        windows.append(_indicators(log, org, t, t + settings.window_days))
        t += settings.window_days
    w = settings.weights
    if settings.weight_mode == "entropy":
        w = entropy_weights([v for ind in windows for v in ind.values()] or list(full_ind.values()))
    full = _scores(full_ind, w)
    leaders = []
    for ind in windows:
        sc = _scores(ind, w)
        leaders.append(max(sc, key=sc.get))

    votes = Counter(leaders)
    if votes:
        top = max(votes.values())
        tied = [s for s, v in votes.items() if v == top]
        chosen = max(tied, key=lambda s: full[s])
    else:
        chosen = max(full, key=full.get)

    return {
        "active_constraint": chosen,
        "stability": votes[chosen] / len(leaders) if leaders else None,
        "windows": len(leaders),
        "votes": dict(votes),
        "score_full_window": full,
        "indicators_full_window": full_ind,
        "weights": list(w),
        "weight_mode": settings.weight_mode,
    }
