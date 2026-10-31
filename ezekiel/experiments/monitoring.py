"""Ciclo de monitoramento e reotimização (fundamentação, Seção do estado de
monitoramento, Eqs. `eq:constraint_state_msg` e `eq:variacao_estado_agente`).

A cada Δt, no intervalo [início, fim) (por padrão, a janela de validação),
calcula-se o estado de cada agente

    estado(a_i, t) = ⟨ρ_i(t), L_i(t), W̄T_i(t)⟩      em [t − Δt, t)

e a variação em relação à observação anterior (Δestado). Só se usa o log
até t — nenhuma informação posterior ao instante da decisão.

Protocolo (valores provisórios — fixar na E1):

  * limiares de mudança: |Δρ| ≥ `rho_abs`; |ΔL| / max(L, 1) ≥ `wip_rel`;
    |ΔW̄T| / max(W̄T, `wt_floor_days`) ≥ `wt_rel`;
  * persistência: o limiar precisa ser excedido em `persistence` janelas
    consecutivas pelo mesmo agente (evita reagir a ruído de uma janela);
  * mudança da restrição ativa: a* recalculado em [t − `constraint_days`,
    t), com a persistência mensal de `identify_constraint`, diferente do a*
    da configuração vigente — dispara diretamente;
  * observações ausentes: janela sem visitas encerradas no agente não tem
    W̄T; usa-se o último valor observado (sem gerar variação);
  * reinicialização: no disparo, recalibra-se o simulador em
    [t − `cal_days`, t), identifica-se a restrição, reestimam-se as funções
    de custo e roda-se o ADOPT de novo. As tabelas ficam fixas durante cada
    execução do ADOPT; o monitoramento só decide QUANDO reiniciar.
    Intervalo mínimo entre reinícios `min_interval_days`; no máximo
    `max_reoptimizations`.

Avaliação: cada configuração nova é comparada à vigente no trecho até o
próximo disparo (ou o fim), simulando os dois a partir do estado real em t
com números aleatórios comuns (P80 pareado por réplica).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ezekiel.data.eventlog import EventLog
from ezekiel.dcop.adopt import solve_adopt
from ezekiel.dcop.costs import estimate_costs
from ezekiel.dcop.dependencies import candidate_dependencies
from ezekiel.dcop.organization import Organization
from ezekiel.evaluation import stats as st
from ezekiel.mining.constraint import agent_states, identify_constraint
from ezekiel.simulation.calibration import at_state, build_model
from ezekiel.simulation.engine import Simulator

logger = logging.getLogger(__name__)


@dataclass
class MonitoringSettings:
    enabled: bool = True
    window_days: float = 30.0            # Δt
    rho_abs: float = 0.15
    wip_rel: float = 0.25
    wt_rel: float = 0.25
    wt_floor_days: float = 1.0
    persistence: int = 2
    constraint_days: float = 90.0
    min_interval_days: float = 60.0
    max_reoptimizations: int = 2
    eval_reps: int = 10


def _delta(prev: dict, cur: dict, s: MonitoringSettings) -> dict:
    out = {"rho": cur["rho"] - prev["rho"], "wip": cur["wip"] - prev["wip"],
           "wt_mean": (cur["wt_mean"] - prev["wt_mean"])
           if cur["wt_mean"] is not None and prev["wt_mean"] is not None else 0.0}
    out["exceeds"] = sorted(k for k, ok in (
        ("rho", abs(out["rho"]) >= s.rho_abs),
        ("wip", abs(out["wip"]) / max(prev["wip"], 1.0) >= s.wip_rel),
        ("wt_mean", prev["wt_mean"] is not None
         and abs(out["wt_mean"]) / max(prev["wt_mean"], s.wt_floor_days) >= s.wt_rel),
    ) if ok)
    return out


def _decide(log: EventLog, organization: Organization, t: float, horizon: float, s, sim: Simulator) -> dict:
    """Recalibra em [t − cal_days, t) e resolve uma nova instância com ADOPT."""
    model, _ = build_model(log, t - s.cal_days, t, s.calibration, sim)
    vm = at_state(model, log, t, horizon)
    org = organization.bind(vm)
    if not org.agents:
        return {"status": "sem agentes", "vm": vm, "org": org}
    cons = identify_constraint(log, org, t - s.cal_days, t, s.constraint)
    deps = candidate_dependencies(log, org, t - s.cal_days, t, s.dependencies)
    cm = estimate_costs(org, cons["active_constraint"], sim, s.costs, deps, s.dependencies.interaction_tau)
    ad = solve_adopt(cm.instance(s.costs), root=cons["active_constraint"])
    return {"status": "ok", "vm": vm, "org": org, "a_star": cons["active_constraint"],
            "assignment": ad.assignment, "adopt": ad.summary()}


def run_monitoring(log: EventLog, organization: Organization, start: float, end: float,
                   initial: dict, s, sim: Simulator) -> dict:
    """`initial`: {"org": BoundOrganization, "a_star": str, "assignment": dict}
    da decisão em `start`. `s`: ExperimentSettings (usa `s.monitoring`)."""
    ms: MonitoringSettings = s.monitoring
    cur = dict(initial)
    org = cur["org"]
    timeline, decisions = [], []
    prev, streak = None, {a: 0 for a in org.agents}
    last_reopt = start
    t = start + ms.window_days
    while t <= end + 1e-9:
        states = agent_states(log, org, t, ms.window_days)
        if prev is not None:
            for a in states:
                if states[a]["wt_mean"] is None and a in prev:   # ausente: último observado
                    states[a]["wt_mean"] = prev[a]["wt_mean"]
        # agentes novos após uma reotimização não têm janela anterior
        deltas = {a: _delta(prev[a], states[a], ms) for a in states if a in prev} if prev else {}
        for a in org.agents:
            streak[a] = streak.get(a, 0) + 1 if deltas.get(a, {}).get("exceeds") else 0
        trailing = identify_constraint(log, org, max(t - ms.constraint_days, log.t_min), t, s.constraint)
        reasons = [f"variação persistente em {a}" for a in org.agents if streak[a] >= ms.persistence]
        if trailing["active_constraint"] != cur["a_star"]:
            reasons.append(f"restrição ativa mudou: {cur['a_star']} → {trailing['active_constraint']}")
        row = {"t": str(log.date(t).date()), "states": states, "deltas": deltas,
               "active_constraint_trailing": trailing["active_constraint"], "triggers": reasons}
        can = (len(decisions) < ms.max_reoptimizations and t - last_reopt >= ms.min_interval_days
               and end - t >= ms.window_days)
        if reasons and can:
            row["reoptimized"] = True
            dec = _decide(log, organization, t, end - t, s, sim)
            entry = {"t": row["t"], "t_days": t, "reasons": reasons, "status": dec["status"]}
            if dec["status"] == "ok":
                entry.update(a_star=dec["a_star"], assignment={a: list(v) for a, v in dec["assignment"].items()},
                             adopt=dec["adopt"])
                entry["_dec"] = dec
                cur = {"org": dec["org"], "a_star": dec["a_star"], "assignment": dec["assignment"]}
                org = dec["org"]
                streak = {a: 0 for a in org.agents}
            decisions.append(entry)
            last_reopt = t
        timeline.append(row)
        prev = states
        t += ms.window_days

    # Avaliação por trecho: nova configuração vs a vigente antes do disparo.
    seeds = [s.eval_seed_base + 50_000 + r for r in range(ms.eval_reps)]
    previous = initial["assignment"]
    oks = [d for d in decisions if d["status"] == "ok"]
    for i, d in enumerate(oks):
        dec = d.pop("_dec")
        seg_end = oks[i + 1]["t_days"] if i + 1 < len(oks) else end
        vm = dec["vm"]
        vm.horizon = seg_end - d["t_days"]
        o = dec["org"]
        old = {a: previous.get(a, o.baseline_value(a)) for a in o.agents}
        old = {a: (v if v in o.domain(a, s.costs.wip_factors, s.costs.policies, s.costs.capacity_shares)
                   else o.baseline_value(a)) for a, v in old.items()}
        keep, new = sim.run_many([vm.with_assignment(o.expand(old)), vm.with_assignment(o.expand(dec["assignment"]))],
                                 seeds)
        d["segment_days"] = vm.horizon
        d["changed_configuration"] = old != dec["assignment"]
        d["new_vs_previous_p80"] = st.paired_compare([r.metrics()["lt_p80"] for r in keep],
                                                     [r.metrics()["lt_p80"] for r in new], "lower", s.alpha)
        previous = dec["assignment"]
    for d in decisions:
        d.pop("_dec", None)
        d.pop("t_days", None)
    return {"settings": ms.__dict__, "timeline": timeline, "reoptimizations": decisions,
            "n_triggers": sum(bool(r["triggers"]) for r in timeline)}
