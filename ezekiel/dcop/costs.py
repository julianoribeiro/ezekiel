"""Estimação, por simulação, das funções de custo da instância DCOP e da
configuração C1 (decisões locais não coordenadas).

Agentes e variáveis (metodologia, "Agentes, variáveis e domínios"): os
agentes são as UNIDADES ORGANIZACIONAIS do modelo organizacional
(dcop/organization.py), não os estágios. Cada agente controla
x_i = (fator de limite de WIP, política de fila[, fração de capacidade]),
aplicado aos estágios sob sua responsabilidade:

  * limite de WIP de cada estágio do agente = fator × capacidade do
    estágio, fator ∈ wip_factors ∪ {sem limite};
  * política de fila de todos os seus estágios;
  * (opcional) fração da capacidade total do agente alocada ao seu
    primeiro estágio — o total do agente é conservado, de modo que
    nenhuma configuração ganha recursos que as outras não têm.

Custos (normalizados pela linha de base C0, calculada no mesmo estado
inicial — informação disponível no instante da decisão):

  f_ij(x_i, x_j) = γ1·W̃T + γ2·P̃80(LT) − γ3·T̃H
      simulando o sistema com os agentes i e j em (x_i, x_j) e os demais
      em C0;

  termo ToC de WIP a montante de a* (α): custo unário dos agentes a
      montante, α·WIP_i(x_i)/WIP_i(C0), com WIP_i somado sobre os estágios
      do agente;
  termo ToC de ociosidade de a* (β): β·Idle_{a*} (ociosidade dos
      servidores do agente, ponderada pela capacidade), somado à aresta
      entre a* e seu alimentador imediato; se a* não tiver agente a
      montante, vira custo unário de a*.

Pesos γ: fixos (`gamma_mode="fixed"`, padrão) ou estimados na própria
estimação de custos (`gamma_mode="calibration"`): γ_k ∝ 1/desvio-padrão da
métrica normalizada k entre todas as configurações simuladas, reescalados
para somar 3 (como γ = (1, 1, 1)). Assim nenhuma métrica domina só por
variar mais. As simulações de estimação usam sementes e estado próprios,
nunca as réplicas de avaliação.

Cada entrada é a média de `reps` réplicas com sementes PRÓPRIAS da
estimação (distintas das sementes de avaliação), com erro-padrão
registrado. Após estimadas, as tabelas ficam fixas durante o ADOPT.

C1 — cada agente escolhe o valor que minimiza a permanência média nos
SEUS estágios (informação local), supondo os demais na política vigente,
sem função de custo compartilhada.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from ezekiel.dcop.dependencies import interaction_strength
from ezekiel.dcop.instance import DCOPInstance
from ezekiel.dcop.organization import BoundOrganization, Value
from ezekiel.simulation.engine import POLICIES, Simulator

logger = logging.getLogger(__name__)

BASELINE_POLICY = "FIFO"


@dataclass
class CostSettings:
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)   # γ1 WT, γ2 P80, γ3 TH
    alpha: float = 0.5
    beta: float = 0.5
    wip_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    policies: tuple[str, ...] = POLICIES
    capacity_shares: tuple[float, ...] = (0.75, 1.0, 1.25)
    reps: int = 5
    seed_base: int = 500_000          # sementes da estimação ≠ sementes de avaliação
    edge_min_flow_share: float = 0.05
    gamma_mode: str = "fixed"         # "fixed" | "calibration"


@dataclass
class CostModel:
    org: BoundOrganization
    constraint: str
    domains: dict[str, list[Value]]
    dependencies: list[dict]
    baseline: dict
    unary_metrics: dict[str, dict[Value, dict]] = field(default_factory=dict)
    pair_metrics: dict[tuple[str, str], dict[tuple[Value, Value], dict]] = field(default_factory=dict)
    sims_run: int = 0
    gamma: tuple[float, float, float] | None = None   # γ efetivos (None = CostSettings.gamma)

    @property
    def agents(self) -> list[str]:
        return self.org.agents

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Arestas efetivas: pares candidatos cujas decisões interagem na
        função de custo (teste de interação); sem a coluna `edge`, todos os
        pares listados (comportamento anterior)."""
        return [tuple(r["agents"]) for r in self.dependencies if r.get("edge", True)]

    def instance(self, s: CostSettings, alpha: float | None = None, beta: float | None = None) -> DCOPInstance:
        alpha = s.alpha if alpha is None else alpha
        beta = s.beta if beta is None else beta
        g1, g2, g3 = self.gamma or s.gamma
        b = self.baseline
        org = self.org

        def f(m: dict) -> float:
            return (g1 * m["wt_mean"] / max(b["wt_mean"], 1e-9)
                    + g2 * m["lt_p80"] / max(b["lt_p80"], 1e-9)
                    - g3 * m["th_week"] / max(b["th_week"], 1e-9))

        binary = {e: {k: f(m) for k, m in table.items()} for e, table in self.pair_metrics.items()
                  if e in set(self.edges)}
        unary: dict[str, dict[Value, float]] = {a: {v: 0.0 for v in self.domains[a]} for a in self.agents}
        # Agente sem vizinhos (ex.: uma só unidade decisória): sem arestas, o
        # desempenho do sistema sumiria da função objetivo — entra como custo
        # unário f_i(x_i), simulado com os demais em C0.
        connected = {a for e in self.edges for a in e}
        for a in self.agents:
            if a not in connected:
                for v, m in self.unary_metrics[a].items():
                    unary[a][v] += f(m)

        cidx = self.agents.index(self.constraint)
        upstream = self.agents[:cidx]
        if alpha > 0:
            for a in upstream:
                base_wip = max(org.agent_wip(b, a), 1e-9)
                for v, m in self.unary_metrics[a].items():
                    unary[a][v] += alpha * org.agent_wip(m, a) / base_wip
        if beta > 0:
            feeder = upstream[-1] if upstream else None
            edge = (feeder, self.constraint) if feeder is not None and (feeder, self.constraint) in binary else None
            if edge:
                for k, m in self.pair_metrics[edge].items():
                    binary[edge][k] += beta * org.agent_idle(m, self.constraint)
            else:
                for v, m in self.unary_metrics[self.constraint].items():
                    unary[self.constraint][v] += beta * org.agent_idle(m, self.constraint)
        return DCOPInstance(self.agents, self.domains, unary, binary)

    def local_choice(self) -> dict[str, Value]:
        """C1: argmin da permanência nos próprios estágios (empate → valor vigente)."""
        out = {}
        for a in self.agents:
            base_v = self.org.baseline_value(a)
            table = self.unary_metrics[a]
            out[a] = min(table, key=lambda v: (round(self.org.agent_sojourn(table[v], a), 9), v != base_v))
        return out


def _summ(results) -> dict:
    ms = [r.metrics() for r in results]
    out = {k: float(np.nanmean([m[k] for m in ms])) for k in ("wt_mean", "lt_p80", "th_week")}
    out["se_lt_p80"] = float(np.nanstd([m["lt_p80"] for m in ms], ddof=1) / np.sqrt(len(ms))) if len(ms) > 1 else 0.0
    stages = results[0].wip_avg
    out["wip"] = {k: float(np.mean([r.wip_avg[k] for r in results])) for k in stages}
    out["idle"] = {k: float(np.mean([r.idle[k] for r in results])) for k in results[0].idle}
    soj, soj_n = {}, {}
    for k in stages:
        vals = [r.sojourn_mean[k] for r in results if k in r.sojourn_mean]
        soj[k] = float(np.mean(vals)) if vals else float("inf")
        soj_n[k] = float(np.mean([r.sojourn_n.get(k, 0) for r in results]))
    out["sojourn"], out["sojourn_n"] = soj, soj_n
    return out


MECHANISM = {
    "fluxo": "precedência: limite de WIP e prioridade de um agente alteram chegadas, bloqueio e "
             "ociosidade nos estágios do outro",
    "sincronização": "itens nos estágios de um agente aguardam a conclusão de itens nos estágios do outro",
    "recurso compartilhado": "as mesmas pessoas atuam nos dois agentes: as decisões competem pela "
                             "mesma capacidade humana",
    "correlação de filas": "variações de WIP se propagam entre os agentes (propagação de atraso)",
    "domínio": "interface declarada no modelo organizacional",
}
# Mecanismos que o simulador sempre representa (os demais podem justificar a
# candidatura, mas o teste de interação só os observa quando a calibração
# liga a sincronização ou os pools de pessoas — ver estimate_costs).
SIMULATED = {"fluxo", "correlação de filas"}


def calibration_gamma(metrics: list[dict], baseline: dict) -> tuple[float, float, float]:
    """γ_k ∝ 1/sd da métrica normalizada pela linha de base, somando 3."""
    keys = ("wt_mean", "lt_p80", "th_week")
    inv = []
    for k in keys:
        x = np.array([m[k] / max(baseline[k], 1e-9) for m in metrics], float)
        x = x[np.isfinite(x)]
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        inv.append(1.0 / sd if sd > 1e-9 else 0.0)
    if not any(inv):
        return (1.0, 1.0, 1.0)
    # métricas que não variam não influenciam a escolha: peso médio das demais
    mean_inv = float(np.mean([v for v in inv if v > 0]))
    inv = [v if v > 0 else mean_inv for v in inv]
    tot = sum(inv)
    return tuple(3.0 * v / tot for v in inv)


def estimate_costs(org: BoundOrganization, constraint: str, sim: Simulator, s: CostSettings,
                   dependencies: list[dict] | None = None, interaction_tau: float = 0.01) -> CostModel:
    """`dependencies`: pares candidatos com evidências (dcop/dependencies.py).
    Sem eles, usa o fluxo direto entre os estágios (comportamento anterior)."""
    model = org.model
    seeds = [s.seed_base + r for r in range(s.reps)]
    agents = org.agents
    base_assign = {a: org.baseline_value(a) for a in agents}
    domains = {a: org.domain(a, s.wip_factors, s.policies, s.capacity_shares) for a in agents}
    deps = dependencies if dependencies is not None else org.dependencies(s.edge_min_flow_share)
    edges = [tuple(r["agents"]) for r in deps if r.get("candidate", True)]

    jobs: list[tuple[str, object, dict]] = [("base", None, base_assign)]
    for a in agents:
        for v in domains[a]:
            jobs.append(("u", (a, v), {**base_assign, a: v}))
    for (a, b) in edges:
        for va, vb in itertools.product(domains[a], domains[b]):
            jobs.append(("p", ((a, b), (va, vb)), {**base_assign, a: va, b: vb}))

    # cache por atribuição (as combinações iguais à linha de base se repetem)
    def key_of(asg: dict) -> tuple:
        return tuple(sorted(asg.items(), key=lambda kv: kv[0]))

    uniq: dict[tuple, dict] = {}
    for _, _, asg in jobs:
        uniq.setdefault(key_of(asg), asg)
    keys = list(uniq)
    logger.info(f"Estimando custos: {len(keys)} configurações × {len(seeds)} réplicas")
    results = sim.run_many([model.with_assignment(org.expand(uniq[k])) for k in keys], seeds)
    by_key = {k: _summ(r) for k, r in zip(keys, results)}

    cm = CostModel(org, constraint, domains, deps, baseline=by_key[key_of(base_assign)],
                   sims_run=len(keys) * len(seeds))
    for kind, key, asg in jobs:
        if kind == "u":
            a, v = key
            cm.unary_metrics.setdefault(a, {})[v] = by_key[key_of(asg)]
        elif kind == "p":
            e, vv = key
            cm.pair_metrics.setdefault(e, {})[vv] = by_key[key_of(asg)]

    if s.gamma_mode == "calibration":
        cm.gamma = calibration_gamma(list(by_key.values()), cm.baseline)

    if dependencies is not None:
        # Etapa 2 (mecanismo): as decisões do par participam da MESMA função
        # de custo? Mede-se a interação na função de custo simulada.
        g1, g2, g3 = cm.gamma or s.gamma
        b = cm.baseline

        def f(m: dict) -> float:
            return (g1 * m["wt_mean"] / max(b["wt_mean"], 1e-9) + g2 * m["lt_p80"] / max(b["lt_p80"], 1e-9)
                    - g3 * m["th_week"] / max(b["th_week"], 1e-9))

        for r in deps:
            a_, b_ = r["agents"]
            r["mechanism"] = [MECHANISM[x] for x in r.get("sources_passed", [])]
            r["variables"] = [f"x_{a_}", f"x_{b_}"]
            if not r.get("candidate"):
                r.update(edge=False, interaction=None, decision="sem aresta: nenhuma evidência acima dos limiares")
                continue
            simulated = set(SIMULATED)
            if model.sync:
                simulated.add("sincronização")
            if any(st_.pool for st_ in model.stages):
                simulated.add("recurso compartilhado")
            table = {k: f(m) for k, m in cm.pair_metrics[(a_, b_)].items()}
            inter = interaction_strength(table, base_assign[a_], base_assign[b_])
            r["interaction"] = inter
            r["function"] = f"f({a_}, {b_}) estimada por simulação (Eq. de custo)"
            if inter >= interaction_tau:
                r.update(edge=True, decision="aresta: as decisões interagem na função de custo")
            else:
                unmodeled = [x for x in r["sources_passed"] if x not in simulated]
                r.update(edge=False, decision=(
                    "sem aresta: decisões separáveis na função de custo simulada"
                    + (f"; evidência de {', '.join(unmodeled)} não representada no simulador"
                       if unmodeled else "")))
    return cm
