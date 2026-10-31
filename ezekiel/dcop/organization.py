"""Modelo organizacional: quem são os agentes e o que cada um controla.

Fundamentação (Seção de agentes) e metodologia ("Unidade de análise e
unidade de decisão"; "Agentes, variáveis e domínios"; ameaça
"Representação dos agentes"):

  * agente = UNIDADE DECISÓRIA ORGANIZACIONAL (equipe, serviço, unidade
    funcional) que controla ao menos uma decisão operacional e dispõe de
    informação local; estágios, filas e recursos passivos compõem o
    AMBIENTE e só são agentes quando possuem autonomia decisória;
  * para cada agente documentam-se responsabilidade operacional,
    informação local, decisões controladas, recursos relacionados e
    interfaces com outras unidades;
  * variáveis: limites de WIP, regras de priorização ou frações
    discretizadas de capacidade, conforme a intervenção possível;
  * cada variável pertence a um único agente (α: X → A);
  * diferentes granularidades são comparadas quando há mais de uma
    decomposição organizacional plausível.

Uma `Organization` é declarada em JSON (config/organizacao/*.json). `bind`
a associa a um
modelo de simulação calibrado: unidades sem estágio de capacidade finita
no recorte não têm decisão controlável e ficam fora do DCOP, com o motivo
registrado.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ezekiel.simulation.engine import SimModel

DECISIONS = ("wip_limit", "priority", "capacity")
ORG_DIR = Path(__file__).resolve().parents[2] / "config" / "organizacao"

# Valor de decisão de um agente: (fator de WIP | None, política, fração de capacidade)
#   fator de WIP — limite de cada estágio do agente = fator × capacidade do estágio
#                  (None = sem limite, política vigente);
#   fração de capacidade — parcela dos servidores do agente alocada ao seu
#                  PRIMEIRO estágio, relativa à alocação observada (1,0 = vigente);
#                  o total do agente é conservado.
Value = tuple


@dataclass(frozen=True)
class AgentSpec:
    name: str
    stages: tuple[str, ...]
    responsibility: str
    local_information: tuple[str, ...]
    decisions: tuple[str, ...] = ("wip_limit", "priority")
    resources: tuple[str, ...] = ()
    evidence: str = ""   # origem: explícita (dados) | especialista | premissa
    # interfaces declaradas por especialistas: ({"with": agente, "mechanism": "..."}, ...)
    declared_interfaces: tuple = ()

    def __post_init__(self) -> None:
        bad = set(self.decisions) - set(DECISIONS)
        if bad:
            raise ValueError(f"{self.name}: decisões desconhecidas {bad}")


@dataclass
class Organization:
    name: str
    description: str
    agents: list[AgentSpec]
    environment_notes: dict[str, str] = field(default_factory=dict)  # estágio → por que não é agente

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for a in self.agents:
            for s in a.stages:
                if s in seen:
                    raise ValueError(f"Estágio {s} atribuído a dois agentes ({seen[s]}, {a.name}): "
                                     "cada variável deve pertencer a um único agente.")
                seen[s] = a.name

    @classmethod
    def from_dict(cls, d: dict) -> "Organization":
        agents = [AgentSpec(
            name=a["name"], stages=tuple(a["stages"]), responsibility=a.get("responsibility", ""),
            local_information=tuple(a.get("local_information", ())),
            decisions=tuple(a.get("decisions", ("wip_limit", "priority"))),
            resources=tuple(a.get("resources", ())), evidence=a.get("evidence", ""),
            declared_interfaces=tuple(a.get("interfaces", ())),
        ) for a in d["agents"]]
        return cls(d["name"], d.get("description", ""), agents, d.get("environment", {}))

    @classmethod
    def load(cls, name_or_path: str) -> "Organization":
        p = Path(name_or_path)
        if not p.suffix:
            p = ORG_DIR / f"{name_or_path}.json"
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "agents": [asdict(a) for a in self.agents], "environment": self.environment_notes}

    def bind(self, model: SimModel) -> "BoundOrganization":
        return BoundOrganization(self, model)


class BoundOrganization:
    """Organização associada a um modelo calibrado (estágios presentes, capacidades)."""

    def __init__(self, org: Organization, model: SimModel) -> None:
        self.org = org
        self.model = model
        order = {s: i for i, s in enumerate(model.stage_names)}
        finite = {s.name for s in model.stages if s.servers}
        self.controlled: dict[str, tuple[str, ...]] = {}
        self.excluded: dict[str, str] = {}
        for a in org.agents:
            st = tuple(sorted((s for s in a.stages if s in finite), key=order.get))
            if not st:
                present = [s for s in a.stages if s in order]
                self.excluded[a.name] = (
                    "nenhum estágio sob sua responsabilidade tem atividade no recorte"
                    if not present else
                    "seus estágios no recorte são atrasos puros (sem capacidade controlável)")
                continue
            self.controlled[a.name] = st
        self.specs = {a.name: a for a in org.agents}
        # agentes em ordem de fluxo (primeiro estágio controlado)
        self.agents: list[str] = sorted(self.controlled, key=lambda a: order[self.controlled[a][0]])
        owned = {s for st in self.controlled.values() for s in st}
        self.environment: dict[str, str] = {}
        for s in model.stage_names:
            if s in owned:
                continue
            self.environment[s] = org.environment_notes.get(
                s, "sem unidade decisória com controle sobre o estágio no modelo organizacional")
        self.owner = {s: a for a, st in self.controlled.items() for s in st}

    # ── variáveis e domínios ─────────────────────────────────────────────
    def baseline_value(self, agent: str) -> Value:
        return (None, "FIFO", 1.0) if self._has_capacity(agent) else (None, "FIFO")

    def _has_capacity(self, agent: str) -> bool:
        return "capacity" in self.specs[agent].decisions and len(self.controlled[agent]) >= 2

    def domain(self, agent: str, wip_factors, policies, capacity_shares=(0.75, 1.0, 1.25)) -> list[Value]:
        dec = self.specs[agent].decisions
        wips: list = [None]
        if "wip_limit" in dec:
            # fatores que produzem os mesmos limites em todos os estágios do
            # agente (capacidades pequenas) são o mesmo valor — mantém o primeiro
            seen, wips = set(), []
            for f in wip_factors:
                lim = tuple(max(c, int(round(c * f))) for c in
                            (self.model.stage(s).servers for s in self.controlled[agent]))
                if lim not in seen:
                    seen.add(lim)
                    wips.append(f)
            wips.append(None)
        pols = list(policies) if "priority" in dec else ["FIFO"]
        if self._has_capacity(agent):
            return [(w, p, c) for w in wips for p in pols for c in capacity_shares]
        return [(w, p) for w in wips for p in pols]

    def expand(self, assignment: dict[str, Value]) -> dict[str, tuple]:
        """Atribuição por agente → (limite de WIP, política[, servidores]) por estágio."""
        out: dict[str, tuple] = {}
        for agent, value in assignment.items():
            stages = self.controlled[agent]
            factor, policy = value[0], value[1]
            servers = {s: self.model.stage(s).servers for s in stages}
            if len(value) > 2 and value[2] != 1.0:
                servers = self._reallocate(stages, servers, value[2])
            for s in stages:
                wip = None if factor is None else max(servers[s], int(round(servers[s] * factor)))
                out[s] = (wip, policy, servers[s])
        return out

    @staticmethod
    def _reallocate(stages: tuple[str, ...], servers: dict[str, int], share: float) -> dict[str, int]:
        total = sum(servers.values())
        first = stages[0]
        new_first = min(max(1, int(round(servers[first] * share))), total - (len(stages) - 1))
        rest = total - new_first
        others = stages[1:]
        base_rest = sum(servers[s] for s in others)
        out = {first: new_first}
        alloc = 0
        for i, s in enumerate(others):
            if i == len(others) - 1:
                out[s] = max(1, rest - alloc)
            else:
                out[s] = max(1, int(round(rest * servers[s] / max(base_rest, 1))))
                alloc += out[s]
        return out

    # ── métricas agregadas por agente ────────────────────────────────────
    def agent_wip(self, m: dict, agent: str) -> float:
        return float(sum(m["wip"].get(s, 0.0) for s in self.controlled[agent]))

    def agent_sojourn(self, m: dict, agent: str) -> float:
        """Permanência média por visita nos estágios do agente (ponderada pelo nº de visitas)."""
        st = self.controlled[agent]
        n = sum(m["sojourn_n"].get(s, 0.0) for s in st)
        if n <= 0:
            return float("inf")
        return float(sum(m["sojourn"].get(s, 0.0) * m["sojourn_n"].get(s, 0.0) for s in st) / n)

    def agent_idle(self, m: dict, agent: str) -> float:
        st = self.controlled[agent]
        cap = {s: self.model.stage(s).servers for s in st}
        tot = sum(cap.values())
        return float(sum(m["idle"].get(s, 0.0) * cap[s] for s in st) / tot) if tot else 0.0

    # ── dependências entre agentes ───────────────────────────────────────
    def dependencies(self, min_share: float) -> list[dict]:
        """Matriz de dependências (metodologia, "Grafo de restrições"): para
        cada par de agentes, evidência, mecanismo proposto, variáveis
        envolvidas e função de custo correspondente. Hoje a única evidência
        computada é o fluxo direto de itens entre os estágios dos agentes."""
        rows = []
        rank = {a: i for i, a in enumerate(self.agents)}
        for i, a in enumerate(self.agents):
            for b in self.agents[i + 1:]:
                flows = []
                for s in self.controlled[a]:
                    for t in self.controlled[b]:
                        p_st = self.model.routing.get(s, {}).get(t, 0.0)
                        p_ts = self.model.routing.get(t, {}).get(s, 0.0)
                        if p_st > 0:
                            flows.append((s, t, p_st))
                        if p_ts > 0:
                            flows.append((t, s, p_ts))
                adjacent = rank[b] == rank[a] + 1
                strong = [f for f in flows if f[2] >= min_share]
                if not strong and not adjacent:
                    continue
                rows.append({
                    "agents": [a, b],
                    "evidence": ("fluxo direto de itens: " + "; ".join(
                        f"{x}→{y} {p:.0%} das saídas" for x, y, p in sorted(flows, key=lambda f: -f[2])))
                    if flows else "adjacência na ordem do fluxo (sem transição direta observada)",
                    "mechanism": "limite de WIP e prioridade de um agente alteram chegadas, bloqueio "
                                 "e ociosidade nos estágios do outro",
                    "variables": [f"x_{a}", f"x_{b}"],
                    "function": f"f({a}, {b}) estimada por simulação (Eq. de custo)",
                })
        return rows

    def documentation(self) -> list[dict]:
        """Quadro de documentação dos agentes (metodologia, "Agentes, variáveis e domínios")."""
        inter = {a: set() for a in self.agents}
        for r in self.dependencies(0.0):
            a, b = r["agents"]
            inter[a].add(b)
            inter[b].add(a)
        out = []
        for a in self.agents:
            sp = self.specs[a]
            out.append({
                "agent": a, "stages": list(self.controlled[a]),
                "responsibility": sp.responsibility,
                "local_information": list(sp.local_information),
                "decisions": [d for d in sp.decisions if d != "capacity" or self._has_capacity(a)],
                "resources": list(sp.resources) or [f"capacidade de {s}: {self.model.stage(s).servers}"
                                                    for s in self.controlled[a]],
                "interfaces": sorted(inter[a]),
                "evidence": sp.evidence,
            })
        return out
