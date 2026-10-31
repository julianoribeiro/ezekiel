"""Instância DCOP: agentes, domínios, custos unários e binários, pseudo-árvore.

Função global (metodologia, Eq. `eq:dcop_global_goal`):

    J(x) = Σ_j f_j(x_escopo(f_j)) + α·W̃IP_montante(a*) + β·Ĩdle(a*)

Aqui todos os termos são decompostos em custos unários e binários (ver
dcop/costs.py), de modo que a instância é um DCOP binário padrão — o mesmo
problema é entregue ao ADOPT (C2) e ao solucionador centralizado (C3),
"com as mesmas variáveis, domínios, restrições e função objetivo".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

Value = Hashable


@dataclass
class DCOPInstance:
    agents: list[str]
    domains: dict[str, list[Value]]
    unary: dict[str, dict[Value, float]] = field(default_factory=dict)
    # chave (a, b) com a antes de b em `agents`; valor indexado por (v_a, v_b)
    binary: dict[tuple[str, str], dict[tuple[Value, Value], float]] = field(default_factory=dict)

    def neighbors(self, a: str) -> list[str]:
        out = []
        for (x, y) in self.binary:
            if x == a:
                out.append(y)
            elif y == a:
                out.append(x)
        return out

    def pair_cost(self, a: str, va: Value, b: str, vb: Value) -> float:
        if (a, b) in self.binary:
            return self.binary[(a, b)][(va, vb)]
        return self.binary[(b, a)][(vb, va)]

    def cost(self, assignment: dict[str, Value]) -> float:
        total = sum(self.unary.get(a, {}).get(assignment[a], 0.0) for a in self.agents)
        for (a, b), table in self.binary.items():
            total += table[(assignment[a], assignment[b])]
        return total


@dataclass
class PseudoTree:
    root: str
    parent: dict[str, str | None]
    children: dict[str, list[str]]
    pseudo_parents: dict[str, list[str]]
    pseudo_children: dict[str, list[str]]
    order: list[str]   # pré-ordem da DFS

    def depth(self) -> int:
        d = {self.root: 0}
        for a in self.order:
            if self.parent[a] is not None:
                d[a] = d[self.parent[a]] + 1
        return max(d.values()) + 1


def build_pseudo_tree(inst: DCOPInstance, root: str) -> PseudoTree:
    """Pseudo-árvore por DFS a partir da raiz (a restrição ativa).

    Vizinhos são visitados na ordem de `inst.agents` (ordem do fluxo), o
    que mantém a cadeia montante/jusante como ramos da raiz.
    """
    parent: dict[str, str | None] = {root: None}
    children = {a: [] for a in inst.agents}
    pseudo_parents = {a: [] for a in inst.agents}
    pseudo_children = {a: [] for a in inst.agents}
    order: list[str] = []
    rank = {a: i for i, a in enumerate(inst.agents)}
    ancestors: dict[str, set[str]] = {root: set()}

    def visit(a: str) -> None:
        order.append(a)
        for n in sorted(inst.neighbors(a), key=lambda x: abs(rank[x] - rank[a])):
            if n not in parent:
                parent[n] = a
                children[a].append(n)
                ancestors[n] = ancestors[a] | {a}
                visit(n)
            elif n in ancestors[a] and n != parent[a] and n not in pseudo_parents[a]:
                pseudo_parents[a].append(n)
                pseudo_children[n].append(a)

    visit(root)
    for a in inst.agents:          # componentes desconexos viram novas raízes
        if a not in parent:
            parent[a] = None
            ancestors[a] = set()
            visit(a)
    return PseudoTree(root, parent, children, pseudo_parents, pseudo_children, order)
