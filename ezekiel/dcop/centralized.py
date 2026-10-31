"""C3 — referência centralizada (metodologia, "Configurações de comparação").

Solucionador com informação global sobre as MESMAS variáveis, domínios e
função objetivo da instância DCOP. Usa eliminação de variáveis na ordem
inversa da pseudo-árvore (equivalente centralizado do DPOP): exato, com
custo exponencial apenas na largura induzida — para cadeias e árvores do
pipeline, polinomial.

`brute_force` enumera todo o espaço e serve só para verificação em
instâncias pequenas (testes).
"""

from __future__ import annotations

import itertools
import time
import tracemalloc
from dataclasses import dataclass

from ezekiel.dcop.instance import DCOPInstance, Value, build_pseudo_tree


@dataclass
class CentralResult:
    assignment: dict[str, Value]
    cost: float
    wall_time_s: float
    peak_memory_kb: float
    max_separator: int

    def summary(self) -> dict:
        return {
            "assignment": {k: list(v) if isinstance(v, tuple) else v for k, v in self.assignment.items()},
            "cost": self.cost, "wall_time_s": self.wall_time_s,
            "peak_memory_kb": self.peak_memory_kb, "max_separator": self.max_separator,
        }


class TooLarge(Exception):
    """A tabela UTIL de algum nó excede o limite de entradas."""


def separator_sizes(inst: DCOPInstance, root: str | None = None) -> dict[str, list[str]]:
    pt = build_pseudo_tree(inst, root or inst.agents[0])
    sep: dict[str, list[str]] = {}
    for x in reversed(pt.order):
        s = set(([pt.parent[x]] if pt.parent[x] else []) + pt.pseudo_parents[x])
        for c in pt.children[x]:
            s |= set(sep[c]) - {x}
        sep[x] = [a for a in pt.order if a in s]
    return sep


def solve_centralized(
    inst: DCOPInstance, root: str | None = None, track_memory: bool = True,
    max_table_entries: float | None = None,
) -> CentralResult:
    root = root or inst.agents[0]
    if max_table_entries is not None:
        for x, sp in separator_sizes(inst, root).items():
            size = len(inst.domains[x])
            for a in sp:
                size *= len(inst.domains[a])
            if size > max_table_entries:
                raise TooLarge(f"nó {x}: {size:.3g} entradas")
    pt = build_pseudo_tree(inst, root)
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()

    # Fase UTIL (folhas → raiz): util[x] é função do separador de x.
    sep: dict[str, list[str]] = {}
    util: dict[str, dict[tuple, float]] = {}
    best: dict[str, dict[tuple, Value]] = {}
    for x in reversed(pt.order):
        higher = ([pt.parent[x]] if pt.parent[x] else []) + pt.pseudo_parents[x]
        s = set(higher)
        for c in pt.children[x]:
            s |= set(sep[c]) - {x}
        sep[x] = [a for a in pt.order if a in s]   # ordem estável (ancestrais)
        table: dict[tuple, float] = {}
        arg: dict[tuple, Value] = {}
        for ctx in itertools.product(*(inst.domains[a] for a in sep[x])):
            env = dict(zip(sep[x], ctx))
            best_v, best_c = None, float("inf")
            for d in inst.domains[x]:
                c = inst.unary.get(x, {}).get(d, 0.0)
                for h in higher:
                    c += inst.pair_cost(x, d, h, env[h])
                env[x] = d
                for ch in pt.children[x]:
                    c += util[ch][tuple(env[a] for a in sep[ch])]
                if c < best_c:
                    best_v, best_c = d, c
            table[ctx] = best_c
            arg[ctx] = best_v
        util[x], best[x] = table, arg

    # Fase VALUE (raiz → folhas)
    assignment: dict[str, Value] = {}
    for x in pt.order:
        assignment[x] = best[x][tuple(assignment[a] for a in sep[x])]

    wall = time.perf_counter() - t0
    peak = 0.0
    if track_memory:
        peak = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
    return CentralResult(
        assignment=assignment, cost=inst.cost(assignment), wall_time_s=wall,
        peak_memory_kb=peak, max_separator=max((len(v) for v in sep.values()), default=0),
    )


def brute_force(inst: DCOPInstance) -> tuple[dict[str, Value], float]:
    best, best_c = None, float("inf")
    for vals in itertools.product(*(inst.domains[a] for a in inst.agents)):
        a = dict(zip(inst.agents, vals))
        c = inst.cost(a)
        if c < best_c:
            best, best_c = a, c
    return best, best_c
