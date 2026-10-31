"""ADOPT — Asynchronous Distributed OPTimization (Modi et al., 2005).

Implementação fiel ao pseudocódigo do artigo (handlers de VALUE, COST,
THRESHOLD e TERMINATE; procedimentos backTrack e manutenção dos invariantes
de limiar, alocação e limiar dos filhos), executada em ambiente
controlado por ciclos síncronos: em cada ciclo, cada agente processa as
mensagens recebidas no ciclo anterior e executa um backTrack. Como na
metodologia, a execução assíncrona não é confundida com comunicação física:
medem-se ciclos, número e volume lógico das mensagens.

ADOPT é completo: ao terminar, a atribuição é ótima para a instância
(verificado contra o solucionador centralizado nos testes). Custos reais
são comparados com tolerância numérica.

A mensagem CONSTRAINT-STATE da versão anterior foi removida: ela não
alterava nada no algoritmo e as funções de custo permanecem fixas durante
a execução (metodologia: "as funções permanecerão fixas durante uma
execução do ADOPT").
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field

from ezekiel.dcop.instance import DCOPInstance, PseudoTree, Value, build_pseudo_tree

INF = math.inf
_TOL = 1e-9


def _eq(a: float, b: float) -> bool:
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= _TOL * max(1.0, abs(a), abs(b))


def _compatible(c1: dict, c2: dict) -> bool:
    return all(c2[k] == v for k, v in c1.items() if k in c2)


@dataclass
class Msg:
    kind: str           # VALUE | COST | THRESHOLD | TERMINATE
    src: str
    dst: str
    payload: tuple

    def size(self) -> int:
        """Volume lógico: nº de pares (variável, valor) + escalares."""
        if self.kind == "VALUE":
            return 1
        if self.kind == "COST":
            return len(self.payload[0]) + 2
        if self.kind == "THRESHOLD":
            return len(self.payload[1]) + 1
        return len(self.payload[0])


class _Agent:
    def __init__(self, name: str, inst: DCOPInstance, pt: PseudoTree) -> None:
        self.name = name
        self.inst = inst
        self.domain = inst.domains[name]
        self.parent = pt.parent[name]
        self.children = pt.children[name]
        self.higher = ([self.parent] if self.parent else []) + pt.pseudo_parents[name]
        self.lower = self.children + pt.pseudo_children[name]
        self.neighbors = set(self.higher) | set(self.lower)
        self.is_root = self.parent is None
        self.unary = inst.unary.get(name, {})

        self.ctx: dict[str, Value] = {}
        self.threshold = 0.0
        self.lb = {d: {c: 0.0 for c in self.children} for d in self.domain}
        self.ub = {d: {c: INF for c in self.children} for d in self.domain}
        self.t = {d: {c: 0.0 for c in self.children} for d in self.domain}
        self.cctx: dict = {d: {c: {} for c in self.children} for d in self.domain}
        self.term_received = False
        self.terminated = False
        self.inbox: list[Msg] = []
        self.out: list[Msg] = []
        self.di = min(self.domain, key=self.LB_d)

    # ── custos ──────────────────────────────────────────────────────────
    def delta(self, d: Value) -> float:
        c = self.unary.get(d, 0.0)
        for h in self.higher:
            if h in self.ctx:
                c += self.inst.pair_cost(self.name, d, h, self.ctx[h])
        return c

    def LB_d(self, d: Value) -> float:
        return self.delta(d) + sum(self.lb[d].values())

    def UB_d(self, d: Value) -> float:
        return self.delta(d) + sum(self.ub[d].values())

    def LB(self) -> float:
        return min(self.LB_d(d) for d in self.domain)

    def UB(self) -> float:
        return min(self.UB_d(d) for d in self.domain)

    # ── invariantes ─────────────────────────────────────────────────────
    def _reset_incompatible(self) -> None:
        for d in self.domain:
            for c in self.children:
                if not _compatible(self.cctx[d][c], self.ctx):
                    self.lb[d][c], self.t[d][c], self.ub[d][c], self.cctx[d][c] = 0.0, 0.0, INF, {}

    def maintain_threshold(self) -> None:
        lb, ub = self.LB(), self.UB()
        if self.threshold < lb:
            self.threshold = lb
        if self.threshold > ub:
            self.threshold = ub

    def maintain_child_threshold(self) -> None:
        for d in self.domain:
            for c in self.children:
                if self.lb[d][c] > self.t[d][c]:
                    self.t[d][c] = self.lb[d][c]
                if self.t[d][c] > self.ub[d][c]:
                    self.t[d][c] = self.ub[d][c]

    def maintain_allocation(self) -> None:
        d = self.di
        diff = self.threshold - (self.delta(d) + sum(self.t[d].values()))
        if diff > _TOL:
            for c in self.children:
                room = self.ub[d][c] - self.t[d][c]
                inc = min(room, diff)
                if inc > 0:
                    self.t[d][c] += inc
                    diff -= inc
                if diff <= _TOL:
                    break
        elif diff < -_TOL:
            need = -diff
            for c in self.children:
                room = self.t[d][c] - self.lb[d][c]
                dec = min(room, need)
                if dec > 0:
                    self.t[d][c] -= dec
                    need -= dec
                if need <= _TOL:
                    break
        for c in self.children:
            self.out.append(Msg("THRESHOLD", self.name, c, (self.t[d][c], dict(self.ctx))))

    # ── handlers ────────────────────────────────────────────────────────
    def on_value(self, m: Msg) -> None:
        if self.term_received:
            return
        self.ctx[m.src] = m.payload[0]
        self._reset_incompatible()
        self.maintain_threshold()

    def on_threshold(self, m: Msg) -> None:
        t, context = m.payload
        if _compatible(context, self.ctx):
            self.threshold = t
            self.maintain_threshold()

    def on_terminate(self, m: Msg) -> None:
        self.term_received = True
        self.ctx = {k: v for k, v in m.payload[0].items() if k != self.name}
        self._reset_incompatible()
        self.maintain_threshold()

    def on_cost(self, m: Msg) -> None:
        context, lb, ub = m.payload
        context = dict(context)
        if self.name not in context:
            return
        d = context.pop(self.name)
        if not self.term_received:
            for k, v in context.items():
                if k not in self.neighbors:
                    self.ctx[k] = v
            self._reset_incompatible()
        if _compatible(context, self.ctx):
            self.lb[d][m.src] = lb
            self.ub[d][m.src] = ub
            self.cctx[d][m.src] = context
            self.maintain_child_threshold()
            self.maintain_threshold()

    # ── backTrack ───────────────────────────────────────────────────────
    def backtrack(self) -> None:
        ub = self.UB()
        if _eq(self.threshold, ub):
            self.di = min(self.domain, key=self.UB_d)
        elif self.LB_d(self.di) > self.threshold + _TOL * max(1.0, abs(self.threshold)):
            self.di = min(self.domain, key=self.LB_d)
        for n in self.lower:
            self.out.append(Msg("VALUE", self.name, n, (self.di,)))
        self.maintain_allocation()
        if _eq(self.threshold, ub) and (self.term_received or self.is_root):
            final = dict(self.ctx)
            final[self.name] = self.di
            for c in self.children:
                self.out.append(Msg("TERMINATE", self.name, c, (final,)))
            self.terminated = True
            return
        if self.parent:
            self.out.append(Msg("COST", self.name, self.parent, (dict(self.ctx), self.LB(), self.UB())))

    def step(self) -> None:
        for m in self.inbox:
            getattr(self, "on_" + m.kind.lower())(m)
        self.inbox = []
        self.backtrack()


@dataclass
class AdoptResult:
    assignment: dict[str, Value]
    cost: float
    converged: bool
    cycles: int
    messages: int
    message_volume: int
    messages_by_type: dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    peak_memory_kb: float = 0.0
    root: str | None = None
    tree_depth: int = 0

    def summary(self) -> dict:
        return {
            "assignment": {k: list(v) if isinstance(v, tuple) else v for k, v in self.assignment.items()},
            "cost": self.cost, "converged": self.converged, "cycles": self.cycles,
            "messages": self.messages, "message_volume": self.message_volume,
            "messages_by_type": self.messages_by_type, "wall_time_s": self.wall_time_s,
            "peak_memory_kb": self.peak_memory_kb, "root": self.root, "tree_depth": self.tree_depth,
        }


def solve_adopt(
    inst: DCOPInstance, root: str | None = None, max_cycles: int = 100_000,
    track_memory: bool = True, time_limit_s: float | None = None,
) -> AdoptResult:
    root = root or inst.agents[0]
    pt = build_pseudo_tree(inst, root)
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()

    agents = {a: _Agent(a, inst, pt) for a in inst.agents}
    n_msgs = vol = 0
    by_type = {"VALUE": 0, "COST": 0, "THRESHOLD": 0, "TERMINATE": 0}
    for ag in agents.values():          # Initialize → backTrack
        ag.maintain_threshold()
        ag.backtrack()

    cycles = 0
    while cycles < max_cycles and not all(a.terminated for a in agents.values()):
        cycles += 1
        pending = []
        for ag in agents.values():
            pending.extend(ag.out)
            ag.out = []
        if not pending:
            break
        if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
            break
        for m in pending:
            n_msgs += 1
            vol += m.size()
            by_type[m.kind] += 1
            if not agents[m.dst].terminated:
                agents[m.dst].inbox.append(m)
        for ag in agents.values():
            if ag.inbox and not ag.terminated:
                ag.step()

    wall = time.perf_counter() - t0
    peak = 0.0
    if track_memory:
        peak = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
    assignment = {a: ag.di for a, ag in agents.items()}
    return AdoptResult(
        assignment=assignment, cost=inst.cost(assignment),
        converged=all(a.terminated for a in agents.values()),
        cycles=cycles, messages=n_msgs, message_volume=vol, messages_by_type=by_type,
        wall_time_s=wall, peak_memory_kb=peak, root=root, tree_depth=pt.depth(),
    )
