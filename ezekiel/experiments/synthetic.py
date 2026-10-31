"""H4 — viabilidade computacional em cenários sintéticos.

H4 (introdução): "para instâncias pertencentes aos intervalos de escala
definidos no protocolo, a coordenação distribuída alcança soluções dentro
de uma margem de qualidade previamente estabelecida em relação ao baseline
centralizado, respeitando limites operacionais de tempo, memória por nó e
volume de mensagens."

Cenários sintéticos (metodologia, "Geração de cenários sintéticos") são
gerados por um modelo estrutural parametrizado — não por duplicação de
traços — variando nº de agentes e densidade de interdependências:

  * cadeia de n estágios (fluxo principal) + arestas extras entre estágios
    não adjacentes com probabilidade `density` (retrabalho, saltos);
  * cada agente i tem capacidade c_i e carga ρ_i; variável x_i = (nível de
    WIP w ∈ {0..W−1}, política p ∈ {0..P−1}); domínio de tamanho W·P
    (padrão 5×3 = 15, igual ao das instâncias empíricas);
  * custo unário: retenção h_i·w + congestionamento ρ_i/(1 + w);
  * custo binário na cadeia: desequilíbrio de WIP entre estágios
    adjacentes (bloqueio/inanição) k_ij·|w_i − w_j| + interação de
    políticas r_ij[p_i, p_j]; nas arestas extras, só acoplamento leve.

Para cada instância mede-se, para o ADOPT (C2) e para o solucionador
centralizado (C3): custo, lacuna de otimalidade, ciclos, nº e volume de
mensagens, tempo de parede e pico de memória (tracemalloc; memória "por nó"
= pico do processo / nº de agentes, pois a execução é simulada num processo).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np

from ezekiel.dcop.adopt import solve_adopt
from ezekiel.dcop.centralized import TooLarge, solve_centralized
from ezekiel.dcop.instance import DCOPInstance


@dataclass
class H4Settings:
    sizes: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15)
    densities: tuple[float, ...] = (0.0, 0.1, 0.2)
    wip_levels: int = 5
    policies: int = 3
    seeds: int = 5
    max_cycles: int = 200_000
    time_limit_s: float = 30.0
    central_max_entries: float = 5e6           # acima disso, C3 é registrada como inviável
    # ATENÇÃO: critérios provisórios — fixar a priori com o orientador.
    scale_range: tuple[int, int] = (3, 9)       # intervalo de escala do protocolo (nº de agentes)
    max_gap: float = 0.01                       # lacuna relativa máxima vs C3
    max_time_s: float = 10.0
    max_memory_per_agent_kb: float = 10_240.0
    max_messages: int = 1_000_000


def generate_instance(n: int, density: float, seed: int, wip_levels: int = 5, policies: int = 3) -> DCOPInstance:
    rng = random.Random(seed)
    agents = [f"a{i}" for i in range(n)]
    dom = [(w, p) for w in range(wip_levels) for p in range(policies)]
    domains = {a: list(dom) for a in agents}
    rho = {a: rng.uniform(0.5, 0.95) for a in agents}
    hold = {a: rng.uniform(0.05, 0.3) for a in agents}
    unary = {a: {(w, p): hold[a] * w + rho[a] / (1 + w) + rng.uniform(0, 0.05) for (w, p) in dom} for a in agents}

    binary = {}
    for i in range(n - 1):
        a, b = agents[i], agents[i + 1]
        k = rng.uniform(0.1, 1.0)
        r = [[rng.uniform(0, 0.3) for _ in range(policies)] for _ in range(policies)]
        binary[(a, b)] = {(va, vb): k * abs(va[0] - vb[0]) + r[va[1]][vb[1]] for va in dom for vb in dom}
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < density:
                a, b = agents[i], agents[j]
                k = rng.uniform(0.0, 0.3)
                binary[(a, b)] = {(va, vb): k * abs(va[0] - vb[0]) + rng.uniform(0, 0.05) for va in dom for vb in dom}
    inst = DCOPInstance(agents, domains, unary, binary)
    inst.root_hint = max(agents, key=rho.get)   # "restrição" = maior carga
    return inst


def run_h4(s: H4Settings = H4Settings(), progress=None) -> dict:
    rows = []
    for n in s.sizes:
        for dens in s.densities:
            for seed in range(s.seeds):
                inst = generate_instance(n, dens, 1000 * n + int(dens * 100) + seed, s.wip_levels, s.policies)
                root = inst.root_hint
                ad = solve_adopt(inst, root=root, max_cycles=s.max_cycles, time_limit_s=s.time_limit_s)
                try:
                    c3 = solve_centralized(inst, root=root, max_table_entries=s.central_max_entries)
                    gap = (ad.cost - c3.cost) / abs(c3.cost) if c3.cost else 0.0
                except TooLarge:
                    c3, gap = None, None
                rows.append({
                    "n_agents": n, "density": dens, "seed": seed, "edges": len(inst.binary),
                    "domain_size": len(inst.domains[inst.agents[0]]),
                    "adopt_converged": ad.converged, "adopt_cycles": ad.cycles,
                    "adopt_messages": ad.messages, "adopt_message_volume": ad.message_volume,
                    "adopt_time_s": ad.wall_time_s, "adopt_peak_kb": ad.peak_memory_kb,
                    "adopt_peak_kb_per_agent": ad.peak_memory_kb / n,
                    "tree_depth": ad.tree_depth,
                    "central_feasible": c3 is not None,
                    "central_time_s": c3.wall_time_s if c3 else None,
                    "central_peak_kb": c3.peak_memory_kb if c3 else None,
                    "central_max_separator": c3.max_separator if c3 else None,
                    "gap": gap,
                })
                if progress:
                    progress(rows[-1])

    def ok(r: dict) -> bool:
        return (r["adopt_converged"] and (r["gap"] is None or r["gap"] <= s.max_gap)
                and r["adopt_time_s"] <= s.max_time_s
                and r["adopt_peak_kb_per_agent"] <= s.max_memory_per_agent_kb
                and r["adopt_messages"] <= s.max_messages)

    lo, hi = s.scale_range
    in_scope = [r for r in rows if lo <= r["n_agents"] <= hi]
    by_size = {}
    for n in s.sizes:
        rs = [r for r in rows if r["n_agents"] == n]
        by_size[n] = {
            "instances": len(rs),
            "converged_rate": float(np.mean([r["adopt_converged"] for r in rs])),
            "within_limits_rate": float(np.mean([ok(r) for r in rs])),
            "median_messages": float(np.median([r["adopt_messages"] for r in rs])),
            "median_cycles": float(np.median([r["adopt_cycles"] for r in rs])),
            "median_time_s": float(np.median([r["adopt_time_s"] for r in rs])),
            "central_feasible_rate": float(np.mean([r["central_feasible"] for r in rs])),
            "median_central_time_s": float(np.median([r["central_time_s"] for r in rs if r["central_time_s"] is not None]))
            if any(r["central_time_s"] is not None for r in rs) else None,
            "max_gap": max((r["gap"] for r in rs if r["gap"] is not None), default=None),
        }
    # Regressão log(mensagens) ~ n: taxa de crescimento por agente adicional.
    xs = np.array([r["n_agents"] for r in rows], float)
    ys = np.log(np.array([max(r["adopt_messages"], 1) for r in rows], float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "validated": bool(in_scope and all(ok(r) for r in in_scope)),
        "in_scope_instances": len(in_scope),
        "in_scope_within_limits": sum(ok(r) for r in in_scope),
        "by_size": by_size,
        "message_growth": {"log_messages_per_agent": float(slope), "factor_per_agent": float(np.exp(slope))},
        "rows": rows,
        "settings": asdict(s),
    }
