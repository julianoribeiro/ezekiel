"""Verificação do coordenador (metodologia, "Verificação do coordenador"):
ADOPT e o solucionador centralizado devem coincidir com a enumeração
exaustiva em instâncias pequenas."""

import random

import pytest

from ezekiel.dcop.adopt import solve_adopt
from ezekiel.dcop.centralized import brute_force, solve_centralized
from ezekiel.dcop.instance import DCOPInstance, build_pseudo_tree


def _random_instance(seed: int) -> DCOPInstance:
    r = random.Random(seed)
    n, dom = r.randint(2, 6), r.randint(2, 4)
    agents = [f"a{i}" for i in range(n)]
    domains = {a: list(range(dom)) for a in agents}
    pairs = [(agents[i], agents[i + 1]) for i in range(n - 1)]
    pairs += [(agents[i], agents[j]) for i in range(n) for j in range(i + 2, n) if r.random() < 0.3]
    binary = {p: {(x, y): r.random() * 10 for x in domains[p[0]] for y in domains[p[1]]} for p in pairs}
    unary = {a: {d: r.random() * 3 for d in domains[a]} for a in agents}
    return DCOPInstance(agents, domains, unary, binary)


@pytest.mark.parametrize("seed", range(150))
def test_adopt_and_centralized_are_optimal(seed):
    inst = _random_instance(seed)
    root = random.Random(seed).choice(inst.agents)
    _, best = brute_force(inst)
    ad = solve_adopt(inst, root=root, track_memory=False)
    c3 = solve_centralized(inst, root=root, track_memory=False)
    assert ad.converged
    assert ad.cost == pytest.approx(best, abs=1e-6)
    assert c3.cost == pytest.approx(best, abs=1e-6)


def test_pseudo_tree_rooted_at_constraint_on_chain():
    agents = ["a", "b", "c", "d"]
    dom = {a: [0, 1] for a in agents}
    binary = {(agents[i], agents[i + 1]): {(x, y): 0.0 for x in [0, 1] for y in [0, 1]} for i in range(3)}
    pt = build_pseudo_tree(DCOPInstance(agents, dom, {}, binary), root="c")
    assert pt.parent["c"] is None
    assert set(pt.children["c"]) == {"b", "d"}
    assert pt.parent["a"] == "b"
    assert not any(pt.pseudo_parents.values())


def test_disconnected_component_with_edge_converges():
    """Raiz isolada e outro componente com aresta (a restrição ativa sem vizinhos)."""
    agents = ["a", "b", "c"]
    dom = {a: [0, 1] for a in agents}
    binary = {("b", "c"): {(x, y): float(x != y) for x in [0, 1] for y in [0, 1]}}
    inst = DCOPInstance(agents, dom, {"a": {0: 1.0, 1: 0.0}}, binary)
    pt = build_pseudo_tree(inst, root="a")
    assert pt.depth() == 2
    ad = solve_adopt(inst, root="a", track_memory=False)
    assert ad.converged and ad.cost == pytest.approx(brute_force(inst)[1])


def test_isolated_agent_keeps_system_cost():
    """Com um só agente, a função objetivo deve conter o desempenho do sistema."""
    from ezekiel.dcop.costs import CostModel, CostSettings
    from ezekiel.dcop.organization import AgentSpec, Organization
    from ezekiel.simulation.engine import SimModel, StageModel

    model = SimModel([StageModel("Dev", 2, [1.0], [0.0])], 1.0, {"Dev": 1.0}, {"Dev": {"DONE": 1.0}})
    org = Organization("t", "", [AgentSpec("Equipe", ("Dev",), "", ())]).bind(model)
    good, bad = (None, "FIFO"), (1.0, "FIFO")
    m = lambda p80: {"wt_mean": 1.0, "lt_p80": p80, "th_week": 1.0, "wip": {"Dev": 1.0},
                     "idle": {"Dev": 0.0}, "sojourn": {"Dev": 1.0}, "sojourn_n": {"Dev": 1.0}}
    cm = CostModel(org, "Equipe", {"Equipe": [good, bad]}, [], baseline=m(10.0),
                   unary_metrics={"Equipe": {good: m(10.0), bad: m(20.0)}})
    inst = cm.instance(CostSettings())
    assert inst.unary["Equipe"][good] < inst.unary["Equipe"][bad]


def test_calibration_gamma_inverse_spread():
    from ezekiel.dcop.costs import calibration_gamma
    base = {"wt_mean": 10.0, "lt_p80": 10.0, "th_week": 10.0}
    ms = [{"wt_mean": 10 + d, "lt_p80": 10 + 2 * d, "th_week": 10.0} for d in (-1, 0, 1)]
    g = calibration_gamma(ms, base)
    assert abs(sum(g) - 3.0) < 1e-9
    assert g[0] > g[1]                       # WT varia menos que P80 → mais peso
    assert g[2] > 0                          # vazão constante: peso médio das demais


def test_entropy_weights():
    from ezekiel.mining.constraint import entropy_weights
    rows = [{"rho": 0.5, "wt": 1.0, "wip": 1.0, "block": 0.0},
            {"rho": 0.5, "wt": 9.0, "wip": 1.0, "block": 0.0}]
    w = entropy_weights(rows)
    assert abs(sum(w) - 1) < 1e-9 and w[1] == max(w) and w[0] == 0.0
