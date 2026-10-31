"""Modelo organizacional (fundamentação, Seção de agentes; metodologia,
"Agentes, variáveis e domínios"): agentes são unidades organizacionais,
estágios são ambiente."""

import pytest

from ezekiel.data.eventlog import DONE
from ezekiel.dcop.organization import AgentSpec, Organization
from ezekiel.simulation.engine import SimModel, StageModel


def _model():
    st = [
        StageModel("Backlog", None, [1.0], [0.0], counts_as_wait=True),
        StageModel("Development", 4, [1.0], [0.0]),
        StageModel("Code Review", 2, [1.0], [0.0]),
        StageModel("QA Testing", 3, [1.0], [0.0]),
    ]
    routing = {"Backlog": {"Development": 1.0}, "Development": {"Code Review": 0.9, "QA Testing": 0.1},
               "Code Review": {"QA Testing": 0.8, "Development": 0.2}, "QA Testing": {DONE: 1.0}}
    return SimModel(st, 1.0, {"Backlog": 1.0}, routing)


def _org(decisions=("wip_limit", "priority")):
    return Organization("t", "", [
        AgentSpec("Dev", ("Development", "Code Review"), "desenvolver", ("board",), decisions),
        AgentSpec("QA", ("QA Testing",), "testar", ("board",)),
        AgentSpec("Operações", ("Staging",), "homologar", ()),
    ], {"Backlog": "fila de entrada"})


def test_stage_cannot_belong_to_two_agents():
    with pytest.raises(ValueError):
        Organization("x", "", [AgentSpec("A", ("Development",), "", ()), AgentSpec("B", ("Development",), "", ())])


def test_bind_separates_agents_and_environment():
    b = _org().bind(_model())
    assert b.agents == ["Dev", "QA"]
    assert b.controlled["Dev"] == ("Development", "Code Review")
    assert "Operações" in b.excluded                       # sem estágio ativo no recorte
    assert b.environment == {"Backlog": "fila de entrada"}


def test_expand_applies_agent_decision_to_its_stages():
    b = _org().bind(_model())
    out = b.expand({"Dev": (1.5, "urgency"), "QA": (None, "FIFO")})
    assert out["Development"] == (6, "urgency", 4)
    assert out["Code Review"] == (3, "urgency", 2)
    assert out["QA Testing"] == (None, "FIFO", 3)


def test_capacity_reallocation_conserves_agent_total():
    b = _org(("wip_limit", "priority", "capacity")).bind(_model())
    assert len(b.domain("Dev", (1.0,), ("FIFO",))[0]) == 3
    out = b.expand({"Dev": (None, "FIFO", 0.5)})
    assert out["Development"][2] + out["Code Review"][2] == 6
    assert out["Development"][2] == 2


def test_dependencies_document_evidence():
    deps = _org().bind(_model()).dependencies(0.05)
    assert [d["agents"] for d in deps] == [["Dev", "QA"]]
    assert "Code Review→QA Testing" in deps[0]["evidence"]
    assert set(deps[0]) >= {"evidence", "mechanism", "variables", "function"}


def test_documentation_has_methodology_fields():
    doc = _org().bind(_model()).documentation()
    assert set(doc[0]) >= {"agent", "responsibility", "local_information", "decisions",
                           "resources", "interfaces", "evidence"}
    assert doc[0]["interfaces"] == ["QA"]


def test_domain_drops_equivalent_factors():
    m = _model()
    m.stages[2].servers = 1
    org = Organization("t", "", [AgentSpec("CR", ("Code Review",), "", ())]).bind(m)
    wips = sorted({v[0] for v in org.domain("CR", (1.0, 1.5, 2.0, 3.0), ("FIFO",)) if v[0] is not None})
    assert wips == [1.0, 1.5, 3.0]                       # 1,5 e 2,0 dão o mesmo limite (2)


def test_builtin_organizations_load():
    for name in ("por_papel", "por_estagio"):
        assert Organization.load(name).agents
