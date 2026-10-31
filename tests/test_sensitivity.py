"""Análise de sensibilidade (metodologia, "Análise de sensibilidade e robustez")."""

from ezekiel.experiments.hypotheses import ExperimentSettings
from ezekiel.experiments.sensitivity import _robustness, variant


def test_variant_changes_only_the_factor():
    base = ExperimentSettings()
    v = variant(base, "capacity_quantile", 0.8)
    assert v.calibration.server_quantile == 0.8 and base.calibration.server_quantile == 0.9
    c = variant(base, "couplings", True)
    assert c.calibration.synchronization and c.calibration.shared_people
    assert not base.calibration.synchronization


def test_robustness_flags():
    ref = {"active_constraint": "A", "c2": {"A": 1},
           "H2": {"delta_pct": -3.0, "practical": "inconclusivo", "validated": False},
           "H3": {"delta_pct": 0.0, "practical": "equivalência prática", "validated": False}}
    var = {"active_constraint": "B", "c2": {"A": 1},
           "H2": {"delta_pct": 1.0, "practical": "inconclusivo", "validated": False},
           "H3": {"delta_pct": 0.0, "practical": "equivalência prática", "validated": False}}
    r = _robustness(ref, var)
    assert not r["same_constraint"] and r["same_c2"]
    assert not r["H2_direction_same"] and r["H2_practical_same"] and r["H3_direction_same"]
