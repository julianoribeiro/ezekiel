"""Protocolo de monitoramento (fundamentação, Eq. da variação do estado)."""

from ezekiel.experiments.monitoring import MonitoringSettings, _delta


def test_delta_thresholds():
    s = MonitoringSettings()
    prev = {"rho": 0.5, "wip": 10.0, "wt_mean": 4.0}
    assert _delta(prev, {"rho": 0.55, "wip": 11.0, "wt_mean": 4.5}, s)["exceeds"] == []
    d = _delta(prev, {"rho": 0.8, "wip": 14.0, "wt_mean": 6.0}, s)
    assert d["exceeds"] == ["rho", "wip", "wt_mean"]


def test_delta_small_values_use_floors():
    s = MonitoringSettings()
    prev = {"rho": 0.1, "wip": 0.2, "wt_mean": 0.1}
    # WIP e espera minúsculos: variações relativas enormes, mas abaixo dos pisos
    assert _delta(prev, {"rho": 0.1, "wip": 0.4, "wt_mean": 0.3}, s)["exceeds"] == []


def test_delta_missing_wait_is_not_a_change():
    s = MonitoringSettings()
    prev = {"rho": 0.5, "wip": 10.0, "wt_mean": None}
    assert _delta(prev, {"rho": 0.5, "wip": 10.0, "wt_mean": 9.0}, s)["exceeds"] == []
