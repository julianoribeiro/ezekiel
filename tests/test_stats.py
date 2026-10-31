import numpy as np
import pytest

from ezekiel.evaluation.stats import holm, non_inferiority, paired_compare


def test_holm_matches_manual():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_detects_improvement_and_uses_replica_pairs():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 30)
    treat = ref - 5 + rng.normal(0, 1, 30)       # melhora pareada de ~5%
    r = paired_compare(ref, treat, "lower", delta_rel=0.02)
    assert r["test"] == "t pareado"
    assert r["p_value"] < 0.001
    assert r["delta_pct"] == pytest.approx(-5, abs=1)
    assert r["practical"] == "melhoria relevante"


def test_identical_configurations_are_not_significant():
    x = np.arange(30, dtype=float)
    r = paired_compare(x, x.copy(), "lower")
    assert r["test"] == "degenerate" and r["p_value"] == 1.0


def test_non_inferiority():
    ref = np.full(30, 10.0) + np.linspace(-0.1, 0.1, 30)
    assert non_inferiority(ref, ref - 0.1, 0.05)["non_inferior"]
    assert not non_inferiority(ref, ref - 1.0, 0.05)["non_inferior"]


def test_required_replications_meets_half_width():
    from ezekiel.evaluation.stats import half_width, required_replications
    rng = np.random.default_rng(0)
    d = rng.normal(1.0, 2.0, 10)
    r = required_replications(d, h_target=0.5)
    sd = d.std(ddof=1)
    from scipy import stats as sps
    assert sps.t.ppf(0.975, r - 1) * sd / np.sqrt(r) <= 0.5
    assert sps.t.ppf(0.975, r - 2) * sd / np.sqrt(r - 1) > 0.5          # mínimo
    assert half_width(d) > 0
    assert required_replications(np.zeros(10), 0.1) == 10              # sem variância: o piloto basta


def test_manifest_records_code_data_and_settings(tmp_path):
    import json
    from ezekiel import manifest
    from ezekiel.experiments.hypotheses import ExperimentSettings
    data = tmp_path / "log.json"
    data.write_text("{}")
    (tmp_path / "log.meta.json").write_text('{"source": "teste"}')
    p = manifest.write(tmp_path / "m.json", ExperimentSettings(), [data])
    m = json.loads(p.read_text())
    assert len(m["code"]["sha256"]) == 64 and m["data"][0]["meta"]["content"]["source"] == "teste"
    assert m["settings"]["costs"]["seed_base"] == m["seeds"]["cost_estimation_base"]
