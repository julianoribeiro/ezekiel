"""Análise de sensibilidade e robustez (metodologia, "Análise de
sensibilidade e robustez").

  "Para cada parâmetro θ_k, serão definidos um valor de referência θ_k^0 e
   um intervalo plausível [θ_k^−, θ_k^+] [...]. Inicialmente, será
   conduzida uma análise local, na qual cada parâmetro será variado
   individualmente [...]. Posteriormente, parâmetros relevantes serão
   variados simultaneamente por meio de delineamentos fatoriais."

Cada fator tem um nível de referência (as configurações padrão) e níveis
baixo/alto. Os intervalos abaixo são PROVISÓRIOS (derivados das escolhas
de operacionalização e de variações plausíveis, não dos resultados): fixar
com o orientador na E1. Para cada variante, roda-se o pipeline completo
(H1, restrição ativa, DCOP, C0–C3, H2, H3) e registram-se:

  * mudança da restrição ativa e da configuração escolhida por C2;
  * Δ% do P80 de C2 vs C1 (H2) e de C2 vs C2_sem_ToC (H3);
  * robustez de DIREÇÃO: o sinal da diferença se mantém em relação à
    referência;
  * robustez PRÁTICA: a classificação (melhoria relevante, equivalência
    prática, inconclusivo, piora relevante) se mantém.

Fontes de variação cobertas (metodologia): pesos da função objetivo;
janela de monitoramento; limiar e persistência da restrição; chegadas e
serviço; demanda; capacidade; horizonte; associação multiobjeto
(acoplamentos); tratamento de casos censurados/anômalos (fim do período
ativo); e as escolhas estruturais desta rodada (calendário, multitarefa,
curvas por classe, pesos estimados).
"""

from __future__ import annotations

import copy
import itertools
import logging
import time
from dataclasses import replace

from ezekiel.data.eventlog import EventLog
from ezekiel.experiments.hypotheses import ExperimentSettings, run_dataset, to_jsonable
from ezekiel.simulation.engine import Simulator

logger = logging.getLogger(__name__)


def _set(s: ExperimentSettings, path: str, value) -> ExperimentSettings:
    """Cópia de `s` com o campo `a.b` = value."""
    s = copy.deepcopy(s)
    obj = s
    *head, last = path.split(".")
    for h in head:
        obj = getattr(obj, h)
    setattr(obj, last, value)
    return s


# fator → (campo, níveis {rótulo: valor}); a referência é o valor em ExperimentSettings()
FACTORS: dict[str, tuple[str, dict]] = {
    "gamma": ("costs.gamma", {"WT×2": (2.0, 1.0, 1.0), "P80×2": (1.0, 2.0, 1.0), "TH×2": (1.0, 1.0, 2.0)}),
    "gamma_mode": ("costs.gamma_mode", {"calibração": "calibration"}),
    "alpha": ("costs.alpha", {"0": 0.0, "0,25": 0.25, "1": 1.0}),
    "beta": ("costs.beta", {"0": 0.0, "0,25": 0.25, "1": 1.0}),
    "constraint_weights": ("constraint.weight_mode", {"entropia": "entropy"}),
    "constraint_window": ("constraint.window_days", {"15 d": 15.0, "60 d": 60.0}),
    "monitoring_window": ("monitoring.window_days", {"15 d": 15.0, "60 d": 60.0}),
    "monitoring_persistence": ("monitoring.persistence", {"1": 1, "3": 3}),
    "demand": ("demand_scale", {"−20%": 0.8, "+20%": 1.2}),
    "service": ("service_factor", {"−10%": 0.9, "+10%": 1.1}),
    "capacity_quantile": ("calibration.server_quantile", {"P80": 0.8, "P95": 0.95}),
    "horizon": ("val_days", {"90 d": 90.0, "270 d": 270.0}),
    "calibration_window": ("cal_days", {"180 d": 180.0, "730 d": 730.0}),
    "active_end": ("active_end_quantile", {"0,95": 0.95, "1,0": 1.0}),
    "couplings": ("calibration.synchronization+shared_people", {"ligados": True}),
    "calendar": ("calendar", {"dias úteis": "business"}),
    "multitask": ("calibration.multitask", {"estimado": "estimated", "κ=0,2": 0.2}),
    "by_class": ("calibration.by_class", {"curvas por classe": True}),
}


def variant(s: ExperimentSettings, factor: str, level) -> ExperimentSettings:
    path, _ = FACTORS[factor]
    if path == "calibration.synchronization+shared_people":
        s = _set(s, "calibration.synchronization", level)
        return _set(s, "calibration.shared_people", level)
    return _set(s, path, level)


def _summary(res: dict) -> dict:
    e = res.get("experiments", {})
    out = {"H1": res["H1"]["status"]}
    if e.get("status") != "ok":
        out["experiments"] = e.get("status")
        return out
    h2, h3 = e["H2"]["primary_p80"], e["H3"]["primary_p80"]
    out.update(
        active_constraint=e["constraint"]["active_constraint"],
        c2=e["configurations"]["C2"]["by_agent"],
        edges=e["edges"],
        H2={"delta_pct": h2.get("delta_pct"), "p": h2.get("p_value"), "practical": h2.get("practical"),
            "validated": e["H2"]["validated_unadjusted"]},
        H3={"delta_pct": h3.get("delta_pct"), "p": h3.get("p_value"), "practical": h3.get("practical"),
            "validated": e["H3"]["validated_unadjusted"]},
        reps=e.get("replications", {}).get("reps"),
    )
    return out


def _sign(x) -> int:
    return 0 if x is None or abs(x) < 1e-9 else (1 if x > 0 else -1)


def _robustness(ref: dict, var: dict) -> dict:
    if "H2" not in ref or "H2" not in var:
        return {}
    out = {"same_constraint": ref["active_constraint"] == var["active_constraint"],
           "same_c2": ref["c2"] == var["c2"]}
    for h in ("H2", "H3"):
        out[f"{h}_direction_same"] = _sign(ref[h]["delta_pct"]) == _sign(var[h]["delta_pct"])
        out[f"{h}_practical_same"] = ref[h]["practical"] == var[h]["practical"]
        out[f"{h}_decision_same"] = ref[h]["validated"] == var[h]["validated"]
    return out


def local_analysis(log: EventLog, label: str, base: ExperimentSettings, sim: Simulator,
                   factors: list[str] | None = None, progress=None) -> dict:
    """Um fator por vez, demais na referência."""
    t0 = time.perf_counter()
    ref = _summary(run_dataset(log, label, base, sim))
    rows = []
    for f in factors or list(FACTORS):
        for lab, val in FACTORS[f][1].items():
            t1 = time.perf_counter()
            try:
                v = _summary(run_dataset(log, label, variant(base, f, val), sim))
            except Exception as exc:  # noqa: BLE001
                v = {"error": repr(exc)}
            row = {"factor": f, "level": lab, "value": val, "result": v,
                   "robustness": _robustness(ref, v), "elapsed_s": time.perf_counter() - t1}
            rows.append(row)
            if progress:
                progress(row)
    return to_jsonable({"label": label, "reference": ref, "local": rows, "elapsed_s": time.perf_counter() - t0})


def factorial_analysis(log: EventLog, label: str, base: ExperimentSettings, sim: Simulator,
                       factors: dict[str, tuple[str, str]], progress=None) -> dict:
    """Fatorial completo 2^k: `factors` = {fator: (nível baixo, nível alto)},
    com rótulos de FACTORS ("ref" = valor de referência)."""
    names = list(factors)
    rows = []
    for combo in itertools.product((0, 1), repeat=len(names)):
        s = base
        levels = {}
        for f, hi in zip(names, combo):
            lab = factors[f][hi]
            levels[f] = lab
            if lab != "ref":
                s = variant(s, f, FACTORS[f][1][lab])
        try:
            r = _summary(run_dataset(log, label, s, sim))
        except Exception as exc:  # noqa: BLE001
            r = {"error": repr(exc)}
        rows.append({"levels": levels, "result": r})
        if progress:
            progress(rows[-1])
    # efeitos principais e interações de 2ª ordem sobre o Δ% de H2
    def y(r):
        return (r["result"].get("H2") or {}).get("delta_pct")
    effects = {}
    ys = [y(r) for r in rows]
    if all(v is not None for v in ys):
        k = len(names)
        signs = list(itertools.product((-1, 1), repeat=k))
        for i, f in enumerate(names):
            effects[f] = sum(sg[i] * v for sg, v in zip(signs, ys)) / 2 ** (k - 1)
        for i, j in itertools.combinations(range(k), 2):
            effects[f"{names[i]}×{names[j]}"] = sum(sg[i] * sg[j] * v for sg, v in zip(signs, ys)) / 2 ** (k - 1)
    return to_jsonable({"label": label, "factors": factors, "runs": rows, "effects_on_H2_delta_pct": effects})


def quick(s: ExperimentSettings) -> ExperimentSettings:
    """Domínio e réplicas reduzidos (exploração)."""
    from ezekiel.dcop.costs import CostSettings
    s = replace(s, reps=10, pilot_reps=None)
    s.costs = CostSettings(wip_factors=(1.0, 2.0), policies=("FIFO", "urgency"), reps=3)
    s.monitoring = replace(s.monitoring, enabled=False)
    return s
