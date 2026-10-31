"""Execução das hipóteses H1--H4 sobre um log (introdução, Seção de
hipóteses; metodologia, "Delineamento experimental").

Fluxo por conjunto de dados:

  1. Janelas temporais. O fim do período ativo do log define a última
     janela de validação; para trás dela fica a janela de calibração.
     Com `folds > 1`, repete-se com origens anteriores (validação temporal
     progressiva, preservando a ordem cronológica).
  2. H1 — em cada dobra: calibra em [cal_start, split), simula a janela de
     validação [split, split + val_days) a partir do estado reconstruído em
     `split` e compara com o observado (evaluation/validation.py).
  3. Instância DCOP (dobra mais recente) — agentes = UNIDADES
     ORGANIZACIONAIS do modelo organizacional (`organization`; ver
     dcop/organization.py), cada uma controlando os estágios sob sua
     responsabilidade; estágios sem unidade decisória são ambiente.
     Restrição ativa entre os agentes pela Eq. de identificação operacional
     com persistência temporal; custos estimados por simulação
     (dcop/costs.py). O resultado documenta cada agente (responsabilidade,
     informação local, decisões, recursos, interfaces, evidência), a matriz
     de dependências e o estado ⟨ρ, L, W̄T⟩ de cada agente no instante da
     decisão.
  4. Configurações (todas simuladas no MESMO modelo, mesmo estado inicial,
     mesmas sementes — números aleatórios comuns):
        C0       política vigente (sem limite de WIP, FIFO)
        C1       decisões locais não coordenadas
        C2       coordenação distribuída (ADOPT, função com termos ToC)
        C2-sem-ToC  ADOPT com α = β = 0 (ablação para H3)
        C3       referência centralizada exata (mesma função de C2)
     O histórico observado NÃO é comparado diretamente com C1--C3
     (metodologia: só depois de verificado se o simulador reproduz C0).
  5. H2 — C2 vs C1: redução do P80 do lead time (primário) sem
     deterioração relevante da vazão (não inferioridade com margem).
  6. H3 — C2 vs C2-sem-ToC: melhora do P80 sem piora significativa de
     espera, WIP e eficiência de fluxo (secundárias).
  7. Monitoramento e reotimização (experiments/monitoring.py): ao longo da
     janela de validação, estado ⟨ρ, L, W̄T⟩ por agente a cada Δt; quando os
     limiares do protocolo são excedidos (ou a restrição ativa muda), nova
     calibração e nova execução do ADOPT, avaliada contra a configuração
     vigente no trecho seguinte.
  8. H4 (parte empírica) — custo, ciclos, mensagens, memória e lacuna de
     otimalidade do ADOPT em relação a C3. A varredura de escala fica em
     experiments/synthetic.py.

As decisões por conjunto de dados aqui são NÃO ajustadas; o ajuste de Holm
entre conjuntos de dados (família por hipótese) é feito na agregação
(scripts/run_hypotheses.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ezekiel.data.eventlog import EventLog
from ezekiel.dcop.adopt import solve_adopt
from ezekiel.dcop.centralized import solve_centralized
from ezekiel.dcop.costs import CostSettings, estimate_costs
from ezekiel.dcop.dependencies import DependencySettings, candidate_dependencies
from ezekiel.dcop.organization import Organization
from ezekiel.evaluation import stats as st
from ezekiel.evaluation.validation import ValidationCriteria, validate_window
from ezekiel.experiments.monitoring import MonitoringSettings, run_monitoring
from ezekiel.ocel.loader import COLUMN_TO_STAGE
from ezekiel.ocel.ocel2 import synchronization_indicators
from ezekiel.mining.discovery import discover
from ezekiel.mining.constraint import ConstraintSettings, agent_states, identify_constraint
from ezekiel.simulation.calibration import CalibrationSettings, at_state, build_model
from ezekiel.simulation.engine import Simulator

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSettings:
    cal_days: float = 365.0
    val_days: float = 180.0
    folds: int = 1
    active_end_quantile: float = 0.99
    end_date: str | None = None                 # força o fim (YYYY-MM-DD)
    reps: int = 30                              # réplicas da validação (H1) e R sem piloto
    # Estudo-piloto (metodologia, "Número de réplicas e precisão"): R0
    # réplicas de todas as configurações; R é o menor número com meia largura
    # do IC das diferenças pareadas do P80 (C2−C1 e C2−C2_sem_ToC) ≤
    # precision_rel × P80 médio da referência. None = R fixo (`reps`).
    # ATENÇÃO: precisão provisória — fixar com o orientador.
    pilot_reps: int | None = 10
    precision_rel: float = 0.025
    reps_max: int = 100
    eval_seed_base: int = 10_000
    # Modelo organizacional: nome em config/organizacao/ ou caminho para JSON.
    # No Jira as equipes não são observáveis — "por_papel" é premissa
    # declarada; "por_estagio" serve para comparar granularidades.
    organization: str = "por_papel"
    monitoring_window_days: float = 30.0         # Δt do estado ⟨ρ, L, W̄T⟩ dos agentes
    alpha: float = st.ALPHA
    # ATENÇÃO: margens provisórias — fixar a priori com o orientador.
    h2_throughput_ni_margin: float = 0.05       # não inferioridade da vazão (5%)
    practical_delta_p80: float = 0.05           # relevância prática do P80 (5%)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    constraint: ConstraintSettings = field(default_factory=ConstraintSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    dependencies: DependencySettings = field(default_factory=DependencySettings)
    validation: ValidationCriteria = field(default_factory=ValidationCriteria)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    # Calendário de trabalho: "calendar" (dias corridos, padrão — projetos
    # públicos com contribuidores globais) ou "business" (dias úteis, sem
    # fins de semana). Em "business", todas as durações destas configurações
    # (dadas em dias corridos) são convertidas para dias úteis (× 5/7).
    calendar: str = "calendar"
    # Perturbações para a análise de sensibilidade (1,0 = sem perturbação):
    # aplicadas ao modelo dos experimentos (C0–C3), não à validação de H1.
    demand_scale: float = 1.0          # × taxa de chegada
    service_factor: float = 1.0        # × escala do tempo de serviço


def in_log_time(s: ExperimentSettings) -> ExperimentSettings:
    """Converte as durações das configurações (dias corridos) para a
    unidade do calendário escolhido."""
    if s.calendar != "business":
        return s
    k = 5.0 / 7.0
    from dataclasses import replace
    return replace(
        s, cal_days=s.cal_days * k, val_days=s.val_days * k,
        monitoring_window_days=s.monitoring_window_days * k,
        constraint=replace(s.constraint, window_days=s.constraint.window_days * k),
        validation=replace(s.validation, throughput_block_days=s.validation.throughput_block_days * k),
        monitoring=replace(s.monitoring, window_days=s.monitoring.window_days * k,
                           constraint_days=s.monitoring.constraint_days * k,
                           min_interval_days=s.monitoring.min_interval_days * k),
    )


def _fmt_value(v: tuple) -> dict:
    out = {"wip_factor": v[0], "policy": v[1]}
    if len(v) > 2:
        out["capacity_share_first_stage"] = v[2]
    return out


def _fmt_config(org, assignment: dict) -> dict:
    stages = org.expand(assignment)
    return {
        "by_agent": {a: _fmt_value(v) for a, v in assignment.items()},
        "by_stage": {st: {"wip_limit": v[0], "policy": v[1], "servers": v[2]} for st, v in stages.items()},
    }


def _per_replica(results) -> dict[str, list[float]]:
    rows = [r.metrics() for r in results]
    return {k: [row[k] for row in rows] for k in rows[0]}


def _per_class(results) -> dict[str, dict[str, list[float]]]:
    """Métricas por classe de serviço, por réplica: {classe: {métrica: [r1..rR]}}."""
    rows = [r.metrics_by_class() for r in results]
    classes = sorted({c for row in rows for c in row})
    out = {}
    for c in classes:
        keys = next(iter(row[c] for row in rows if c in row))
        out[c] = {k: [row.get(c, {}).get(k, float("nan")) for row in rows] for k in keys}
    return out


def _end_time(log: EventLog, s: ExperimentSettings) -> float:
    if s.end_date:
        import pandas as pd
        from ezekiel.data.eventlog import to_days
        return log.to_days_of(float(to_days(pd.Series([pd.Timestamp(s.end_date, tz="UTC")]))[0]))
    return log.active_end(s.active_end_quantile)


def observed_utilization(log: EventLog, model, diag: dict, a: float, b: float) -> dict[str, float]:
    """Ocupação observada dos estágios finitos em [a, b): casos ativos
    (sem os estacionados, pelo mesmo corte de idade da calibração),
    limitados à capacidade, divididos pela capacidade."""
    out = {}
    for st in model.stages:
        if not st.servers:
            continue
        cap = diag["stage_params"].get(st.name, {}).get("active_age_cap_days") or np.inf
        conc = log.concurrency_samples(st.name, a, b, max_age=cap)
        out[st.name] = float(np.minimum(conc, st.servers).mean() / st.servers)
    return out


def run_h1(log: EventLog, s: ExperimentSettings, sim: Simulator) -> tuple[list[dict], dict]:
    """Retorna (dobras, contexto da dobra mais recente para os experimentos)."""
    t_end = _end_time(log, s)
    folds, ctx = [], None
    seeds = [s.eval_seed_base + r for r in range(s.reps)]
    for k in range(s.folds):
        val_end = t_end - k * s.val_days
        split = val_end - s.val_days
        cal_start = split - s.cal_days
        fold = {"fold": k, "calibration": [str(log.date(cal_start).date()), str(log.date(split).date())],
                "validation": [str(log.date(split).date()), str(log.date(val_end).date())]}
        if cal_start < log.t_min:
            fold["status"] = "janela de calibração anterior ao início do log"
            folds.append(fold)
            continue
        try:
            model, diag = build_model(log, cal_start, split, s.calibration, sim)
        except ValueError as exc:
            fold["status"] = f"calibração impossível: {exc}"
            folds.append(fold)
            continue
        vm = at_state(model, log, split, s.val_days)
        reps = sim.run(vm, seeds)
        util = observed_utilization(log, model, diag, split, val_end)
        res = validate_window(log.observed_metrics(split, val_end), reps, s.validation,
                              horizon=s.val_days, observed_utilization=util)
        fold.update(res)
        if model.classes:
            fold["by_class"] = {
                "observed": log.observed_metrics_by_class(split, val_end, model.classes),
                "simulated": {c: {k: st.describe(v) for k, v in m.items()}
                              for c, m in _per_class(reps).items()},
            }
        fold["calibration_diagnostics"] = diag
        fold["initial_cases"] = len(vm.initial_cases)
        folds.append(fold)
        if k == 0:
            ctx = {"model": model, "val_model": vm, "cal_start": cal_start, "split": split, "val_end": val_end}
    return folds, ctx


def _perturbed(vm, s: ExperimentSettings):
    if s.demand_scale == 1.0 and s.service_factor == 1.0:
        return vm
    import copy
    m = copy.copy(vm)
    m.stages = [copy.copy(x) for x in vm.stages]
    m.arrival_rate = vm.arrival_rate * s.demand_scale
    for x in m.stages:
        x.service_scale *= s.service_factor
    return m


def run_experiments(log: EventLog, ctx: dict, s: ExperimentSettings, sim: Simulator) -> dict:
    model, vm = ctx["model"], _perturbed(ctx["val_model"], s)
    organization = Organization.load(s.organization)
    org = organization.bind(vm)
    if not org.agents:
        return {"status": "sem unidades decisórias: nenhum agente do modelo organizacional controla "
                          "estágio de capacidade finita no recorte",
                "organization": organization.to_dict(), "excluded_agents": org.excluded,
                "environment": org.environment}

    cons = identify_constraint(log, org, ctx["cal_start"], ctx["split"], s.constraint)
    a_star = cons["active_constraint"]
    states = agent_states(log, org, ctx["split"], s.monitoring_window_days)

    # Grafo de restrições por evidências + teste de interação (dcop/dependencies.py)
    deps = candidate_dependencies(log, org, ctx["cal_start"], ctx["split"], s.dependencies)
    t0 = time.perf_counter()
    cm = estimate_costs(org, a_star, sim, s.costs, deps, s.dependencies.interaction_tau)
    cost_time = time.perf_counter() - t0

    inst_full = cm.instance(s.costs)
    inst_abl = cm.instance(s.costs, alpha=0.0, beta=0.0)
    c1 = cm.local_choice()
    ad_full = solve_adopt(inst_full, root=a_star)
    ad_abl = solve_adopt(inst_abl, root=a_star)
    c3 = solve_centralized(inst_full, root=a_star)

    base = {a: org.baseline_value(a) for a in org.agents}
    configs = {"C0": base, "C1": c1, "C2": ad_full.assignment,
               "C2_sem_ToC": ad_abl.assignment, "C3": c3.assignment}
    names = list(configs)
    models = [vm.with_assignment(org.expand(configs[n])) for n in names]
    if s.pilot_reps:
        pilot = sim.run_many(models, [s.eval_seed_base + r for r in range(s.pilot_reps)])
        pp = {n: [r.metrics()["lt_p80"] for r in res] for n, res in zip(names, pilot)}
        plan = {}
        for ref, treat in (("C1", "C2"), ("C2_sem_ToC", "C2")):
            d = np.asarray(pp[treat]) - np.asarray(pp[ref])
            h_t = s.precision_rel * abs(float(np.nanmean(pp[ref])))
            plan[f"{treat}-{ref}"] = {"sd_diff": float(np.nanstd(d, ddof=1)), "h_target": h_t,
                                      "required": st.required_replications(d, h_t, s.alpha, s.reps_max)}
        R = min(max([s.pilot_reps] + [p["required"] for p in plan.values()]), s.reps_max)
        extra = sim.run_many(models, [s.eval_seed_base + r for r in range(s.pilot_reps, R)]) if R > s.pilot_reps \
            else [[] for _ in names]
        sims = [p + e for p, e in zip(pilot, extra)]
        replication = {"rule": "meia largura do IC da diferença pareada do P80 ≤ precision_rel × P80 da referência",
                       "pilot_reps": s.pilot_reps, "precision_rel": s.precision_rel, "reps_max": s.reps_max,
                       "plan": plan, "reps": R, "capped": R == s.reps_max}
    else:
        R = s.reps
        sims = sim.run_many(models, [s.eval_seed_base + r for r in range(R)])
        replication = {"rule": "R fixo", "reps": R}
    per = {n: _per_replica(r) for n, r in zip(names, sims)}
    per_cls = {n: _per_class(r) for n, r in zip(names, sims)}

    # ── H2: C2 vs C1 ────────────────────────────────────────────────────
    h2_p80 = st.paired_compare(per["C1"]["lt_p80"], per["C2"]["lt_p80"], "lower", s.alpha, s.practical_delta_p80)
    h2_th = st.non_inferiority(per["C1"]["th_week"], per["C2"]["th_week"], s.h2_throughput_ni_margin, "higher", s.alpha)
    same_12 = configs["C1"] == configs["C2"]
    h2 = {
        "comparison": "C2 (ADOPT) vs C1 (decisões locais)",
        "primary_p80": h2_p80, "throughput_non_inferiority": h2_th,
        "identical_configurations": same_12,
        "validated_unadjusted": bool(h2_p80["p_value"] < s.alpha and h2_th["non_inferior"]),
        "c2_vs_c0_p80": st.paired_compare(per["C0"]["lt_p80"], per["C2"]["lt_p80"], "lower", s.alpha, s.practical_delta_p80),
        "c1_vs_c0_p80": st.paired_compare(per["C0"]["lt_p80"], per["C1"]["lt_p80"], "lower", s.alpha, s.practical_delta_p80),
    }

    # ── H3: C2 vs C2 sem termos ToC ────────────────────────────────────
    same_abl = configs["C2"] == configs["C2_sem_ToC"]
    h3_p80 = st.paired_compare(per["C2_sem_ToC"]["lt_p80"], per["C2"]["lt_p80"], "lower", s.alpha, s.practical_delta_p80)
    secondary = {}
    for metric, better in (("wt_mean", "lower"), ("wip_total", "lower"), ("fe", "higher")):
        worse = "higher" if better == "lower" else "lower"
        r = st.paired_compare(per["C2_sem_ToC"][metric], per["C2"][metric], worse, s.alpha)
        secondary[metric] = {"p_worse": r["p_value"], "delta_pct": r.get("delta_pct"),
                             "significantly_worse": bool(r["p_value"] < s.alpha)}
    h3 = {
        "comparison": "C2 (com termos ToC) vs C2 sem termos ToC (α=β=0)",
        "primary_p80": h3_p80, "secondary_no_deterioration": secondary,
        "identical_configurations": same_abl,
        "validated_unadjusted": bool(not same_abl and h3_p80["p_value"] < s.alpha
                                     and not any(v["significantly_worse"] for v in secondary.values())),
    }

    # ── H4 (empírica): ADOPT vs C3 ─────────────────────────────────────
    gap = (ad_full.cost - c3.cost) / abs(c3.cost) if c3.cost else 0.0
    h4 = {"adopt": ad_full.summary(), "centralized": c3.summary(),
          "optimality_gap": gap, "cost_estimation_time_s": cost_time,
          "cost_estimation_simulations": cm.sims_run,
          "adopt_ablation": ad_abl.summary()}

    monitoring = None
    if s.monitoring.enabled:
        try:
            monitoring = run_monitoring(log, organization, ctx["split"], ctx["val_end"],
                                        {"org": org, "a_star": a_star, "assignment": ad_full.assignment}, s, sim)
        except Exception as exc:  # noqa: BLE001 — o monitoramento não invalida H2–H4
            logger.exception("monitoramento falhou")
            monitoring = {"status": f"erro: {exc!r}"}

    return {
        "status": "ok",
        "organization": organization.name,
        "monitoring": monitoring,
        "agents": org.agents,
        "agent_documentation": org.documentation(),
        "excluded_agents": org.excluded,
        "environment": org.environment,
        "dependency_matrix": cm.dependencies,
        "agent_states_at_decision": states,
        "constraint": cons,
        "edges": [list(e) for e in cm.edges],
        "domains": {a: [list(v) for v in d] for a, d in cm.domains.items()},
        "cost_baseline": {k: v for k, v in cm.baseline.items() if k in ("wt_mean", "lt_p80", "th_week")},
        "configurations": {n: _fmt_config(org, c) for n, c in configs.items()},
        "metrics": {n: {k: st.describe(v) for k, v in per[n].items()} for n in names},
        # Classes de serviço: descritivo (não entra nas famílias de H2/H3).
        "metrics_by_class": {n: {c: {k: st.describe(v) for k, v in m.items()} for c, m in per_cls[n].items()}
                             for n in names},
        "by_class_c2_vs_c1_p80": {
            c: st.paired_compare(per_cls["C1"][c]["lt_p80"], per_cls["C2"].get(c, {}).get("lt_p80", []),
                                 "lower", s.alpha, s.practical_delta_p80)
            for c in per_cls["C1"] if c in per_cls["C2"]},
        "per_replica": per,
        "H2": h2, "H3": h3, "H4_empirical": h4,
        "replications": replication,
        "settings": {"costs": asdict(s.costs), "reps": R,
                     "gamma_effective": list(cm.gamma or s.costs.gamma),
                     "constraint_weights_effective": cons["weights"]},
    }


def run_dataset(log: EventLog, label: str, s: ExperimentSettings, sim: Simulator) -> dict:
    t0 = time.perf_counter()
    if s.calendar == "business":
        log = log.with_business_calendar()
    s_cal, s = s, in_log_time(s)
    out: dict = {"label": label, "log": log.describe()}
    if log.ocel is not None:
        # Camada centrada em objetos (metodologia, "Validação do log" e
        # "Caracterização empírica"): validação estrutural, projeções por
        # tipo de objeto e indicadores de sincronização entre objetos que a
        # simulação não representa (simplificação declarada e quantificada).
        done = set(log.profile.completion_statuses) | {c for c, st_ in COLUMN_TO_STAGE.items() if st_ == "Deploy"}
        out["object_centric"] = {
            "validation": log.ocel.validate(),
            "characterization": log.ocel.characterize(),
            "synchronization": synchronization_indicators(log.ocel, done),
        }
    folds, ctx = run_h1(log, s, sim)
    statuses = [f.get("status") for f in folds]
    approved = sum(x == "aprovada" for x in statuses)
    evaluable = sum(x in ("aprovada", "reprovada") for x in statuses)
    out["H1"] = {
        "folds": folds,
        "approved_folds": approved,
        "evaluable_folds": evaluable,
        "validated": bool(evaluable and approved == evaluable),
        "status": "dados insuficientes" if evaluable == 0 else ("aprovada" if approved == evaluable else "reprovada"),
    }
    if ctx is not None:
        # Descoberta de processos na janela de calibração (mining/discovery.py)
        try:
            out["process_discovery"] = discover(log, ctx["cal_start"], ctx["split"], ctx["model"].classes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("descoberta falhou")
            out["process_discovery"] = {"status": f"erro: {exc!r}"}
    if ctx is None or folds[0].get("status") == "dados insuficientes":
        out["experiments"] = {"status": "não executado — janela mais recente sem dados suficientes"}
    else:
        out["experiments"] = run_experiments(log, ctx, s, sim)
        out["experiments"]["simulator_validated_H1"] = out["H1"]["validated"]
    out["elapsed_s"] = time.perf_counter() - t0
    out["settings"] = {k: v for k, v in asdict(s_cal).items() if not isinstance(v, dict)}
    out["settings"]["time_unit"] = "dias úteis" if log.calendar == "business" else "dias corridos"
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if np.isnan(f) or np.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
