"""H1 — fidelidade da simulação em janela temporal de validação.

H1 (introdução): "o modelo de simulação calibrado com dados de treinamento
reproduz, em uma janela temporal de validação não utilizada na calibração,
as métricas de fluxo do sistema dentro de uma margem de erro previamente
estabelecida."

Validação de saída (metodologia, "Verificação e validação"): não se usa um
único limiar médio. São comparados, entre observado (janela de validação) e
simulado (R réplicas a partir do estado reconstruído no início da janela):

  * erro relativo de P50, P80 e P90 do lead time dos casos concluídos;
  * distância entre distribuições: KS e Wasserstein normalizada pela média
    observada (sobre o conjunto das réplicas);
  * erro relativo da vazão semanal e do WIP médio total;
  * cobertura de quantis: o quantil observado cai dentro do intervalo
    [2,5%; 97,5%] dos quantis das réplicas.

Métricas estendidas (metodologia, "Verificação e validação": "serão
comparadas distribuições de tempo de atravessamento e espera, vazão por
janela, WIP, utilização e frequências de rota"):

  * espera: erro relativo da média e Wasserstein normalizada. No Jira, a
    espera observada inclui status de fila dentro de estágios finitos
    (ex.: "Patch Available"), que o simulador trata como permanência no
    estágio — a comparação é reportada, mas é viesada por construção;
  * vazão por janela: conclusões em blocos de `throughput_block_days`;
    erro percentual médio e cobertura dos blocos observados pelo intervalo
    [2,5%; 97,5%] das réplicas;
  * WIP por estágio: maior erro relativo entre estágios com WIP observado
    relevante;
  * utilização: maior diferença absoluta, por estágio finito, entre a
    ocupação observada (casos ativos limitados à capacidade) e a simulada;
  * frequências de rota: distância de variação total entre as
    distribuições de próximo destino de cada estágio, ponderada pelas
    visitas observadas, e a maior entre estágios;
  * retrabalho: diferença absoluta na fração de movimentos para estágio
    anterior;
  * dependências entre objetos: não representadas no simulador — apenas
    sinalizadas (os indicadores ficam na caracterização OCEL).

As métricas estendidas têm critérios provisórios próprios, avaliados em
`checks_extended`. Só entram na decisão de H1 com `extended_enforced=True`
(a decidir na E1, com os demais critérios).

Os critérios de aceitação vêm de `ValidationCriteria` — devem ser fixados
a priori (antes da análise confirmatória). A H1 é aceita na janela só se
TODOS os critérios forem atendidos. Com validação temporal progressiva
(várias origens), reporta-se cada dobra e a fração de dobras aprovadas.

Suficiência de dados: janelas com menos de `min_completions` casos
concluídos observados são classificadas como "dados insuficientes" — não
como rejeição de H1 — e ficam fora da análise confirmatória.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class ValidationCriteria:
    # ATENÇÃO: valores provisórios — fixar com o orientador antes da análise confirmatória.
    max_rel_err_p50: float = 0.25
    max_rel_err_p80: float = 0.25
    max_rel_err_p90: float = 0.35
    max_rel_err_throughput: float = 0.20
    max_rel_err_wip: float = 0.15
    max_wasserstein_norm: float = 0.35
    min_quantile_coverage: float = 2 / 3
    min_completions: int = 30
    # Estendidos — provisórios; só decidem H1 se extended_enforced.
    max_rel_err_wait_mean: float = 0.35
    max_route_tv: float = 0.15
    max_abs_err_rework: float = 0.05
    max_abs_err_utilization: float = 0.15
    max_rel_err_stage_wip: float = 0.30
    min_wip_for_stage_check: float = 1.0
    throughput_block_days: float = 28.0
    min_block_throughput_coverage: float = 2 / 3
    extended_enforced: bool = False


def _rel(sim: float, obs: float) -> float:
    return abs(sim - obs) / abs(obs) if obs else float("inf")


def _tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _normalize(counts: dict, stages: set) -> dict:
    out: dict[str, float] = {}
    for k, v in counts.items():
        k = k if k in stages else "DONE"
        out[k] = out.get(k, 0.0) + v
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot else {}


def extended_metrics(observed: dict, replicas, crit: ValidationCriteria, horizon: float,
                     observed_utilization: dict | None = None) -> dict:
    """Métricas estendidas da validação de saída (ver docstring do módulo)."""
    out: dict = {}
    # espera
    w_obs = np.asarray(observed.get("waits", []), float)
    w_sim = np.concatenate([r.waits for r in replicas]) if replicas else np.array([])
    if len(w_obs) and len(w_sim):
        out["rel_err_wait_mean"] = _rel(float(w_sim.mean()), float(w_obs.mean()))
        out["wasserstein_wait_norm"] = float(stats.wasserstein_distance(w_obs, w_sim) / max(w_obs.mean(), 1e-9))
        out["wait_mean_obs"], out["wait_mean_sim"] = float(w_obs.mean()), float(w_sim.mean())

    # vazão por bloco
    blk = crit.throughput_block_days
    nb = int(horizon // blk)
    if nb >= 2:
        edges = np.arange(nb + 1) * blk
        obs_b = np.histogram(np.asarray(observed.get("done_times", []), float), edges)[0]
        sim_b = np.array([np.histogram(r.done_times, edges)[0] for r in replicas])
        lo, hi = np.quantile(sim_b, 0.025, axis=0), np.quantile(sim_b, 0.975, axis=0)
        mean_b = sim_b.mean(axis=0)
        ok = obs_b > 0
        out["throughput_blocks_obs"] = obs_b.tolist()
        out["throughput_blocks_sim_mean"] = mean_b.tolist()
        out["throughput_block_mape"] = float(np.mean(np.abs(mean_b[ok] - obs_b[ok]) / obs_b[ok])) if ok.any() else None
        out["throughput_block_coverage"] = float(np.mean((obs_b >= lo) & (obs_b <= hi)))

    # WIP por estágio
    wip_obs = observed.get("wip_avg", {})
    stages = list(replicas[0].wip_avg) if replicas else []
    errs = {}
    for s in stages:
        o = wip_obs.get(s, 0.0)
        if o >= crit.min_wip_for_stage_check:
            errs[s] = _rel(float(np.mean([r.wip_avg[s] for r in replicas])), o)
    out["rel_err_stage_wip"] = errs
    out["max_rel_err_stage_wip"] = max(errs.values()) if errs else None

    # utilização
    if observed_utilization:
        diffs = {s: abs(float(np.mean([r.utilization.get(s, 0.0) for r in replicas])) - u)
                 for s, u in observed_utilization.items() if s in replicas[0].utilization}
        out["utilization_obs"] = observed_utilization
        out["utilization_sim"] = {s: float(np.mean([r.utilization[s] for r in replicas]))
                                  for s in replicas[0].utilization}
        out["max_abs_err_utilization"] = max(diffs.values()) if diffs else None

    # rotas e retrabalho
    st_set = set(stages)
    obs_tr = observed.get("transitions", {})
    sim_tr: dict[str, dict[str, float]] = {}
    for r in replicas:
        for s, row in r.transitions.items():
            for k, v in row.items():
                sim_tr.setdefault(s, {})[k] = sim_tr.setdefault(s, {}).get(k, 0) + v
    tv, wts = {}, {}
    for s in stages:
        if s in obs_tr and s in sim_tr:
            tv[s] = _tv(_normalize(obs_tr[s], st_set), _normalize(sim_tr[s], st_set))
            wts[s] = sum(obs_tr[s].values())
    out["route_tv_by_stage"] = tv
    out["route_tv_weighted"] = (float(sum(tv[s] * wts[s] for s in tv) / sum(wts.values()))
                                if tv and sum(wts.values()) else None)
    out["route_tv_max"] = max(tv.values()) if tv else None
    n_obs = sum(sum(v.values()) for v in obs_tr.values())
    n_sim = sum(sum(v.values()) for v in sim_tr.values())
    if n_obs and n_sim:
        out["rework_rate_obs"] = observed.get("rework", 0) / n_obs
        out["rework_rate_sim"] = sum(r.rework for r in replicas) / n_sim
        out["abs_err_rework"] = abs(out["rework_rate_sim"] - out["rework_rate_obs"])
    out["object_dependencies"] = "não representadas no simulador (ver caracterização OCEL)"

    def chk(v, lim, le=True):
        return None if v is None else bool(v <= lim if le else v >= lim)
    out_checks = {
        "wait_mean": chk(out.get("rel_err_wait_mean"), crit.max_rel_err_wait_mean),
        "route_frequencies": chk(out["route_tv_weighted"], crit.max_route_tv),
        "rework": chk(out.get("abs_err_rework"), crit.max_abs_err_rework),
        "utilization": chk(out.get("max_abs_err_utilization"), crit.max_abs_err_utilization),
        "stage_wip": chk(out["max_rel_err_stage_wip"], crit.max_rel_err_stage_wip),
        "throughput_blocks": chk(out.get("throughput_block_coverage"), crit.min_block_throughput_coverage, le=False),
    }
    return {"metrics": out, "checks": out_checks}


def validate_window(observed: dict, replicas, crit: ValidationCriteria, horizon: float | None = None,
                    observed_utilization: dict | None = None) -> dict:
    lt_obs = np.asarray(observed["lead_times"], float)
    if len(lt_obs) < crit.min_completions:
        return {"status": "dados insuficientes", "observed_completions": int(len(lt_obs)),
                "criteria": asdict(crit)}

    qs = (0.5, 0.8, 0.9)
    obs_q = {q: float(np.quantile(lt_obs, q)) for q in qs}
    rep_q = {q: np.array([np.quantile(r.lead_times, q) for r in replicas if len(r.lead_times)]) for q in qs}
    sim_q = {q: float(np.mean(rep_q[q])) if len(rep_q[q]) else float("nan") for q in qs}
    covered = {q: bool(len(rep_q[q]) and np.quantile(rep_q[q], 0.025) <= obs_q[q] <= np.quantile(rep_q[q], 0.975))
               for q in qs}

    lt_sim = np.concatenate([r.lead_times for r in replicas]) if replicas else np.array([])
    ks = stats.ks_2samp(lt_obs, lt_sim) if len(lt_sim) else None
    wass = stats.wasserstein_distance(lt_obs, lt_sim) / max(lt_obs.mean(), 1e-9) if len(lt_sim) else float("inf")

    th_sim = float(np.mean([r.throughput_week for r in replicas]))
    wip_sim = float(np.mean([sum(r.wip_avg.values()) for r in replicas]))
    th_obs, wip_obs = observed["throughput_week"], observed["wip_total"]

    metrics = {
        "rel_err_p50": _rel(sim_q[0.5], obs_q[0.5]),
        "rel_err_p80": _rel(sim_q[0.8], obs_q[0.8]),
        "rel_err_p90": _rel(sim_q[0.9], obs_q[0.9]),
        "rel_err_throughput": _rel(th_sim, th_obs),
        "rel_err_wip": _rel(wip_sim, wip_obs),
        "wasserstein_norm": float(wass),
        "ks_statistic": float(ks.statistic) if ks else None,
        "quantile_coverage": sum(covered.values()) / len(qs),
    }
    checks = {
        "p50": metrics["rel_err_p50"] <= crit.max_rel_err_p50,
        "p80": metrics["rel_err_p80"] <= crit.max_rel_err_p80,
        "p90": metrics["rel_err_p90"] <= crit.max_rel_err_p90,
        "throughput": metrics["rel_err_throughput"] <= crit.max_rel_err_throughput,
        "wip": metrics["rel_err_wip"] <= crit.max_rel_err_wip,
        "wasserstein": metrics["wasserstein_norm"] <= crit.max_wasserstein_norm,
        "coverage": metrics["quantile_coverage"] >= crit.min_quantile_coverage,
    }
    ext = extended_metrics(observed, replicas, crit, horizon, observed_utilization) if horizon else None
    passed = all(checks.values())
    if ext is not None and crit.extended_enforced:
        passed = passed and all(v is not False for v in ext["checks"].values())
    return {
        "status": "aprovada" if passed else "reprovada",
        "checks": checks,
        "checks_extended": ext["checks"] if ext else None,
        "status_extended": (None if ext is None else
                            "aprovada" if all(checks.values()) and all(v is not False for v in ext["checks"].values())
                            else "reprovada"),
        "extended": ext["metrics"] if ext else None,
        "metrics": metrics,
        "observed": {"p50": obs_q[0.5], "p80": obs_q[0.8], "p90": obs_q[0.9],
                     "mean": float(lt_obs.mean()), "throughput_week": th_obs, "wip_total": wip_obs,
                     "completions": int(len(lt_obs))},
        "simulated": {"p50": sim_q[0.5], "p80": sim_q[0.8], "p90": sim_q[0.9],
                      "mean": float(lt_sim.mean()) if len(lt_sim) else None,
                      "throughput_week": th_sim, "wip_total": wip_sim,
                      "parked_mean": float(np.mean([r.parked for r in replicas]))},
        "quantile_covered": {str(q): v for q, v in covered.items()},
        "criteria": asdict(crit),
    }
