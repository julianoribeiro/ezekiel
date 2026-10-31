"""Calibração do simulador a partir de uma janela de calibração do log.

Separação temporal (metodologia, "Estimação dos parâmetros"): todos os
parâmetros vêm de [cal_start, cal_end); a janela de validação começa em
cal_end e nunca é usada aqui. O estado inicial da simulação de validação é
o estado do log em cal_end — informação disponível no instante da decisão,
sem vazamento dos desfechos da janela de validação.

Parâmetros estimados:
  * taxa de chegada e estágio de entrada (casos iniciados na janela);
  * matriz de roteamento (visitas encerradas na janela);
  * distribuição empírica de permanência por estágio (visitas encerradas
    na janela) — reamostrada pelo simulador;
  * capacidade de cada estágio finito = quantil (padrão P90) do nº de casos
    simultâneos observado na janela;
  * fator de escala do tempo de serviço por estágio finito, ajustado por
    simulação da própria janela de calibração para que a permanência média
    simulada reproduza a observada (o tempo empírico já contém a espera
    histórica; sem o ajuste, a fila simulada seria contada duas vezes).

Estágios com fila e serviço separados (perfil com `split_wait_service` e
evidência suficiente no estágio — colunas de fila e de trabalho
observadas): a curva de sobrevivência usa só o tempo em colunas de
TRABALHO; a capacidade inicial é o quantil da simultaneidade em colunas de
trabalho; e, em vez do fator de escala, ajusta-se o número de servidores
para que a espera simulada no estágio reproduza a espera observada nas
colunas de fila. A fila deixa de ser um insumo e passa a emergir da
disputa por capacidade — condição para que limites de WIP e políticas de
fila tenham efeito no modelo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ezekiel.data.eventlog import DONE, WORK_STAGES, EventLog
from ezekiel.simulation.engine import SimModel, Simulator, StageModel

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSettings:
    # Atrasos puros (sem recurso limitante identificável no log): a fila de
    # entrada e a espera pela janela de implantação.
    unbounded_stages: tuple[str, ...] = ("Backlog",)
    server_quantile: float = 0.9
    min_stage_visits: int = 20
    scale_iterations: int = 6
    scale_reps: int = 5
    min_service_days: float = 1.0 / 1440  # 1 minuto
    # Classes de serviço (tipo do item): classes com ao menos min_class_cases
    # casos iniciados na janela; `by_class` estima curvas de permanência por
    # classe (estágios sem fila/serviço separados, com visitas suficientes).
    min_class_cases: int = 30
    max_classes: int = 4
    by_class: bool = False
    # Multitarefa: "off" (κ = 0, padrão), "estimated" (κ estimado na janela
    # de calibração, usado só se a associação for positiva e significativa)
    # ou um valor fixo de κ.
    multitask: str | float = "off"
    multitask_alpha: float = 0.05
    # Acoplamentos (engine.py): sincronização entre objetos (bloqueios do
    # OCEL) e pessoas compartilhadas entre estágios (pool de servidores).
    # Desligados por padrão — a decidir na E1 e examinar na sensibilidade.
    synchronization: bool = False
    min_sync_prob: float = 0.01
    min_sync_visits: int = 5
    shared_people: bool = False
    pool_min_jaccard: float = 0.10
    pool_max_share: float = 0.5


def km_from_durations(left: np.ndarray, obs: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier com truncamento à esquerda (`left`), duração observada
    (`obs`) e indicador de evento. Retorna (instantes, S(t) após cada um)."""
    ev_t = np.sort(obs[event])
    if len(ev_t) == 0:
        return np.array([]), np.array([])
    times, d = np.unique(ev_t, return_counts=True)
    left_s, obs_s = np.sort(left), np.sort(obs)
    # Em risco em τ: entrou antes de τ (left < τ) e ainda não saiu (obs ≥ τ).
    # Para τ = 0 (durações nulas — ex.: passagem pela coluna de fila sem
    # esperar), os casos sem truncamento (left = 0) também estão em risco;
    # sem esse ajuste, o primeiro evento em 0 zerava a curva inteira.
    entered = np.searchsorted(left_s, times, side="left")
    entered = np.where(times <= 0, np.searchsorted(left_s, 0.0, side="right"), entered)
    at_risk = entered - np.searchsorted(obs_s, times, side="left")
    surv = np.cumprod(1.0 - d / np.maximum(at_risk, d))
    return times, surv


def kaplan_meier(enter: np.ndarray, exit_: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Curva de sobrevivência da permanência no estágio, estimada em [a, b).

    Cada visita que esteve no estágio em algum momento da janela contribui:
      truncamento à esquerda: entra no conjunto de risco com a idade que já
        tinha em a (visitas iniciadas antes da janela);
      evento: saída dentro da janela;
      censura à direita: visita ainda no estágio em b (ou nunca encerrada).
    Retorna (instantes de evento, S(t) logo após cada instante).
    """
    exit_ = np.where(np.isnan(exit_), np.inf, exit_)
    m = (enter < b) & (exit_ >= a)
    enter, exit_ = enter[m], exit_[m]
    left = np.maximum(a - enter, 0.0)
    is_event = exit_ < b
    obs = np.where(is_event, exit_, b) - enter
    return km_from_durations(left, obs, is_event)


def phase_kaplan_meier(log: EventLog, stage: str, phase: str, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """KM do tempo acumulado numa fase (work = colunas de trabalho; queue =
    colunas de fila) por visita, com truncamento à esquerda e censura."""
    v = log.visits[log.visits["stage"] == stage]
    exit_ = v["exit"].fillna(np.inf).to_numpy()
    m = (v["enter"].to_numpy() < b) & (exit_ >= a)
    v = v[m]
    exit_ = exit_[m]
    is_event = exit_ < b
    at_a = log.phase_time_until(stage, phase, a)
    at_b = log.phase_time_until(stage, phase, b)
    ids = v["visit_id"].to_numpy()
    total = v["service" if phase == "work" else "wait"].to_numpy()
    left = at_a.reindex(ids).fillna(0.0).to_numpy()
    obs = np.where(is_event, total, at_b.reindex(ids).fillna(0.0).to_numpy())
    obs = np.maximum(obs, left)
    return km_from_durations(left, obs, is_event)


def service_kaplan_meier(log: EventLog, stage: str, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """KM do tempo de TRABALHO por visita (exclui o tempo em colunas de fila)."""
    return phase_kaplan_meier(log, stage, "work", a, b)


def initial_cases(log: EventLog, t: float, stage_models: list[StageModel]) -> list[tuple]:
    """Estado inicial: (estágio, idade do caso, idade da visita, serviço já
    decorrido, estacionado, fila já decorrida, espera acumulada). Em estágios
    com fila/serviço separados, o serviço decorrido é o tempo já passado em
    colunas de trabalho, e casos parados na fila há mais que `queue_age_cap`
    ficam estacionados. A espera acumulada antes de t entra na espera do
    caso, como a idade entra no lead time: sem ela, a espera simulada dos
    casos iniciais só contaria o trecho dentro do horizonte."""
    by = {s.name: s for s in stage_models}
    st = log.state_at(t)
    st = st[st["stage"].isin(by)]
    wait0 = log.case_wait_until(t)
    out = []
    split_stages = [n for n, m in by.items() if m.split]
    work_done = {n: log.phase_time_until(n, "work", t) for n in split_stages}
    queue_done = {n: log.phase_time_until(n, "queue", t) for n in split_stages}
    for row in st.itertuples(index=False):
        m = by[row.stage]
        w0 = float(wait0.get(row.case, 0.0))
        if m.split:
            elapsed = float(work_done[row.stage].get(row.visit_id, 0.0))
            q_el = float(queue_done[row.stage].get(row.visit_id, 0.0))
            in_queue = row.phase == "queue"
            parked = bool(in_queue and q_el > m.queue_age_cap)
            # em fila: atraso exógeno residual condicionado ao já esperado;
            # em trabalho: a fila já passou (None)
            out.append((row.stage, float(row.case_age), float(row.stage_age), elapsed, parked,
                        q_el if in_queue else None, w0))
        else:
            out.append((row.stage, float(row.case_age), float(row.stage_age), float(row.stage_age), False,
                        None, w0))
    return out


def estimate_multitask(log: EventLog, stage: str, servers: int, a: float, b: float,
                       bins: int = 5, alpha: float = 0.05) -> dict:
    """κ da penalidade de multitarefa a partir das visitas encerradas em
    [a, b): x = casos no estágio na entrada / servidores; y = permanência.
    Agrupa x em quantis, toma a mediana de y por grupo e ajusta, pela origem,
    m_j/m_0 − 1 = κ·max(0, x̄_j − 1). Usa κ só se a correlação de Spearman
    entre x e y for positiva e significativa (associação, não causalidade:
    filas longas também esperam mais — por isso o κ é apenas um candidato,
    a ser examinado na análise de sensibilidade)."""
    from scipy import stats as sps
    v = log.visits_ending_in(a, b)
    v = v[v["stage"] == stage]
    if len(v) < 5 * bins or not servers:
        return {"kappa": 0.0, "status": "visitas insuficientes", "n": int(len(v))}
    enters = np.sort(log.visits.loc[log.visits["stage"] == stage, "enter"].to_numpy())
    exits = np.sort(log.visits.loc[log.visits["stage"] == stage, "exit"].fillna(np.inf).to_numpy())
    e = v["enter"].to_numpy()
    load = (np.searchsorted(enters, e, side="left") - np.searchsorted(exits, e, side="right")) / servers
    y = (v["exit"] - v["enter"]).to_numpy()
    rho, p = sps.spearmanr(load, y)
    q = np.quantile(load, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, load, side="right") - 1, 0, bins - 1)
    xs, ms = [], []
    for j in range(bins):
        m = idx == j
        if m.sum():
            xs.append(float(load[m].mean()))
            ms.append(float(np.median(y[m])))
    m0 = ms[0] if ms and ms[0] > 0 else None
    exc = np.maximum(np.asarray(xs) - 1.0, 0.0)
    kappa = 0.0
    if m0 and exc.dot(exc) > 0:
        kappa = float(exc.dot(np.asarray(ms) / m0 - 1.0) / exc.dot(exc))
    used = bool(np.isfinite(rho) and rho > 0 and p < alpha and kappa > 0)
    return {"kappa": min(max(kappa, 0.0), 1.0) if used else 0.0, "kappa_raw": kappa,
            "spearman": float(rho) if np.isfinite(rho) else None, "p_value": float(p) if np.isfinite(p) else None,
            "used": used, "n": int(len(v))}


BLOCKING_QUALIFIERS = ("is blocked by", "depends on")


def sync_rules(log: EventLog, stages: list[str], blocked_stages: list[str], a: float, b: float,
               min_prob: float = 0.01, min_visits: int = 5) -> dict[str, list[tuple[str, float]]]:
    """p(S, S') = fração das visitas a S encerradas em [a, b) cujo item tem
    um bloqueador (relação O2O explícita) com visita a S' sobreposta."""
    if log.ocel is None:
        return {}
    q = log.ocel.o2o
    pairs = q[q["qualifier"].isin(BLOCKING_QUALIFIERS)][["source_id", "target_id"]]
    if pairs.empty:
        return {}
    pairs = pairs.rename(columns={"source_id": "item", "target_id": "blocker"}).drop_duplicates()
    v = log.visits.assign(item=log.visits["case"].str.rsplit("#", n=1).str[0])
    ended = v[(v["exit"] >= a) & (v["exit"] < b) & v["stage"].isin(blocked_stages)]
    if ended.empty:
        return {}
    bv = v[v["stage"].isin(stages)][["item", "stage", "enter", "exit"]].rename(
        columns={"item": "blocker", "stage": "b_stage", "enter": "b_enter", "exit": "b_exit"})
    m = ended[["visit_id", "item", "stage", "enter", "exit"]].merge(pairs, on="item").merge(bv, on="blocker")
    m = m[(m["b_stage"] != m["stage"]) & (m["b_enter"] < m["exit"])
          & (m["b_exit"].fillna(np.inf) > m["enter"])]
    hits = m.drop_duplicates(["visit_id", "b_stage"]).groupby(["stage", "b_stage"]).size()
    n = ended["stage"].value_counts()
    rules: dict[str, list[tuple[str, float]]] = {}
    for (s, s2), k in hits.items():
        p = float(k) / float(n[s])
        if k >= min_visits and p >= min_prob:
            rules.setdefault(s, []).append((s2, p))
    return {s: sorted(r, key=lambda x: -x[1]) for s, r in rules.items()}


def people_pools(log: EventLog, finite: list[str], a: float, b: float,
                 min_jaccard: float = 0.10, max_share: float = 0.5) -> tuple[dict, dict]:
    """Pools de pessoas compartilhadas entre estágios finitos (Jaccard das
    pessoas anonimizadas que movem itens para fora de cada estágio em
    [a, b) ≥ `min_jaccard`). Retorna ({estágio: (pool, fração)}, Jaccards)."""
    from ezekiel.dcop.dependencies import _actors
    people = {s: _actors(log, [s], a, b) for s in finite}
    jac = {}
    for i, x in enumerate(finite):
        for y in finite[i + 1:]:
            u = people[x] | people[y]
            if u:
                jac[(x, y)] = len(people[x] & people[y]) / len(u)
    edges = [e for e, j in jac.items() if j >= min_jaccard]
    comp = {s: s for s in finite}

    def root(s):
        while comp[s] != s:
            s = comp[s]
        return s
    for x, y in edges:
        comp[root(x)] = root(y)
    groups: dict[str, list[str]] = {}
    for s in finite:
        groups.setdefault(root(s), []).append(s)
    out = {}
    for k, (_, members) in enumerate(sorted((g, m) for g, m in groups.items() if len(m) > 1)):
        for s in members:
            share = max(j for e, j in jac.items() if s in e and j >= min_jaccard)
            out[s] = (f"pool-{k + 1}", min(share, max_share))
    return out, {f"{x}|{y}": j for (x, y), j in jac.items()}


def initial_classes(log: EventLog, t: float, stage_models: list[StageModel], classes: list[str]) -> list[str]:
    """Classe de cada caso inicial, na mesma ordem de `initial_cases`."""
    names = {s.name for s in stage_models}
    st = log.state_at(t)
    st = st[st["stage"].isin(names)]
    return log.case_classes(classes).reindex(st["case"]).fillna("Outros").tolist()


def build_model(
    log: EventLog,
    cal_start: float,
    cal_end: float,
    settings: CalibrationSettings,
    simulator: Simulator | None = None,
) -> tuple[SimModel, dict]:
    """Calibra o modelo em [cal_start, cal_end). Retorna (modelo, diagnóstico).

    O modelo devolvido começa em cal_end (estado reconstruído) com horizonte
    a ser definido pelo chamador.
    """
    days = cal_end - cal_start
    vis = log.visits_ending_in(cal_start, cal_end)
    counts = vis["stage"].value_counts()
    stages = [s for s in WORK_STAGES if counts.get(s, 0) >= settings.min_stage_visits]
    if not stages:
        raise ValueError("Nenhum estágio com visitas suficientes na janela de calibração.")

    started = log.started_in(cal_start, cal_end)
    entry = started["entry_stage"][started["entry_stage"].isin(stages)].value_counts(normalize=True)
    arrival_rate = len(started) / days

    routing: dict[str, dict[str, float]] = {}
    for s in stages:
        nxt = vis.loc[vis["stage"] == s, "next"]
        nxt = nxt.where(nxt.isin(stages + [DONE]), DONE)  # estágio residual → conclusão
        p = nxt.value_counts(normalize=True)
        routing[s] = {k: float(v) for k, v in p.items()}

    classes = log.service_classes(cal_start, cal_end, settings.min_class_cases, settings.max_classes)
    case_cls = log.case_classes(classes)
    class_probs = {}
    if classes:
        cp = case_cls.reindex(started["case"]).value_counts(normalize=True)
        class_probs = {str(k): float(v) for k, v in cp.items()}

    stage_models: list[StageModel] = []
    diag_stages: dict[str, dict] = {}
    all_v = log.visits
    for s in stages:
        sv_end = vis[vis["stage"] == s]
        dur = (sv_end["exit"] - sv_end["enter"]).to_numpy()
        unbounded = s in settings.unbounded_stages
        split = bool(
            log.profile.split_wait_service and not unbounded
            and (sv_end["service"] > 0).sum() >= settings.min_stage_visits
            and (sv_end["wait"] > 0).sum() >= settings.min_stage_visits
        )
        q_times = q_surv = np.array([])
        if split:
            times, surv = service_kaplan_meier(log, s, cal_start, cal_end)
            q_times, q_surv = phase_kaplan_meier(log, s, "queue", cal_start, cal_end)
            q_times = np.maximum(q_times, settings.min_service_days)
            served = sv_end["service"].to_numpy()
            active_cap = float(np.quantile(served, 0.95)) if len(served) else np.inf
            conc = log.work_concurrency(s, cal_start, cal_end, max_age=active_cap)
            queue_cap = float(np.quantile(sv_end["wait"].to_numpy(), 0.95))
        else:
            sv = all_v[all_v["stage"] == s]
            times, surv = kaplan_meier(sv["enter"].to_numpy(), sv["exit"].to_numpy(), cal_start, cal_end)
            # Capacidade: simultaneidade dos casos ATIVOS — exclui os que já
            # passaram do P95 das permanências concluídas (os "estacionados",
            # que não consomem capacidade).
            active_cap = float(np.quantile(dur, 0.95)) if len(dur) else np.inf
            conc = log.concurrency_samples(s, cal_start, cal_end, max_age=active_cap)
            queue_cap = math.inf
        times = np.maximum(times, settings.min_service_days)
        servers = None if unbounded else max(1, int(math.ceil(np.quantile(conc, settings.server_quantile))))
        class_km: dict[str, tuple[list[float], list[float]]] = {}
        if settings.by_class and classes and not split:
            sv = all_v[all_v["stage"] == s]
            sv_cls = case_cls.reindex(sv["case"]).to_numpy()
            ended_cls = case_cls.reindex(sv_end["case"]).to_numpy()
            for c in classes + ["Outros"]:
                if (ended_cls == c).sum() < settings.min_stage_visits:
                    continue
                m = sv_cls == c
                ct, cs = kaplan_meier(sv["enter"].to_numpy()[m], sv["exit"].to_numpy()[m], cal_start, cal_end)
                class_km[c] = (np.maximum(ct, settings.min_service_days).tolist(), cs.tolist())
        st_model = StageModel(
            name=s, servers=servers, km_times=times.tolist(), km_surv=surv.tolist(),
            counts_as_wait=unbounded, split=split, queue_age_cap=queue_cap,
            queue_km_times=q_times.tolist(), queue_km_surv=q_surv.tolist(), class_km=class_km,
        )
        mt = None
        if servers and settings.multitask != "off":
            if settings.multitask == "estimated":
                mt = estimate_multitask(log, s, servers, cal_start, cal_end, alpha=settings.multitask_alpha)
                st_model.multitask = mt["kappa"]
            else:
                st_model.multitask = float(settings.multitask)
        stage_models.append(st_model)
        diag_stages[s] = {
            "servers_initial": servers,
            "split_wait_service": split,
            "visits_ended": int(len(dur)),
            "sojourn_mean_obs_ended": float(dur.mean()) if len(dur) else None,
            "sojourn_p50_obs_ended": float(np.median(dur)) if len(dur) else None,
            "queue_wait_mean_obs_ended": float(sv_end["wait"].mean()) if len(sv_end) else None,
            "service_mean_obs_ended": float(sv_end["service"].mean()) if len(sv_end) else None,
            "km_p_parked": st_model.p_parked,
            "km_mean_if_leaves": st_model.mean_service,
            "active_age_cap_days": active_cap,
            "queue_age_cap_days": queue_cap if math.isfinite(queue_cap) else None,
            "concurrency_active_mean": float(conc.mean()),
            "concurrency_active_p90": float(np.quantile(conc, 0.9)),
            "class_curves": sorted(class_km),
            "multitask_kappa": st_model.multitask,
            "multitask_estimation": mt,
        }

    sync, pools, jac = {}, {}, {}
    if settings.synchronization:
        sync = sync_rules(log, stages, [s for s in stages if s not in settings.unbounded_stages],
                          cal_start, cal_end, settings.min_sync_prob, settings.min_sync_visits)
    if settings.shared_people:
        finite_names = [m.name for m in stage_models if m.servers]
        pools, jac = people_pools(log, finite_names, cal_start, cal_end,
                                  settings.pool_min_jaccard, settings.pool_max_share)
        for m in stage_models:
            if m.name in pools:
                m.pool, m.pool_share = pools[m.name]

    model = SimModel(
        stages=stage_models,
        arrival_rate=arrival_rate,
        entry_probs={k: float(v) for k, v in entry.items()},
        routing=routing,
        class_probs=class_probs,
        classes=classes,
        days_per_week=log.days_per_week,
        sync=sync,
    )

    # ── Ajuste, simulando a própria janela de calibração ────────────────
    #   estágios finitos sem separação: fator de escala do serviço
    #     (permanência simulada ≈ observada);
    #   estágios com fila/serviço separados: escala da componente exógena
    #     da espera (espera simulada total ≈ espera observada nas colunas de
    #     fila). A capacidade fica no quantil da simultaneidade observada em
    #     colunas de trabalho — ajustá-la para reproduzir a espera levava a
    #     um único servidor por estágio, artefato sem sentido operacional:
    #     a espera observada não é explicada pela disputa por capacidade.
    finite = [s for s in model.stages if s.servers]
    scale_trace = []
    if finite and simulator is not None and settings.scale_iterations > 0:
        cal_model = SimModel(
            stages=model.stages, arrival_rate=arrival_rate, entry_probs=model.entry_probs,
            routing=routing, initial_cases=initial_cases(log, cal_start, model.stages), horizon=days,
            class_probs=class_probs,
            initial_classes=initial_classes(log, cal_start, model.stages, classes) if classes else [],
            days_per_week=log.days_per_week, sync=sync,
        )
        obs_m = log.observed_metrics(cal_start, cal_end)
        obs_soj, obs_wait = obs_m["sojourn_mean"], obs_m["queue_wait_mean"]
        eps = 0.05  # dias — evita razões instáveis quando a espera é quase nula
        for it in range(settings.scale_iterations):
            res = simulator.run(cal_model, list(range(settings.scale_reps)))
            gaps = {}
            for st in finite:
                if st.split:
                    tot = [r.stage_wait[st.name] for r in res if st.name in r.stage_wait]
                    endo = [r.stage_endo_wait[st.name] for r in res if st.name in r.stage_endo_wait]
                    if not tot or st.name not in obs_wait:
                        continue
                    obs_w = obs_wait[st.name]
                    tot_m, endo_m = float(np.mean(tot)), float(np.mean(endo))
                    gaps[st.name] = (obs_w + eps) / (tot_m + eps)
                    # (1) a disputa por capacidade, sozinha, não pode exceder a
                    #     espera observada: se exceder metade dela, há
                    #     servidores de menos — aumenta-se a capacidade;
                    if endo_m > 0.5 * obs_w + eps:
                        st.servers = int(math.ceil(st.servers * min(((endo_m + eps) / (0.5 * obs_w + eps)) ** 0.5, 2.0)))
                    # (2) a componente exógena cobre o restante da espera.
                    target = max(obs_w - min(endo_m, 0.5 * obs_w), 0.0)
                    exo_m = max(tot_m - endo_m, 0.0)
                    st.queue_scale *= min(max(((target + eps) / (exo_m + eps)) ** 0.5, 0.5), 2.0)
                else:
                    sim_vals = [r.sojourn_mean[st.name] for r in res if st.name in r.sojourn_mean]
                    if not sim_vals or st.name not in obs_soj:
                        continue
                    ratio = obs_soj[st.name] / max(float(np.mean(sim_vals)), 1e-9)
                    gaps[st.name] = ratio
                    # passo amortecido (raiz) — a média de permanência é dominada
                    # pela cauda e é ruidosa com poucas réplicas
                    st.service_scale *= min(max(ratio ** 0.5, 0.5), 2.0)
            scale_trace.append({k: round(v, 3) for k, v in gaps.items()})
            if gaps and max(abs(v - 1) for v in gaps.values()) < 0.03:
                break

    diag = {
        "window": [cal_start, cal_end],
        "stages": stages,
        "profile": log.profile.name,
        "arrival_rate_per_day": arrival_rate,
        "cases_started": int(len(started)),
        "entry_probs": model.entry_probs,
        "service_classes": classes,
        "class_probs": class_probs,
        "class_curves_by_stage": settings.by_class,
        "routing": routing,
        "stage_params": diag_stages,
        "servers": {s.name: s.servers for s in model.stages},
        "service_scale": {s.name: s.service_scale for s in model.stages},
        "queue_scale": {s.name: s.queue_scale for s in model.stages if s.split},
        "scale_calibration_trace": scale_trace,
        "synchronization_rules": {s: [list(r) for r in rr] for s, rr in sync.items()},
        "people_pools": {s: list(v) for s, v in pools.items()},
        "people_jaccard_by_stage_pair": jac,
    }
    return model, diag


def at_state(model: SimModel, log: EventLog, t: float, horizon: float) -> SimModel:
    """Modelo calibrado inicializado no estado do log em t, com horizonte dado."""
    classes = model.classes
    return SimModel(
        stages=model.stages,
        arrival_rate=model.arrival_rate,
        entry_probs=model.entry_probs,
        routing=model.routing,
        initial_cases=initial_cases(log, t, model.stages),
        horizon=horizon,
        class_probs=model.class_probs,
        initial_classes=initial_classes(log, t, model.stages, classes) if classes else [],
        classes=classes,
        days_per_week=model.days_per_week,
        sync=model.sync,
    )
