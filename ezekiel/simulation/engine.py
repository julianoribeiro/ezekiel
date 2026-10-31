"""Simulação de eventos discretos do fluxo de trabalho (SimPy).

Modelo (metodologia, Seções "Construção e validação do modelo de simulação"
e "Condições iniciais e período de aquecimento"):

  * Simulação TERMINANTE de uma janela histórica, inicializada com o estado
    reconstruído do log no instante inicial (casos em andamento, com sua
    idade no sistema e no estágio corrente). Nenhum período é descartado.
  * Chegadas Poisson com a taxa da janela de calibração; estágio de entrada
    sorteado da distribuição empírica.
  * Roteamento por cadeia de Markov estimada do log (inclui saltos de
    estágio, retrabalho e conclusão direta), em vez de impor a todo caso a
    passagem pelos sete estágios.
  * Tempos por estágio amostrados da curva de SOBREVIVÊNCIA empírica de
    permanência (Kaplan-Meier com truncamento à esquerda e censura à
    direita — ver calibration.py), com fator de escala calibrado. Para
    casos iniciais, o residual é amostrado condicionalmente à idade já
    decorrida no estágio. A massa de probabilidade que a curva não esgota
    (itens que nunca saíram do estágio no período observado) gera casos
    ESTACIONADOS: permanecem no estágio — contam no WIP observado, como no
    log — mas não ocupam servidor nem vaga de WIP. Sem isso, itens com
    anos de idade eram "concluídos" pela simulação e inflavam vazão e
    lead time.
  * Estágios com capacidade finita (`servers`) e estágios de atraso puro
    (`servers=None`, ex.: Backlog — a espera por triagem não é limitada por
    um recurso identificável no log).
  * Limite de WIP com bloqueio Kanban real: ao avançar, o caso só libera a
    vaga do estágio atual depois de obter vaga no próximo. Movimentos para
    trás (retrabalho) liberam a vaga antes de pedir a nova — o que impede
    impasse circular.
  * Disciplinas de fila: FIFO (ordem de chegada ao estágio), urgency (caso
    mais antigo no sistema primeiro) e cost-of-delay (valor exógeno
    Exp(1) por caso — proxy, pois o log não registra valor de negócio).
  * Números aleatórios comuns: cada caso tem seu próprio gerador, semeado
    pelo par (réplica, índice do caso); chegadas usam um gerador separado.
    Assim a réplica r de qualquer política recebe exatamente as mesmas
    chegadas, rotas e tempos de serviço — requisito do desenho pareado.

Classes de serviço (fundamentação, "Agregação e interpretação sistêmica":
a média pode ocultar diferenças entre classes de serviço): cada caso tem
uma classe (tipo de item). Casos iniciais trazem a classe real; chegadas
sorteiam a classe da proporção da janela de calibração com um gerador
próprio do caso, que não altera os demais sorteios (números aleatórios
comuns preservados). Estágios podem ter curvas de permanência por classe
(`StageModel.class_km`); sem elas, a curva é comum a todas as classes.

Acoplamentos entre unidades (metodologia, "dependências de sincronização
entre objetos" no modelo conceitual), opcionais e estimados na calibração:

  * sincronização: ao iniciar uma visita a um estágio S, o caso fica
    bloqueado com probabilidade p(S, S') e espera a SAÍDA de um caso real
    (sorteado entre os ativos, não estacionados) do estágio S' — o
    bloqueador. O tempo de liberação depende do que acontece em S' (WIP,
    política, capacidade), o que acopla as decisões dos agentes;
  * pessoas compartilhadas: estágios com as mesmas pessoas cedem uma
    fração `pool_share` de seus servidores a um pool comum (`pool`). O caso
    usa o primeiro servidor livre, dedicado ou do pool; a carga de um
    estágio passa a disputar a capacidade do outro.

Os sorteios desses mecanismos usam geradores próprios do caso — números
aleatórios comuns preservados entre configurações.

Métricas por réplica (unidade estatística da análise — metodologia,
"Unidade estatística e independência das observações"): distribuição de
lead time dos casos concluídos no horizonte, espera, vazão, WIP médio no
tempo por estágio, ociosidade dos estágios finitos e casos em aberto ao fim.
"""

from __future__ import annotations

import bisect
import copy
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import simpy

from ezekiel.data.eventlog import DONE

POLICIES = ("FIFO", "urgency", "cost-of-delay")


@dataclass
class StageModel:
    name: str
    servers: int | None              # None = atraso puro (sem recurso limitante)
    km_times: list[float]            # instantes de saída (dias), crescentes
    km_surv: list[float]             # S(t) logo após cada instante (não crescente)
    service_scale: float = 1.0
    wip_limit: int | None = None     # None = sem limite (política histórica)
    policy: str = "FIFO"
    counts_as_wait: bool = False     # tempo no estágio é espera (fila de entrada)
    # Estágio com fila e serviço separados (status "Ready to X" distintos):
    # a curva KM principal é só do tempo em colunas de TRABALHO. A espera no
    # estágio tem duas componentes: EXÓGENA — atraso amostrado da curva KM
    # do tempo em colunas de fila (prioridades, lotes, bloqueios externos,
    # que a capacidade não explica), com escala calibrada — e ENDÓGENA —
    # disputa pelos servidores. Casos iniciais parados na fila há mais que
    # `queue_age_cap` dias são tratados como estacionados.
    split: bool = False
    queue_age_cap: float = math.inf
    queue_km_times: list[float] = field(default_factory=list)
    queue_km_surv: list[float] = field(default_factory=list)
    queue_scale: float = 1.0
    class_km: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    # Penalidade de multitarefa: no início do serviço, a duração é
    # multiplicada por 1 + κ·max(0, L/c − 1), com L = casos no estágio e c =
    # servidores. κ = 0 desliga (o WIP excedente só espera em fila).
    multitask: float = 0.0
    pool: str | None = None          # pool de pessoas compartilhadas com outros estágios
    pool_share: float = 0.0          # fração dos servidores cedida ao pool

    def __post_init__(self) -> None:
        if len(self.km_times) != len(self.km_surv):
            raise ValueError(f"{self.name}: km_times e km_surv com tamanhos diferentes")

    @property
    def p_parked(self) -> float:
        """Probabilidade de um caso novo nunca sair do estágio (massa residual)."""
        return self.km_surv[-1] if self.km_surv else 1.0

    def survival(self, x: float) -> float:
        i = bisect.bisect_right(self.km_times, x / self.service_scale)
        return 1.0 if i == 0 else self.km_surv[i - 1]

    @property
    def mean_service(self) -> float:
        """Média da permanência condicionada a sair do estágio."""
        t, s = self.km_times, self.km_surv
        if not t:
            return float("nan")
        mass = np.diff(np.r_[1.0, s]) * -1.0
        tot = mass.sum()
        return float((np.asarray(t) * mass).sum() / tot) * self.service_scale if tot > 0 else float("nan")


@dataclass
class SimModel:
    stages: list[StageModel]                 # em ordem de fluxo
    arrival_rate: float                      # casos/dia
    entry_probs: dict[str, float]
    routing: dict[str, dict[str, float]]     # estágio → {próximo | DONE: prob}
    initial_cases: list[tuple] = field(default_factory=list)
    # (estágio, idade do caso, idade da visita[, serviço já decorrido,
    #  estacionado[, fila já decorrida | None se o caso está em trabalho]]),
    # em dias. Sem os campos opcionais: serviço decorrido = idade da visita.
    horizon: float = 180.0
    class_probs: dict[str, float] = field(default_factory=dict)   # classes das chegadas
    initial_classes: list[str | None] = field(default_factory=list)  # paralela a initial_cases
    classes: list[str] = field(default_factory=list)                 # classes nomeadas (resto: "Outros")
    days_per_week: float = 7.0        # 5 em tempo útil (calendário de trabalho)
    sync: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # S → [(S', p)]

    def stage(self, name: str) -> StageModel:
        return next(s for s in self.stages if s.name == name)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def with_assignment(self, assignment: dict[str, tuple]) -> "SimModel":
        """Cópia do modelo com (limite de WIP, política[, servidores]) por estágio."""
        m = copy.copy(self)
        m.stages = [copy.copy(s) for s in self.stages]
        for s in m.stages:
            if s.name in assignment:
                v = assignment[s.name]
                s.wip_limit, s.policy = v[0], v[1]
                if len(v) > 2 and v[2] is not None:
                    s.servers = v[2]
        return m


@dataclass
class ReplicaResult:
    lead_times: np.ndarray
    waits: np.ndarray
    throughput_week: float
    wip_avg: dict[str, float]
    sojourn_mean: dict[str, float]
    idle: dict[str, float]
    open_at_end: int
    parked: int = 0
    stage_wait: dict[str, float] = field(default_factory=dict)  # espera média por visita
    stage_endo_wait: dict[str, float] = field(default_factory=dict)  # só disputa por servidor/vaga
    sojourn_n: dict[str, int] = field(default_factory=dict)   # visitas encerradas por estágio
    transitions: dict[str, dict[str, int]] = field(default_factory=dict)  # estágio → {próximo|DONE: n}
    rework: int = 0                  # movimentos para estágio anterior (retrabalho)
    done_times: np.ndarray = field(default_factory=lambda: np.array([]))  # instantes de conclusão
    utilization: dict[str, float] = field(default_factory=dict)  # serviço / capacidade, estágios finitos
    sync_blocks: int = 0             # visitas que esperaram por bloqueador
    sync_wait_mean: float = 0.0      # espera média por bloqueador (entre as bloqueadas)
    lead_times_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    waits_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    wip_by_class: dict[str, float] = field(default_factory=dict)

    def metrics_by_class(self) -> dict[str, dict[str, float]]:
        out = {}
        for c, lt in self.lead_times_by_class.items():
            has = len(lt) > 0
            w = self.waits_by_class.get(c, np.array([]))
            lt_mean = float(lt.mean()) if has else float("nan")
            wt = float(w.mean()) if has else float("nan")
            out[c] = {
                "lt_p50": float(np.quantile(lt, 0.5)) if has else float("nan"),
                "lt_p80": float(np.quantile(lt, 0.8)) if has else float("nan"),
                "lt_p90": float(np.quantile(lt, 0.9)) if has else float("nan"),
                "wt_mean": wt,
                "fe": 1.0 - wt / lt_mean if has and lt_mean > 0 else float("nan"),
                "completions": float(len(lt)),
                "wip": self.wip_by_class.get(c, 0.0),
            }
        return out

    def metrics(self) -> dict[str, float]:
        lt = self.lead_times
        has = len(lt) > 0
        lt_mean = float(lt.mean()) if has else float("nan")
        wt_mean = float(self.waits.mean()) if has else float("nan")
        return {
            "lt_mean": lt_mean,
            "lt_p50": float(np.quantile(lt, 0.5)) if has else float("nan"),
            "lt_p80": float(np.quantile(lt, 0.8)) if has else float("nan"),
            "lt_p90": float(np.quantile(lt, 0.9)) if has else float("nan"),
            "wt_mean": wt_mean,
            "fe": 1.0 - wt_mean / lt_mean if has and lt_mean > 0 else float("nan"),
            "th_week": self.throughput_week,
            "wip_total": float(sum(self.wip_avg.values())),
            "completions": float(len(lt)),
            "open_at_end": float(self.open_at_end),
            "parked": float(self.parked),
        }


# ---------------------------------------------------------------------- #
# Uma réplica
# ---------------------------------------------------------------------- #

def _draw_km(times: list[float], surv: list[float], scale: float, rng: random.Random, elapsed: float) -> float:
    """Duração restante amostrada de uma curva KM, condicionada à idade já
    decorrida; math.inf = nunca termina (caso estacionado)."""
    if not times:
        return 0.0
    i = bisect.bisect_right(times, elapsed / scale)
    s_x = 1.0 if i == 0 else surv[i - 1]
    if s_x <= 0:
        return 1e-6
    u = rng.random() * s_x
    # primeiro índice com S(t) <= u  (surv é não crescente)
    lo, hi = 0, len(surv)
    while lo < hi:
        mid = (lo + hi) // 2
        if surv[mid] <= u:
            hi = mid
        else:
            lo = mid + 1
    if lo == len(surv):
        return math.inf
    return max(times[lo] * scale - elapsed, 1e-6)


def _draw_service(st: StageModel, rng: random.Random, elapsed: float, cls: str | None = None) -> float:
    """Permanência (ou serviço, em estágios separados) restante; inf = estacionado."""
    if cls is not None and cls in st.class_km:
        t, s = st.class_km[cls]
        return _draw_km(t, s, st.service_scale, rng, elapsed)
    return _draw_km(st.km_times, st.km_surv, st.service_scale, rng, elapsed)


def run_replica(model: SimModel, seed: int) -> ReplicaResult:
    env = simpy.Environment()
    stages = {s.name: s for s in model.stages}
    order = {s.name: i for i, s in enumerate(model.stages)}

    wip_res = {n: simpy.PriorityResource(env, s.wip_limit) for n, s in stages.items() if s.wip_limit}
    pool_cap: dict[str, int] = {}
    srv_res = {}
    for n, s in stages.items():
        if not s.servers:
            continue
        k = min(s.servers, int(round(s.pool_share * s.servers))) if s.pool else 0
        if k:
            pool_cap[s.pool] = pool_cap.get(s.pool, 0) + k
        if s.servers - k > 0:
            srv_res[n] = simpy.PriorityResource(env, s.servers - k)
    pool_res = {p: simpy.PriorityResource(env, c) for p, c in pool_cap.items()}
    present: dict[str, dict[int, simpy.Event]] = {n: {} for n in stages}   # casos no estágio
    parked_ids: set[int] = set()
    sync_wait = [0.0, 0]

    routes = {
        n: (list(np.cumsum(list(r.values()))), list(r.keys()))
        for n, r in model.routing.items()
    }
    entry = (list(np.cumsum(list(model.entry_probs.values()))), list(model.entry_probs.keys()))

    count = {n: 0 for n in stages}
    area = {n: 0.0 for n in stages}
    last = {n: 0.0 for n in stages}
    # Ocupação dos servidores integrada no tempo (inclui serviços ainda em
    # curso no fim do horizonte).
    in_srv = {n: 0 for n in stages}
    busy = {n: 0.0 for n in stages}
    last_srv = {n: 0.0 for n in stages}
    soj_sum = {n: 0.0 for n in stages}
    soj_n = {n: 0 for n in stages}
    wait_sum = {n: 0.0 for n in stages}
    wait_n = {n: 0 for n in stages}
    endo_sum = {n: 0.0 for n in stages}
    lead_times: list[float] = []
    waits: list[float] = []
    done_at: list[float] = []
    by_cls_lt: dict[str, list[float]] = {}
    by_cls_wt: dict[str, list[float]] = {}
    cls_n: dict[str, int] = {}
    cls_area: dict[str, float] = {}
    cls_last: dict[str, float] = {}

    def bump_cls(c: str, delta: int) -> None:
        cls_area[c] = cls_area.get(c, 0.0) + cls_n.get(c, 0) * (env.now - cls_last.get(c, 0.0))
        cls_last[c] = env.now
        cls_n[c] = cls_n.get(c, 0) + delta
    trans = {n: {} for n in stages}
    rework = [0]
    started = [0]
    parked = [0]

    def bump(n: str, delta: int, cid: int | None = None) -> None:
        area[n] += count[n] * (env.now - last[n])
        last[n] = env.now
        count[n] += delta
        if cid is not None:
            if delta > 0:
                present[n][cid] = env.event()
            elif delta < 0:
                ev = present[n].pop(cid, None)
                if ev is not None and not ev.triggered:
                    ev.succeed()

    def bump_srv(n: str, delta: int) -> None:
        busy[n] += in_srv[n] * (env.now - last_srv[n])
        last_srv[n] = env.now
        in_srv[n] += delta

    def pick(table: tuple[list[float], list[str]], u: float) -> str:
        cum, keys = table
        i = bisect.bisect_left(cum, u * cum[-1])
        return keys[min(i, len(keys) - 1)]

    def prio(st: StageModel, start: float, cod: float, t_arrival: float) -> float:
        if st.policy == "urgency":
            return start
        if st.policy == "cost-of-delay":
            return -cod
        return t_arrival

    def case(cid: int, start: float, stage: str, stage_age: float,
             elapsed: float | None = None, parked_init: bool = False,
             queue_elapsed: float | None = 0.0, cls: str | None = None, wait0: float = 0.0):
        rng = random.Random(seed * 1_000_003 + cid)
        cod = rng.expovariate(1.0)
        started[0] += 1
        cls = cls or "—"
        bump_cls(cls, +1)
        wait = wait0                      # espera acumulada antes do horizonte
        t_arrival = env.now - stage_age
        if parked_init:
            bump(stage, +1, cid)
            parked_ids.add(cid)
            parked[0] += 1
            yield env.event()
            return
        slot = None
        stage_wait = 0.0
        stage_endo = 0.0
        if stage in wip_res:
            t0 = env.now
            slot = wip_res[stage].request(priority=prio(stages[stage], start, cod, t_arrival))
            yield slot
            wait += env.now - t0
            stage_wait += env.now - t0
            stage_endo += env.now - t0
        bump(stage, +1, cid)
        t_enter = env.now - stage_age
        elapsed = stage_age if elapsed is None else elapsed
        visit_no = 0

        while True:
            st = stages[stage]
            visit_no += 1
            if model.sync and elapsed == 0 and stage in model.sync:
                srng = random.Random(seed * 104_729 + cid * 13 + visit_no)
                for blk_stage, p in model.sync[stage]:
                    if srng.random() >= p:
                        continue
                    cands = sorted(c for c in present.get(blk_stage, {}) if c != cid and c not in parked_ids)
                    if cands:
                        t0 = env.now
                        yield present[blk_stage][srng.choice(cands)]
                        wait += env.now - t0
                        stage_wait += env.now - t0
                        sync_wait[0] += env.now - t0
                        sync_wait[1] += 1
                    break
            if st.split and st.queue_km_times and queue_elapsed is not None:
                dq = _draw_km(st.queue_km_times, st.queue_km_surv, st.queue_scale, rng, queue_elapsed)
                if dq == math.inf:
                    if slot is not None:
                        wip_res[stage].release(slot)
                    parked_ids.add(cid)
                    parked[0] += 1
                    yield env.event()
                    return
                yield env.timeout(dq)
                wait += dq
                stage_wait += dq
            queue_elapsed = 0.0
            d = _draw_service(st, rng, elapsed, cls)
            if d == math.inf:
                # Estacionado: continua no estágio (WIP), sem servidor/vaga.
                if slot is not None:
                    wip_res[stage].release(slot)
                parked_ids.add(cid)
                parked[0] += 1
                yield env.event()
                return
            if st.servers:
                t0 = env.now
                pr = prio(st, start, cod, t_arrival)
                reqs = [(r, r.request(priority=pr)) for r in
                        (srv_res.get(stage), pool_res.get(st.pool) if st.pool else None) if r is not None]
                if len(reqs) == 1:
                    res, req = reqs[0]
                    yield req
                else:   # primeiro servidor livre: dedicado ou do pool compartilhado
                    got = yield simpy.AnyOf(env, [q for _, q in reqs])
                    res, req = next((r, q) for r, q in reqs if q in got)
                    for r, q in reqs:
                        if q is not req:
                            if q.triggered:
                                r.release(q)
                            else:
                                q.cancel()
                if st.multitask > 0:
                    d *= 1.0 + st.multitask * max(0.0, count[stage] / st.servers - 1.0)
                wait += env.now - t0
                stage_wait += env.now - t0
                stage_endo += env.now - t0
                bump_srv(stage, +1)
                yield env.timeout(d)
                bump_srv(stage, -1)
                res.release(req)
            else:
                yield env.timeout(d)
                if st.counts_as_wait:
                    wait += d
            elapsed = 0.0

            nxt = pick(routes[stage], rng.random()) if stage in routes else DONE
            if nxt not in stages:
                nxt = DONE
            trans[stage][nxt] = trans[stage].get(nxt, 0) + 1
            if nxt != DONE and order[nxt] < order[stage]:
                rework[0] += 1

            if nxt == DONE:
                if slot is not None:
                    wip_res[stage].release(slot)
                bump(stage, -1, cid)
                soj_sum[stage] += env.now - t_enter
                soj_n[stage] += 1
                wait_sum[stage] += stage_wait
                endo_sum[stage] += stage_endo
                wait_n[stage] += 1
                lead_times.append(env.now - start)
                waits.append(wait)
                done_at.append(env.now)
                by_cls_lt.setdefault(cls, []).append(env.now - start)
                by_cls_wt.setdefault(cls, []).append(wait)
                bump_cls(cls, -1)
                return

            wait_sum[stage] += stage_wait
            endo_sum[stage] += stage_endo
            wait_n[stage] += 1
            stage_wait = 0.0
            stage_endo = 0.0
            t_next = env.now
            new_slot = None
            if nxt in wip_res:
                forward = order[nxt] > order[stage]
                if not forward:
                    if slot is not None:
                        wip_res[stage].release(slot)
                    bump(stage, -1, cid)
                    soj_sum[stage] += env.now - t_enter
                    soj_n[stage] += 1
                t0 = env.now
                new_slot = wip_res[nxt].request(priority=prio(stages[nxt], start, cod, t_next))
                yield new_slot
                wait += env.now - t0
                stage_wait += env.now - t0   # espera por vaga conta para o estágio de destino
                stage_endo += env.now - t0
                if forward:
                    if slot is not None:
                        wip_res[stage].release(slot)
                    bump(stage, -1, cid)
                    soj_sum[stage] += env.now - t_enter
                    soj_n[stage] += 1
            else:
                if slot is not None:
                    wip_res[stage].release(slot)
                bump(stage, -1, cid)
                soj_sum[stage] += env.now - t_enter
                soj_n[stage] += 1

            bump(nxt, +1, cid)
            stage, slot, t_enter, t_arrival = nxt, new_slot, env.now, t_next

    def arrivals():
        rng = random.Random(seed)
        cid = len(model.initial_cases)
        while True:
            yield env.timeout(rng.expovariate(model.arrival_rate))
            cls = None
            if classes:
                # gerador próprio: não altera chegadas nem os sorteios do caso
                cls = pick(classes, random.Random(seed * 7_919 + cid * 31 + 17).random())
            env.process(case(cid, env.now, pick(entry, rng.random()), 0.0, cls=cls))
            cid += 1

    # Casos iniciais: os mais antigos no estágio primeiro (FIFO consistente).
    def _norm(c: tuple) -> tuple:
        c = tuple(c)
        if len(c) == 3:
            c = c + (c[2], False)
        if len(c) == 5:
            c = c + (None,)   # sem informação de fila: caso em trabalho
        if len(c) == 6:
            c = c + (0.0,)    # sem espera acumulada
        return c

    classes = ((list(np.cumsum(list(model.class_probs.values()))), list(model.class_probs.keys()))
               if model.class_probs else None)
    init_cls = list(model.initial_classes) + [None] * (len(model.initial_cases) - len(model.initial_classes))
    init = sorted(((_norm(c), k) for c, k in zip(model.initial_cases, init_cls) if c[0] in stages),
                  key=lambda x: -x[0][2])
    for cid, ((stage, case_age, stage_age, elapsed, parked_init, q_el, w0), k) in enumerate(init):
        env.process(case(cid, -case_age, stage, stage_age, elapsed, parked_init, q_el, cls=k, wait0=w0))
    if model.arrival_rate > 0:
        env.process(arrivals())

    env.run(until=model.horizon)
    for n in stages:
        bump(n, 0)
        bump_srv(n, 0)
    for c in list(cls_n):
        bump_cls(c, 0)

    H = model.horizon
    return ReplicaResult(
        lead_times=np.asarray(lead_times, dtype=float),
        waits=np.asarray(waits, dtype=float),
        throughput_week=len(lead_times) / (H / model.days_per_week),
        wip_avg={n: area[n] / H for n in stages},
        sojourn_mean={n: soj_sum[n] / soj_n[n] for n in stages if soj_n[n]},
        idle={n: 1.0 - min(busy[n] / (stages[n].servers * H), 1.0) for n in stages if stages[n].servers},
        open_at_end=started[0] - len(lead_times),
        parked=parked[0],
        stage_wait={n: wait_sum[n] / wait_n[n] for n in stages if wait_n[n]},
        stage_endo_wait={n: endo_sum[n] / wait_n[n] for n in stages if wait_n[n]},
        sojourn_n={n: soj_n[n] for n in stages},
        transitions=trans,
        rework=rework[0],
        done_times=np.asarray(done_at, dtype=float),
        utilization={n: min(busy[n] / (stages[n].servers * H), 1.0) for n in stages if stages[n].servers},
        sync_blocks=sync_wait[1],
        sync_wait_mean=sync_wait[0] / sync_wait[1] if sync_wait[1] else 0.0,
        lead_times_by_class={c: np.asarray(v, float) for c, v in by_cls_lt.items()},
        waits_by_class={c: np.asarray(v, float) for c, v in by_cls_wt.items()},
        wip_by_class={c: a / H for c, a in cls_area.items()},
    )


def _run_job(args: tuple[SimModel, int]) -> ReplicaResult:
    model, seed = args
    return run_replica(model, seed)


# ---------------------------------------------------------------------- #
# Execução em lote (paralela)
# ---------------------------------------------------------------------- #

class Simulator:
    """Executa réplicas em paralelo. Use como context manager."""

    def __init__(self, n_jobs: int | None = None) -> None:
        self.n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "Simulator":
        if self.n_jobs > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool:
            self._pool.shutdown()
            self._pool = None

    def run_batch(self, jobs: list[tuple[SimModel, int]]) -> list[ReplicaResult]:
        if self._pool is None:
            return [_run_job(j) for j in jobs]
        chunk = max(1, len(jobs) // (self.n_jobs * 4))
        return list(self._pool.map(_run_job, jobs, chunksize=chunk))

    def run(self, model: SimModel, seeds: list[int]) -> list[ReplicaResult]:
        return self.run_batch([(model, s) for s in seeds])

    def run_many(
        self, models: list[SimModel], seeds: list[int]
    ) -> list[list[ReplicaResult]]:
        """Roda vários modelos com as mesmas sementes (números aleatórios comuns)."""
        flat = self.run_batch([(m, s) for m in models for s in seeds])
        k = len(seeds)
        return [flat[i * k:(i + 1) * k] for i in range(len(models))]


def mean_metrics(results: list[ReplicaResult]) -> dict[str, float]:
    """Média, entre réplicas, das métricas-resumo de cada réplica."""
    rows = [r.metrics() for r in results]
    return {k: float(np.nanmean([row[k] for row in rows])) for k in rows[0]}
