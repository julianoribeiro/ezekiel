"""Simulação estocástica do pipeline de desenvolvimento via SimPy.

Modela o pipeline como uma rede de filas M/M/c/K (capacidade finita — Kanban):
  Backlog → Desenvolvimento → Revisão de Código → Testes → Implantação

Cada estágio possui dois recursos SimPy:
  wip_res    — capacity = wip_limit  → slot Kanban (backpressure real)
  server_res — capacity = n_servers  → processamento paralelo

O item só entra no estágio quando há slot WIP disponível; caso contrário aguarda
fora do estágio (backpressure). Isso torna o wip_limit efetivo na simulação,
ao contrário de uma fila M/M/c ilimitada.

Parâmetros calibrados a partir do MiningResult (Camada 3).
Suporta análise what-if via SimConfig.

Hipótese H1 da tese: MAE entre simulação e dados reais < 15%.
"""

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Generator

import simpy

from ezekiel.mining.analyzer import MiningResult, PIPELINE_STAGES

logger = logging.getLogger(__name__)

# Tempo de simulação em dias
_SIM_DURATION_DAYS = 365


@dataclass
class StageConfig:
    """Configuração de um estágio do pipeline na simulação."""
    name: str
    n_servers: int          # capacidade (número de recursos paralelos)
    wip_limit: int          # limite WIP (kanban) — enforced via simpy.Resource
    mean_service_time: float   # tempo médio de serviço (dias) — parâmetro calibrado
    cv_service_time: float = 1.0   # coeficiente de variação (1.0 = exponencial)
    priority_policy: str = "FIFO"   # FIFO | urgency | cost-of-delay


@dataclass
class SimConfig:
    """Configuração completa de um cenário de simulação."""
    stages: list[StageConfig] = field(default_factory=list)
    arrival_rate: float = 1.4   # itens por dia (calibrado dos logs)
    sim_duration: float = _SIM_DURATION_DAYS
    seed: int = 42
    n_runs: int = 30             # réplicas para estimativas com IC


@dataclass
class SimRunResult:
    """Resultado de uma única réplica de simulação."""
    lead_times: list[float] = field(default_factory=list)
    waiting_times: list[float] = field(default_factory=list)
    throughput_per_week: float = 0.0
    stage_wip: dict[str, float] = field(default_factory=dict)
    stage_utilization: dict[str, float] = field(default_factory=dict)


@dataclass
class SimResult:
    """Resultado agregado de n_runs réplicas."""
    mean_lead_time: float = 0.0
    mean_waiting_time: float = 0.0
    mean_throughput: float = 0.0
    flow_efficiency: float = 0.0
    stage_wip: dict[str, float] = field(default_factory=dict)
    stage_utilization: dict[str, float] = field(default_factory=dict)
    # Para cálculo do MAE (H1) e testes estatísticos (H2, H3)
    lead_time_samples: list[float] = field(default_factory=list)
    throughput_samples: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "lead_time_days": round(self.mean_lead_time, 1),
            "waiting_time_days": round(self.mean_waiting_time, 1),
            "throughput_per_week": round(self.mean_throughput, 1),
            "flow_efficiency_pct": round(self.flow_efficiency * 100, 1),
            "stage_wip": {k: round(v, 1) for k, v in self.stage_wip.items()},
            "stage_utilization": {k: round(v, 3) for k, v in self.stage_utilization.items()},
        }


class SimulationEngine:
    """Motor de simulação estocástica baseado em SimPy."""

    def calibrate(
        self,
        mining_result: MiningResult,
        calibration_iterations: int = 8,
        calibration_runs: int = 5,
    ) -> SimConfig:
        """
        Cria um SimConfig calibrado a partir das métricas de Process Mining.

        Estimativa inicial pela fórmula de Little (rede de filas aberta):
            E[S] ≈ ρ / λ   (garante ρ_sim = ρ_histórico com o λ real)

        Essa fórmula fechada assume uma rede ABERTA (chegadas em cada etapa
        na taxa nominal `arrival_rate`). O pipeline real é modelado como uma
        rede em SÉRIE com WIP finito por etapa (backpressure Kanban via
        `wip_res`) — o bloqueio a montante reduz a taxa de chegada efetiva em
        cada etapa a jusante, então a utilização obtida na simulação fica
        sistematicamente ABAIXO do alvo, com gap crescente rio abaixo. Em
        etapas próximas da saturação, mesmo um gap pequeno é amplificado
        pelo fator 1/(1-ρ) do tempo de fila e eleva o erro do lead time.

        Por isso a estimativa fechada serve só de ponto de partida: em
        seguida rodamos `calibration_iterations` rodadas de simulação curta
        (calibration_runs réplicas cada) medindo a utilização de fato obtida
        — já sob o efeito do bloqueio da rede — e reescalamos
        mean_service_time proporcionalmente (alvo/obtido) até convergir.
        Etapas sem sinal real no log (ex.: Code Review, Deploy — ver
        diagnóstico de dados na Camada 1) não têm alvo de utilização
        (ρ≈0) e não entram no ajuste, para não amplificar ruído.
        """
        arrival_rate = (
            mining_result.mean_throughput_per_week / 7
            if mining_result.mean_throughput_per_week > 0
            else 1.0
        )

        stages: list[StageConfig] = []
        targets: dict[str, float] = {}
        for m in mining_result.stage_metrics:
            rho = min(m.utilization, 0.95)
            mean_service = max(rho / arrival_rate, 0.05)
            wip_limit = max(2, round(m.mean_wip) + 1)

            stages.append(StageConfig(
                name=m.stage,
                n_servers=1,
                wip_limit=wip_limit,
                mean_service_time=round(mean_service, 3),
            ))
            targets[m.stage] = rho

        cfg = SimConfig(stages=stages, arrival_rate=arrival_rate)

        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.n_runs = calibration_runs
        for iteration in range(calibration_iterations):
            result = self.run(trial_cfg)
            max_gap = 0.0
            for stage_cfg in trial_cfg.stages:
                target = targets.get(stage_cfg.name, 0.0)
                if target <= 1e-6:
                    continue  # sem sinal real — não há alvo para calibrar
                achieved = result.stage_utilization.get(stage_cfg.name, 0.0)
                if achieved <= 1e-6:
                    continue
                max_gap = max(max_gap, abs(achieved - target))
                # Ajuste proporcional (controle tipo Newton amortecido):
                # se a utilização obtida ficou abaixo do alvo, aumenta o
                # tempo de serviço; limitado a [0.5x, 2x] por rodada para
                # evitar oscilação.
                factor = min(max(target / achieved, 0.5), 2.0)
                stage_cfg.mean_service_time = round(stage_cfg.mean_service_time * factor, 4)

            if max_gap < 0.01:
                logger.info(
                    f"Calibração iterativa convergiu em {iteration + 1} "
                    f"iteração(ões) — maior gap de utilização: {max_gap:.4f}"
                )
                break
        else:
            logger.warning(
                f"Calibração iterativa não convergiu em {calibration_iterations} "
                f"iterações — maior gap de utilização: {max_gap:.4f}"
            )

        for final_stage, calibrated_stage in zip(cfg.stages, trial_cfg.stages):
            final_stage.mean_service_time = calibrated_stage.mean_service_time

        return cfg

    def run(self, config: SimConfig) -> SimResult:
        """Executa n_runs réplicas e retorna resultado agregado."""
        all_lt: list[float] = []
        all_wt: list[float] = []
        all_tp: list[float] = []
        all_wip: dict[str, list[float]] = {s.name: [] for s in config.stages}
        all_util: dict[str, list[float]] = {s.name: [] for s in config.stages}

        for run_idx in range(config.n_runs):
            seed = config.seed + run_idx
            run_result = self._single_run(config, seed)
            all_lt.extend(run_result.lead_times)
            all_wt.extend(run_result.waiting_times)
            all_tp.append(run_result.throughput_per_week)
            for stage in config.stages:
                all_wip[stage.name].append(run_result.stage_wip.get(stage.name, 0.0))
                all_util[stage.name].append(run_result.stage_utilization.get(stage.name, 0.0))

        def mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        mean_lt = mean(all_lt)
        mean_wt = mean(all_wt)
        mean_tp = mean(all_tp)
        flow_eff = 1 - (mean_wt / mean_lt) if mean_lt > 0 else 0.0

        return SimResult(
            mean_lead_time=round(mean_lt, 2),
            mean_waiting_time=round(mean_wt, 2),
            mean_throughput=round(mean_tp, 2),
            flow_efficiency=round(flow_eff, 3),
            stage_wip={k: round(mean(v), 2) for k, v in all_wip.items()},
            stage_utilization={k: round(mean(v), 3) for k, v in all_util.items()},
            lead_time_samples=all_lt,
            throughput_samples=all_tp,
        )

    @staticmethod
    def _queue_priority(policy: str, item_id: int, arrival_time: float, cod_value: float) -> float:
        """Calcula a prioridade do item para simpy.PriorityResource.

        simpy.PriorityResource serve o menor valor de prioridade primeiro.

        FIFO        : prioridade = item_id (sequência de chegada ao pipeline)
        urgency     : prioridade = arrival_time (chegada global mais antiga → menor
                      valor → atendida primeiro — "Oldest Job First")
        cost-of-delay: prioridade = -cod_value (maior valor de negócio → número mais
                      negativo → atendido primeiro — proxy WSJF/CD3)
        """
        if policy == "urgency":
            return arrival_time
        if policy == "cost-of-delay":
            return -cod_value
        return float(item_id)   # FIFO padrão

    def _single_run(self, config: SimConfig, seed: int) -> SimRunResult:
        """Uma réplica de simulação — modelo M/M/c/K com política de fila configurável.

        Dois recursos PriorityResource por estágio:
          wip_res    — capacity = wip_limit  → slot Kanban (backpressure real)
          server_res — capacity = n_servers  → processamento paralelo

        A disciplina de fila (FIFO / urgency / cost-of-delay) é implementada via
        o parâmetro `priority` do PriorityResource, calculado por _queue_priority().
        Cada estágio pode ter sua própria política (configurável pelo agente DCOP).
        """
        rng = random.Random(seed)
        env = simpy.Environment()

        # PriorityResource em todos os estágios — FIFO é caso especial (priority=item_id)
        wip_res: dict[str, simpy.PriorityResource] = {
            sc.name: simpy.PriorityResource(env, capacity=sc.wip_limit)
            for sc in config.stages
        }
        server_res: dict[str, simpy.PriorityResource] = {
            sc.name: simpy.PriorityResource(env, capacity=sc.n_servers)
            for sc in config.stages
        }

        lead_times: list[float] = []
        waiting_times: list[float] = []
        items_completed = [0]
        wip_snapshots: dict[str, list[float]] = {sc.name: [] for sc in config.stages}
        stage_busy_time: dict[str, float] = {sc.name: 0.0 for sc in config.stages}
        stage_map = {sc.name: sc for sc in config.stages}

        def item_process(
            env: simpy.Environment,
            item_id: int,
            arrival_time: float,
            cod_value: float,
        ) -> Generator:
            total_waiting = 0.0

            for stage_cfg in config.stages:
                name = stage_cfg.name
                pri = self._queue_priority(
                    stage_cfg.priority_policy, item_id, arrival_time, cod_value
                )

                # ── Fase 1: aguarda slot WIP (backpressure Kanban) ──────────
                wait_start = env.now
                with wip_res[name].request(priority=pri) as wip_req:
                    yield wip_req

                    wip_snapshots[name].append(wip_res[name].count)

                    # ── Fase 2: aguarda servidor disponível ─────────────────
                    with server_res[name].request(priority=pri) as srv_req:
                        yield srv_req

                        # A espera termina quando o serviço de fato começa —
                        # antes, só a fila pelo slot WIP (fase 1) era contada;
                        # a fila pelo servidor (fase 2, onde a maior parte do
                        # atraso acontece sob alta utilização) era descartada,
                        # subestimando waiting_time e inflando artificialmente
                        # a flow_efficiency reportada.
                        service_start = env.now
                        total_waiting += service_start - wait_start

                        mu = stage_cfg.mean_service_time
                        service_time = rng.expovariate(1.0 / mu) if mu > 0 else 0.1
                        yield env.timeout(service_time)
                        stage_busy_time[name] += env.now - service_start

            lead_times.append(env.now - arrival_time)
            waiting_times.append(total_waiting)
            items_completed[0] += 1

        def arrival_generator(env: simpy.Environment) -> Generator:
            item_id = 0
            while True:
                yield env.timeout(rng.expovariate(config.arrival_rate))
                item_id += 1
                t = env.now
                # CoD amostrado na chegada: Exp(1), média=1 — representa valor de negócio relativo
                cod = rng.expovariate(1.0)
                env.process(item_process(env, item_id, t, cod))

        env.process(arrival_generator(env))
        env.run(until=config.sim_duration)

        weeks = config.sim_duration / 7
        tp = items_completed[0] / weeks if weeks > 0 else 0.0

        stage_wip = {
            name: (sum(snaps) / len(snaps) if snaps else 0.0)
            for name, snaps in wip_snapshots.items()
        }
        stage_util = {
            name: min(busy / (config.sim_duration * stage_map[name].n_servers), 1.0)
            for name, busy in stage_busy_time.items()
        }

        return SimRunResult(
            lead_times=lead_times,
            waiting_times=waiting_times,
            throughput_per_week=tp,
            stage_wip=stage_wip,
            stage_utilization=stage_util,
        )

    def compute_mae(self, sim_result: SimResult, mining_result: MiningResult) -> float:
        """
        Calcula o MAE entre simulação e dados históricos (Hipótese H1).

        MAE = |sim_LT - obs_LT| / obs_LT  (normalizado)

        A tese exige MAE < 15%.
        """
        obs_lt = mining_result.mean_lead_time_days
        sim_lt = sim_result.mean_lead_time
        if obs_lt == 0:
            return 0.0
        mae = abs(sim_lt - obs_lt) / obs_lt
        return round(mae, 4)


def build_whatif_config(
    base_config: SimConfig,
    constraint_stage: str | None = None,
    extra_servers: int = 2,
    wip_reduction_pct: float = 0.30,
    # Parâmetros legados (mantidos para compatibilidade)
    extra_reviewers: int = 0,
    wip_review_limit: int | None = None,
    ci_automation_pct: float | None = None,
) -> SimConfig:
    """
    Gera um SimConfig modificado para análise what-if (cenário S1).

    Aplica intervenções no estágio gargalo identificado pelo mining:
      - Aumenta a capacidade (n_servers) do estágio
      - Reduz o WIP limit proporcionalmente (kanban mais restrito)
      - Reduz o tempo de serviço nos estágios de testes via automação CI
    """
    import copy
    cfg = copy.deepcopy(base_config)

    for stage in cfg.stages:
        if constraint_stage and stage.name == constraint_stage:
            stage.n_servers += extra_servers
            new_wip = max(2, round(stage.wip_limit * (1 - wip_reduction_pct)))
            stage.wip_limit = wip_review_limit if wip_review_limit is not None else new_wip

        # Legado: intervenção direta em Revisão de Código
        elif not constraint_stage and stage.name == "Revisão de Código":
            if extra_reviewers > 0:
                stage.n_servers += extra_reviewers
            if wip_review_limit is not None:
                stage.wip_limit = wip_review_limit

        # Automação CI reduz tempo de serviço em Testes
        if stage.name == "Testes" and ci_automation_pct is not None:
            reduction = ci_automation_pct / 100
            stage.mean_service_time *= max(1 - reduction * 0.5, 0.1)

    return cfg
