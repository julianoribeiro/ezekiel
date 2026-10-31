"""Coordenador DCOP — orquestra os 5 agentes ADOPT até convergência.

Responsabilidades:
  1. Constrói a pseudo-árvore com raiz = restrição ativa (ToC)
  2. Estima as funções de custo f_ij via simulação (Eq. 4.2 da tese)
  3. Executa ciclos assíncronos até convergência ou max_cycles
  4. Retorna a atribuição ótima e o custo global final

Convergência: todos os agentes mantêm o mesmo valor por STABLE_CYCLES ciclos
consecutivos e nenhuma mensagem de backtrack está pendente.
"""

import logging
from dataclasses import dataclass, field

from ezekiel.config import settings
from ezekiel.legacy.dcop.agent import DCOPAgent, AGENT_DOMAIN, WIP_DOMAIN
from ezekiel.legacy.dcop.messages import Message
from ezekiel.mining.analyzer import MiningResult, PIPELINE_STAGES
from ezekiel.legacy.engine import SimulationEngine, SimConfig, StageConfig

logger = logging.getLogger(__name__)

STABLE_CYCLES = 5  # ciclos consecutivos estáveis para declarar convergência


@dataclass
class DCOPResult:
    """Resultado da otimização DCOP/ADOPT."""
    assignments: list[dict] = field(default_factory=list)
    global_cost: float = 0.0
    cycles_to_converge: int = 0
    converged: bool = False
    message_log: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "converged": self.converged,
            "cycles": self.cycles_to_converge,
            "global_cost": round(self.global_cost, 2),
            "assignments": [
                {
                    "stage": a["stage"],
                    "wip_limit": a["wip_limit"],
                    "priority": a["priority_policy"],
                    "is_constraint": a["is_constraint"],
                }
                for a in self.assignments
            ],
        }


def apply_dcop_assignments(base_config: SimConfig, dcop_result: DCOPResult) -> SimConfig:
    """Aplica a atribuição ótima do DCOP a um SimConfig (gera a config do cenário S2).

    Fonte única de verdade — usada tanto pelo CLI (`ezekiel.pipeline`) quanto
    pela API (`api.main`). Antes, a API tinha sua própria heurística
    (reduzir WIP e fixar 2 servidores nas 2 etapas de maior constraint_score)
    que nunca lia `dcop_result.assignments` — o S2 exibido pela API não
    refletia o que o ADOPT de fato calculou.
    """
    import copy
    cfg = copy.deepcopy(base_config)
    for assignment in dcop_result.assignments:
        for stage_cfg in cfg.stages:
            if stage_cfg.name == assignment["stage"]:
                stage_cfg.wip_limit = assignment["wip_limit"]
                stage_cfg.priority_policy = assignment["priority_policy"]
                stage_cfg.n_servers = max(1, assignment["wip_limit"] // 2)
    return cfg


class DCOPCoordinator:
    """Coordenador do protocolo ADOPT para o pipeline de 5 agentes."""

    def __init__(self, sim_engine: SimulationEngine | None = None) -> None:
        self.sim_engine = sim_engine or SimulationEngine()

    def run(
        self,
        mining_result: MiningResult,
        base_sim_config: SimConfig,
        max_cycles: int | None = None,
        gamma1: float | None = None,
        gamma2: float | None = None,
        gamma3: float | None = None,
    ) -> DCOPResult:
        """
        Executa o ADOPT completo:
          1. Estima funções de custo f_ij via simulação
          2. Constrói pseudo-árvore
          3. Executa ciclos até convergência
        """
        max_cycles = max_cycles or settings.dcop_max_cycles
        g1 = gamma1 if gamma1 is not None else settings.dcop_gamma1
        g2 = gamma2 if gamma2 is not None else settings.dcop_gamma2
        g3 = gamma3 if gamma3 is not None else settings.dcop_gamma3

        constraint_stage = (
            mining_result.active_constraint.stage
            if mining_result.active_constraint
            else "Revisão de Código"
        )

        logger.info(
            f"Iniciando DCOP/ADOPT — restrição ativa: {constraint_stage}, "
            f"max_cycles={max_cycles}"
        )

        # Estima as funções de custo f_ij para todos os pares de estágios adjacentes
        cost_tables = self._estimate_cost_tables(
            base_sim_config, mining_result, g1, g2, g3
        )

        # Constrói os agentes e a pseudo-árvore
        agents = self._build_agents(
            mining_result, constraint_stage, cost_tables, g1, g2, g3
        )

        # Inicializa estado operacional dos agentes
        for agent in agents.values():
            stage_m = next(
                (m for m in mining_result.stage_metrics if m.stage == agent.stage_name),
                None,
            )
            if stage_m:
                agent.update_operational_state(
                    stage_m.utilization,
                    stage_m.mean_wip,
                    stage_m.mean_waiting_time_days,
                )

        # Executa ciclos ADOPT
        message_log: list[dict] = []
        stable_count = 0
        prev_assignments: dict[str, tuple] = {}

        for cycle in range(1, max_cycles + 1):
            # Coleta mensagens geradas por todos os agentes
            all_messages: list[Message] = []
            for agent in agents.values():
                msgs = agent.step()
                all_messages.extend(msgs)

            # Entrega mensagens aos destinatários
            for msg in all_messages:
                if msg.receiver_id in agents:
                    agents[msg.receiver_id].receive(msg)
                message_log.append({
                    "cycle": cycle,
                    "from": msg.sender_id,
                    "to": msg.receiver_id,
                    "type": msg.msg_type.name,
                    "payload": msg.payload,
                })

            # Verifica convergência
            current_assignments = {
                aid: agent.state.current_value for aid, agent in agents.items()
            }
            if current_assignments == prev_assignments:
                stable_count += 1
            else:
                stable_count = 0
            prev_assignments = dict(current_assignments)

            if stable_count >= STABLE_CYCLES:
                logger.info(f"[OK] ADOPT convergiu em {cycle} ciclos.")
                final_assignments = [a.get_assignment() for a in agents.values()]
                global_cost = self._compute_global_cost(agents, cost_tables)
                return DCOPResult(
                    assignments=final_assignments,
                    global_cost=global_cost,
                    cycles_to_converge=cycle,
                    converged=True,
                    message_log=message_log[-50:],  # últimas 50 mensagens
                )

        logger.warning(f"ADOPT não convergiu em {max_cycles} ciclos.")
        final_assignments = [a.get_assignment() for a in agents.values()]
        global_cost = self._compute_global_cost(agents, cost_tables)
        return DCOPResult(
            assignments=final_assignments,
            global_cost=global_cost,
            cycles_to_converge=max_cycles,
            converged=False,
            message_log=message_log[-50:],
        )

    # ------------------------------------------------------------------ #
    # Estimativa das funções de custo via simulação
    # ------------------------------------------------------------------ #

    def _estimate_cost_tables(
        self,
        base_config: SimConfig,
        mining_result: MiningResult,
        gamma1: float,
        gamma2: float,
        gamma3: float,
    ) -> dict[tuple[str, str], dict[tuple, dict[tuple, float]]]:
        """
        Estima f_ij(x_i, x_j) para cada par de estágios adjacentes via simulação.

        Equação 4.2 da tese:
          f_ij(x_i, x_j) = E_sim[γ1·WT_ij + γ2·LT_ij − γ3·TH_ij]

        Para eficiência, amostramos um subconjunto do domínio (WIP apenas,
        política fixada em FIFO para a estimativa de custo). O domínio
        amostrado (WIP_DOMAIN, 1..5) é o mesmo domínio de candidatos que os
        agentes de fato avaliam (AGENT_DOMAIN) — antes as tabelas cobriam
        wip_i em [2,3,4,5] e wip_j em [1,2,3,4], deixando WIP=1 e WIP=5 fora
        do alcance conforme o papel, o que fazia a maioria das consultas de
        cost_fn não encontrar entrada e cair no fallback linear em WIP.
        """
        import copy
        logger.info("Estimando funções de custo f_ij via simulação...")

        # Pares de estágios adjacentes (arestas da pseudo-árvore)
        pairs = list(zip(PIPELINE_STAGES[:-1], PIPELINE_STAGES[1:]))

        cost_tables: dict = {}

        for stage_i, stage_j in pairs:
            table: dict[tuple, dict[tuple, float]] = {}

            for wip_i in WIP_DOMAIN:
                for wip_j in WIP_DOMAIN:
                    # Cria config modificada para este par
                    cfg = copy.deepcopy(base_config)
                    for stage_cfg in cfg.stages:
                        if stage_cfg.name == stage_i:
                            stage_cfg.wip_limit = wip_i
                            stage_cfg.n_servers = max(1, wip_i // 2)
                        elif stage_cfg.name == stage_j:
                            stage_cfg.wip_limit = wip_j
                            stage_cfg.n_servers = max(1, wip_j // 2)

                    # Executa simulação rápida (5 runs para estimativa)
                    cfg.n_runs = 5
                    sim = self.sim_engine.run(cfg)

                    # Extrai métricas da interface i→j
                    wt_ij = sim.stage_wip.get(stage_j, 0.0) * sim.mean_lead_time / max(sum(sim.stage_wip.values()), 1)
                    lt_ij = sim.mean_lead_time
                    th_ij = sim.mean_throughput

                    cost = gamma1 * wt_ij + gamma2 * lt_ij - gamma3 * th_ij

                    xi = (wip_i, "FIFO")
                    xj = (wip_j, "FIFO")
                    table.setdefault(xi, {})[xj] = max(cost, 0.0)

            cost_tables[(stage_i, stage_j)] = table

        logger.info(f"Funções de custo estimadas para {len(cost_tables)} pares de estágios.")
        return cost_tables

    # ------------------------------------------------------------------ #
    # Construção dos agentes e pseudo-árvore
    # ------------------------------------------------------------------ #

    def _build_agents(
        self,
        mining_result: MiningResult,
        constraint_stage: str,
        cost_tables: dict,
        gamma1: float,
        gamma2: float,
        gamma3: float,
    ) -> dict[str, DCOPAgent]:
        """
        Constrói os 5 agentes com a pseudo-árvore enraizada na restrição ativa.

        A hierarquia da pseudo-árvore segue a posição no fluxo de valor,
        com a restrição ativa como raiz (Seção 4.5.1 da tese).
        """
        # ID dos agentes
        agent_ids = {stage: f"a{i+1}" for i, stage in enumerate(PIPELINE_STAGES)}

        # Pseudo-árvore linear: raiz = restrição ativa
        # Estágios antes da restrição → ancestrais
        # Estágios depois → descendentes
        constraint_idx = PIPELINE_STAGES.index(constraint_stage)

        # Estrutura pai/filho na pseudo-árvore
        # (linear: cada nó tem no máximo 1 pai e 1 filho)
        parent_map: dict[str, str | None] = {}
        children_map: dict[str, list[str]] = {s: [] for s in PIPELINE_STAGES}

        for i, stage in enumerate(PIPELINE_STAGES):
            if i == 0:
                parent_map[stage] = None
            else:
                parent_map[stage] = PIPELINE_STAGES[i - 1]
                children_map[PIPELINE_STAGES[i - 1]].append(stage)

        agents: dict[str, DCOPAgent] = {}

        for stage in PIPELINE_STAGES:
            aid = agent_ids[stage]
            parent_stage = parent_map[stage]
            parent_aid = agent_ids[parent_stage] if parent_stage else None
            children_aids = [agent_ids[c] for c in children_map[stage]]
            is_constraint = stage == constraint_stage

            # Cria função de custo específica para este agente
            cost_fn = self._make_cost_fn(stage, cost_tables)

            # agent_id do vizinho → estágio do vizinho (pai + filhos), para o
            # cost_fn saber em qual par (stage_i, stage_j) da tabela consultar.
            neighbor_stages: dict[str, str] = {}
            if parent_aid:
                neighbor_stages[parent_aid] = parent_stage
            for child_stage, child_aid in zip(children_map[stage], children_aids):
                neighbor_stages[child_aid] = child_stage

            agent = DCOPAgent(
                agent_id=aid,
                stage_name=stage,
                parent_id=parent_aid,
                children_ids=children_aids,
                pseudo_children_ids=[],
                cost_fn=cost_fn,
                is_constraint=is_constraint,
                gamma1=gamma1,
                gamma2=gamma2,
                gamma3=gamma3,
                neighbor_stages=neighbor_stages,
            )
            agents[aid] = agent

        return agents

    def _make_cost_fn(
        self,
        stage_i: str,
        cost_tables: dict,
    ) -> callable:
        """Retorna a função de custo f_ij para o agente do estágio `stage_i`.

        Ciente de direção: o mesmo agente pode ser o lado esquerdo (stage_i)
        do par com seu filho, e o lado direito (stage_j) do par com seu pai.
        Antes, o lookup não sabia o estágio do vizinho e pegava a primeira
        tabela que citasse `stage_i` em qualquer posição, aplicando a chave
        errada na maioria das vezes (e caindo no fallback linear em WIP).
        """

        def cost_fn(neighbor_stage: str | None, own_value: tuple, neighbor_value: tuple) -> float:
            wip_own, _ = own_value
            wip_nb, _ = neighbor_value
            own_key = (wip_own, "FIFO")
            nb_key = (wip_nb, "FIFO")

            # Papel "esquerdo": este agente é stage_i do par (stage_i=próprio, stage_j=vizinho)
            table = cost_tables.get((stage_i, neighbor_stage))
            if table is not None:
                val = table.get(own_key, {}).get(nb_key)
                if val is not None:
                    return val

            # Papel "direito": este agente é stage_j do par (stage_i=vizinho, stage_j=próprio)
            table = cost_tables.get((neighbor_stage, stage_i))
            if table is not None:
                val = table.get(nb_key, {}).get(own_key)
                if val is not None:
                    return val

            # Fallback: não deveria ocorrer depois que as tabelas passaram a
            # cobrir todo o WIP_DOMAIN — mantido por segurança (par
            # desconhecido, ex.: vizinho fora da pseudo-árvore).
            return float(wip_own + wip_nb)

        return cost_fn

    # ------------------------------------------------------------------ #
    # Custo global
    # ------------------------------------------------------------------ #

    def _compute_global_cost(
        self,
        agents: dict[str, DCOPAgent],
        cost_tables: dict,
    ) -> float:
        """Soma o custo f_ij sobre todas as arestas da pseudo-árvore."""
        total = 0.0
        stage_to_agent = {a.stage_name: a for a in agents.values()}

        for (stage_i, stage_j), table in cost_tables.items():
            agent_i = stage_to_agent.get(stage_i)
            agent_j = stage_to_agent.get(stage_j)
            if not agent_i or not agent_j:
                continue
            wi, _ = agent_i.state.current_value
            wj, _ = agent_j.state.current_value
            xi_key = (wi, "FIFO")
            xj_key = (wj, "FIFO")
            cost = table.get(xi_key, {}).get(xj_key, 0.0)
            total += cost

        return round(total, 3)
