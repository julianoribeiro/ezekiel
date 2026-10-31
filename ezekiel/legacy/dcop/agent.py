"""Agente DCOP/ADOPT para otimização distribuída do pipeline.

Cada agente representa um estágio do pipeline de desenvolvimento:
  a1 — Backlog
  a2 — Desenvolvimento
  a3 — Revisão de Código  (restrição ativa ★)
  a4 — Testes
  a5 — Implantação

Variáveis controladas por cada agente (Seção 4.5.1 da tese):
  x_i = (w_i, p_i)
    w_i ∈ {1,2,3,4,5}           limite WIP
    p_i ∈ {FIFO, urgency, cod}  política de priorização (FIFO / urgência / cost-of-delay)

Função de custo entre agentes i e j (Equação 4.2 da tese):
  f_ij(x_i, x_j) = E_sim[γ1·WT_ij + γ2·LT_ij − γ3·TH_ij]

Protocolo de mensagens: VALUE, COST, THRESHOLD, CONSTRAINT-STATE.
Estrutura: pseudo-árvore com raiz = restrição ativa (ToC).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ezekiel.legacy.dcop.messages import Message, MessageType

logger = logging.getLogger(__name__)

# Domínios discretos das variáveis de decisão (conforme tese)
WIP_DOMAIN = [1, 2, 3, 4, 5]
PRIORITY_DOMAIN = ["FIFO", "urgency", "cost-of-delay"]

# Produto cartesiano do domínio de cada agente
AGENT_DOMAIN: list[tuple[int, str]] = [
    (w, p) for w in WIP_DOMAIN for p in PRIORITY_DOMAIN
]


@dataclass
class AgentState:
    """Estado interno de um agente ADOPT."""
    agent_id: str
    stage_name: str

    # Atribuição corrente: (wip_limit, priority_policy)
    current_value: tuple[int, str] = (3, "FIFO")

    # Limites ADOPT
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    threshold: float = float("inf")

    # Custo local acumulado (somado dos filhos)
    context_cost: float = 0.0

    # Contexto de atribuições dos vizinhos (para avaliação de f_ij)
    neighbor_context: dict[str, tuple[int, str]] = field(default_factory=dict)

    # Estado operacional corrente — atualizado via CONSTRAINT-STATE
    utilization: float = 0.0    # ρ_i
    wip_current: float = 0.0    # L_i
    waiting_time: float = 0.0   # WT_i

    # Caixa de entrada de mensagens
    inbox: list[Message] = field(default_factory=list)
    # Mensagens enviadas neste ciclo
    outbox: list[Message] = field(default_factory=list)


class DCOPAgent:
    """
    Agente ADOPT para um estágio do pipeline.

    O agente executa de forma assíncrona: em cada ciclo recebe mensagens,
    avalia o custo local, atualiza seus bounds e decide se muda de valor.
    """

    def __init__(
        self,
        agent_id: str,
        stage_name: str,
        parent_id: str | None,
        children_ids: list[str],
        pseudo_children_ids: list[str],
        cost_fn: Callable[[str, tuple, tuple], float],
        is_constraint: bool = False,
        gamma1: float = 1.0,
        gamma2: float = 0.5,
        gamma3: float = 0.3,
        neighbor_stages: dict[str, str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.stage_name = stage_name
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.pseudo_children_ids = pseudo_children_ids
        self.cost_fn = cost_fn   # f_ij(neighbor_stage, x_i, x_j) estimado pela simulação
        self.is_constraint = is_constraint  # True se for a restrição ativa (raiz)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        # agent_id do vizinho → nome do estágio do vizinho. Necessário para o
        # cost_fn saber de qual par (stage_i, stage_j) da tabela de custo
        # consultar — antes o lookup não sabia o estágio do vizinho e testava
        # o primeiro par que citasse o próprio estágio, na maioria das vezes
        # errando o par/direção certos.
        self.neighbor_stages: dict[str, str] = neighbor_stages or {}

        self.state = AgentState(agent_id=agent_id, stage_name=stage_name)

    # ------------------------------------------------------------------ #
    # Interface principal
    # ------------------------------------------------------------------ #

    def receive(self, msg: Message) -> None:
        self.state.inbox.append(msg)

    def step(self) -> list[Message]:
        """
        Executa um ciclo ADOPT:
          1. Processa mensagens recebidas
          2. Avalia custo local para cada valor do domínio
          3. Atualiza lower/upper bounds
          4. Decide se muda de valor
          5. Gera mensagens de saída
        """
        self.state.outbox = []
        self._process_inbox()
        self._update_value()
        self._propagate()
        msgs = list(self.state.outbox)
        self.state.inbox = []
        return msgs

    def get_assignment(self) -> dict:
        w, p = self.state.current_value
        return {
            "agent_id": self.agent_id,
            "stage": self.stage_name,
            "wip_limit": w,
            "priority_policy": p,
            "is_constraint": self.is_constraint,
            "lower_bound": round(self.state.lower_bound, 4) if self.state.lower_bound != float("inf") else None,
            "upper_bound": round(self.state.upper_bound, 4) if self.state.upper_bound != float("inf") else None,
            "utilization": round(self.state.utilization, 3),
            "waiting_time": round(self.state.waiting_time, 2),
        }

    # ------------------------------------------------------------------ #
    # Processamento de mensagens
    # ------------------------------------------------------------------ #

    def _process_inbox(self) -> None:
        for msg in self.state.inbox:
            if msg.msg_type == MessageType.VALUE:
                self._handle_value(msg)
            elif msg.msg_type == MessageType.COST:
                self._handle_cost(msg)
            elif msg.msg_type == MessageType.THRESHOLD:
                self._handle_threshold(msg)
            elif msg.msg_type == MessageType.CONSTRAINT_STATE:
                self._handle_constraint_state(msg)

    def _handle_value(self, msg: Message) -> None:
        sender = msg.sender_id
        value = msg.payload.get("value")
        if value is not None:
            self.state.neighbor_context[sender] = tuple(value) if isinstance(value, list) else value

    def _handle_cost(self, msg: Message) -> None:
        self.state.context_cost = msg.payload.get("cost", 0.0)
        # A mensagem COST já carrega o valor corrente do filho (payload["value"]),
        # mas até aqui era descartado — o pai nunca aprendia a atribuição do
        # filho, só o custo agregado. Sem isso, um agente sem pai (a raiz da
        # pseudo-árvore, tipicamente a restrição ativa) nunca recebe nenhuma
        # VALUE (que só flui pai→filho) e seu neighbor_context ficava vazio
        # para sempre, caindo no fallback trivial de _evaluate_local_cost
        # (custo = WIP, sempre minimizado em WIP=1, ignorando qualquer troca
        # real estimada pela simulação).
        value = msg.payload.get("value")
        if value is not None:
            self.state.neighbor_context[msg.sender_id] = (
                tuple(value) if isinstance(value, list) else value
            )

    def _handle_threshold(self, msg: Message) -> None:
        self.state.threshold = msg.payload.get("threshold", float("inf"))

    def _handle_constraint_state(self, msg: Message) -> None:
        """Recebe estado operacional do vizinho — apenas para diagnóstico."""
        sender = msg.sender_id
        payload = msg.payload
        logger.debug(
            f"[{self.agent_id}] CONSTRAINT-STATE de {sender}: "
            f"ρ={payload.get('utilization', 0):.2f}, "
            f"L={payload.get('wip', 0):.1f}, "
            f"WT={payload.get('waiting_time', 0):.1f}d"
        )

    # ------------------------------------------------------------------ #
    # Atualização de valor (ADOPT backtracking)
    # ------------------------------------------------------------------ #

    def _update_value(self) -> None:
        """
        Avalia todos os valores do domínio e seleciona o que minimiza o custo local.
        Atualiza lower e upper bounds conforme ADOPT.
        """
        best_value = self.state.current_value
        best_cost = float("inf")

        for candidate in AGENT_DOMAIN:
            local_cost = self._evaluate_local_cost(candidate)
            if local_cost < best_cost:
                best_cost = local_cost
                best_value = candidate

        self.state.lower_bound = best_cost
        self.state.upper_bound = best_cost + self.state.context_cost

        # Muda de valor se encontrar algo estritamente melhor e abaixo do threshold
        if best_cost < self._current_cost() and best_cost <= self.state.threshold:
            old = self.state.current_value
            self.state.current_value = best_value
            if old != best_value:
                logger.debug(
                    f"[{self.agent_id}] Mudou de {old} → {best_value} "
                    f"(custo: {self._current_cost():.3f} → {best_cost:.3f})"
                )

    def _evaluate_local_cost(self, candidate: tuple[int, str]) -> float:
        """
        Avalia f_ij(x_i, x_j) para o candidato, somando sobre todos os vizinhos.

        f_ij é estimado pela simulação (injetado via cost_fn no construtor).
        """
        total = 0.0
        for neighbor_id, neighbor_value in self.state.neighbor_context.items():
            neighbor_stage = self.neighbor_stages.get(neighbor_id)
            total += self.cost_fn(neighbor_stage, candidate, neighbor_value)
        # Fallback só se aplica no(s) primeiro(s) ciclo(s), antes de qualquer
        # VALUE/COST ter chegado — depois disso todo agente (incluindo a
        # raiz, via o valor embutido na mensagem COST do filho) tem contexto.
        if not self.state.neighbor_context:
            wip, _ = candidate
            total = float(wip)
        return total

    def _current_cost(self) -> float:
        return self._evaluate_local_cost(self.state.current_value)

    # ------------------------------------------------------------------ #
    # Propagação de mensagens
    # ------------------------------------------------------------------ #

    def _propagate(self) -> None:
        """Gera e enfileira as mensagens de saída para este ciclo."""
        w, p = self.state.current_value

        # VALUE → todos os filhos e pseudo-filhos
        for child_id in self.children_ids + self.pseudo_children_ids:
            self.state.outbox.append(
                Message.value(self.agent_id, child_id, f"x_{self.agent_id}", [w, p])
            )

        # COST → pai (se não for raiz)
        if self.parent_id:
            self.state.outbox.append(
                Message.cost(
                    self.agent_id,
                    self.parent_id,
                    self.state.lower_bound,
                    {"value": [w, p]},
                )
            )

        # THRESHOLD → filhos
        if self.children_ids:
            child_threshold = (
                self.state.threshold - self.state.lower_bound
            ) / len(self.children_ids)
            for child_id in self.children_ids:
                self.state.outbox.append(
                    Message.threshold(self.agent_id, child_id, max(child_threshold, 0.0))
                )

        # CONSTRAINT-STATE → todos os vizinhos (se tiver estado operacional)
        if self.state.utilization > 0:
            for neighbor_id in self.children_ids + ([self.parent_id] if self.parent_id else []):
                self.state.outbox.append(
                    Message.constraint_state(
                        self.agent_id,
                        neighbor_id,
                        self.state.utilization,
                        self.state.wip_current,
                        self.state.waiting_time,
                    )
                )

    def update_operational_state(
        self, utilization: float, wip: float, waiting_time: float
    ) -> None:
        """Atualiza o estado operacional do agente a partir dos logs/simulação."""
        self.state.utilization = utilization
        self.state.wip_current = wip
        self.state.waiting_time = waiting_time
