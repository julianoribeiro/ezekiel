"""Tipos de mensagens do protocolo ADOPT adaptado.

O protocolo original (Modi et al., 2005) define três tipos:
  VALUE     — informa o valor atual da variável do agente
  COST      — informa o custo acumulado a partir de um filho
  THRESHOLD — informa o limiar de custo máximo aceito

Esta tese acrescenta um quarto tipo (Equação 4.3):
  CONSTRAINT-STATE — transmite o estado operacional corrente do estágio
                     ⟨ρ_i, L_i, WT_i⟩ para monitoramento e diagnóstico.
                     NÃO altera as funções de custo durante a execução
                     (garantia de terminação preservada — vide Seção 4.5.1).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MessageType(Enum):
    VALUE = auto()
    COST = auto()
    THRESHOLD = auto()
    CONSTRAINT_STATE = auto()   # extensão da tese — Eq. 4.3


@dataclass
class Message:
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    payload: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construtores semânticos
    # ------------------------------------------------------------------ #

    @classmethod
    def value(cls, sender: str, receiver: str, variable: str, domain_value: Any) -> "Message":
        """VALUE(x_i, d_i) — agente informa sua atribuição atual."""
        return cls(
            msg_type=MessageType.VALUE,
            sender_id=sender,
            receiver_id=receiver,
            payload={"variable": variable, "value": domain_value},
        )

    @classmethod
    def cost(cls, sender: str, receiver: str, cost_value: float, context: dict) -> "Message":
        """COST(c_ij) — filho informa custo acumulado ao pai."""
        return cls(
            msg_type=MessageType.COST,
            sender_id=sender,
            receiver_id=receiver,
            payload={"cost": cost_value, "context": context},
        )

    @classmethod
    def threshold(cls, sender: str, receiver: str, threshold_value: float) -> "Message":
        """THRESHOLD(t_i) — pai informa limiar de custo máximo ao filho."""
        return cls(
            msg_type=MessageType.THRESHOLD,
            sender_id=sender,
            receiver_id=receiver,
            payload={"threshold": threshold_value},
        )

    @classmethod
    def constraint_state(
        cls,
        sender: str,
        receiver: str,
        utilization: float,
        wip: float,
        waiting_time: float,
    ) -> "Message":
        """
        CONSTRAINT-STATE(a_i) = ⟨ρ_i, L_i, WT_i⟩  — Equação 4.3 da tese.

        Permite que vizinhos conheçam o estado operacional corrente do estágio
        para diagnóstico e monitoramento. Não altera as funções de custo.
        """
        return cls(
            msg_type=MessageType.CONSTRAINT_STATE,
            sender_id=sender,
            receiver_id=receiver,
            payload={
                "utilization": utilization,     # ρ_i
                "wip": wip,                      # L_i
                "waiting_time": waiting_time,    # WT_i
            },
        )
