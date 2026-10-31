"""Avaliação sistêmica — comparação dos cenários S0, S1, S2.

Valida as hipóteses da tese:
  H1: MAE(simulação, observado) < 15%
  H2: S1 (what-if) produz melhoria estatisticamente significativa vs S0 (p < 0,05)
  H3: S2 (DCOP) é superior a S1 (what-if) (p < 0,05)
  H4: Escalabilidade (análise de tempo de convergência por número de agentes)

Testes estatísticos: Mann-Whitney U (não-paramétrico) sobre amostras de Lead Time
e Throughput das réplicas de simulação.
"""

import logging
from dataclasses import dataclass, field

import scipy.stats as stats

from ezekiel.legacy.engine import SimResult
from ezekiel.mining.analyzer import MiningResult

logger = logging.getLogger(__name__)

SIGNIFICANCE_LEVEL = 0.05


@dataclass
class HypothesisResult:
    name: str
    validated: bool
    statistic: float
    p_value: float
    description: str
    detail: str = ""


@dataclass
class ScenarioMetrics:
    label: str         # S0, S1, S2
    description: str
    lead_time: float
    waiting_time: float
    throughput: float
    flow_efficiency: float
    mean_wip: float
    source: str = "simulation"   # simulation | historical


@dataclass
class EvaluationResult:
    """Resultado completo da avaliação sistêmica."""
    scenarios: list[ScenarioMetrics] = field(default_factory=list)
    hypotheses: list[HypothesisResult] = field(default_factory=list)
    mae_simulation: float = 0.0
    best_scenario: ScenarioMetrics | None = None

    def summary(self) -> dict:
        return {
            "mae_pct": round(self.mae_simulation * 100, 1),
            "h1_validated": self._get_h("H1").validated if self._get_h("H1") else False,
            "h2_validated": self._get_h("H2").validated if self._get_h("H2") else False,
            "h3_validated": self._get_h("H3").validated if self._get_h("H3") else False,
            "best_scenario": self.best_scenario.label if self.best_scenario else None,
            "scenarios": [
                {
                    "label": s.label,
                    "lead_time_days": round(s.lead_time, 1),
                    "waiting_time_days": round(s.waiting_time, 1),
                    "throughput_per_week": round(s.throughput, 1),
                    "flow_efficiency_pct": round(s.flow_efficiency * 100, 1),
                    "mean_wip": round(s.mean_wip, 1),
                }
                for s in self.scenarios
            ],
            "hypotheses": [
                {
                    "name": h.name,
                    "validated": h.validated,
                    "p_value": round(h.p_value, 4),
                    "description": h.description,
                }
                for h in self.hypotheses
            ],
        }

    def _get_h(self, name: str) -> HypothesisResult | None:
        return next((h for h in self.hypotheses if h.name == name), None)


class Evaluator:
    """Executa a avaliação sistêmica dos cenários."""

    def evaluate(
        self,
        mining_result: MiningResult,
        sim_s0: SimResult,      # S0: baseline simulado
        sim_s1: SimResult,      # S1: what-if
        sim_s2: SimResult,      # S2: DCOP coordenado
    ) -> EvaluationResult:
        logger.info("Iniciando avaliação sistêmica S0 / S1 / S2...")

        # Constrói ScenarioMetrics para cada cenário
        s0_hist = ScenarioMetrics(
            label="S0",
            description="Baseline histórico",
            lead_time=mining_result.mean_lead_time_days,
            waiting_time=mining_result.mean_waiting_time_days,
            throughput=mining_result.mean_throughput_per_week,
            flow_efficiency=mining_result.flow_efficiency,
            mean_wip=sum(m.mean_wip for m in mining_result.stage_metrics),
            source="historical",
        )

        s0_sim = ScenarioMetrics(
            label="S0-sim",
            description="Baseline simulado (calibração)",
            lead_time=sim_s0.mean_lead_time,
            waiting_time=sim_s0.mean_waiting_time,
            throughput=sim_s0.mean_throughput,
            flow_efficiency=sim_s0.flow_efficiency,
            mean_wip=sum(sim_s0.stage_wip.values()),
            source="simulation",
        )

        s1 = ScenarioMetrics(
            label="S1",
            description="What-if (+revisores, WIP limitado)",
            lead_time=sim_s1.mean_lead_time,
            waiting_time=sim_s1.mean_waiting_time,
            throughput=sim_s1.mean_throughput,
            flow_efficiency=sim_s1.flow_efficiency,
            mean_wip=sum(sim_s1.stage_wip.values()),
            source="simulation",
        )

        s2 = ScenarioMetrics(
            label="S2",
            description="DCOP/ADOPT coordenado",
            lead_time=sim_s2.mean_lead_time,
            waiting_time=sim_s2.mean_waiting_time,
            throughput=sim_s2.mean_throughput,
            flow_efficiency=sim_s2.flow_efficiency,
            mean_wip=sum(sim_s2.stage_wip.values()),
            source="simulation",
        )

        # MAE (H1)
        mae = self._compute_mae(s0_hist, s0_sim)

        # Hipóteses
        h1 = self._test_h1(mae)
        h2 = self._test_h2(sim_s0, sim_s1)
        h3 = self._test_h3(sim_s1, sim_s2)

        # Melhor cenário = menor lead time
        best = min([s0_hist, s1, s2], key=lambda s: s.lead_time)

        result = EvaluationResult(
            scenarios=[s0_hist, s0_sim, s1, s2],
            hypotheses=[h1, h2, h3],
            mae_simulation=mae,
            best_scenario=best,
        )

        logger.info(f"[OK] Avaliação concluída. Melhor cenário: {best.label}")
        for h in [h1, h2, h3]:
            status = "✓ VALIDADA" if h.validated else "✗ NÃO VALIDADA"
            logger.info(f"  {h.name}: {status} (p={h.p_value:.4f}) — {h.description}")

        return result

    # ------------------------------------------------------------------ #
    # Hipótese H1 — Acurácia da simulação (MAE < 15%)
    # ------------------------------------------------------------------ #

    def _compute_mae(self, historical: ScenarioMetrics, simulated: ScenarioMetrics) -> float:
        obs = historical.lead_time
        sim = simulated.lead_time
        return abs(sim - obs) / obs if obs > 0 else 0.0

    def _test_h1(self, mae: float) -> HypothesisResult:
        threshold = 0.15
        validated = mae < threshold
        return HypothesisResult(
            name="H1",
            validated=validated,
            statistic=mae,
            p_value=0.0,  # H1 é um critério de engenharia, não teste estatístico
            description=f"MAE da simulação < 15% (observado: {mae*100:.1f}%)",
            detail=f"threshold={threshold*100:.0f}%, mae={mae*100:.1f}%",
        )

    # ------------------------------------------------------------------ #
    # Hipótese H2 — S1 superior ao S0 (p < 0,05)
    # ------------------------------------------------------------------ #

    def _test_h2(self, s0: SimResult, s1: SimResult) -> HypothesisResult:  # noqa: N802
        """
        Mann-Whitney U unilateral: H0: LT(S1) >= LT(S0), H1: LT(S1) < LT(S0).
        Usa as amostras de lead_time_samples das réplicas de simulação.
        """
        if not s0.lead_time_samples or not s1.lead_time_samples:
            return HypothesisResult(
                name="H2",
                validated=False,
                statistic=0.0,
                p_value=1.0,
                description="S1 (what-if) reduz Lead Time vs S0 (p < 0,05)",
                detail="amostras insuficientes",
            )

        stat, p = stats.mannwhitneyu(
            s1.lead_time_samples, s0.lead_time_samples, alternative="less"
        )
        validated = bool(p < SIGNIFICANCE_LEVEL)

        return HypothesisResult(
            name="H2",
            validated=validated,
            statistic=float(stat),
            p_value=float(p),
            description=f"S1 (what-if) reduz Lead Time vs S0 (p={p:.4f})",
        )

    # ------------------------------------------------------------------ #
    # Hipótese H3 — S2 superior a S1 (p < 0,05)
    # ------------------------------------------------------------------ #

    def _test_h3(self, s1: SimResult, s2: SimResult) -> HypothesisResult:
        """
        Mann-Whitney U unilateral: H0: LT(S2) >= LT(S1), H1: LT(S2) < LT(S1).
        """
        if not s1.lead_time_samples or not s2.lead_time_samples:
            return HypothesisResult(
                name="H3",
                validated=False,
                statistic=0.0,
                p_value=1.0,
                description="S2 (DCOP) é superior a S1 (what-if) (p < 0,05)",
                detail="amostras insuficientes",
            )

        stat, p = stats.mannwhitneyu(
            s2.lead_time_samples, s1.lead_time_samples, alternative="less"
        )
        validated = bool(p < SIGNIFICANCE_LEVEL)

        return HypothesisResult(
            name="H3",
            validated=validated,
            statistic=float(stat),
            p_value=float(p),
            description=f"S2 (DCOP) é superior a S1 (what-if) (p={p:.4f})",
        )
