"""Orquestrador do pipeline completo de 5 camadas.

Executa sequencialmente:
  Camada 1 → Camada 2 → Camada 3 → Camada 4a/4b → Camada 5

Pode ser invocado via CLI ou pela API FastAPI.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ezekiel.config import settings
from ezekiel.legacy.dcop.coordinator import DCOPCoordinator, DCOPResult, apply_dcop_assignments
from ezekiel.legacy.evaluator import Evaluator, EvaluationResult
from ezekiel.mining.analyzer import MiningResult, ProcessMiningAnalyzer
from ezekiel.ocel.loader import OCELLoader
from ezekiel.legacy.engine import SimulationEngine, SimResult, build_whatif_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    ocel: dict | None = None
    mining: MiningResult | None = None
    sim_s0: SimResult | None = None
    sim_s1: SimResult | None = None
    sim_s2: SimResult | None = None
    dcop: DCOPResult | None = None
    evaluation: EvaluationResult | None = None
    mae: float = 0.0

    def summary(self) -> dict:
        out: dict = {}
        if self.mining:
            out["mining"] = self.mining.summary()
        if self.sim_s0:
            out["sim_s0"] = self.sim_s0.summary()
        if self.sim_s1:
            out["sim_s1"] = self.sim_s1.summary()
        if self.sim_s2:
            out["sim_s2"] = self.sim_s2.summary()
        if self.dcop:
            out["dcop"] = self.dcop.summary()
        if self.evaluation:
            out["evaluation"] = self.evaluation.summary()
        return out


class Pipeline:
    """Pipeline completo de otimização de processos interdependentes."""

    def __init__(self) -> None:
        self.sim_engine = SimulationEngine()
        self.dcop_coordinator = DCOPCoordinator(self.sim_engine)

    def run(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        output_dir: str | None = None,
        ocel_dir: str | None = None,
        ocel_path: str | None = None,
    ) -> PipelineResult:
        """
        Executa o pipeline completo.

        Modos de entrada (em ordem de prioridade):
          ocel_dir  — carrega todos os .json da pasta (modo principal)
          ocel_path — carrega um único arquivo OCEL já gerado
        """
        output = Path(output_dir or settings.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = PipelineResult()

        # ── Camada 1 + 2: Fonte de dados ───────────────────────────────
        if ocel_dir:
            logger.info(f"═══ Camadas 1+2 — Carregando OCELs de: {ocel_dir} ═══")
            loader = OCELLoader()
            result.ocel = loader.load_dir(ocel_dir)
            summary = loader.summary(result.ocel)
            logger.info(f"Resumo: {summary}")

        elif ocel_path:
            logger.info(f"═══ Camadas 1+2 — Carregando OCEL de: {ocel_path} ═══")
            with open(ocel_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Formato legado (ocel:time), normaliza via loader
            loader = OCELLoader()
            result.ocel = loader.load_dir(Path(ocel_path).parent)

        else:
            raise ValueError("Informe ocel_dir ou ocel_path.")

        # ── Camada 3: Process Mining ────────────────────────────────────
        logger.info("═══ Camada 3 — Process Mining + ToC ═══")
        analyzer = ProcessMiningAnalyzer()
        result.mining = analyzer.analyze(result.ocel)

        # ── Camada 4a: Simulação S0 (baseline) ─────────────────────────
        logger.info("═══ Camada 4a — Simulação S0 (baseline) ═══")
        base_config = self.sim_engine.calibrate(result.mining)
        result.sim_s0 = self.sim_engine.run(base_config)

        mae = self.sim_engine.compute_mae(result.sim_s0, result.mining)
        result.mae = mae
        logger.info(f"MAE simulação vs histórico: {mae*100:.1f}% (H1: < 15%)")

        # ── Camada 4a: Simulação S1 (what-if) ───────────────────────────
        # Aplica intervenção no estágio gargalo identificado pelo mining
        logger.info("═══ Camada 4a — Simulação S1 (what-if) ═══")
        constraint_stage = (
            result.mining.active_constraint.stage
            if result.mining.active_constraint
            else "Revisão de Código"
        )
        logger.info(f"What-if: reduzindo WIP do gargalo '{constraint_stage}'")
        whatif_config = build_whatif_config(
            base_config,
            constraint_stage=constraint_stage,
            extra_servers=2,
            wip_reduction_pct=0.30,   # reduz 30% o WIP do gargalo
        )
        result.sim_s1 = self.sim_engine.run(whatif_config)

        # ── Camada 4b: DCOP/ADOPT → S2 ──────────────────────────────────
        logger.info("═══ Camada 4b — DCOP/ADOPT ═══")
        result.dcop = self.dcop_coordinator.run(result.mining, base_config)

        # Simula S2 com a configuração ótima do DCOP
        dcop_config = apply_dcop_assignments(base_config, result.dcop)
        result.sim_s2 = self.sim_engine.run(dcop_config)

        # ── Camada 5: Avaliação ──────────────────────────────────────────
        logger.info("═══ Camada 5 — Avaliação Sistêmica ═══")
        evaluator = Evaluator()
        result.evaluation = evaluator.evaluate(
            result.mining, result.sim_s0, result.sim_s1, result.sim_s2
        )

        # Salva resultado
        result_file = output / "pipeline_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.summary(), f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Resultado salvo em: {result_file}")

        return result
