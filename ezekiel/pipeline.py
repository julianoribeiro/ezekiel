"""Orquestrador do pipeline — testa as hipóteses H1--H4 da tese.

  Camadas 1+2  leitura de OCEL 2.0 (The Public Jira Dataset) → log de casos
  Camada 3     log de casos/visitas, identificação da restrição ativa
  Camada 4a    simulação calibrada e validada em janela temporal (H1)
  Camada 4b    DCOP/ADOPT com custos estimados por simulação (C2), C1, C3
  Camada 5     avaliação estatística pareada de H2, H3 e H4

A implementação anterior (S0/S1/S2, MAE < 15%) está em `ezekiel.legacy`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ezekiel.config import settings
from ezekiel.data.eventlog import JIRA_PROFILE, PROFILES, EventLog, LogProfile
from ezekiel.experiments.hypotheses import ExperimentSettings, run_dataset, to_jsonable
from ezekiel.ocel.loader import OCELLoader
from ezekiel.ocel.ocel2 import ObjectCentricLog
from ezekiel.simulation.engine import Simulator

logger = logging.getLogger(__name__)

# Mantido por compatibilidade — use JIRA_PROFILE.
JIRA_COMPLETION_STATUSES = JIRA_PROFILE.completion_statuses


def _is_builder_format(raw: dict) -> bool:
    ev = next(iter(raw.get("ocel:events", {}).values()), {})
    return "ocel:activity" in ev


def load_eventlog(
    ocel_path: str | Path | None = None,
    ocel_dir: str | Path | None = None,
    completion_statuses: frozenset[str] = frozenset(),
    profile: LogProfile | str | None = None,
) -> tuple[EventLog, str]:
    """Carrega o log de casos. `profile`: "jira" (padrão) — ver
    `LogProfile`. `completion_statuses` só é usado sem perfil explícito
    (compatibilidade)."""
    if isinstance(profile, str):
        profile = PROFILES[profile]
    if profile is None:
        profile = (LogProfile("custom", completion_statuses=completion_statuses)
                   if completion_statuses else JIRA_PROFILE)
    loader = OCELLoader()
    if ocel_path:
        p = Path(ocel_path)
        raw = loader._load_file(p)
        if ObjectCentricLog.is_ocel2(raw):   # OCEL 2.0 padrão, multiobjeto
            mp = p.with_suffix(".meta.json")
            meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            return EventLog.from_ocel2(ObjectCentricLog.from_json(raw, meta), profile), p.stem
        if _is_builder_format(raw):          # OCEL sintético (formato legado)
            return EventLog.from_ocel(raw, profile), p.stem
        events, _, _ = loader._normalize(raw, source=p.stem)
        return EventLog.from_ocel({"ocel:events": events}, profile), p.stem
    if ocel_dir:
        ocel = loader.load_dir(ocel_dir)
        return EventLog.from_ocel(ocel, profile), Path(ocel_dir).name

    raise ValueError("Informe ocel_path ou ocel_dir (logs de bases públicas).")


class Pipeline:
    def run(
        self,
        output_dir: str | None = None,
        ocel_dir: str | None = None,
        ocel_path: str | None = None,
        experiment: ExperimentSettings | None = None,
        completion_statuses: frozenset[str] = frozenset(),
        n_jobs: int | None = None,
        profile: str | None = None,
    ) -> dict:
        out = Path(output_dir or settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log, label = load_eventlog(ocel_path, ocel_dir, completion_statuses, profile)
        s = experiment or ExperimentSettings()
        with Simulator(n_jobs) as sim:
            result = to_jsonable(run_dataset(log, label, s, sim))
        path = out / "hypotheses_result.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info(f"Resultado salvo em: {path}")
        return result
