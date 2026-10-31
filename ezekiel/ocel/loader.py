"""Loader para logs OCEL (formato legado) convertidos de issues do Jira
(ocel-jira-*.json, gerados por scripts/build_jira_ocel.py a partir do
changelog de status do The Public Jira Dataset).

Carrega todos os arquivos .json da pasta ocels/, unifica os logs e
normaliza para a estrutura interna usada pelo ProcessMiningAnalyzer.

Formato real dos arquivos:
  ocel:events:
    <id>:
      ocel:type: "StateTransition"
      ocel:time: "2026-03-26T19:23:31.52Z"
      ocel:attributes: [{name, value}, ...]   # fromColumn, toColumn, fromState, toState
      ocel:relationships: [{objectId, qualifier}]

  ocel:objects:
    <wi-id>:
      ocel:type: "WorkItem"
      ocel:attributes: [{name, time, value}, ...]  # title, workItemType, project, assignedTo
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapeamento de status do Jira (The Public Jira Dataset) → estágio do
# pipeline da tese. Cada Jira público customiza o próprio workflow; os
# status são agrupados pelo estágio de destino.
COLUMN_TO_STAGE: dict[str, str] = {
    # Apache/Zookeeper (workflow padrão do Jira):
    #   Open             → aberta, aguardando alguém puxar (fila de entrada)
    #   In Progress       → desenvolvimento ativo
    #   Patch Available   → patch submetido, aguardando revisão/commit
    #   Resolved          → resolvida, aguardando confirmação/fechamento
    #   Reopened          → devolvida à fila (sinaliza retrabalho)
    #   Closed            → estado terminal
    "Open":              "Backlog",
    "Reopened":          "Backlog",
    "In Progress":       "Development",
    "Patch Available":   "Code Review",
    "Resolved":          "QA Testing",
    "Closed":            "Deploy",

    # Demais 15 repositórios do dataset.
    "New":                     "Backlog",
    "Backlog":                 "Backlog",
    "To Do":                   "Backlog",
    "Blocked":                 "Backlog",
    "Blocked Externally":      "Backlog",
    "Confirmed":                "Backlog",
    "Stalled":                  "Backlog",
    "Triage":                   "Backlog",
    "Needs More Info":           "Backlog",
    "Need More Info":            "Backlog",
    "Awaiting Information":      "Backlog",
    "Returned":                  "Backlog",
    "Pause":                     "Backlog",
    "Define":                    "Backlog",
    "Deferred":                  "Backlog",
    "Raw":                       "Backlog",
    "Refine":                    "Backlog",
    "Ready to Groom":            "Backlog",
    "Needs Scheduling":          "Backlog",
    "Waiting For User Input":   "Backlog",
    "Reported":                 "Backlog",
    "Gathering Interest":        "Backlog",
    "Gathering Impact":          "Backlog",
    "Needs Triage":              "Backlog",
    "Future Consideration":      "Backlog",
    "Long Term Backlog":         "Backlog",
    "Short Term Backlog":        "Backlog",
    "Under Consideration":       "Backlog",

    "Ready For Dev":            "Development",
    "Ready for development":    "Development",
    "Ready for Development":    "Development",
    "Investigating":            "Development",
    "In Development":           "Development",
    "Ready to Develop":         "Development",
    "Development":              "Development",
    "Coding In Progress":       "Development",

    "Product Review":           "Code Review",
    "Reviewing":                "Code Review",
    "In Review":                "Code Review",
    "Ready for Code Review":    "Code Review",
    "In CR Review":             "Code Review",
    "Waiting for Review":       "Code Review",
    "In Code Review":           "Code Review",
    "Awaiting Review":          "Code Review",
    "Pull Request Sent":        "Code Review",

    "In QA":                    "QA Testing",
    "Awaiting Verification":    "QA Testing",
    "Pending QA":               "QA Testing",
    "Unverified":               "QA Testing",
    "To Be Tested":             "QA Testing",
    "Testing In Progress":      "QA Testing",
    "Needs Verification":       "QA Testing",

    "Staging":                  "Staging",
    "Verified":                 "Staging",

    "Pending Release":         "Deploy",
    "Approved":                 "Deploy",
    "Ready For Merge":          "Deploy",
    "Withdrawn":                "Deploy",
    "Published":                "Deploy",
    "Not Being Considered":     "Deploy",
    "Waiting for Release":      "Deploy",

    # ── Variantes (maiúsculas, grafias alternativas) e status adicionais
    # vistos em instâncias menores do dataset — mesma lógica de agrupamento.
    "OPEN":                     "Backlog",      # variante em caixa alta de "Open"
    "RESOLVED":                 "QA Testing",   # variante em caixa alta de "Resolved"
    "CLOSED":                   "Deploy",       # variante em caixa alta de "Closed"
    "ToDo":                     "Backlog",
    "Open - untriaged":         "Backlog",
    "Postponed":                "Backlog",
    "Acknowledged":             "Backlog",
    "Waiting for Response":     "Backlog",
    "Waiting for Feedback":     "Backlog",
    "Assigned":                 "Backlog",
    "Unassigned":               "Backlog",
    "Defered":                  "Backlog",      # grafia alternativa de "Deferred"
    "Draft":                    "Backlog",
    "Pending":                  "Backlog",
    "Monitor / On Hold":        "Backlog",

    "Awaiting Development":     "Development",

    "Under Review":             "Code Review",
    "Technical Review":         "Code Review",
    "Needs Reviewer":           "Code Review",
    "Waiting for Second Review": "Code Review",

    "Quality Review":           "QA Testing",
    "QA in Progress":           "QA Testing",
    "Awaiting Quality Review":  "QA Testing",
    "Integration Test":         "QA Testing",
    "Needs Verfication":        "QA Testing",   # grafia alternativa de "Needs Verification"

    "Awaiting Soak":            "Staging",
    "Soaking":                  "Staging",
    "Passed QA":                "Staging",
    "Implemented":              "Staging",

    "Needs Merge":              "Deploy",
    "Awaiting Deployment":      "Deploy",
    "Deployment":               "Deploy",
    "Integrated":               "Deploy",
    "Inactive - Pending Closure": "Deploy",
    "Awaiting Release":         "Deploy",
    "Done":                     "Deploy",
    # Rejeitadas/inválidas — estado terminal equivalente a Closed, não é
    # retrabalho (o item não volta para nenhuma etapa anterior).
    "Will Not Implement":       "Deploy",
    "Cannot Reproduce":         "Deploy",
    "Invalid":                  "Deploy",
    "Duplicate":                "Deploy",
    "Expected Behavior":        "Deploy",
}

# Status que representam FILA (o item está pronto e aguardando a próxima
# etapa começar a trabalhar nele) vs. status de TRABALHO ATIVO (alguém está
# de fato mexendo no item). Distinção baseada no próprio nome do status
# ("Ready to X" / "Waiting for..." = espera; "In Progress", "In QA" etc. =
# execução) — usada para medir waiting_time por item de
# forma real, a partir do log, em vez de uma suposição fixa de projeto.
WAITING_COLUMNS: frozenset[str] = frozenset({
    "New",
    "Ready to Develop",

    # Jira (Apache/Zookeeper) — fila de entrada, patch aguardando revisão,
    # resolvida aguardando confirmação: espera, não trabalho ativo (ver
    # COLUMN_TO_STAGE).
    "Open",
    "Reopened",
    "Patch Available",
    "Resolved",

    # Jira — demais repositórios. Só as filas reais: "In Review"/"Reviewing"/
    # "In Code Review"/"In CR Review" (alguém revisando agora, não espera),
    # "In QA"/"Testing In Progress"/"Investigating"/"In Development"/
    # "Coding In Progress"/"Staging"/"Verified" (trabalho ativo) ficam de
    # fora de propósito — ver COLUMN_TO_STAGE para o estágio de cada status.
    "Backlog",
    "To Do",
    "Blocked",
    "Blocked Externally",
    "Confirmed",
    "Stalled",
    "Triage",
    "Needs More Info",
    "Need More Info",
    "Awaiting Information",
    "Returned",
    "Pause",
    "Define",
    "Deferred",
    "Raw",
    "Refine",
    "Ready to Groom",
    "Needs Scheduling",
    "Waiting For User Input",
    "Reported",
    "Gathering Interest",
    "Gathering Impact",
    "Needs Triage",
    "Future Consideration",
    "Long Term Backlog",
    "Short Term Backlog",
    "Under Consideration",
    "Ready For Dev",
    "Ready for development",
    "Ready for Development",
    "Ready for Code Review",
    "Waiting for Review",
    "Awaiting Review",
    "Pull Request Sent",
    "Awaiting Verification",
    "Pending QA",
    "Unverified",
    "To Be Tested",
    "Needs Verification",
    "Pending Release",
    "Waiting for Release",

    "OPEN",
    "RESOLVED",
    "ToDo",
    "Open - untriaged",
    "Postponed",
    "Acknowledged",
    "Waiting for Response",
    "Waiting for Feedback",
    "Assigned",
    "Unassigned",
    "Defered",
    "Draft",
    "Pending",
    "Monitor / On Hold",
    "Awaiting Development",
    "Needs Reviewer",
    "Waiting for Second Review",
    "Awaiting Quality Review",
    "Needs Verfication",
    "Awaiting Soak",
    "Needs Merge",
    "Awaiting Deployment",
    "Inactive - Pending Closure",
    "Awaiting Release",
})


class OCELLoader:
    """Carrega e normaliza arquivos OCEL 2.0 da pasta ocels/."""

    def load_dir(self, ocel_dir: str | Path) -> dict:
        """
        Carrega todos os .json da pasta, unifica e retorna log normalizado.

        O log normalizado usa a estrutura interna do ProcessMiningAnalyzer:
          events[id] = {
            "activity": <stage>,           # estágio mapeado
            "ocel:timestamp": <iso_str>,
            "ocel:omap": [wi_id, ...],
            "raw": {...}                   # evento original
          }
          objects[id] = {
            "ocel:type": "WorkItem",
            "ocel:ovmap": {title, workItemType, project, assignedTo}
          }
        """
        ocel_dir = Path(ocel_dir)
        files = sorted(ocel_dir.glob("*.json"))

        if not files:
            raise FileNotFoundError(f"Nenhum arquivo .json encontrado em: {ocel_dir}")

        logger.info(f"Carregando {len(files)} arquivo(s) OCEL de: {ocel_dir}")

        merged_events: dict = {}
        merged_objects: dict = {}
        skipped = 0

        for fpath in files:
            raw = self._load_file(fpath)
            events, objects, n_skipped = self._normalize(raw, source=fpath.stem)
            merged_events.update(events)
            merged_objects.update(objects)
            skipped += n_skipped
            logger.info(
                f"  {fpath.name}: {len(events)} eventos úteis, "
                f"{len(objects)} objetos ({n_skipped} transições ignoradas)"
            )

        logger.info(
            f"[OK] Total: {len(merged_events)} eventos, "
            f"{len(merged_objects)} objetos, {skipped} transições ignoradas."
        )

        return {
            "ocel:global-log": {
                "ocel:version": "2.0",
                "ocel:ordering": "timestamp",
            },
            "ocel:events": merged_events,
            "ocel:objects": merged_objects,
            "_sources": [f.name for f in files],
        }

    def _load_file(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _normalize(
        self, raw: dict, source: str
    ) -> tuple[dict, dict, int]:
        """Normaliza um arquivo OCEL 2.0 para formato interno."""
        events_raw = raw.get("ocel:events", {})
        objects_raw = raw.get("ocel:objects", {})

        events: dict = {}
        skipped = 0

        for event_id, ev in events_raw.items():
            # Extrai atributos do evento
            attrs = {a["name"]: a["value"] for a in ev.get("ocel:attributes", [])}
            to_column = attrs.get("toColumn", "")
            from_column = attrs.get("fromColumn", "")

            # Mapeia coluna → estágio
            stage = COLUMN_TO_STAGE.get(to_column)
            if not stage:
                skipped += 1
                continue

            # Extrai timestamp
            timestamp = ev.get("ocel:time", "")

            # Extrai objetos relacionados
            omap = [rel["objectId"] for rel in ev.get("ocel:relationships", [])]

            uid = f"{source}__{event_id}"
            events[uid] = {
                "activity": stage,
                "ocel:timestamp": timestamp,
                "ocel:omap": omap,
                "wi_ids": omap,  # todos os relacionados são WorkItems aqui
                "raw": {
                    "fromColumn": from_column,
                    "toColumn": to_column,
                    "fromState": attrs.get("fromState", ""),
                    "toState": attrs.get("toState", ""),
                },
            }

        # Normaliza objetos
        objects: dict = {}
        for obj_id, obj in objects_raw.items():
            attrs = {a["name"]: a["value"] for a in obj.get("ocel:attributes", [])}
            # Adiciona prefixo de source se necessário para evitar colisão
            uid = obj_id
            objects[uid] = {
                "ocel:type": obj.get("ocel:type", "WorkItem"),
                "ocel:ovmap": {
                    "title": attrs.get("title", ""),
                    "workItemType": attrs.get("workItemType", ""),
                    "project": attrs.get("project", ""),
                    "assignedTo": attrs.get("assignedTo", ""),
                },
            }

        return events, objects, skipped

    def summary(self, ocel: dict) -> dict:
        """Retorna resumo estatístico do log carregado."""
        events = ocel.get("ocel:events", {})
        objects = ocel.get("ocel:objects", {})

        from collections import Counter
        stage_counts = Counter(ev["activity"] for ev in events.values())
        type_counts = Counter(
            obj.get("ocel:ovmap", {}).get("workItemType", "?")
            for obj in objects.values()
        )
        project_counts = Counter(
            obj.get("ocel:ovmap", {}).get("project", "?")
            for obj in objects.values()
        )

        return {
            "sources": ocel.get("_sources", []),
            "total_events": len(events),
            "total_objects": len(objects),
            "events_per_stage": dict(stage_counts),
            "work_item_types": dict(type_counts),
            "projects": dict(project_counts),
        }
