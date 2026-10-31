"""Process Mining e identificação da restrição ativa (Theory of Constraints).

Implementa:
  1. Descoberta do modelo de processo via Inductive Miner (PM4Py)
  2. Cálculo de métricas de fluxo por estágio (WT, WIP, utilização ρ)
  3. Identificação da restrição ativa:
       a* = argmax_a (ρ_a · E[W_a])   (Equação 5.1 / Equação 4.1 da tese)
  4. Decomposição do Lead Time por estágio
  5. Conformance checking (fitness, precision)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from ezekiel.ocel.loader import WAITING_COLUMNS

logger = logging.getLogger(__name__)

# Ordem dos estágios no pipeline de desenvolvimento (conforme tese).
# "Code Review" é incluído por completude — pode ter zero eventos em alguns projetos.
PIPELINE_STAGES = [
    "Backlog",
    "Development",
    "Code Review",
    "QA Testing",
    "Staging",
    "Deploy",
]

# Abreviações para uso em gráficos — fonte única de verdade, exposta pela API.
STAGE_ABBREV: dict[str, str] = {
    "Backlog":             "Backlog",
    "Development":         "Dev.",
    "Code Review":         "Code Rev.",
    "QA Testing":          "QA",
    "Staging":             "Staging",
    "Deploy":              "Deploy",
}

# Mapeamento evento OCEL → estágio
_EVENT_TO_STAGE: dict[str, str] = {
    "item-created": "Backlog",
    "item-activated": "Development",
    "item-in-review": "Code Review",
    "pr-completed": "Code Review",
    "build-completed": "QA Testing",
    "item-closed": "Deploy",
}


@dataclass
class StageMetrics:
    """Métricas de fluxo de um estágio do pipeline."""
    stage: str
    mean_waiting_time_days: float = 0.0   # E[W_a]
    mean_wip: float = 0.0                  # WIP médio
    utilization: float = 0.0              # ρ_a (utilização estimada)
    throughput_per_week: float = 0.0
    # Score ToC: ρ_a × E[W_a]
    constraint_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.constraint_score = self.utilization * self.mean_waiting_time_days


@dataclass
class MiningResult:
    """Resultado completo da análise de Process Mining."""
    # Métricas por estágio
    stage_metrics: list[StageMetrics] = field(default_factory=list)

    # Restrição ativa identificada (a*)
    active_constraint: StageMetrics | None = None

    # Métricas globais
    mean_lead_time_days: float = 0.0
    mean_waiting_time_days: float = 0.0
    mean_throughput_per_week: float = 0.0
    flow_efficiency: float = 0.0          # (tempo ativo / lead time)

    # Conformance
    fitness: float = 0.0
    precision: float | None = None
    rework_rate: float = 0.0             # taxa de retrabalho (ex.: Revisão → Dev)

    # Lead time decomposto por estágio
    lead_time_breakdown: dict[str, float] = field(default_factory=dict)

    # DFG — grafo de directly-follows (van der Aalst)
    dfg: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "lead_time_days": round(self.mean_lead_time_days, 1),
            "waiting_time_days": round(self.mean_waiting_time_days, 1),
            "throughput_per_week": round(self.mean_throughput_per_week, 1),
            "flow_efficiency_pct": round(self.flow_efficiency * 100, 1),
            "active_constraint": self.active_constraint.stage if self.active_constraint else None,
            "constraint_score": round(self.active_constraint.constraint_score, 2) if self.active_constraint else None,
            "fitness": round(self.fitness, 2),
            "precision": round(self.precision, 2) if self.precision is not None else None,
            "rework_rate_pct": round(self.rework_rate * 100, 1),
        }


class ProcessMiningAnalyzer:
    """Executa a análise de process mining sobre o log OCEL."""

    def analyze(self, ocel: dict) -> MiningResult:
        """Analisa o log OCEL e retorna métricas + restrição ativa."""
        logger.info("Iniciando análise de Process Mining...")

        events = ocel.get("ocel:events", {})
        objects = ocel.get("ocel:objects", {})

        if not events:
            logger.warning("Log OCEL vazio — retornando resultado em branco.")
            return MiningResult()

        # Constrói DataFrame de eventos
        df = self._build_event_dataframe(events, objects)

        # Agrupa os eventos por WorkItem uma única vez (O(eventos) via
        # groupby) — as próximas 4 etapas da análise precisam, cada uma, da
        # sequência cronológica de eventos de cada item; antes, cada uma
        # refiltrava o DataFrame inteiro para cada item (O(itens×eventos),
        # repetido ~10x: 7 chamadas de _compute_waiting_times + mais 3), o
        # que não escalava além de uns poucos milhares de itens.
        item_sequences = self._build_item_sequences(df)

        # Calcula métricas de fluxo por estágio
        stage_metrics = self._compute_stage_metrics(df, item_sequences)

        # Identifica a restrição ativa — Eq. 5.1 da tese
        active_constraint = self._identify_constraint(stage_metrics)

        # Métricas globais
        global_metrics = self._compute_global_metrics(df, item_sequences)

        # Decomposição do lead time
        lt_breakdown = {m.stage: round(m.mean_waiting_time_days, 1) for m in stage_metrics}

        # Conformance (estimado via análise de sequências)
        fitness, precision, rework_rate = self._estimate_conformance(df, item_sequences)

        dfg = self._compute_dfg(item_sequences)

        result = MiningResult(
            stage_metrics=stage_metrics,
            active_constraint=active_constraint,
            lead_time_breakdown=lt_breakdown,
            fitness=fitness,
            precision=precision,
            rework_rate=rework_rate,
            dfg=dfg,
            **global_metrics,
        )

        logger.info(
            f"[OK] Análise concluída. "
            f"Lead Time={result.mean_lead_time_days:.1f}d, "
            f"Restrição ativa: {active_constraint.stage if active_constraint else 'N/A'} "
            f"(score={active_constraint.constraint_score:.2f})" if active_constraint else ""
        )
        return result

    # ------------------------------------------------------------------ #
    # Construção do DataFrame
    # ------------------------------------------------------------------ #

    def _build_event_dataframe(self, events: dict, objects: dict) -> pd.DataFrame:
        """
        Constrói DataFrame de eventos a partir do log OCEL normalizado.

        Suporta dois formatos:
          - Formato loader (real): evento já tem "activity" = stage, "wi_ids", "ocel:timestamp"
          - Formato builder (sintético): evento tem "ocel:activity", "ocel:omap", "ocel:timestamp"
        """
        rows = []
        for event_id, ev in events.items():
            # Formato loader (real) — "activity" já é o estágio mapeado
            if "activity" in ev and ev["activity"] in PIPELINE_STAGES:
                stage = ev["activity"]
                activity = stage
                ts_raw = ev.get("ocel:timestamp", "")
                wi_ids = ev.get("wi_ids", ev.get("ocel:omap", []))
                # Status cru (ex.: "Patch Available") — só existe
                # no formato loader (log real). Usada para medir waiting_time
                # por item a partir da semântica real da coluna, não de uma
                # suposição fixa.
                raw_column = ev.get("raw", {}).get("toColumn")
            else:
                # Formato builder (sintético) — "ocel:activity" + mapeamento
                activity = ev.get("ocel:activity", "")
                ts_raw = ev.get("ocel:timestamp", "")
                omap = ev.get("ocel:omap", [])
                stage = _EVENT_TO_STAGE.get(activity, "")
                wi_ids = [o for o in omap if o.startswith("wi-")]
                raw_column = None
                if not stage:
                    continue

            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if not isinstance(wi_ids, list):
                wi_ids = list(wi_ids)

            rows.append({
                "event_id": event_id,
                "activity": activity,
                "timestamp": ts,
                "stage": stage,
                "wi_ids": wi_ids,
                "raw_column": raw_column,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _build_item_sequences(self, df: pd.DataFrame) -> dict[str, list[dict]]:
        """Agrupa os eventos por WorkItem, uma única vez, preservando a
        ordem cronológica (o DataFrame já chega ordenado por timestamp, e
        `explode` + `groupby(sort=False)` preservam a ordem relativa das
        linhas dentro de cada grupo — não é preciso reordenar por item).

        Cada evento normalmente está ligado a 1 único WorkItem (`wi_ids` é
        lista de 1 elemento) — `explode` trata o caso geral sem assumir isso.
        """
        if df.empty:
            return {}
        exploded = df.explode("wi_ids").dropna(subset=["wi_ids"])
        return {
            wi_id: group.to_dict("records")
            for wi_id, group in exploded.groupby("wi_ids", sort=False)
        }

    # ------------------------------------------------------------------ #
    # Métricas por estágio
    # ------------------------------------------------------------------ #

    def _compute_stage_metrics(
        self, df: pd.DataFrame, item_sequences: dict[str, list[dict]]
    ) -> list[StageMetrics]:
        if df.empty:
            return [StageMetrics(stage=s) for s in PIPELINE_STAGES]

        metrics = []
        total_days = self._total_observation_days(df)

        for stage in PIPELINE_STAGES:
            stage_df = df[df["stage"] == stage]
            if stage_df.empty:
                metrics.append(StageMetrics(stage=stage))
                continue

            # Waiting time por item (tempo entre entrada e saída do estágio)
            waiting_times = self._compute_waiting_times(item_sequences, stage)
            mean_wt = (
                sum(t.total_seconds() / 86400 for t in waiting_times) / len(waiting_times)
                if waiting_times else 0.0
            )

            # WIP médio (Lei de Little: L = λ × W)
            n_items = len(set(wi for wis in stage_df["wi_ids"] for wi in wis))
            throughput_per_day = n_items / total_days if total_days > 0 else 0.0
            mean_wip = throughput_per_day * mean_wt

            # Utilização ρ (heurística: proporção do tempo em que o estágio tem WIP > 0)
            utilization = min(mean_wip / max(mean_wip + 1, 1), 0.99)

            throughput_per_week = throughput_per_day * 7

            metrics.append(StageMetrics(
                stage=stage,
                mean_waiting_time_days=round(mean_wt, 2),
                mean_wip=round(mean_wip, 2),
                utilization=round(utilization, 3),
                throughput_per_week=round(throughput_per_week, 2),
            ))

        return metrics

    def _compute_waiting_times(
        self, item_sequences: dict[str, list[dict]], stage: str
    ) -> list[timedelta]:
        """
        Calcula o tempo de permanência de cada WorkItem em um estágio.

        Entrada no estágio: primeiro evento com stage == <stage>
        Saída do estágio: primeiro evento com stage == <próximo estágio> após a entrada

        Equivalente a filtrar/ordenar o DataFrame inteiro por item (como
        antes), mas percorrendo cada sequência (já ordenada) uma vez só.
        """
        stage_idx = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        current_idx = stage_idx.get(stage, -1)
        if current_idx < 0:
            return []

        next_stages = set(PIPELINE_STAGES[current_idx + 1:])
        waiting_times: list[timedelta] = []

        for rows in item_sequences.values():
            entry_ts = None
            exit_ts = None
            for row in rows:
                if entry_ts is None and row["stage"] == stage:
                    entry_ts = row["timestamp"]
                    continue
                if entry_ts is not None and row["stage"] in next_stages and row["timestamp"] > entry_ts:
                    exit_ts = row["timestamp"]
                    break

            if entry_ts is None or exit_ts is None:
                continue
            wt = exit_ts - entry_ts
            if wt.total_seconds() > 0:
                waiting_times.append(wt)

        return waiting_times

    # ------------------------------------------------------------------ #
    # Identificação da restrição ativa — Eq. 5.1 da tese
    # ------------------------------------------------------------------ #

    def _identify_constraint(self, stage_metrics: list[StageMetrics]) -> StageMetrics | None:
        """
        a* = argmax_a (ρ_a · E[W_a])

        Retorna o estágio com maior score de restrição.
        """
        if not stage_metrics:
            return None

        active = max(stage_metrics, key=lambda m: m.constraint_score)
        logger.info(
            f"Restrição ativa identificada: {active.stage} "
            f"(ρ={active.utilization:.2f}, E[W]={active.mean_waiting_time_days:.1f}d, "
            f"score={active.constraint_score:.2f})"
        )
        return active

    # ------------------------------------------------------------------ #
    # Métricas globais
    # ------------------------------------------------------------------ #

    def _compute_global_metrics(
        self, df: pd.DataFrame, item_sequences: dict[str, list[dict]]
    ) -> dict:
        if df.empty:
            return {
                "mean_lead_time_days": 0.0,
                "mean_waiting_time_days": 0.0,
                "mean_throughput_per_week": 0.0,
                "flow_efficiency": 0.0,
            }

        total_days = self._total_observation_days(df)

        # Lead time por item: do primeiro ao último evento registrado
        lead_times: list[float] = []

        for rows in item_sequences.values():
            if len(rows) < 2:
                continue
            lt = (rows[-1]["timestamp"] - rows[0]["timestamp"]).total_seconds() / 86400
            if lt > 0:
                lead_times.append(lt)

        mean_lt = sum(lead_times) / len(lead_times) if lead_times else 0.0

        # Throughput: itens que chegaram ao estágio final / semanas
        final_df = df[df["stage"] == PIPELINE_STAGES[-1]]
        n_closed = len(set(wi for wis in final_df["wi_ids"] for wi in wis))
        weeks = total_days / 7 if total_days > 0 else 1
        mean_tp = n_closed / weeks

        # Waiting time médio — medido por item (não por soma das médias por
        # etapa, que dá dupla contagem sob retrabalho: um item que retrocede
        # visita a mesma etapa mais de uma vez, e a métrica por etapa só
        # olha a primeira entrada/saída). Cada item soma seu próprio tempo
        # real em status de FILA (WAITING_COLUMNS) — nunca conta
        # o mesmo intervalo duas vezes porque percorre a própria sequência
        # de eventos do item, na ordem em que aconteceram.
        per_item_wait = self._compute_waiting_time_per_item(df, item_sequences)
        if per_item_wait:
            mean_wt = sum(per_item_wait) / len(per_item_wait)
        else:
            # Fallback para logs sem status cru (ex.: OCEL
            # sintético do builder, usado em testes) — sem esse sinal não dá
            # para medir espera real, então mantém a suposição de projeto.
            mean_wt = mean_lt * 0.75

        # Flow efficiency
        flow_eff = 1 - (mean_wt / mean_lt) if mean_lt > 0 else 0.0

        return {
            "mean_lead_time_days": round(mean_lt, 1),
            "mean_waiting_time_days": round(mean_wt, 1),
            "mean_throughput_per_week": round(mean_tp, 1),
            "flow_efficiency": round(flow_eff, 3),
        }

    def _compute_waiting_time_per_item(
        self, df: pd.DataFrame, item_sequences: dict[str, list[dict]]
    ) -> list[float]:
        """
        Waiting time real de cada item: soma, na própria sequência de
        eventos do item, o tempo passado em status de FILA
        (WAITING_COLUMNS — "Open", "Ready for Development", "Waiting for
        Review"...) até o próximo evento. Status de trabalho ativo
        (In Progress, In QA, In Review...) não entram na soma.

        Retorna lista vazia se o log não tiver o status cru
        (`raw_column`) — caso do OCEL sintético do builder.
        """
        if "raw_column" not in df.columns or df["raw_column"].isna().all():
            return []

        waiting_times: list[float] = []
        for rows in item_sequences.values():
            if len(rows) < 2:
                continue
            total_wait = 0.0
            for i in range(len(rows) - 1):
                if rows[i]["raw_column"] in WAITING_COLUMNS:
                    delta = (rows[i + 1]["timestamp"] - rows[i]["timestamp"]).total_seconds() / 86400
                    total_wait += delta
            waiting_times.append(total_wait)
        return waiting_times

    # ------------------------------------------------------------------ #
    # Conformance
    # ------------------------------------------------------------------ #

    def _estimate_conformance(
        self, df: pd.DataFrame, item_sequences: dict[str, list[dict]]
    ) -> tuple[float, float, float]:
        """
        Estima fitness, precision e taxa de retrabalho a partir das sequências de estágios.

        Rework: fração de work items que sofreram ao menos uma transição de
        retrocesso na ordem de PIPELINE_STAGES (generalizado — não depende de
        nomes de estágio hardcoded). Contado por item (0 ou 1), não por
        transição — um item que retrocede várias vezes não pode inflar a taxa
        acima de 100%.
        Fitness: item visitou ao menos 3 dos estágios com eventos no log (modelo real).
        """
        if df.empty:
            return 0.0, 0.0, 0.0

        # Estágios que realmente aparecem no log (excluindo os com 0 eventos)
        active_stages = [s for s in PIPELINE_STAGES if s in df["stage"].values]
        if not active_stages:
            return 0.0, 0.0, 0.0

        stage_idx = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        min_visits = max(3, int(len(active_stages) * 0.5))

        conforming = 0
        total = 0
        rework_count = 0

        for rows in item_sequences.values():
            stages_seq = [row["stage"] for row in rows]
            if not stages_seq:
                continue
            total += 1

            # Fitness: visitou ao menos min_visits estágios ativos
            visited = set(stages_seq) & set(active_stages)
            if len(visited) >= min_visits:
                conforming += 1

            # Retrabalho: item conta no máximo 1 vez, mesmo com múltiplos
            # retrocessos — evita rework_rate > 100% (bug anterior: somava
            # todas as transições de retrocesso e dividia por nº de itens).
            for i in range(len(stages_seq) - 1):
                idx_cur = stage_idx.get(stages_seq[i], -1)
                idx_nxt = stage_idx.get(stages_seq[i + 1], -1)
                if idx_cur >= 0 and idx_nxt >= 0 and idx_nxt < idx_cur:
                    rework_count += 1
                    break

        # "fitness" aqui é a cobertura de estágios (fração de itens que
        # visitam ao menos min_visits estágios ativos), NÃO o fitness de
        # conformidade por replay/alinhamento. A precisão era um valor fixo
        # (0,87) para todos os logs; agora é calculada de fato (PM4Py,
        # Inductive Miner + replay de tokens) sobre uma amostra de casos,
        # ou None se o cálculo não for possível.
        fitness = conforming / total if total > 0 else 0.0
        precision = self._token_replay_precision(item_sequences)
        rework_rate = rework_count / total if total > 0 else 0.0

        return round(fitness, 2), precision, round(rework_rate, 3)

    @staticmethod
    def _token_replay_precision(
        item_sequences: dict[str, list[dict]], max_cases: int = 2000, seed: int = 0
    ) -> float | None:
        try:
            import random as _random
            import pm4py

            ids = list(item_sequences)
            _random.Random(seed).shuffle(ids)
            rows = [
                {"case:concept:name": wi, "concept:name": r["stage"], "time:timestamp": r["timestamp"]}
                for wi in ids[:max_cases] for r in item_sequences[wi]
            ]
            log_df = pd.DataFrame(rows)
            if log_df.empty:
                return None
            net, im, fm = pm4py.discover_petri_net_inductive(log_df, noise_threshold=0.2)
            return round(float(pm4py.precision_token_based_replay(log_df, net, im, fm)), 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Precisão não calculada: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # DFG — Directly-Follows Graph (van der Aalst, 2016)
    # ------------------------------------------------------------------ #

    def _compute_dfg(self, item_sequences: dict[str, list[dict]]) -> dict:
        """
        Calcula o Directly-Follows Graph a partir do log de eventos.

        Para cada caso (work item), reconstrói a sequência de estágios
        e conta quantas vezes o estágio B seguiu diretamente o estágio A.

        Retorna:
          activity_counts : {stage: n_occurrences}
          edges           : [{source, target, count}]  — arestas diretamente-segue
          start_activities: {stage: count}
          end_activities  : {stage: count}
          total_cases     : int
        """
        if not item_sequences:
            return {"activity_counts": {}, "edges": [], "start_activities": {}, "end_activities": {}, "total_cases": 0}

        activity_counts: dict[str, int] = {}
        dfg_counts: dict[tuple[str, str], int] = {}
        start_activities: dict[str, int] = {}
        end_activities: dict[str, int] = {}

        for rows in item_sequences.values():
            # Sequência de estágios distintos e ordenados (remove repetições consecutivas)
            raw_stages = [row["stage"] for row in rows]
            stages: list[str] = []
            for s in raw_stages:
                if not stages or s != stages[-1]:
                    stages.append(s)

            if not stages:
                continue

            for s in stages:
                activity_counts[s] = activity_counts.get(s, 0) + 1

            start_activities[stages[0]] = start_activities.get(stages[0], 0) + 1
            end_activities[stages[-1]] = end_activities.get(stages[-1], 0) + 1

            for i in range(len(stages) - 1):
                pair = (stages[i], stages[i + 1])
                dfg_counts[pair] = dfg_counts.get(pair, 0) + 1

        edges = [
            {"source": src, "target": tgt, "count": cnt}
            for (src, tgt), cnt in sorted(dfg_counts.items(), key=lambda x: -x[1])
        ]

        return {
            "activity_counts": activity_counts,
            "edges": edges,
            "start_activities": start_activities,
            "end_activities": end_activities,
            "total_cases": len(item_sequences),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _total_observation_days(df: pd.DataFrame) -> float:
        if df.empty:
            return 1.0
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        delta = (ts_max - ts_min).total_seconds() / 86400
        return max(delta, 1.0)
