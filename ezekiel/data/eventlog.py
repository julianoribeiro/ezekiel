"""Log de casos e visitas a estágios, reconstruído do OCEL normalizado.

É a base empírica comum de H1--H4. A partir das transições de estado de
cada WorkItem, reconstrói:

  casos   — um *episódio* de trabalho: começa na criação (ou na reabertura)
            do item e termina na entrada em um estágio terminal. Um item
            reaberto gera um novo caso, de modo que o tempo em que ficou
            fechado não é contado como trabalho.
  visitas — permanência contígua de um caso em um estágio não terminal:
            entrada, saída (NaN se ainda aberta) e próximo destino
            (outro estágio ou DONE).

Todos os instantes são representados em dias (float) desde 1970-01-01 UTC,
o que mantém as operações por janela vetorizadas mesmo para logs com
centenas de milhares de itens.

Segmentos de fila e de trabalho (metodologia, "Estimação dos parâmetros":
tempos de serviço e de espera não são agregados quando há evidência para
separá-los): cada visita é decomposta nos intervalos passados em status de
FILA ("Open", "Waiting for …" — WAITING_COLUMNS) e em status de TRABALHO.
A espera observada de um caso é a soma dos seus segmentos de fila.

Perfil da fonte (`LogProfile`): o que encerra um caso depende da semântica
dos status — no Jira, "Resolved" encerra o trabalho e o "Closed" posterior
é administrativo.

Censura (fundamentação, Seção de simulação): casos e visitas ainda abertos
no fim do log são mantidos com saída NaN — nunca tratados como concluídos —
e as métricas observadas por janela usam o mesmo critério de censura das
métricas simuladas (só entram na distribuição de lead time os casos
concluídos dentro da janela).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ezekiel.mining.analyzer import PIPELINE_STAGES
from ezekiel.ocel.loader import COLUMN_TO_STAGE, WAITING_COLUMNS

DONE = "DONE"
TERMINAL_STAGES: frozenset[str] = frozenset({"Deploy"})
ALL_STAGES: list[str] = [s for s in PIPELINE_STAGES if s not in TERMINAL_STAGES] + [
    s for s in PIPELINE_STAGES if s in TERMINAL_STAGES
]
WORK_STAGES: list[str] = [s for s in ALL_STAGES if s not in TERMINAL_STAGES]
# Eventos de logs sintéticos (testes) sem atributos de status.
_ACTIVITY_TO_STAGE: dict[str, str] = {
    "item-created": "Backlog", "item-activated": "Development",
    "item-in-review": "Code Review", "item-closed": "Deploy",
}

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_days(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def from_days(t: float) -> pd.Timestamp:
    return _EPOCH + pd.Timedelta(days=float(t))


# Calendário de trabalho (metodologia, "Estimação dos parâmetros":
# calendários e finais de semana na conversão do tempo cronológico em tempo
# operacional). Tempo útil = dias corridos sem sábados e domingos (UTC); não
# há calendário de feriados comum aos projetos públicos (contribuidores em
# vários países). 1970-01-01 foi quinta-feira: u = t + 3 conta a partir de
# uma segunda-feira.

def business_days(t):
    """Dias corridos desde a época → dias úteis desde a época (monótona)."""
    u = np.asarray(t, float) + 3.0
    w = np.floor(u / 7.0)
    return 5.0 * w + np.minimum(u - 7.0 * w, 5.0)


def from_business_days(x):
    """Inversa de `business_days` (instantes de fim de semana → segunda-feira)."""
    x = np.asarray(x, float)
    w = np.floor(x / 5.0)
    return 7.0 * w + (x - 5.0 * w) - 3.0


@dataclass(frozen=True)
class LogProfile:
    """Semântica dos status da fonte."""
    name: str
    completion_statuses: frozenset[str] = frozenset()        # status crus que encerram o caso
    column_overrides: dict[str, str] = field(default_factory=dict)  # status cru → estágio
    split_wait_service: bool = False   # separar fila/trabalho na calibração (se houver evidência)


JIRA_PROFILE = LogProfile("jira", completion_statuses=frozenset({"Resolved", "RESOLVED"}))
PROFILES = {"jira": JIRA_PROFILE}


@dataclass
class EventLog:
    cases: pd.DataFrame     # case, item, start, end (NaN = aberto), entry_stage, wait
    visits: pd.DataFrame    # visit_id, case, stage, enter, exit (NaN = aberta), next, wait, service
    segments: pd.DataFrame  # visit_id, stage, phase (queue|work), start, end (NaN = aberto)
    t_min: float
    t_max: float
    profile: LogProfile = field(default_factory=lambda: LogProfile("custom"))
    ocel: object | None = None   # ObjectCentricLog de origem (OCEL 2.0), quando houver
    calendar: str = "calendar"   # "calendar" (dias corridos) | "business" (dias úteis)

    @property
    def days_per_week(self) -> float:
        return 5.0 if self.calendar == "business" else 7.0

    def date(self, t: float) -> pd.Timestamp:
        """Data de um instante do log (desfaz o calendário de trabalho)."""
        return from_days(float(from_business_days(t)) if self.calendar == "business" else t)

    def to_days_of(self, t_calendar: float) -> float:
        """Instante em dias corridos → unidade de tempo do log."""
        return float(business_days(t_calendar)) if self.calendar == "business" else t_calendar

    def with_business_calendar(self) -> "EventLog":
        """Cópia do log em tempo útil (todas as durações sem fins de semana)."""
        if self.calendar == "business":
            return self
        f = business_days
        c, v, sg = self.cases.copy(), self.visits.copy(), self.segments.copy()
        c["start"], c["end"] = f(c["start"]), f(c["end"])
        v["enter"], v["exit"] = f(v["enter"]), f(v["exit"])
        sg["start"], sg["end"] = f(sg["start"]), f(sg["end"])
        dur = (sg["end"] - sg["start"]).fillna(0.0)
        ph = dur.groupby([sg["visit_id"], sg["phase"]]).sum().unstack(fill_value=0.0)
        v["wait"] = v["visit_id"].map(ph.get("queue", pd.Series(dtype=float))).fillna(0.0)
        v["service"] = v["visit_id"].map(ph.get("work", pd.Series(dtype=float))).fillna(0.0)
        c["wait"] = c["case"].map(v.groupby("case")["wait"].sum()).fillna(0.0)
        return EventLog(cases=c, visits=v, segments=sg, t_min=float(f(self.t_min)), t_max=float(f(self.t_max)),
                        profile=self.profile, ocel=self.ocel, calendar="business")

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #

    @classmethod
    def from_ocel(
        cls, ocel: dict, profile: LogProfile | None = None,
        completion_statuses: frozenset[str] = frozenset(),
    ) -> "EventLog":
        """Constrói o log a partir de um OCEL em formato legado:

          * normalizado pelo OCELLoader (Jira convertido): evento com
            "activity" = estágio e "raw.toColumn";
          * sintético (testes): "ocel:activity", "ocel:omap" e, opcionalmente,
            "ocel:vmap" com toColumn.

        Só eventos de transição de WorkItem definem visitas.

        `completion_statuses` (atalho legado) equivale a um perfil com esses
        status de conclusão. Ver `LogProfile`.
        """
        profile = profile or LogProfile("custom", completion_statuses=completion_statuses)
        terminal_stage = next(iter(TERMINAL_STAGES))
        rows: list[tuple[str, str, str, str | None]] = []
        for ev in ocel.get("ocel:events", {}).values():
            if "activity" in ev:                          # formato do loader
                stage = ev["activity"]
                column = ev.get("raw", {}).get("toColumn")
                state = ev.get("raw", {}).get("toState")
                items = ev.get("wi_ids") or ev.get("ocel:omap") or []
            else:                                         # formato sintético
                vmap = ev.get("ocel:vmap") or {}
                column, state = vmap.get("toColumn"), vmap.get("toState")
                stage = (COLUMN_TO_STAGE.get(column or "")
                         or _ACTIVITY_TO_STAGE.get(ev.get("ocel:activity", "")))
                items = [o for o in ev.get("ocel:omap", []) if str(o).startswith("wi-")]
            if column in profile.column_overrides:
                stage = profile.column_overrides[column]
            if profile.completion_statuses and (
                column in profile.completion_statuses or state in profile.completion_statuses
            ):
                stage = terminal_stage
            if stage not in ALL_STAGES:
                continue
            ts = ev.get("ocel:timestamp") or ev.get("ocel:time", "")
            for wi in items:
                rows.append((wi, stage, ts, column))
        if not rows:
            raise ValueError("Log OCEL sem eventos mapeados para estágios do pipeline.")

        df = pd.DataFrame(rows, columns=["item", "stage", "ts", "column"])
        df["t"] = to_days(pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce"))
        df = df.dropna(subset=["t"]).drop(columns="ts")
        log = cls.from_transitions(df)
        log.profile = profile
        return log

    @classmethod
    def from_ocel2(cls, ocel, profile: LogProfile | None = None) -> "EventLog":
        """Log de casos a partir de um OCEL 2.0 multiobjeto: projeção sobre
        os itens de trabalho (transições de status). Os demais objetos e
        relações permanecem em `self.ocel` para caracterização, validação e
        indicadores de sincronização."""
        profile = profile or LogProfile("custom")
        tr = ocel.workitem_transitions()
        terminal_stage = next(iter(TERMINAL_STAGES))
        col = tr["column"].fillna("").astype(str)
        state = tr["state"].fillna("").astype(str)
        stage = pd.Series([profile.column_overrides.get(c) or COLUMN_TO_STAGE.get(c) for c in col],
                          index=tr.index)
        if profile.completion_statuses:
            done = col.isin(profile.completion_statuses) | state.isin(profile.completion_statuses)
            stage = stage.where(~done, terminal_stage)
        df = pd.DataFrame({"item": tr["item"], "stage": stage, "column": col,
                           "t": to_days(pd.to_datetime(tr["time"], utc=True))})
        df = df[df["stage"].isin(ALL_STAGES)].dropna(subset=["t"])
        if df.empty:
            raise ValueError("OCEL sem transições de itens de trabalho mapeadas para estágios.")
        log = cls.from_transitions(df)
        log.profile = profile
        log.ocel = ocel
        types = ocel.item_attribute("workItemType")
        log.cases["item_type"] = log.cases["item"].map(types)
        return log

    @classmethod
    def from_transitions(cls, df: pd.DataFrame) -> "EventLog":
        """Constrói o log a partir de transições (colunas item, stage, t em
        dias e, opcionalmente, column = coluna/status crus)."""
        df = df.sort_values(["item", "t"], kind="mergesort").reset_index(drop=True)
        if "column" not in df:
            df["column"] = None

        item = df["item"].to_numpy()
        stage = df["stage"].to_numpy()
        t = df["t"].to_numpy(dtype=float)
        col = df["column"].fillna("").astype(str)
        is_queue = col.isin(WAITING_COLUMNS)
        phase = np.where(is_queue.to_numpy(), "queue", "work")

        # Segmento = intervalo entre uma transição e a seguinte do mesmo item.
        same_next_raw = np.r_[item[1:] == item[:-1], False]
        seg_end = np.where(same_next_raw, np.r_[t[1:], [np.nan]], np.nan)

        # Repetições consecutivas do mesmo estágio no mesmo item continuam a
        # mesma visita (ex.: "Ready to Develop" → "Development").
        new_item = np.r_[True, item[1:] != item[:-1]]
        keep = new_item | np.r_[True, stage[1:] != stage[:-1]]
        row_visit = np.cumsum(keep) - 1                   # visita (linha mantida) de cada transição
        item, stage, t = item[keep], stage[keep], t[keep]
        new_item = np.r_[True, item[1:] != item[:-1]]
        visit_id = np.arange(len(item))

        terminal = np.isin(stage, list(TERMINAL_STAGES))
        # Índice do episódio: nº de entradas em estágio terminal ANTES da
        # linha, dentro do mesmo item (a linha terminal pertence ao
        # episódio que ela encerra).
        term_int = terminal.astype(np.int64)
        item_start_idx = np.maximum.accumulate(np.where(new_item, np.arange(len(item)), 0))
        csum = np.cumsum(term_int)
        before_item = np.where(item_start_idx > 0, csum[item_start_idx - 1], 0)
        ep = csum - term_int - before_item
        case = np.char.add(np.char.add(item.astype(str), "#"), ep.astype(str))

        same_next = np.r_[item[1:] == item[:-1], False]
        nxt_stage = np.r_[stage[1:], [""]]
        nxt_t = np.r_[t[1:], [np.nan]]

        # Visitas: linhas não terminais.
        vis_mask = ~terminal
        v_exit = np.where(same_next, nxt_t, np.nan)[vis_mask]
        v_next_raw = np.where(same_next, nxt_stage, "")[vis_mask]
        v_next = np.where(np.isin(v_next_raw, list(TERMINAL_STAGES)), DONE, v_next_raw)
        v_next = np.where(v_next == "", None, v_next)
        visits = pd.DataFrame({
            "visit_id": visit_id[vis_mask],
            "case": case[vis_mask],
            "stage": stage[vis_mask],
            "enter": t[vis_mask],
            "exit": v_exit,
            "next": v_next,
        })

        segments = pd.DataFrame({
            "visit_id": row_visit, "phase": phase,
            "start": df["t"].to_numpy(dtype=float), "end": seg_end,
        })
        segments = segments[np.isin(segments["visit_id"].to_numpy(), visits["visit_id"].to_numpy())]
        segments = segments.merge(visits[["visit_id", "stage"]], on="visit_id", how="left")
        dur = (segments["end"] - segments["start"]).fillna(0.0)
        by_phase = dur.groupby([segments["visit_id"], segments["phase"]]).sum().unstack(fill_value=0.0)
        visits["wait"] = visits["visit_id"].map(by_phase.get("queue", pd.Series(dtype=float))).fillna(0.0)
        visits["service"] = visits["visit_id"].map(by_phase.get("work", pd.Series(dtype=float))).fillna(0.0)

        # Casos: início = primeira visita não terminal do episódio;
        # fim = linha terminal do episódio (se houver).
        starts = visits.groupby("case", sort=False).agg(
            start=("enter", "min"), entry_stage=("stage", "first"), wait=("wait", "sum")
        )
        ends = pd.Series(t[terminal], index=case[terminal], name="end")
        ends = ends[~ends.index.duplicated(keep="first")]
        cases = starts.join(ends, how="left").reset_index(names="case")
        cases["item"] = cases["case"].str.rsplit("#", n=1).str[0]
        cases = cases[["case", "item", "start", "end", "entry_stage", "wait"]]

        return cls(cases=cases, visits=visits, segments=segments.reset_index(drop=True),
                   t_min=float(t.min()), t_max=float(t.max()))

    # ------------------------------------------------------------------ #
    # Consultas por janela
    # ------------------------------------------------------------------ #

    def completed_in(self, a: float, b: float) -> pd.DataFrame:
        c = self.cases
        return c[(c["end"] >= a) & (c["end"] < b)]

    def started_in(self, a: float, b: float) -> pd.DataFrame:
        c = self.cases
        return c[(c["start"] >= a) & (c["start"] < b)]

    def visits_ending_in(self, a: float, b: float) -> pd.DataFrame:
        v = self.visits
        return v[(v["exit"] >= a) & (v["exit"] < b)]

    def time_avg_wip(self, a: float, b: float) -> dict[str, float]:
        """WIP médio no tempo, por estágio, na janela [a, b)."""
        v = self.visits
        exit_ = v["exit"].fillna(np.inf).to_numpy()
        overlap = np.minimum(exit_, b) - np.maximum(v["enter"].to_numpy(), a)
        overlap = np.clip(overlap, 0.0, None)
        s = pd.Series(overlap, index=v["stage"].to_numpy()).groupby(level=0).sum() / (b - a)
        return {st: float(s.get(st, 0.0)) for st in WORK_STAGES}

    def concurrency_samples(
        self, stage: str, a: float, b: float, step: float = 0.25, max_age: float = np.inf
    ) -> np.ndarray:
        """Nº de casos simultâneos no estágio em uma grade regular de [a, b).

        `max_age`: só conta o caso enquanto sua idade no estágio for menor
        que esse valor (exclui itens estacionados).
        """
        v = self.visits[self.visits["stage"] == stage]
        if v.empty:
            return np.zeros(1)
        grid = np.arange(a, b, step)
        enters = np.sort(v["enter"].to_numpy())
        exits = np.sort(np.minimum(v["exit"].fillna(np.inf).to_numpy(), v["enter"].to_numpy() + max_age))
        return (np.searchsorted(enters, grid, side="right")
                - np.searchsorted(exits, grid, side="right")).astype(float)

    def work_concurrency(self, stage: str, a: float, b: float, step: float = 0.25,
                         max_age: float = np.inf) -> np.ndarray:
        """Nº de casos simultâneos em colunas de TRABALHO do estágio (grade regular)."""
        sg = self.segments[(self.segments["stage"] == stage) & (self.segments["phase"] == "work")]
        if sg.empty:
            return np.zeros(1)
        grid = np.arange(a, b, step)
        st = np.sort(sg["start"].to_numpy())
        en = np.sort(np.minimum(sg["end"].fillna(np.inf).to_numpy(), sg["start"].to_numpy() + max_age))
        return (np.searchsorted(st, grid, side="right") - np.searchsorted(en, grid, side="right")).astype(float)

    def phase_time_until(self, stage: str, phase: str, T: float) -> pd.Series:
        """Tempo acumulado em `phase` por visita do estágio, até o instante T."""
        sg = self.segments[(self.segments["stage"] == stage) & (self.segments["phase"] == phase)]
        contrib = np.clip(np.minimum(sg["end"].fillna(np.inf).to_numpy(), T) - sg["start"].to_numpy(), 0.0, None)
        return pd.Series(contrib, index=sg["visit_id"].to_numpy()).groupby(level=0).sum()

    def current_phase_at(self, t: float) -> pd.Series:
        """Fase (queue|work) do segmento ativo de cada visita no instante t."""
        sg = self.segments
        act = sg[(sg["start"] <= t) & (sg["end"].isna() | (sg["end"] > t))]
        return act.drop_duplicates("visit_id", keep="last").set_index("visit_id")["phase"]

    def state_at(self, t: float) -> pd.DataFrame:
        """Casos em andamento no instante t: estágio, idade do caso e idade da visita."""
        v = self.visits
        active = v[(v["enter"] <= t) & (v["exit"].isna() | (v["exit"] > t))]
        starts = self.cases.set_index("case")["start"]
        out = pd.DataFrame({
            "visit_id": active["visit_id"].to_numpy(),
            "case": active["case"].to_numpy(),
            "stage": active["stage"].to_numpy(),
            "stage_age": t - active["enter"].to_numpy(),
        })
        out["case_age"] = t - starts.reindex(out["case"]).to_numpy()
        out["phase"] = out["visit_id"].map(self.current_phase_at(t)).fillna("work").to_numpy()
        return out

    def observed_metrics(self, a: float, b: float) -> dict:
        """Métricas de fluxo observadas na janela [a, b) — mesma definição do simulador."""
        done = self.completed_in(a, b)
        lt = (done["end"] - done["start"]).to_numpy()
        soj = self.visits_ending_in(a, b)
        soj = (soj["exit"] - soj["enter"]).groupby(soj["stage"]).mean()
        wip = self.time_avg_wip(a, b)
        ended = self.visits_ending_in(a, b)
        order = {s: i for i, s in enumerate(ALL_STAGES)}
        trans: dict[str, dict[str, int]] = {}
        for (s, n), k in ended.groupby(["stage", "next"]).size().items():
            trans.setdefault(s, {})[n] = int(k)
        back = ended["next"].map(order).lt(ended["stage"].map(order)) & ended["next"].ne(DONE)
        return {
            "lead_times": lt,
            "done_times": (done["end"] - a).to_numpy(),
            "transitions": trans,
            "rework": int(back.sum()),
            "waits": done["wait"].to_numpy(),
            "queue_wait_mean": {k: float(v) for k, v in
                                self.visits_ending_in(a, b).groupby("stage")["wait"].mean().items()},
            "completions": int(len(lt)),
            "arrivals": int(len(self.started_in(a, b))),
            "throughput_week": len(lt) / ((b - a) / self.days_per_week),
            "wip_avg": wip,
            "wip_total": float(sum(wip.values())),
            "sojourn_mean": {k: float(v) for k, v in soj.items()},
        }

    def case_classes(self, classes: list[str] | None = None) -> pd.Series:
        """Classe de serviço de cada caso (tipo do item); fora de `classes`, "Outros"."""
        t = self.cases.get("item_type")
        if t is None:
            return pd.Series("—", index=self.cases["case"].to_numpy())
        t = t.fillna("—").astype(str)
        if classes is not None:
            t = t.where(t.isin(classes), "Outros")
        return pd.Series(t.to_numpy(), index=self.cases["case"].to_numpy())

    def service_classes(self, a: float, b: float, min_cases: int = 30, max_classes: int = 4) -> list[str]:
        """Classes com ao menos `min_cases` casos iniciados em [a, b), as mais
        frequentes primeiro (no máximo `max_classes`)."""
        if "item_type" not in self.cases:
            return []
        vc = self.started_in(a, b)["item_type"].fillna("—").value_counts()
        return [str(k) for k, v in vc.items() if v >= min_cases][:max_classes]

    def observed_metrics_by_class(self, a: float, b: float, classes: list[str]) -> dict[str, dict]:
        """LT, WT, TH, FE e WIP por classe de serviço na janela [a, b)."""
        cls = self.case_classes(classes)
        done = self.completed_in(a, b)
        done = done.assign(cls=cls.reindex(done["case"]).to_numpy())
        v = self.visits
        exit_ = v["exit"].fillna(np.inf).to_numpy()
        overlap = np.clip(np.minimum(exit_, b) - np.maximum(v["enter"].to_numpy(), a), 0.0, None)
        wip = pd.Series(overlap, index=cls.reindex(v["case"]).to_numpy()).groupby(level=0).sum() / (b - a)
        out = {}
        for c in list(classes) + ["Outros"]:
            d = done[done["cls"] == c]
            lt = (d["end"] - d["start"]).to_numpy()
            has = len(lt) > 0
            wt = float(d["wait"].mean()) if has else float("nan")
            lt_mean = float(lt.mean()) if has else float("nan")
            out[c] = {
                "completions": int(len(lt)),
                "lt_p50": float(np.quantile(lt, 0.5)) if has else None,
                "lt_p80": float(np.quantile(lt, 0.8)) if has else None,
                "lt_p90": float(np.quantile(lt, 0.9)) if has else None,
                "wt_mean": wt if has else None,
                "fe": 1.0 - wt / lt_mean if has and lt_mean > 0 else None,
                "th_week": len(lt) / ((b - a) / self.days_per_week),
                "wip": float(wip.get(c, 0.0)),
            }
        return out

    def active_end(self, quantile: float = 0.99) -> float:
        """Fim do período de atividade regular do log.

        O último evento não é um bom fim de janela: repositórios migrados
        (ex.: Spring → GitHub em 2019) ou com coleta encerrada têm uma cauda
        final de poucos eventos. Usa-se o quantil dos instantes de entrada
        nas visitas e de conclusão dos casos.
        """
        t = np.concatenate([self.visits["enter"].to_numpy(), self.cases["end"].dropna().to_numpy()])
        return float(np.quantile(t, quantile))

    def describe(self) -> dict:
        c = self.cases
        lt = (c["end"] - c["start"]).dropna()
        return {
            "period": [str(self.date(self.t_min).date()), str(self.date(self.t_max).date())],
            "calendar": self.calendar,
            "items": int(c["item"].nunique()),
            "cases": int(len(c)),
            "cases_completed": int(c["end"].notna().sum()),
            "visits": int(len(self.visits)),
            "visits_per_stage": {k: int(v) for k, v in self.visits["stage"].value_counts().items()},
            "lead_time_days": {
                "mean": float(lt.mean()) if len(lt) else None,
                "p50": float(lt.quantile(0.5)) if len(lt) else None,
                "p80": float(lt.quantile(0.8)) if len(lt) else None,
                "p90": float(lt.quantile(0.9)) if len(lt) else None,
            },
        }
