"""Log de eventos centrado em objetos (OCEL 2.0) — leitura, escrita,
validação e caracterização (metodologia, Seções "Estruturação do log
centrado em objetos" e "Caracterização empírica e mineração de processos").

Formato em disco: OCEL 2.0 JSON padrão (ocel-standard.org), legível pelo
PM4Py (`pm4py.read_ocel2_json`):

    objectTypes / eventTypes   tipos declarados com seus atributos
    objects                    id, type, attributes[{name,time,value}],
                               relationships[{objectId, qualifier}]  (O2O)
    events                     id, type, time, attributes[{name,value}],
                               relationships[{objectId, qualifier}]  (E2O)

Arquivo lateral `<nome>.meta.json` (não faz parte do padrão):
    relation_evidence   qualificador → nível de evidência (metodologia,
                        "Regras de associação multiobjeto"):
                        "explicita" | "derivada" | "inferida"
    inferred_relations  relações INFERIDAS (proximidade temporal etc.),
                        mantidas FORA do log principal, com regra,
                        parâmetros e confiança — só entram após validação
                        com especialistas ou análise de sensibilidade
    source, conversion  procedência e regras de transformação

Em memória, o log é um conjunto de tabelas (pandas): events, objects,
e2o (evento–objeto, com qualificador) e o2o (objeto–objeto).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EVIDENCE_LEVELS = ("explicita", "derivada", "inferida")


@dataclass
class ObjectCentricLog:
    events: pd.DataFrame        # event_id, type, time (datetime UTC), attrs (dict)
    objects: pd.DataFrame       # object_id, type, attrs (dict)
    e2o: pd.DataFrame           # event_id, object_id, qualifier
    o2o: pd.DataFrame           # source_id, target_id, qualifier
    object_types: dict[str, list[dict]] = field(default_factory=dict)
    event_types: dict[str, list[dict]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # E/S
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, d: dict, meta: dict | None = None) -> "ObjectCentricLog":
        ev_rows, e2o_rows = [], []
        for e in d.get("events", []):
            attrs = {a["name"]: a.get("value") for a in e.get("attributes", [])}
            ev_rows.append((e["id"], e["type"], e["time"], attrs))
            for r in e.get("relationships", []):
                e2o_rows.append((e["id"], r["objectId"], r.get("qualifier", "")))
        ob_rows, o2o_rows = [], []
        for o in d.get("objects", []):
            attrs = {a["name"]: a.get("value") for a in o.get("attributes", [])}
            ob_rows.append((o["id"], o["type"], attrs))
            for r in o.get("relationships", []):
                o2o_rows.append((o["id"], r["objectId"], r.get("qualifier", "")))
        events = pd.DataFrame(ev_rows, columns=["event_id", "type", "time", "attrs"])
        events["time"] = pd.to_datetime(events["time"], utc=True, format="ISO8601", errors="coerce")
        return cls(
            events=events,
            objects=pd.DataFrame(ob_rows, columns=["object_id", "type", "attrs"]),
            e2o=pd.DataFrame(e2o_rows, columns=["event_id", "object_id", "qualifier"]),
            o2o=pd.DataFrame(o2o_rows, columns=["source_id", "target_id", "qualifier"]),
            object_types={t["name"]: t.get("attributes", []) for t in d.get("objectTypes", [])},
            event_types={t["name"]: t.get("attributes", []) for t in d.get("eventTypes", [])},
            meta=meta or {},
        )

    @classmethod
    def read(cls, path: str | Path) -> "ObjectCentricLog":
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        mp = p.with_suffix(".meta.json")
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        return cls.from_json(d, meta)

    @staticmethod
    def is_ocel2(d: dict) -> bool:
        return "objectTypes" in d and "events" in d and isinstance(d.get("events"), list)

    # ------------------------------------------------------------------ #
    # Validação (metodologia, "Validação do log")
    # ------------------------------------------------------------------ #

    def validate(self) -> dict:
        """Validação estrutural: tipos declarados, unicidade de ids,
        integridade referencial, marcas temporais, cardinalidades e nível
        de evidência de cada qualificador de relação."""
        issues: list[str] = []
        dup_ev = int(self.events["event_id"].duplicated().sum())
        dup_ob = int(self.objects["object_id"].duplicated().sum())
        if dup_ev:
            issues.append(f"{dup_ev} ids de evento duplicados")
        if dup_ob:
            issues.append(f"{dup_ob} ids de objeto duplicados")
        undeclared_ev = sorted(set(self.events["type"]) - set(self.event_types))
        undeclared_ob = sorted(set(self.objects["type"]) - set(self.object_types))
        if undeclared_ev:
            issues.append(f"tipos de evento não declarados: {undeclared_ev}")
        if undeclared_ob:
            issues.append(f"tipos de objeto não declarados: {undeclared_ob}")
        bad_time = int(self.events["time"].isna().sum())
        if bad_time:
            issues.append(f"{bad_time} eventos com marca temporal inválida")
        ids = set(self.objects["object_id"])
        e2o_orphan = int((~self.e2o["object_id"].isin(ids)).sum())
        o2o_orphan = int((~self.o2o["source_id"].isin(ids) | ~self.o2o["target_id"].isin(ids)).sum())
        if e2o_orphan:
            issues.append(f"{e2o_orphan} relações evento–objeto apontam para objeto inexistente")
        if o2o_orphan:
            issues.append(f"{o2o_orphan} relações objeto–objeto com extremidade inexistente")
        no_obj = int((~self.events["event_id"].isin(set(self.e2o["event_id"]))).sum())
        if no_obj:
            issues.append(f"{no_obj} eventos sem objeto relacionado")
        ev_levels = self.meta.get("relation_evidence", {})
        quals = set(self.e2o["qualifier"]) | set(self.o2o["qualifier"])
        no_level = sorted(q for q in quals if q not in ev_levels)
        if no_level:
            issues.append(f"qualificadores sem nível de evidência declarado: {no_level}")
        inferred_in_log = sorted(q for q in quals if ev_levels.get(q) == "inferida")
        if inferred_in_log:
            issues.append(f"relações inferidas no log principal: {inferred_in_log}")
        return {
            "valid": not issues,
            "issues": issues,
            "counts": {"events": int(len(self.events)), "objects": int(len(self.objects)),
                       "e2o": int(len(self.e2o)), "o2o": int(len(self.o2o)),
                       "inferred_relations_outside_log": len(self.meta.get("inferred_relations", []))},
        }

    # ------------------------------------------------------------------ #
    # Caracterização por tipo de objeto
    # ------------------------------------------------------------------ #

    def flatten(self, object_type: str) -> pd.DataFrame:
        """Projeção (achatamento) do log sobre um tipo de objeto: um caso por
        objeto, com os eventos relacionados a ele em ordem temporal."""
        obj = self.objects.loc[self.objects["type"] == object_type, "object_id"]
        rel = self.e2o[self.e2o["object_id"].isin(set(obj))]
        df = rel.merge(self.events[["event_id", "type", "time", "attrs"]], on="event_id")
        return df.rename(columns={"object_id": "case", "type": "activity"}).sort_values(["case", "time"])

    def characterize(self, max_variants: int = 5) -> dict:
        out: dict = {"object_types": {}, "event_types": dict(Counter(self.events["type"])), "o2o": {}}
        by_ev = self.e2o.merge(self.objects[["object_id", "type"]], on="object_id")
        for ot, n in Counter(self.objects["type"]).items():
            flat = self.flatten(ot)
            per_obj = flat.groupby("case")["activity"].apply(tuple)
            variants = Counter(per_obj)
            dfg = Counter()
            for seq in per_obj:
                for a, b in zip(seq, seq[1:]):
                    dfg[f"{a} → {b}"] += 1
            span = flat.groupby("case")["time"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
            out["object_types"][ot] = {
                "objects": int(n),
                "objects_with_events": int(flat["case"].nunique()),
                "events_per_object_mean": float(per_obj.map(len).mean()) if len(per_obj) else 0.0,
                "lifetime_days_median": float(span.median()) if len(span) else None,
                "event_types": dict(Counter(flat["activity"])),
                "variants": int(len(variants)),
                "top_variants": [{"variant": list(v), "count": c} for v, c in variants.most_common(max_variants)],
                "dfg_top_edges": dict(dfg.most_common(10)),
                "related_event_types": dict(Counter(by_ev.loc[by_ev["type"] == ot, "event_id"].map(
                    self.events.set_index("event_id")["type"]))),
            }
        o2o = self.o2o.merge(self.objects[["object_id", "type"]].rename(columns={"object_id": "source_id", "type": "st"}),
                             on="source_id").merge(
            self.objects[["object_id", "type"]].rename(columns={"object_id": "target_id", "type": "tt"}), on="target_id")
        for (st, q, tt), n in Counter(zip(o2o["st"], o2o["qualifier"], o2o["tt"])).items():
            out["o2o"][f"{st} —{q}→ {tt}"] = int(n)
        return out

    TRANSITION_EVENTS = ("item-created", "item-activated", "item-status-changed", "item-closed")

    def workitem_transitions(self, object_type: str = "WorkItem") -> pd.DataFrame:
        """Transições dos itens de trabalho (projeção usada pelo log de
        casos): item, coluna/status de destino, estado de destino, instante.
        Jira: toStatus/status (toColumn em logs legados)."""
        flat = self.flatten(object_type)
        flat = flat[flat["activity"].isin(self.TRANSITION_EVENTS)]

        def column(a: dict):
            c = a.get("toColumn") or a.get("toStatus") or a.get("status")
            return c

        return pd.DataFrame({"item": flat["case"].to_numpy(),
                             "column": flat["attrs"].map(column).to_numpy(),
                             "state": flat["attrs"].map(lambda a: a.get("toState")).to_numpy(),
                             "actor": flat["attrs"].map(lambda a: a.get("actor")).to_numpy(),
                             "time": flat["time"].to_numpy()})

    def item_attribute(self, name: str, object_type: str = "WorkItem") -> pd.Series:
        ob = self.objects[self.objects["type"] == object_type]
        return pd.Series(ob["attrs"].map(lambda a: a.get(name)).to_numpy(), index=ob["object_id"].to_numpy())


def synchronization_indicators(ocel: ObjectCentricLog, completion_statuses: set[str]) -> dict:
    """Indicadores de dependência entre objetos que a simulação NÃO
    representa (simplificação declarada; metodologia, "Especificação
    conceitual": simplificações mantidas quando não alteram os mecanismos
    relevantes às comparações). Quantificam o quanto a simplificação pesa:

      * itens com vínculo de bloqueio/dependência para outro item;
      * fração de itens concluídos ANTES de um item que os bloqueia;
      * espera entre a conclusão do item e a liberação da versão de
        correção (sincronização com o objeto Version).
    """
    tr = ocel.workitem_transitions()
    done = tr[tr["column"].isin(completion_statuses)].groupby("item")["time"].min()
    items = set(ocel.objects.loc[ocel.objects["type"] == "WorkItem", "object_id"])
    q = ocel.o2o
    block = q[q["qualifier"].isin(["is blocked by", "depends on"]) & q["source_id"].isin(items) & q["target_id"].isin(items)]
    blocked_items = set(block["source_id"])
    viol = 0
    checked = 0
    for s, t in zip(block["source_id"], block["target_id"]):
        if s in done.index and t in done.index:
            checked += 1
            viol += int(done[s] < done[t])
    rel = q[(q["qualifier"] == "fix version") & q["source_id"].isin(items)]
    vattr = ocel.objects.set_index("object_id")["attrs"]
    waits = []
    for s, v in zip(rel["source_id"], rel["target_id"]):
        rd = vattr.get(v, {}).get("releaseDate") if v in vattr.index else None
        if s in done.index and rd:
            try:   # datas de liberação inválidas no dataset (ex.: ano 0015) são ignoradas
                w = (pd.Timestamp(rd, tz="UTC") - done[s]).total_seconds() / 86400
            except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
                continue
            if w >= 0:
                waits.append(w)
    n = len(items) or 1
    return {
        "items": len(items),
        "items_with_blocking_link": len(blocked_items),
        "share_items_with_blocking_link": len(blocked_items) / n,
        "blocking_pairs_both_completed": checked,
        "share_completed_before_blocker": (viol / checked) if checked else None,
        "items_with_released_fix_version": len(waits),
        "wait_to_release_days_median": float(np.median(waits)) if waits else None,
        "wait_to_release_days_p80": float(np.quantile(waits, 0.8)) if waits else None,
    }
