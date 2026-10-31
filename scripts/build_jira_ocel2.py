"""Converte um projeto do The Public Jira Dataset para OCEL 2.0 MULTIOBJETO.

Tipos de objeto (metodologia, "Tipos de objetos e eventos" — a lista
definitiva depende da plataforma; este é o mapeamento para o Jira):

  WorkItem   issue (unidade de fluxo)
  Version    versão de release (fixVersions / versions) — sincronização de
             entrega: itens aguardam a liberação da versão
  Sprint     iteração (campo customizado "Sprint" do repositório)
  Component  componente do produto
  Team       equipe (campo customizado "Team" do repositório, quando existe)

Relações objeto–objeto (O2O), todas EXPLÍCITAS (campos da própria issue):
  WorkItem —blocks / is blocked by / depends on / is dependency of /
           duplicates / is duplicated by / parent of / child of /
           related: <categoria>→ WorkItem       (issuelinks, subtasks, parent;
                                                  categorias de Montgomery et al.)
  WorkItem —fix version / affects version→ Version
  WorkItem —component→ Component
  WorkItem —sprint→ Sprint
  WorkItem —team→ Team

Eventos (E2O com qualificador). O atributo `actor` das mudanças de status é
o autor da mudança, já anonimizado por UUID no dataset — base da evidência
de recurso compartilhado entre unidades (metodologia, "dependências
candidatas").
  item-created         criação; atributo status = status inicial
  item-status-changed  mudança de status (fromStatus, toStatus)
  item-linked          vínculo criado (changelog "Link"): item, linked item
  item-version-set     versão de correção atribuída (changelog "Fix Version")
  item-sprint-set      item colocado em sprint (changelog "Sprint")
  version-released     liberação da versão (releaseDate)
  sprint-started / sprint-completed

As transições de status reproduzem exatamente as do conversor anterior
(scripts/build_jira_ocel.py): o log de casos dos itens de trabalho é o mesmo.

Uso:
  python scripts/build_jira_ocel2.py --repo MongoDB --project SERVER --out ocels/ocel2/mongodb-server.json
  (requer mongod em 127.0.0.1:27018 com a coleção restaurada; ver build_jira_ocel.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF = ROOT / "ocels" / "ThePublicJiraDataset" / "0. DataDefinition"

CANONICAL = {   # categoria (Montgomery et al.) → (saída, entrada)
    "Block": ("blocks", "is blocked by"),
    "Depend": ("depends on", "is dependency of"),
    "Duplicate": ("duplicates", "is duplicated by"),
    "Parent-Child": ("parent of", "child of"),
}
EVIDENCE = {q: "explicita" for q in [
    "item", "linked item", "fix version", "affects version", "component", "sprint", "team", "version",
    "blocks", "is blocked by", "depends on", "is dependency of", "duplicates", "is duplicated by",
    "parent of", "child of",
]}

_SPRINT_RE = re.compile(r"\[(.*)\]\s*$")


def _field_ids(repo: str) -> tuple[list[str], list[str]]:
    info = json.loads((DEF / "jira_field_information.json").read_text(encoding="utf-8"))
    fields = info.get(repo, [])
    sprint = [f["id"] for f in fields if (f.get("name") or "").strip().lower() == "sprint"]
    team = [f["id"] for f in fields if (f.get("name") or "").strip().lower() in ("team", "scrum team")]
    return sprint, team


def _link_category() -> dict[str, str]:
    m = json.loads((DEF / "jira_issue_linktype_mapping.json").read_text(encoding="utf-8"))
    return m.get("link types", {})


def _parse_sprint(v) -> dict | None:
    if isinstance(v, dict):
        return {"id": str(v.get("id")), "name": v.get("name"), "state": v.get("state"),
                "startDate": v.get("startDate"), "endDate": v.get("endDate"), "completeDate": v.get("completeDate")}
    if isinstance(v, str):
        m = _SPRINT_RE.search(v)
        if not m:
            return None
        kv = dict(p.split("=", 1) for p in m.group(1).split(",") if "=" in p)
        if "id" not in kv:
            return None
        nul = lambda x: None if x in (None, "", "<null>") else x  # noqa: E731
        return {"id": kv["id"], "name": nul(kv.get("name")), "state": nul(kv.get("state")),
                "startDate": nul(kv.get("startDate")), "endDate": nul(kv.get("endDate")),
                "completeDate": nul(kv.get("completeDate"))}
    return None


def _team_names(v) -> list[str]:
    if v in (None, "", [], {}):
        return []
    if isinstance(v, list):
        return [n for x in v for n in _team_names(x)]
    if isinstance(v, dict):
        n = v.get("value") or v.get("name") or v.get("title")
        return [str(n)] if n else []
    return [str(v)]


def build(issues: list[dict], repo: str) -> tuple[dict, dict]:
    sprint_fields, team_fields = _field_ids(repo)
    category = _link_category()
    keys = {i["key"]: f"jira-wi-{i['id']}" for i in issues}
    objects: dict[str, dict] = {}
    o2o: dict[str, set] = {}
    events: list[dict] = []
    unmatched_links = 0

    def obj(oid: str, otype: str, attrs: dict, time: str = "") -> None:
        if oid not in objects:
            objects[oid] = {"id": oid, "type": otype,
                            "attributes": [{"name": k, "time": time or "1970-01-01T00:00:00Z", "value": v}
                                           for k, v in attrs.items() if v is not None]}
            o2o[oid] = set()

    def ev(etype: str, time: str, rels: list[tuple[str, str]], attrs: dict | None = None) -> None:
        if not time:
            return
        n = len(events) + 1
        events.append({"id": f"e{n}", "type": etype, "time": time,
                       "attributes": [{"name": k, "value": v} for k, v in (attrs or {}).items()],
                       "relationships": [{"objectId": o, "qualifier": q} for o, q in rels]})

    excluded: list[dict] = []
    version_ids: dict[str, str] = {}
    sprint_ids: dict[str, str] = {}

    for issue in issues:
        f = issue.get("fields", {})
        wi = keys[issue["key"]]
        created = f.get("created", "")
        obj(wi, "WorkItem", {
            "key": issue["key"],
            "workItemType": (f.get("issuetype") or {}).get("name"),
            "priority": (f.get("priority") or {}).get("name"),
            "project": (f.get("project") or {}).get("key"),
        }, created)

        for fld, q in (("fixVersions", "fix version"), ("versions", "affects version")):
            for v in f.get(fld) or []:
                vid = f"jira-ver-{v.get('id')}"
                obj(vid, "Version", {"name": v.get("name"), "released": v.get("released"),
                                     "releaseDate": v.get("releaseDate")})
                version_ids[str(v.get("id"))] = vid
                o2o[wi].add((vid, q))
        for c in f.get("components") or []:
            cid = f"jira-comp-{c.get('id')}"
            obj(cid, "Component", {"name": c.get("name")})
            o2o[wi].add((cid, "component"))
        for sf in sprint_fields:
            for raw in f.get(sf) or []:
                sp = _parse_sprint(raw)
                if sp:
                    sid = f"jira-sprint-{sp['id']}"
                    obj(sid, "Sprint", {"name": sp["name"], "state": sp["state"], "startDate": sp["startDate"],
                                        "endDate": sp["endDate"], "completeDate": sp["completeDate"]})
                    sprint_ids[sp["id"]] = sid
                    o2o[wi].add((sid, "sprint"))
        for tf in team_fields:
            for name in _team_names(f.get(tf)):
                tid = "jira-team-" + hashlib.sha1(name.encode()).hexdigest()[:10]
                obj(tid, "Team", {"name": name})
                o2o[wi].add((tid, "team"))

        for link in f.get("issuelinks") or []:
            t = link.get("type") or {}
            cat = category.get(t.get("name", ""), t.get("name", "Relate"))
            out, inn = CANONICAL.get(cat, (f"related: {cat}", f"related: {cat}"))
            if "outwardIssue" in link:
                other, q = link["outwardIssue"].get("key"), out
            elif "inwardIssue" in link:
                other, q = link["inwardIssue"].get("key"), inn
            else:
                continue
            if other in keys:
                o2o[wi].add((keys[other], q))
            else:
                unmatched_links += 1
        parent = (f.get("parent") or {}).get("key")
        if parent in keys:
            o2o[wi].add((keys[parent], "child of"))
        for st in f.get("subtasks") or []:
            if st.get("key") in keys:
                o2o[wi].add((keys[st["key"]], "parent of"))

        # ── eventos do item (mesmas transições do conversor anterior) ──
        histories = issue.get("changelog", {}).get("histories", [])
        def actor(h: dict) -> str | None:
            a = h.get("author") or {}
            return a.get("key") or a.get("accountId") or a.get("name")   # UUID anonimizado no dataset

        status_changes = [(h.get("created", ""), it.get("fromString") or "", it.get("toString") or "", actor(h))
                          for h in histories for it in h.get("items", []) if it.get("field") == "status"]
        initial = status_changes[0][1] if status_changes else (f.get("status") or {}).get("name", "")
        if initial:
            ev("item-created", created, [(wi, "item")], {"status": initial})
        for ts, a, b, who in status_changes:
            ev("item-status-changed", ts, [(wi, "item")], {"fromStatus": a, "toStatus": b, "actor": who})
        for h in histories:
            for it in h.get("items", []):
                fld, to = it.get("field"), it.get("to")
                if not to:
                    continue
                if fld == "Link" and to in keys:
                    ev("item-linked", h.get("created", ""), [(wi, "item"), (keys[to], "linked item")],
                       {"link": it.get("toString")})
                elif fld == "Fix Version" and str(to) in version_ids:
                    ev("item-version-set", h.get("created", ""), [(wi, "item"), (version_ids[str(to)], "fix version")])
                elif fld == "Sprint":
                    rels = [(sprint_ids[s.strip()], "sprint") for s in str(to).split(",") if s.strip() in sprint_ids]
                    if rels:
                        ev("item-sprint-set", h.get("created", ""), [(wi, "item"), *rels])

    for oid, o in objects.items():
        a = {x["name"]: x["value"] for x in o["attributes"]}
        if o["type"] == "Version" and a.get("released") and a.get("releaseDate"):
            # Regra de exclusão reproduzível (metodologia, "Qualidade dos
            # dados"): datas anteriores a 1990 são erros de digitação no
            # dataset (ex.: ano 0015) — o evento é descartado e registrado.
            if str(a["releaseDate"])[:4] < "1990":
                excluded.append({"object": oid, "event": "version-released",
                                 "releaseDate": a["releaseDate"], "rule": "data de liberação anterior a 1990"})
                continue
            ev("version-released", a["releaseDate"] + "T00:00:00Z" if len(a["releaseDate"]) == 10 else a["releaseDate"],
               [(oid, "version")])
        if o["type"] == "Sprint":
            ev("sprint-started", a.get("startDate"), [(oid, "sprint")])
            ev("sprint-completed", a.get("completeDate"), [(oid, "sprint")])

    for oid, rels in o2o.items():
        objects[oid]["relationships"] = [{"objectId": t, "qualifier": q} for t, q in sorted(rels)]

    types_attrs = {
        "WorkItem": ["key", "workItemType", "priority", "project"],
        "Version": ["name", "released", "releaseDate"],
        "Component": ["name"],
        "Sprint": ["name", "state", "startDate", "endDate", "completeDate"],
        "Team": ["name"],
    }
    ev_attrs = {
        "item-created": ["status"], "item-status-changed": ["fromStatus", "toStatus", "actor"],
        "item-linked": ["link"], "item-version-set": [], "item-sprint-set": [],
        "version-released": [], "sprint-started": [], "sprint-completed": [],
    }
    present_ot = {o["type"] for o in objects.values()}
    present_et = {e["type"] for e in events}
    ocel = {
        "objectTypes": [{"name": t, "attributes": [{"name": a, "type": "string"} for a in at]}
                        for t, at in types_attrs.items() if t in present_ot],
        "eventTypes": [{"name": t, "attributes": [{"name": a, "type": "string"} for a in at]}
                       for t, at in ev_attrs.items() if t in present_et],
        "objects": list(objects.values()),
        "events": sorted(events, key=lambda e: e["time"]),
    }
    quals = {q for rs in o2o.values() for _, q in rs} | {r["qualifier"] for e in events for r in e["relationships"]}
    meta = {
        "source": {"dataset": "The Public Jira Dataset (Montgomery et al., 2022)", "repository": repo,
                   "sprint_fields": sprint_fields, "team_fields": team_fields},
        "relation_evidence": {q: EVIDENCE.get(q, "explicita") for q in sorted(quals)},
        "inferred_relations": [],
        "conversion": {"issues": len(issues), "links_to_issues_outside_project": unmatched_links,
                       "excluded_events": excluded},
    }
    return ocel, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=27018)
    args = ap.parse_args()
    import pymongo
    col = pymongo.MongoClient("127.0.0.1", args.port)["JiraReposAnon"][args.repo]
    issues = list(col.find({"fields.project.key": args.project}, {"_id": 0}))
    ocel, meta = build(issues, args.repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ocel, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.repo}/{args.project}: {len(issues)} issues, {len(ocel['objects'])} objetos, "
          f"{len(ocel['events'])} eventos → {out}")


if __name__ == "__main__":
    main()
