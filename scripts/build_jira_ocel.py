"""Converte issues de um projeto do Jira (The Public Jira Dataset, Montgomery
et al. 2022) para o formato OCEL legado lido pelo OCELLoader
(ezekiel/ocel/loader.py).

Mapeamento:
  Evento   : cada item "status" de issue.changelog.histories -> StateTransition
             fromColumn/fromState = fromString, toColumn/toState = toString
  Objeto   : cada issue -> WorkItem (title/workItemType/project/assignedTo/url)

Pré-requisitos (dataset bruto é um dump do MongoDB, não um JSON pronto):
  1. Restaurar só a coleção do repositório desejado (dump tem 1 coleção por
     repositório Jira, ex. "Apache", "Qt", "MongoDB" — ver
     ocels/ThePublicJiraDataset/0. DataDefinition/jira_data_sources.json):

       mongod --dbpath <dbpath> --port 27018 --bind_ip 127.0.0.1 &
       mongorestore --port 27018 --gzip \
         --archive="ocels/ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive" \
         --nsInclude="JiraReposAnon.Apache"

     Repare o tamanho: "Apache" sozinho tem ~1M issues — se o disco for
     pequeno, mate o mongorestore (SIGINT) depois de alguns GB; ele grava
     documento a documento, então o que já entrou fica íntegro e usável
     como amostra (mongorestore --nsInclude não filtra por projeto dentro
     da coleção, só por coleção).

  2. Exportar só o projeto escolhido dentro da coleção do repositório:

       mongoexport --port 27018 --db=JiraReposAnon --collection=Apache \
         --query='{"fields.project.key":"ZOOKEEPER"}' \
         --out=zookeeper_issues.json --jsonArray

  3. Rodar este script sobre o JSON exportado.

Uso:
  python scripts/build_jira_ocel.py <issues.json> <output.json>
"""
import json
import sys
from pathlib import Path


def build(issues: list[dict]) -> dict:
    events: dict = {}
    objects: dict = {}

    for issue in issues:
        key = issue["key"]
        issue_id = issue["id"]
        obj_id = f"jira-wi-{issue_id}"
        f = issue.get("fields", {})

        assignee = f.get("assignee") or {}
        objects[obj_id] = {
            "ocel:type": "WorkItem",
            "ocel:attributes": [
                {"name": "title", "time": f.get("created", ""), "value": f.get("summary") or ""},
                {"name": "workItemType", "time": f.get("created", ""), "value": (f.get("issuetype") or {}).get("name", "")},
                {"name": "project", "time": f.get("created", ""), "value": (f.get("project") or {}).get("name", "")},
                {"name": "assignedTo", "time": f.get("created", ""), "value": assignee.get("displayName") or ""},
                {"name": "url", "time": f.get("created", ""), "value": f"https://issues.apache.org/jira/browse/{key}"},
            ],
            "ocel:relationships": [],
        }

        # changelog.histories já vem em ordem cronológica.
        status_changes = [
            (hist.get("created", ""), item.get("fromString") or "", item.get("toString") or "")
            for hist in issue.get("changelog", {}).get("histories", [])
            for item in hist.get("items", [])
            if item.get("field") == "status"
        ]

        idx = 0

        # Evento sintético de criação (fromColumn vazio). Estado inicial = fromString da primeira
        # mudança de status; se a issue nunca mudou de status, usa o
        # status atual.
        initial_status = status_changes[0][1] if status_changes else (f.get("status") or {}).get("name", "")
        if initial_status:
            idx += 1
            events[f"e-{issue_id}-{idx}"] = {
                "ocel:type": "StateTransition",
                "ocel:time": f.get("created", ""),
                "ocel:attributes": [
                    {"name": "fromColumn", "value": ""},
                    {"name": "toColumn", "value": initial_status},
                    {"name": "fromState", "value": ""},
                    {"name": "toState", "value": initial_status},
                ],
                "ocel:relationships": [{"objectId": obj_id, "qualifier": "transitioned"}],
            }

        for ts, from_s, to_s in status_changes:
            idx += 1
            events[f"e-{issue_id}-{idx}"] = {
                "ocel:type": "StateTransition",
                "ocel:time": ts,
                "ocel:attributes": [
                    {"name": "fromColumn", "value": from_s},
                    {"name": "toColumn", "value": to_s},
                    {"name": "fromState", "value": from_s},
                    {"name": "toState", "value": to_s},
                ],
                "ocel:relationships": [{"objectId": obj_id, "qualifier": "transitioned"}],
            }

    return {
        "ocel:version": "2.0",
        "ocel:ordering": "timestamp",
        "ocel:attribute-names": [
            "title", "workItemType", "project", "assignedTo", "url",
            "fromColumn", "toColumn", "fromState", "toState",
        ],
        "ocel:object-types": {"WorkItem": {"attributes": [
            {"name": "title", "type": "string"},
            {"name": "workItemType", "type": "string"},
            {"name": "project", "type": "string"},
            {"name": "assignedTo", "type": "string"},
            {"name": "url", "type": "string"},
        ]}},
        "ocel:event-types": {"StateTransition": {"attributes": [
            {"name": "fromColumn", "type": "string"},
            {"name": "toColumn", "type": "string"},
            {"name": "fromState", "type": "string"},
            {"name": "toState", "type": "string"},
        ]}},
        "ocel:events": events,
        "ocel:objects": objects,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    with open(src, encoding="utf-8") as fh:
        issues = json.load(fh)

    ocel = build(issues)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ocel, fh, ensure_ascii=False)

    print(f"issues: {len(issues)}")
    print(f"objects: {len(ocel['ocel:objects'])}")
    print(f"events: {len(ocel['ocel:events'])}")
    print(f"gravado em: {out}")


if __name__ == "__main__":
    main()
