"""Converte os 16 projetos do The Public Jira Dataset para OCEL 2.0 multiobjeto.

Para cada repositório: restaura só a sua coleção do dump (mongorestore
--nsInclude), converte o projeto selecionado (build_jira_ocel2.py) e apaga a
coleção antes da próxima — nunca mais de um repositório em disco.

Pré-requisito: mongod em 127.0.0.1:27018.
Uso: python scripts/convert_jira_ocel2_all.py [--only apache-zookeeper ...]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pymongo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_jira_ocel2 import build  # noqa: E402

ARCHIVE = ROOT / "ocels" / "ThePublicJiraDataset" / "3. DataDump" / "mongodump-JiraReposAnon.archive"
OUT = ROOT / "ocels" / "ocel2"
PROJECTS = {
    "apache-zookeeper": ("Apache", "ZOOKEEPER"), "mindville-ics": ("Mindville", "ICS"),
    "secondlife-storm": ("SecondLife", "STORM"), "jiraecosystem-amkt": ("JiraEcosystem", "AMKT"),
    "sonatype-nexus": ("Sonatype", "NEXUS"), "inteldaos-daos": ("IntelDAOS", "DAOS"),
    "jfrog-rtfact": ("JFrog", "RTFACT"), "hyperledger-fab": ("Hyperledger", "FAB"),
    "spring-spr": ("Spring", "SPR"), "mariadb-mdev": ("MariaDB", "MDEV"),
    "redhat-jbide": ("RedHat", "JBIDE"), "sakai-sak": ("Sakai", "SAK"),
    "jira-jraserver": ("Jira", "JRASERVER"), "mongodb-server": ("MongoDB", "SERVER"),
    "qt-qtbug": ("Qt", "QTBUG"), "mojang-mc": ("Mojang", "MC"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--port", type=int, default=27018)
    args = ap.parse_args()
    db = pymongo.MongoClient("127.0.0.1", args.port)["JiraReposAnon"]
    OUT.mkdir(parents=True, exist_ok=True)
    for label in args.only or PROJECTS:
        repo, project = PROJECTS[label]
        out = OUT / f"{label}.json"
        if out.exists():
            print(f"[SKIP] {label}: já convertido", flush=True)
            continue
        t0 = time.time()
        if repo not in db.list_collection_names():
            subprocess.run(["mongorestore", "--port", str(args.port), "--gzip", f"--archive={ARCHIVE}",
                            f"--nsInclude=JiraReposAnon.{repo}", "--quiet"], check=True)
        issues = list(db[repo].find({"fields.project.key": project}, {"_id": 0}))
        ocel, meta = build(issues, repo)
        out.write_text(json.dumps(ocel, ensure_ascii=False), encoding="utf-8")
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        db[repo].drop()
        types = {}
        for o in ocel["objects"]:
            types[o["type"]] = types.get(o["type"], 0) + 1
        print(f"[OK] {label} ({time.time() - t0:.0f}s): {len(issues)} issues, objetos {types}, "
              f"{len(ocel['events'])} eventos", flush=True)
        del issues, ocel


if __name__ == "__main__":
    main()
