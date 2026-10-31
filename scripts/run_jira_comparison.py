"""Roda o pipeline completo (Camadas 3-5) isoladamente sobre cada um dos
datasets Jira de porte pequeno/médio gerados por build_jira_ocel.py,
salvando um pipeline_result.json por dataset em output/jira-comparison/.

Não usa Pipeline.run()/--ocel-dir porque esses carregam TODOS os .json de
um diretório e os uniriam num log só — aqui cada dataset precisa ser
analisado isoladamente para a restrição ativa (ToC) de cada projeto não se
misturar com a dos outros. Em vez disso, chama OCELLoader._normalize() em
cada arquivo individualmente (mesma normalização, sem o merge de load_dir).

Uso:
  python scripts/run_jira_comparison.py
"""
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

from ezekiel.ocel.loader import OCELLoader
from ezekiel.mining.analyzer import ProcessMiningAnalyzer
from ezekiel.legacy.engine import SimulationEngine, build_whatif_config
from ezekiel.legacy.dcop.coordinator import DCOPCoordinator, apply_dcop_assignments
from ezekiel.legacy.evaluator import Evaluator

ROOT = Path(__file__).parent.parent
OCELS = ROOT / "ocels"
OUT = ROOT / "output" / "jira-comparison"

# (rótulo, caminho) — datasets de porte pequeno/médio (<= ~27k issues,
# mesma ordem de grandeza do Zookeeper). Os 5 maiores (Sakai, Jira/
# JRASERVER, MongoDB/SERVER, Qt/QTBUG, Mojang/MC) ficam de fora por ora —
# ver conversa: o ProcessMiningAnalyzer faz varreduras O(itens×eventos)
# por item que não escalam para esses volumes sem otimização prévia.
DATASETS = [
    ("apache-zookeeper", OCELS / "ocel-jira-apache-zookeeper.json"),
    ("mindville-ics", OCELS / "jira-repos" / "ocel-jira-mindville-ics.json"),
    ("secondlife-storm", OCELS / "jira-repos" / "ocel-jira-secondlife-storm.json"),
    ("jiraecosystem-amkt", OCELS / "jira-repos" / "ocel-jira-jiraecosystem-amkt.json"),
    ("sonatype-nexus", OCELS / "jira-repos" / "ocel-jira-sonatype-nexus.json"),
    ("inteldaos-daos", OCELS / "jira-repos" / "ocel-jira-inteldaos-daos.json"),
    ("jfrog-rtfact", OCELS / "jira-repos" / "ocel-jira-jfrog-rtfact.json"),
    ("hyperledger-fab", OCELS / "jira-repos" / "ocel-jira-hyperledger-fab.json"),
    ("spring-spr", OCELS / "jira-repos" / "ocel-jira-spring-spr.json"),
    ("mariadb-mdev", OCELS / "jira-repos" / "ocel-jira-mariadb-mdev.json"),
    ("redhat-jbide", OCELS / "jira-repos" / "ocel-jira-redhat-jbide.json"),
    # Datasets grandes — inviáveis antes da otimização de
    # ProcessMiningAnalyzer (varreduras O(itens×eventos) por item);
    # com o groupby único (_build_item_sequences) ficaram tratáveis.
    ("sakai-sak", OCELS / "jira-repos" / "ocel-jira-sakai-sak.json"),
    ("jira-jraserver", OCELS / "jira-repos" / "ocel-jira-jira-jraserver.json"),
    ("mongodb-server", OCELS / "jira-repos" / "ocel-jira-mongodb-server.json"),
    ("qt-qtbug", OCELS / "jira-repos" / "ocel-jira-qt-qtbug.json"),
    ("mojang-mc", OCELS / "jira-repos" / "ocel-jira-mojang-mc.json"),
]


def run_one(label: str, path: Path) -> dict:
    loader = OCELLoader()
    raw = loader._load_file(path)
    events, objects, skipped = loader._normalize(raw, source=path.stem)
    ocel = {
        "ocel:global-log": {"ocel:version": "2.0", "ocel:ordering": "timestamp"},
        "ocel:events": events,
        "ocel:objects": objects,
        "_sources": [path.name],
    }

    sim_engine = SimulationEngine()
    dcop_coordinator = DCOPCoordinator(sim_engine)

    mining = ProcessMiningAnalyzer().analyze(ocel)

    base_cfg = sim_engine.calibrate(mining)
    sim_s0 = sim_engine.run(base_cfg)
    mae = sim_engine.compute_mae(sim_s0, mining)

    constraint_stage = mining.active_constraint.stage if mining.active_constraint else "Development"
    whatif_cfg = build_whatif_config(base_cfg, constraint_stage=constraint_stage, extra_servers=2, wip_reduction_pct=0.30)
    sim_s1 = sim_engine.run(whatif_cfg)

    dcop = dcop_coordinator.run(mining, base_cfg)
    dcop_cfg = apply_dcop_assignments(base_cfg, dcop)
    sim_s2 = sim_engine.run(dcop_cfg)

    evaluation = Evaluator().evaluate(mining, sim_s0, sim_s1, sim_s2)

    return {
        "label": label,
        "n_objects": len(objects),
        "n_events": len(events),
        "skipped": skipped,
        "mining": mining.summary(),
        "sim_s0": sim_s0.summary(),
        "sim_s1": sim_s1.summary(),
        "sim_s2": sim_s2.summary(),
        "dcop": dcop.summary(),
        "evaluation": evaluation.summary(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = []

    for label, path in DATASETS:
        if not path.exists():
            print(f"[SKIP] {label}: arquivo não encontrado em {path}")
            continue

        print(f"=== {label} ===", flush=True)
        t0 = time.time()
        try:
            result = run_one(label, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] {label}: {exc}", flush=True)
            continue
        elapsed = time.time() - t0

        result_dir = OUT / label
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "pipeline_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        m = result["mining"]
        e = result["evaluation"]
        print(
            f"[OK] {label} ({elapsed:.1f}s) — "
            f"LT={m['lead_time_days']}d restrição={m['active_constraint']} "
            f"fitness={m['fitness']} MAE={e['mae_pct']}% "
            f"H1={e['h1_validated']} H2={e['h2_validated']} H3={e['h3_validated']} "
            f"melhor={e['best_scenario']}",
            flush=True,
        )
        comparison.append({"label": label, "elapsed_s": round(elapsed, 1), **m, **{f"eval_{k}": v for k, v in e.items()}})

    with open(OUT / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OK] Comparação salva em {OUT / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
