"""Testa H1--H4 nos 16 repositórios do The Public Jira Dataset e agrega.

Cada repositório é analisado isoladamente (ezekiel.experiments.hypotheses);
depois os resultados são agregados por hipótese, com ajuste de Holm entre
repositórios dentro de cada família (H2-P80, H2-vazão, H3-P80), e a
varredura sintética de H4 é executada (opcional).

Uso:
  python scripts/run_hypotheses.py                      # todos, configuração completa
  python scripts/run_hypotheses.py --only apache-zookeeper mariadb-mdev
  python scripts/run_hypotheses.py --quick              # domínio e réplicas reduzidos
  python scripts/run_hypotheses.py --h4                 # inclui varredura sintética de H4
  python scripts/run_hypotheses.py --aggregate-only     # só reagrega os JSON já salvos

  python scripts/run_hypotheses.py --org por_papel por_estagio   # compara granularidades

Saída: output/hypotheses/<organização>/<repo>.json e summary.json;
output/hypotheses/h4_synthetic.json; output/hypotheses/granularidades.json.
Os arquivos soltos em output/hypotheses/ (sem subpasta) são da rodada anterior,
com estágios como agentes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ezekiel import manifest  # noqa: E402
from ezekiel.dcop.costs import CostSettings  # noqa: E402
from ezekiel.evaluation.stats import ALPHA, holm  # noqa: E402
from ezekiel.experiments.hypotheses import ExperimentSettings, run_dataset, to_jsonable  # noqa: E402
from ezekiel.experiments.synthetic import H4Settings, run_h4  # noqa: E402
from ezekiel.pipeline import load_eventlog  # noqa: E402
from ezekiel.simulation.engine import Simulator  # noqa: E402

OCELS = ROOT / "ocels"
OUT = ROOT / "output" / "hypotheses"

DATASETS = {
    "apache-zookeeper": OCELS / "ocel-jira-apache-zookeeper.json",
    **{name: OCELS / "jira-repos" / f"ocel-jira-{name}.json" for name in [
        "mindville-ics", "secondlife-storm", "jiraecosystem-amkt", "sonatype-nexus",
        "inteldaos-daos", "jfrog-rtfact", "hyperledger-fab", "spring-spr", "mariadb-mdev",
        "redhat-jbide", "sakai-sak", "jira-jraserver", "mongodb-server", "qt-qtbug", "mojang-mc",
    ]},
}


OCEL2 = OCELS / "ocel2"


def ocel_path_for(name: str) -> Path:
    """OCEL 2.0 multiobjeto (scripts/convert_jira_ocel2_all.py) quando existir;
    senão, o OCEL anterior só com WorkItem. A projeção sobre os itens de
    trabalho é idêntica nos dois (verificada em characterize_ocel2.py)."""
    p = OCEL2 / f"{name}.json"
    return p if p.exists() else DATASETS[name]


def aggregate(results: list[dict], alpha: float = ALPHA) -> dict:
    h1 = {r["label"]: r["H1"]["status"] for r in results}
    exp = [r for r in results if r.get("experiments", {}).get("status") == "ok"]

    def fam(getter):
        labels = [r["label"] for r in exp]
        raw = [float(getter(r["experiments"])) for r in exp]
        adj = holm(raw) if raw else []
        return labels, raw, adj

    l2, p2, a2 = fam(lambda e: e["H2"]["primary_p80"]["p_value"])
    _, n2, an2 = fam(lambda e: e["H2"]["throughput_non_inferiority"]["p_value"])
    l3, p3, a3 = fam(lambda e: e["H3"]["primary_p80"]["p_value"])

    per = {}
    for i, r in enumerate(exp):
        e = r["experiments"]
        h3_secondary_ok = not any(v["significantly_worse"] for v in e["H3"]["secondary_no_deterioration"].values())
        per[r["label"]] = {
            "H1": h1[r["label"]],
            "simulator_validated": e.get("simulator_validated_H1"),
            "agents": e["agents"],
            "active_constraint": e["constraint"]["active_constraint"],
            "constraint_stability": e["constraint"]["stability"],
            "H2": {"p80_delta_pct": e["H2"]["primary_p80"].get("delta_pct"),
                   "practical": e["H2"]["primary_p80"].get("practical"),
                   "p_raw": p2[i], "p_holm": a2[i], "th_ni_p_holm": an2[i],
                   "validated": bool(a2[i] < alpha and an2[i] < alpha),
                   "identical_configurations": e["H2"]["identical_configurations"]},
            "H3": {"p80_delta_pct": e["H3"]["primary_p80"].get("delta_pct"),
                   "p_raw": p3[i], "p_holm": a3[i],
                   "validated": bool(a3[i] < alpha and h3_secondary_ok and not e["H3"]["identical_configurations"]),
                   "identical_configurations": e["H3"]["identical_configurations"]},
            "H4_empirical": {"gap": e["H4_empirical"]["optimality_gap"],
                             "adopt_cycles": e["H4_empirical"]["adopt"]["cycles"],
                             "adopt_messages": e["H4_empirical"]["adopt"]["messages"],
                             "adopt_converged": e["H4_empirical"]["adopt"]["converged"]},
        }
    # Análise confirmatória de H2/H3 restrita aos repositórios cujo simulador foi validado (H1).
    confirm = [k for k, v in per.items() if v["simulator_validated"]]
    return {
        "alpha": alpha,
        "H1": {"by_dataset": h1,
               "approved": sum(v == "aprovada" for v in h1.values()),
               "rejected": sum(v == "reprovada" for v in h1.values()),
               "insufficient": sum(v == "dados insuficientes" for v in h1.values())},
        "H2": {"validated_all": sum(per[k]["H2"]["validated"] for k in per), "evaluated": len(per),
               "validated_confirmatory": sum(per[k]["H2"]["validated"] for k in confirm),
               "confirmatory_datasets": confirm},
        "H3": {"validated_all": sum(per[k]["H3"]["validated"] for k in per), "evaluated": len(per),
               "validated_confirmatory": sum(per[k]["H3"]["validated"] for k in confirm),
               "identical_solutions": sum(per[k]["H3"]["identical_configurations"] for k in per)},
        "by_dataset": per,
        "note": ("p-valores ajustados por Holm entre repositórios dentro de cada família; "
                 "a leitura confirmatória de H2/H3 considera só repositórios com H1 aprovada."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--h4", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--calendar", choices=["calendar", "business"], default="calendar",
                    help="tempo corrido (padrão) ou dias úteis")
    ap.add_argument("--couplings", action="store_true",
                    help="simula sincronização entre objetos e pessoas compartilhadas")
    ap.add_argument("--multitask", default="off", help="off | estimated | valor de κ")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--org", nargs="*", default=["por_papel"],
                    help="modelo(s) organizacional(is); vários = comparação de granularidades")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only or list(DATASETS)

    s = ExperimentSettings(folds=args.folds, calendar=args.calendar)
    s.calibration.synchronization = s.calibration.shared_people = args.couplings
    s.calibration.multitask = args.multitask if args.multitask in ("off", "estimated") else float(args.multitask)
    if args.quick:
        s.costs = CostSettings(wip_factors=(1.0, 2.0), policies=("FIFO", "urgency"), reps=3)
        s.reps = 10
    if args.reps:                       # R fixo, sem estudo-piloto
        s.reps = args.reps
        s.pilot_reps = None

    import dataclasses

    def out_dir(org: str) -> Path:
        d = out / Path(org).stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    used: list[Path] = []
    if not args.aggregate_only:
        with Simulator(args.jobs) as sim:
            for name in names:
                path = ocel_path_for(name)
                if not path.exists():
                    print(f"[SKIP] {name}: {path} não encontrado", flush=True)
                    continue
                try:
                    log, _ = load_eventlog(ocel_path=path, profile="jira")
                except Exception as exc:  # noqa: BLE001
                    print(f"[ERRO] {name}: {exc!r}", flush=True)
                    continue
                used.append(path)
                for org in args.org:
                    t0 = time.time()
                    print(f"=== {name} [{org}] ===", flush=True)
                    try:
                        res = to_jsonable(run_dataset(log, name, dataclasses.replace(s, organization=org), sim))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERRO] {name} [{org}]: {exc!r}", flush=True)
                        continue
                    (out_dir(org) / f"{name}.json").write_text(
                        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                    e = res.get("experiments", {})
                    line = f"[OK] {name} [{org}] ({time.time() - t0:.0f}s) H1={res['H1']['status']}"
                    if e.get("status") == "ok":
                        line += (f" agentes={e['agents']} a*={e['constraint']['active_constraint']}"
                                 f" H2={e['H2']['validated_unadjusted']} ({e['H2']['primary_p80'].get('delta_pct')})"
                                 f" H3={e['H3']['validated_unadjusted']} gap={e['H4_empirical']['optimality_gap']:.4f}")
                    else:
                        line += f" experimentos: {e.get('status')}"
                    print(line, flush=True)

    if used:
        for org in args.org:
            p = manifest.write(out_dir(org) / f"manifest_{time.strftime('%Y%m%dT%H%M%S')}.json",
                               dataclasses.replace(s, organization=org), used,
                               {"datasets": [p.stem for p in used]})
            print(f"Manifesto: {p}", flush=True)

    # H4 sintético independe do modelo organizacional: um arquivo só.
    h4_path = out / "h4_synthetic.json"
    if args.h4:
        print("=== H4 sintético ===", flush=True)
        h4 = to_jsonable(run_h4(H4Settings(), progress=lambda r: print(
            f"  n={r['n_agents']} dens={r['density']} seed={r['seed']} conv={r['adopt_converged']} "
            f"msgs={r['adopt_messages']} t={r['adopt_time_s']:.2f}s gap={r['gap']}", flush=True)))
        h4_path.write_text(json.dumps(h4, ensure_ascii=False, indent=1), encoding="utf-8")

    summaries = {}
    for org in args.org:
        d = out_dir(org)
        results = [json.loads((d / f"{n}.json").read_text()) for n in DATASETS if (d / f"{n}.json").exists()]
        if not results:
            continue
        summary = aggregate(results)
        summary["organization"] = org
        if h4_path.exists():
            summary["H4_synthetic"] = json.loads(h4_path.read_text())
        (d / "summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
        summaries[org] = summary
        print(f"[{org}]", json.dumps({k: summary[k] for k in ("H1", "H2", "H3")}, ensure_ascii=False))

    # Comparação de granularidades (metodologia, ameaça "Representação dos agentes")
    if len(summaries) > 1:
        comp = {}
        for name in DATASETS:
            row = {}
            for org, sm in summaries.items():
                v = sm["by_dataset"].get(name)
                if v:
                    row[org] = {"agents": v["agents"], "active_constraint": v["active_constraint"],
                                "H2_delta_pct": v["H2"]["p80_delta_pct"], "H2_validated": v["H2"]["validated"],
                                "H3_validated": v["H3"]["validated"]}
            if row:
                comp[name] = row
        (out / "granularidades.json").write_text(json.dumps(to_jsonable(comp), ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
        print(f"Comparação de granularidades: {out / 'granularidades.json'}")


if __name__ == "__main__":
    main()
