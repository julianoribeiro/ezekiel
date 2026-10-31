"""Análise de sensibilidade e robustez (ezekiel/experiments/sensitivity.py).

Uso:
  python scripts/run_sensitivity.py --only apache-zookeeper              # local, todos os fatores
  python scripts/run_sensitivity.py --only mariadb-mdev --factors demand capacity_quantile --quick
  python scripts/run_sensitivity.py --only mariadb-mdev --factorial demand:−20%/+20% couplings:ref/ligados

Saída: output/sensitivity/<organização>/<repo>.json (local) e <repo>_fatorial.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ezekiel import manifest  # noqa: E402
from ezekiel.experiments.hypotheses import ExperimentSettings  # noqa: E402
from ezekiel.experiments.sensitivity import FACTORS, factorial_analysis, local_analysis, quick  # noqa: E402
from ezekiel.pipeline import load_eventlog  # noqa: E402
from ezekiel.simulation.engine import Simulator  # noqa: E402
from run_hypotheses import DATASETS, ocel_path_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--factors", nargs="*", choices=list(FACTORS))
    ap.add_argument("--factorial", nargs="*", help="fator:nível_baixo/nível_alto (\"ref\" = referência)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--org", default="por_papel")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--out", default=str(ROOT / "output" / "sensitivity"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = ExperimentSettings(organization=args.org)
    if args.quick:
        base = quick(base)
    out = Path(args.out) / Path(args.org).stem
    out.mkdir(parents=True, exist_ok=True)

    def show(row):
        r = row.get("result", {})
        h2 = (r.get("H2") or {}).get("delta_pct")
        rob = row.get("robustness", {})
        print(f"  {row.get('factor', row.get('levels'))} {row.get('level', '')}: H1={r.get('H1')} "
              f"a*={r.get('active_constraint')} H2Δ={h2} dir={rob.get('H2_direction_same')} "
              f"prát={rob.get('H2_practical_same')}", flush=True)

    with Simulator(args.jobs) as sim:
        for name in args.only or list(DATASETS):
            log, _ = load_eventlog(ocel_path=ocel_path_for(name), profile="jira")
            print(f"=== {name} ===", flush=True)
            if args.factorial:
                fac = {}
                for spec in args.factorial:
                    f, lv = spec.split(":")
                    lo, hi = lv.split("/")
                    fac[f] = (lo, hi)
                res = factorial_analysis(log, name, base, sim, fac, progress=show)
                (out / f"{name}_fatorial.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            else:
                res = local_analysis(log, name, base, sim, args.factors, progress=show)
                (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            manifest.write(out / f"{name}_manifest.json", base, [ocel_path_for(name)],
                           {"factors": args.factors, "factorial": args.factorial})


if __name__ == "__main__":
    main()
