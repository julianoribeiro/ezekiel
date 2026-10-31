"""Camada centrada em objetos dos 16 projetos Jira (OCEL 2.0 multiobjeto).

Para cada projeto convertido por convert_jira_ocel2_all.py:
  * validação estrutural do log (metodologia, "Validação do log");
  * caracterização por tipo de objeto: objetos, eventos, variantes, DFG da
    projeção, relações objeto–objeto (metodologia, "Caracterização
    empírica");
  * indicadores de sincronização entre objetos não representados na
    simulação (vínculos de bloqueio, espera pela liberação da versão);
  * verificação de equivalência: a projeção sobre os itens de trabalho deve
    produzir exatamente o mesmo log de casos usado nos experimentos (OCEL
    anterior, só com WorkItem) — se for idêntica, os resultados de H1–H4
    permanecem válidos sem recálculo.

Saída: output/hypotheses/object_centric.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ezekiel.data.eventlog import JIRA_PROFILE  # noqa: E402
from ezekiel.experiments.hypotheses import to_jsonable  # noqa: E402
from ezekiel.ocel.loader import COLUMN_TO_STAGE  # noqa: E402
from ezekiel.ocel.ocel2 import synchronization_indicators  # noqa: E402
from ezekiel.pipeline import load_eventlog  # noqa: E402
from scripts.run_hypotheses import DATASETS  # noqa: E402

OCEL2 = ROOT / "ocels" / "ocel2"


def same_log(a, b) -> bool:
    cols = ["case", "stage", "enter", "exit", "next"]
    x = a.visits.sort_values(["case", "enter"]).reset_index(drop=True)[cols]
    y = b.visits.sort_values(["case", "enter"]).reset_index(drop=True)[cols]
    return (x.shape == y.shape and (x[["case", "stage"]].values == y[["case", "stage"]].values).all()
            and np.allclose(x["enter"], y["enter"]) and np.allclose(x["exit"].fillna(-1), y["exit"].fillna(-1)))


def main() -> None:
    out_path = ROOT / "output" / "hypotheses" / "object_centric.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = set(JIRA_PROFILE.completion_statuses) | {c for c, s in COLUMN_TO_STAGE.items() if s == "Deploy"}
    for label, old_path in DATASETS.items():
        p = OCEL2 / f"{label}.json"
        if label in out and "--refazer" not in sys.argv:
            continue
        if not p.exists():
            print(f"[SKIP] {label}: OCEL 2.0 ainda não convertido", flush=True)
            continue
        new, _ = load_eventlog(ocel_path=p, profile="jira")
        old, _ = load_eventlog(ocel_path=old_path, profile="jira")
        o = new.ocel
        out[label] = to_jsonable({
            "validation": o.validate(),
            "characterization": o.characterize(),
            "synchronization": synchronization_indicators(o, done),
            "item_types": new.cases["item_type"].value_counts().to_dict(),
            "projection_equals_experiment_log": same_log(old, new),
            "meta": o.meta.get("source", {}) | o.meta.get("conversion", {}),
        })
        c = out[label]
        print(f"[OK] {label}: válido={c['validation']['valid']} equivalente={c['projection_equals_experiment_log']} "
              f"objetos={ {k: v['objects'] for k, v in c['characterization']['object_types'].items()} } "
              f"bloqueio={c['synchronization']['share_items_with_blocking_link']:.3f}", flush=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
