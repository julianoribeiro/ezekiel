"""Manifesto de execução (metodologia, "Reprodutibilidade, governança e
aspectos éticos"):

  "Cada execução será associada a uma revisão do código, arquivo de
   configuração, versão do conjunto de dados, ambiente computacional e
   lista de sementes. [...] Conjuntos processados serão [...] acompanhados
   por manifesto contendo origem, transformações, quantidade de registros e
   resumo criptográfico."

O manifesto registra:
  * código: revisão git (e se há alterações não registradas) e um resumo
    SHA-256 de todos os .py do pacote e dos scripts — identifica o código
    mesmo fora do git;
  * ambiente: versão do Python, plataforma e versões dos pacotes usados;
  * dados: para cada log, caminho, tamanho e SHA-256 (o arquivo e o
    `.meta.json`, que traz origem, regras de conversão e contagens);
  * configuração completa, bases das sementes e linha de comando.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("pm4py", "simpy", "pandas", "numpy", "scipy", "pydantic", "pydantic-settings")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted([*ROOT.glob("ezekiel/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("config/**/*.json")]):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_revision() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "ezekiel", "scripts", "config"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        return {"revision": rev.stdout.strip(), "uncommitted_changes": bool(dirty.stdout.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"revision": None, "note": "sem revisão git (repositório sem commits ou git ausente)"}


def data_entry(path: Path) -> dict:
    path = Path(path)
    out = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
           "bytes": path.stat().st_size, "sha256": sha256(path)}
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        out["meta"] = {"path": str(meta.name), "sha256": sha256(meta),
                       "content": json.loads(meta.read_text(encoding="utf-8"))}
    return out


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return _jsonable(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def build(settings, data_paths: list[Path], extra: dict | None = None) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "code": {**git_revision(), "sha256": code_hash()},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions},
        "data": [data_entry(p) for p in data_paths],
        "settings": _jsonable(settings),
        "seeds": {"evaluation_base": getattr(settings, "eval_seed_base", None),
                  "cost_estimation_base": getattr(getattr(settings, "costs", None), "seed_base", None),
                  "rule": "réplica r usa semente base + r; cada caso tem gerador próprio (base, índice)"},
        **(extra or {}),
    }


def write(path: Path, settings, data_paths: list[Path], extra: dict | None = None) -> Path:
    path = Path(path)
    path.write_text(json.dumps(build(settings, data_paths, extra), ensure_ascii=False, indent=1), encoding="utf-8")
    return path
