"""API FastAPI — backend do Ezekiel com endpoints para a interface web."""

import copy
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ezekiel.ocel.loader import OCELLoader
from ezekiel.mining.analyzer import ProcessMiningAnalyzer, MiningResult, STAGE_ABBREV
from ezekiel.legacy.engine import SimulationEngine, SimResult, SimConfig, build_whatif_config
from ezekiel.legacy.dcop.coordinator import DCOPCoordinator, DCOPResult, apply_dcop_assignments
from ezekiel.legacy.evaluator import Evaluator, EvaluationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Ezekiel", version="0.1.0")

# ─── Estado global ────────────────────────────────────────────────────────────

class PipelineState:
    def __init__(self):
        self.status: str = "idle"          # idle | running | done | error
        self.error: str = ""
        self.progress: str = ""
        self.mining: MiningResult | None = None
        self.sim_s0: SimResult | None = None
        self.sim_s1: SimResult | None = None
        self.sim_s2: SimResult | None = None
        self.dcop: DCOPResult | None = None
        self.evaluation: EvaluationResult | None = None
        self.base_cfg: SimConfig | None = None
        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

state = PipelineState()

# ─── Servir HTML ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

# ─── Models ───────────────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    extra_servers: int = 2
    wip_reduction_pct: float = 30.0
    constraint_stage: str | None = None

# ─── Pipeline ─────────────────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
async def run_pipeline(background_tasks: BackgroundTasks):
    if state.status == "running":
        raise HTTPException(409, "Pipeline já está em execução.")
    state.update(status="running", error="")
    background_tasks.add_task(_run_pipeline_task)
    return {"status": "started"}


@app.get("/api/pipeline/status")
async def pipeline_status():
    return {
        "status": state.status,
        "progress": state.progress,
        "error": state.error,
    }

# ─── Mining ───────────────────────────────────────────────────────────────────

@app.get("/api/mining")
async def get_mining():
    _require_done()
    m = state.mining
    return {
        "active_constraint": m.active_constraint.stage if m.active_constraint else None,
        "constraint_score": round(m.active_constraint.constraint_score, 2) if m.active_constraint else 0,
        "lead_time_days": round(m.mean_lead_time_days, 1),
        "waiting_time_days": round(m.mean_waiting_time_days, 1),
        "throughput_per_week": round(m.mean_throughput_per_week, 1),
        "flow_efficiency_pct": round(m.flow_efficiency * 100, 1),
        "fitness": round(m.fitness, 2),
        "rework_rate_pct": round(m.rework_rate * 100, 1),
        "stages": [
            {
                "name": s.stage,
                "abbrev": STAGE_ABBREV.get(s.stage, s.stage),
                "event_count": m.dfg.get("activity_counts", {}).get(s.stage, 0),
                "waiting_time": round(s.mean_waiting_time_days, 1),
                "wip": round(s.mean_wip, 1),
                "utilization": round(s.utilization, 3),
                "throughput": round(s.throughput_per_week, 1),
                "constraint_score": round(s.constraint_score, 3),
                "is_constraint": m.active_constraint and s.stage == m.active_constraint.stage,
            }
            for s in m.stage_metrics
        ],
    }

@app.get("/api/mining/dfg")
async def get_dfg():
    _require_done()
    m = state.mining
    dfg = m.dfg
    stage_map = {sm.stage: sm for sm in m.stage_metrics}

    activities = [
        {
            "id": sm.stage,
            "abbrev": STAGE_ABBREV.get(sm.stage, sm.stage),
            "count": dfg.get("activity_counts", {}).get(sm.stage, 0),
            "mean_duration": round(sm.mean_waiting_time_days, 1),
            "is_constraint": bool(m.active_constraint and sm.stage == m.active_constraint.stage),
            "utilization": round(sm.utilization, 3),
            "constraint_score": round(sm.constraint_score, 3),
        }
        for sm in m.stage_metrics
    ]

    max_count = max((e["count"] for e in dfg.get("edges", [])), default=1)
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "count": e["count"],
            "frequency_pct": round(e["count"] / max_count * 100, 1),
        }
        for e in dfg.get("edges", [])
    ]

    return {
        "activities": activities,
        "edges": edges,
        "start_activities": dfg.get("start_activities", {}),
        "end_activities": dfg.get("end_activities", {}),
        "total_cases": dfg.get("total_cases", 0),
        "fitness": round(m.fitness, 2),
        "rework_rate_pct": round(m.rework_rate * 100, 1),
    }


# ─── Simulação ────────────────────────────────────────────────────────────────

@app.get("/api/simulation/s0")
async def get_s0():
    _require_done()
    return _sim_to_dict(state.sim_s0, "S0", "Baseline histórico")


@app.get("/api/simulation/s1")
async def get_s1():
    _require_done()
    return _sim_to_dict(state.sim_s1, "S1", "What-if (intervenção no gargalo)")


@app.get("/api/simulation/s2")
async def get_s2():
    _require_done()
    return _sim_to_dict(state.sim_s2, "S2", "DCOP/ADOPT coordenado")


@app.post("/api/simulation/whatif")
async def run_whatif(req: WhatIfRequest):
    _require_done()
    engine = SimulationEngine()
    cfg = copy.deepcopy(state.base_cfg)
    cfg.n_runs = 10

    constraint = req.constraint_stage or (
        state.mining.active_constraint.stage if state.mining.active_constraint else None
    )
    cfg = build_whatif_config(
        cfg,
        constraint_stage=constraint,
        extra_servers=req.extra_servers,
        wip_reduction_pct=req.wip_reduction_pct / 100,
    )
    result = engine.run(cfg)
    baseline_lt = state.mining.mean_lead_time_days
    return {
        **_sim_to_dict(result, "whatif", "Simulação interativa"),
        "delta_lead_time_pct": round(
            (result.mean_lead_time - baseline_lt) / baseline_lt * 100, 1
        ) if baseline_lt else 0,
    }

# ─── DCOP ─────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Remove inf/nan de estruturas aninhadas para serialização JSON."""
    if isinstance(obj, float):
        if obj == float("inf") or obj == float("-inf") or obj != obj:
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.get("/api/dcop")
async def get_dcop():
    _require_done()
    d = state.dcop
    return JSONResponse(_sanitize({
        "converged": d.converged,
        "cycles": d.cycles_to_converge,
        "global_cost": round(d.global_cost, 3),
        "assignments": d.assignments,
        "messages": d.message_log,
    }))

# ─── Avaliação ────────────────────────────────────────────────────────────────

@app.get("/api/evaluation")
async def get_evaluation():
    _require_done()
    e = state.evaluation
    m = state.mining
    mae = abs(state.sim_s0.mean_lead_time - m.mean_lead_time_days) / m.mean_lead_time_days if m.mean_lead_time_days else 0

    scenarios = []
    for s in e.scenarios:
        if s.label in ("S0", "S1", "S2"):
            ref_lt = next((sc.lead_time for sc in e.scenarios if sc.label == "S0"), s.lead_time)
            delta = (s.lead_time - ref_lt) / ref_lt * 100 if ref_lt and s.label != "S0" else 0
            scenarios.append({
                "label": s.label,
                "description": s.description,
                "lead_time": round(s.lead_time, 1),
                "waiting_time": round(s.waiting_time, 1),
                "throughput": round(s.throughput, 1),
                "flow_efficiency_pct": round(s.flow_efficiency * 100, 1),
                "mean_wip": round(s.mean_wip, 1),
                "delta_lead_time_pct": round(delta, 1),
            })

    return {
        "mae_pct": round(mae * 100, 1),
        "scenarios": scenarios,
        "hypotheses": [
            {
                "name": h.name,
                "validated": h.validated,
                "p_value": round(h.p_value, 4),
                "description": h.description,
            }
            for h in e.hypotheses
        ],
        "best_scenario": e.best_scenario.label if e.best_scenario else None,
    }

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_done():
    if state.status != "done":
        raise HTTPException(400, "Execute o pipeline primeiro (/api/pipeline/run).")

def _sim_to_dict(r: SimResult, label: str, desc: str) -> dict:
    return {
        "label": label,
        "description": desc,
        "lead_time": round(r.mean_lead_time, 1),
        "waiting_time": round(r.mean_waiting_time, 1),
        "throughput": round(r.mean_throughput, 1),
        "flow_efficiency_pct": round(r.flow_efficiency * 100, 1),
        "stage_wip": {k: round(v, 1) for k, v in r.stage_wip.items()},
        "stage_utilization": {k: round(v, 3) for k, v in r.stage_utilization.items()},
    }

# ─── Background task ──────────────────────────────────────────────────────────

def _run_pipeline_task():
    try:
        ocel_dir = Path(__file__).parent.parent / "ocels"

        state.update(progress="Camada 1+2 — Carregando OCELs...")
        loader = OCELLoader()
        ocel = loader.load_dir(ocel_dir)

        state.update(progress="Camada 3 — Process Mining...")
        mining = ProcessMiningAnalyzer().analyze(ocel)
        state.update(mining=mining)

        state.update(progress="Camada 4a — Simulação S0 (baseline)...")
        engine = SimulationEngine()
        base_cfg = engine.calibrate(mining)
        base_cfg.n_runs = 30
        sim_s0 = engine.run(base_cfg)
        state.update(sim_s0=sim_s0, base_cfg=base_cfg)

        state.update(progress="Camada 4a — Simulação S1 (what-if)...")
        constraint = mining.active_constraint.stage if mining.active_constraint else None
        s1_cfg = build_whatif_config(base_cfg, constraint_stage=constraint, extra_servers=2, wip_reduction_pct=0.3)
        sim_s1 = engine.run(s1_cfg)
        state.update(sim_s1=sim_s1)

        state.update(progress="Camada 4b — DCOP/ADOPT (convergindo agentes)...")
        coord = DCOPCoordinator(engine)
        dcop = coord.run(mining, base_cfg, max_cycles=100)
        state.update(dcop=dcop)

        state.update(progress="Camada 4a — Simulação S2 (DCOP aplicado)...")
        # Usa a atribuição real do ADOPT (result.dcop.assignments) — antes esta
        # rota aplicava uma heurística própria (-3 WIP / 2 servidores nas 2
        # etapas de maior constraint_score) que ignorava a saída do DCOP e
        # divergia do que o CLI (ezekiel.pipeline) reportava para o mesmo cenário.
        dcop_cfg = apply_dcop_assignments(base_cfg, dcop)
        sim_s2 = engine.run(dcop_cfg)
        state.update(sim_s2=sim_s2)

        state.update(progress="Camada 5 — Avaliação sistêmica...")
        evaluation = Evaluator().evaluate(mining, sim_s0, sim_s1, sim_s2)
        state.update(evaluation=evaluation)

        state.update(status="done", progress="Pipeline concluído.")
        logger.info("Pipeline concluído com sucesso.")

    except Exception as e:
        logger.error(f"Erro no pipeline: {e}", exc_info=True)
        state.update(status="error", error=str(e), progress="")
