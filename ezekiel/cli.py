"""CLI do Ezekiel — execução do pipeline via linha de comando."""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ezekiel — Pipeline de Otimização de Processos Interdependentes"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run ───────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Executa o pipeline completo")
    run_parser.add_argument("--output-dir", default="./output", help="Diretório de saída")
    run_parser.add_argument(
        "--ocel-dir",
        default="./ocels",
        help="Pasta com arquivos OCEL 2.0 .json (modo padrão)",
    )
    run_parser.add_argument("--ocel-path", help="Arquivo OCEL único (alternativa ao --ocel-dir)")
    run_parser.add_argument(
        "--profile", choices=["jira"], default=None,
        help="Semântica dos status da fonte (padrão: jira; ver LogProfile)",
    )
    run_parser.add_argument(
        "--completion-status", nargs="*", default=[],
        help="Status crus que encerram o caso, sem perfil (ex.: Resolved)",
    )
    run_parser.add_argument("--cal-days", type=float, default=365.0, help="Janela de calibração (dias)")
    run_parser.add_argument("--val-days", type=float, default=180.0, help="Janela de validação (dias)")
    run_parser.add_argument("--folds", type=int, default=1, help="Origens da validação temporal progressiva")
    run_parser.add_argument("--end-date", help="Fim da última janela de validação (YYYY-MM-DD)")
    run_parser.add_argument("--reps", type=int, default=30, help="Réplicas da validação (H1) e R fixo")
    run_parser.add_argument("--calendar", choices=["calendar", "business"], default="calendar",
                            help="Tempo corrido (padrão) ou dias úteis (calendário de trabalho)")
    run_parser.add_argument("--fixed-reps", action="store_true",
                            help="Usa --reps nas configurações, sem estudo-piloto")
    run_parser.add_argument("--jobs", type=int, help="Processos paralelos de simulação")
    run_parser.add_argument(
        "--org", default="por_papel",
        help="Modelo organizacional (agentes): nome em config/organizacao/ ou caminho de JSON",
    )

    # ── h4 ────────────────────────────────────────────────────────────────────
    h4_parser = subparsers.add_parser("h4", help="Varredura sintética de escalabilidade (H4)")
    h4_parser.add_argument("--output-dir", default="./output", help="Diretório de saída")

    # ── serve ─────────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Inicia o servidor FastAPI")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        import json
        from ezekiel.experiments.hypotheses import ExperimentSettings
        from ezekiel.pipeline import Pipeline
        exp = ExperimentSettings(
            cal_days=args.cal_days, val_days=args.val_days, folds=args.folds,
            end_date=args.end_date, reps=args.reps, organization=args.org,
            pilot_reps=None if args.fixed_reps else 10, calendar=args.calendar,
        )
        result = Pipeline().run(
            output_dir=args.output_dir,
            ocel_dir=args.ocel_dir if not args.ocel_path else None,
            ocel_path=args.ocel_path,
            experiment=exp,
            completion_statuses=frozenset(args.completion_status),
            n_jobs=args.jobs,
            profile=args.profile,
        )
        e = result.get("experiments", {})
        print(json.dumps({
            "H1": result["H1"]["status"],
            "experiments": e.get("status"),
            "H2": e.get("H2", {}).get("validated_unadjusted"),
            "H3": e.get("H3", {}).get("validated_unadjusted"),
            "H4_gap": e.get("H4_empirical", {}).get("optimality_gap"),
        }, indent=2, ensure_ascii=False))

    elif args.command == "h4":
        import json
        from pathlib import Path
        from ezekiel.experiments.hypotheses import to_jsonable
        from ezekiel.experiments.synthetic import run_h4
        res = to_jsonable(run_h4())
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "h4_synthetic.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps({k: res[k] for k in ("validated", "in_scope_instances",
                                                 "in_scope_within_limits", "message_growth")}, indent=2))

    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
