"""Pipeline diario completo: datos, indicadores, noticias, modelos, ranking, paper trading e historial."""

from _common import github_output, parser, settings_from

from src.pipeline.update import run_daily_update


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--force", action="store_true", help="Procesar aunque la sesión ya tenga snapshot")
    args = p.parse_args()
    settings = settings_from(args)
    resultado = run_daily_update(settings, force=args.force)
    github_output(status=resultado["status"], as_of=resultado.get("as_of", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
