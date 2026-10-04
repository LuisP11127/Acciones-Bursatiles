"""Backtest sin reentrenar (usa las predicciones fuera de muestra del último entrenamiento si existen)."""

from _common import parser, settings_from

from src.pipeline.backtest import run_backtest_pipeline


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--update-data", action="store_true")
    args = p.parse_args()
    settings = settings_from(args)
    resultado = run_backtest_pipeline(settings, update_data=args.update_data)
    print(resultado)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
