"""Entrena la red neuronal con validación walk-forward y la promueve solo si mejora."""

from _common import github_output, parser, settings_from

from src.pipeline.train import run_training


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--skip-data-update", action="store_true", help="No actualizar precios antes de entrenar")
    p.add_argument("--quick", action="store_true", help="Configuración mínima para pruebas (pocas épocas)")
    args = p.parse_args()
    extra = {}
    if args.quick:
        extra = {"model": {"neural_network": {"max_epochs": 2, "max_train_samples": 20000, "hidden_size": 16},
                           "walk_forward": {"max_folds": 2}}}
    settings = settings_from(args, extra)
    resultado = run_training(settings, update_data=not args.skip_data_update)
    github_output(status=resultado["status"], promoted=str(resultado.get("promoted", False)).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
