"""Genera los JSON estáticos que consume la web (frontend/public/data por defecto)."""

from _common import parser, settings_from

from src.pipeline.export import run_export


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--out", help="Carpeta de salida (por defecto frontend/public/data)")
    args = p.parse_args()
    settings = settings_from(args)
    resultado = run_export(settings, args.out)
    print(resultado)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
