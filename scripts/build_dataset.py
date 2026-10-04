"""Construye el dataset de entrenamiento (features punto a punto + etiquetas) y guarda un resumen."""

import numpy as np

from _common import parser, settings_from

from src.data.storage import write_json
from src.ml.dataset import build_panel_dataset
from src.pipeline.train import load_analyses


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--update-data", action="store_true", help="Actualizar precios antes de construir el dataset")
    args = p.parse_args()
    settings = settings_from(args)
    settings.paths.ensure()
    mercado, analisis = load_analyses(settings, update_data=args.update_data)
    f = settings.universe["filters"]
    ds = build_panel_dataset(analisis, int(settings.model["features"]["lookback"]), float(f["min_price"]),
                             float(f["min_avg_dollar_volume"]))
    etiquetadas = ds.window_ok & ds.tradable & ~np.isnan(ds.label)
    np.savez_compressed(settings.paths.processed / "dataset.npz", features=ds.features, label=ds.label,
                        trade_return=ds.trade_return, mae=ds.mae, dates=ds.dates.astype(str),
                        symbol_idx=ds.symbol_idx, window_ok=ds.window_ok, tradable=ds.tradable,
                        symbols=np.array(ds.symbols), feature_names=np.array(ds.feature_names))
    write_json(settings.paths.processed / "dataset_summary.json", {
        "rows": int(len(ds)), "symbols": len(ds.symbols), "labeled_samples": int(etiquetadas.sum()),
        "positive_rate": float(np.nanmean(ds.label[etiquetadas])) if etiquetadas.any() else None,
        "start": str(ds.dates.min()), "end": str(ds.dates.max()), "features": ds.feature_names,
        "features_version": settings.model["features"]["version"]})
    print(f"Dataset: {len(ds)} filas, {int(etiquetadas.sum())} ejemplos etiquetados, {len(ds.symbols)} acciones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
