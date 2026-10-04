"""Descarga incremental y validación de precios (solo la etapa de datos)."""

from _common import parser, settings_from

from src.data.market import load_market_data, update_market_data
from src.data.providers import create_market_provider
from src.data.storage import write_json
from src.data.universe import load_universe
from src.logging_utils import summary
from src.market_calendar import last_completed_session


def main() -> int:
    args = parser(__doc__).parse_args()
    settings = settings_from(args)
    settings.paths.ensure()
    summary.reset("update_market")
    cal = settings.data["market_calendar"]
    sesion = last_completed_session(None, cal["close_time"], int(cal["data_ready_delay_minutes"]), cal["timezone"])
    universo = load_universe(settings)
    simbolos = universo["symbol"].tolist() + list(settings.universe["universe"]["reference_symbols"])
    update_market_data(settings, simbolos, create_market_provider(settings, synthetic_end=sesion), as_of=sesion)
    mercado = load_market_data(settings, universo["symbol"].tolist())
    rechazados = {s: r.issues for s, r in mercado.reports.items() if not r.usable_for_signals}
    write_json(settings.paths.results / "data_quality.json", {
        "as_of": mercado.as_of.strftime("%Y-%m-%d"), "checked": len(mercado.reports), "rejected": len(rechazados),
        "rejected_symbols": rechazados, "reports": {s: r.to_dict() for s, r in mercado.reports.items() if r.issues}})
    summary.write(settings.paths.results / "last_market_update_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
