"""Actualiza el archivo de noticias de las oportunidades y acciones en seguimiento."""

from datetime import datetime, timezone

from _common import parser, settings_from

from src.data.providers import create_news_providers
from src.data.storage import read_json
from src.logging_utils import summary
from src.news.scoring import score_news
from src.pipeline.history import archive_news, list_snapshots, load_snapshot


def main() -> int:
    args = parser(__doc__).parse_args()
    settings = settings_from(args)
    settings.paths.ensure()
    summary.reset("update_news")
    fechas = list_snapshots(settings.paths.history)
    snap = load_snapshot(settings.paths.history, fechas[-1]) if fechas else {}
    trades = read_json(settings.paths.state / "trades.json", {"trades": []})["trades"]
    objetivos = {o["symbol"]: o.get("name") or o["symbol"] for o in (snap or {}).get("opportunities", [])}
    objetivos.update({t["symbol"]: t.get("name") or t["symbol"] for t in trades if t["status"] in ("OPEN", "PENDING")})
    cfg = settings.data["news"]
    proveedores = create_news_providers(settings)
    nuevas = []
    for symbol, nombre in list(objetivos.items())[: int(cfg["max_symbols_per_run"])]:
        items = []
        for prov in proveedores:
            items.extend(prov.fetch(symbol, nombre, int(cfg["lookback_days"]), int(cfg["max_items_per_symbol"])))
        analisis = score_news(items, symbol, nombre, cfg, as_of=datetime.now(timezone.utc))
        nuevas.extend({**n, "key": "".join(c for c in n["title"].lower() if c.isalnum())[:90]} for n in analisis["items"])
    summary.count("news_archived", archive_news(settings.paths.news, nuevas))
    summary.write(settings.paths.results / "last_news_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
