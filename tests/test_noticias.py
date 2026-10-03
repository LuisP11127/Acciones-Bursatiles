from datetime import datetime, timedelta, timezone

from bolsa.config import cargar_config
from bolsa.noticias import Noticia, analizar_noticias


def test_sentimiento_financiero_y_recencia():
    cfg = cargar_config()
    ahora = datetime(2026, 10, 3, tzinfo=timezone.utc)
    buenas = [Noticia("Apple beats estimates and raises guidance", fecha=ahora - timedelta(hours=5)),
              Noticia("Analysts upgrade Apple, shares surge to record", fecha=ahora - timedelta(days=1))]
    malas = [Noticia("Apple misses revenue, shares plunge after downgrade", fecha=ahora - timedelta(hours=3)),
             Noticia("Regulators open fraud investigation into Apple", fecha=ahora - timedelta(days=2))]
    positivo = analizar_noticias(buenas, cfg, ahora)
    negativo = analizar_noticias(malas, cfg, ahora)
    assert positivo["sentimiento"] > 0.1 and positivo["puntaje"] > 50
    assert negativo["sentimiento"] < -0.1 and negativo["puntaje"] < 50
    assert positivo["etiqueta"] in ("Positivo", "Muy positivo")


def test_descarta_viejas_y_duplicadas():
    cfg = cargar_config()
    ahora = datetime(2026, 10, 3, tzinfo=timezone.utc)
    lista = [Noticia("Same headline", fecha=ahora), Noticia("Same headline!", fecha=ahora),
             Noticia("Very old news", fecha=ahora - timedelta(days=60))]
    resultado = analizar_noticias(lista, cfg, ahora)
    assert len(resultado["noticias"]) == 1
    assert analizar_noticias([], cfg, ahora)["puntaje"] is None
