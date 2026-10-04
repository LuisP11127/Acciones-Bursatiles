"""Calendario de la bolsa de Nueva York (NYSE / Nasdaq) en su zona horaria.

GitHub Actions solo admite cron en UTC; el workflow se lanza a una hora UTC
segura y este módulo decide si hay una sesión cerrada que procesar.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# Cierres extraordinarios (duelos nacionales, huracanes)
CIERRES_ESPECIALES = {
    date(2012, 10, 29), date(2012, 10, 30), date(2018, 12, 5), date(2025, 1, 9),
}


def _pascua(anio: int) -> date:
    """Domingo de Pascua (algoritmo gregoriano anónimo)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mes = (h + l_ - 7 * m + 114) // 31
    dia = (h + l_ - 7 * m + 114) % 31 + 1
    return date(anio, mes, dia)


def _n_esimo(anio: int, mes: int, dia_semana: int, n: int) -> date:
    """n-ésimo día de la semana del mes (n=-1: el último). Lunes=0."""
    if n > 0:
        d = date(anio, mes, 1)
        d += timedelta(days=(dia_semana - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    siguiente = date(anio + (mes == 12), mes % 12 + 1, 1)
    d = siguiente - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - dia_semana) % 7)


def _observado(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def nyse_holidays(anio: int) -> frozenset[date]:
    festivos = set()
    ano_nuevo = date(anio, 1, 1)
    if ano_nuevo.weekday() == 6:
        festivos.add(ano_nuevo + timedelta(days=1))
    elif ano_nuevo.weekday() != 5:  # si cae en sábado la NYSE no lo traslada al viernes
        festivos.add(ano_nuevo)
    festivos.add(_n_esimo(anio, 1, 0, 3))      # Martin Luther King
    festivos.add(_n_esimo(anio, 2, 0, 3))      # Presidents' Day
    festivos.add(_pascua(anio) - timedelta(days=2))  # Viernes Santo
    festivos.add(_n_esimo(anio, 5, 0, -1))     # Memorial Day
    if anio >= 2022:
        festivos.add(_observado(date(anio, 6, 19)))  # Juneteenth
    festivos.add(_observado(date(anio, 7, 4)))       # Independencia
    festivos.add(_n_esimo(anio, 9, 0, 1))      # Labor Day
    festivos.add(_n_esimo(anio, 11, 3, 4))     # Acción de Gracias
    festivos.add(_observado(date(anio, 12, 25)))     # Navidad
    festivos |= {d for d in CIERRES_ESPECIALES if d.year == anio}
    return frozenset(festivos)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def add_trading_days(d: date, n: int) -> date:
    for _ in range(n):
        d = next_trading_day(d)
    return d


def last_completed_session(now: datetime | None = None, close_time: str = "16:00",
                           delay_minutes: int = 45, tz: str = "America/New_York") -> date:
    """Última sesión cuyo cierre (más un margen para que el proveedor publique los
    datos) ya ocurrió, en la zona horaria del mercado."""
    zona = ZoneInfo(tz)
    ahora = (now or datetime.now(timezone.utc)).astimezone(zona)
    hora, minuto = (int(x) for x in close_time.split(":"))
    listo = datetime.combine(ahora.date(), time(hora, minuto), tzinfo=zona) + timedelta(minutes=delay_minutes)
    hoy = ahora.date()
    if is_trading_day(hoy) and ahora >= listo:
        return hoy
    return previous_trading_day(hoy)
