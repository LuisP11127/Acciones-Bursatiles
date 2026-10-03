"""Gráficos PNG: velas con MA(7), MA(25), MA(99) y volumen, curva de la cartera
y evolución del aprendizaje de la red neuronal."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

C = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta2": "#52514e",
    "tenue": "#898781",
    "rejilla": "#e1e0d9",
    "eje": "#c3c2b7",
    "ma7": "#2a78d6",
    "ma25": "#eb6834",
    "ma99": "#4a3aa7",
    "sube": "#0ca30c",
    "baja": "#d03b3b",
    "serie1": "#2a78d6",
    "serie2": "#eb6834",
}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
ESTILO_MA = (("MA7", "MA(7)", C["ma7"]), ("MA25", "MA(25)", C["ma25"]), ("MA99", "MA(99)", C["ma99"]))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": C["eje"],
    "axes.labelcolor": C["tinta2"],
    "xtick.color": C["tenue"],
    "ytick.color": C["tenue"],
    "axes.titlecolor": C["tinta"],
})


def _fecha_corta(ts: pd.Timestamp) -> str:
    return f"{ts.day} {MESES[ts.month - 1]} {ts.strftime('%y')}"


def _fechas_espanol(ax) -> None:
    """Eje X de fechas en español: "15 sep" en rangos cortos, "sep 26" en largos."""
    inicio, fin = ax.get_xlim()
    if fin - inicio <= 120:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        formato = lambda d: f"{d.day} {MESES[d.month - 1]}"  # noqa: E731
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int((fin - inicio) / 30 / 8) + 1)))
        formato = lambda d: f"{MESES[d.month - 1]} {d.strftime('%y')}"  # noqa: E731
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: formato(mdates.num2date(v))))


def _ejes_limpios(ax, rejilla_y: bool = True) -> None:
    ax.set_facecolor(C["superficie"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(C["eje"])
    if rejilla_y:
        ax.grid(axis="y", color=C["rejilla"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _barras(ax, x: np.ndarray, base: np.ndarray, alto: np.ndarray, colores, ancho: float = 0.62,
            alpha: float = 1.0, zorder: int = 3) -> None:
    """Barras como una sola colección de polígonos (mucho más rápido que ax.bar)."""
    izq, der = x - ancho / 2, x + ancho / 2
    arriba = base + alto
    vertices = np.stack([np.column_stack([izq, base]), np.column_stack([izq, arriba]),
                         np.column_stack([der, arriba]), np.column_stack([der, base])], axis=1)
    ax.add_collection(PolyCollection(vertices, facecolors=colores, edgecolors="none", alpha=alpha,
                                     zorder=zorder), autolim=True)
    ax.autoscale_view()


def _formato_precio(valor: float) -> str:
    return f"${valor:,.2f}" if valor < 1000 else f"${valor:,.0f}"


def _formato_volumen(valor: float, _pos=None) -> str:
    for limite, sufijo in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(valor) >= limite:
            return f"{valor / limite:.0f}{sufijo}"
    return f"{valor:.0f}"


def grafico_accion(df: pd.DataFrame, ticker: str, ruta: str | Path, nombre: str = "",
                   dias: int = 180, marcas: list[dict] | None = None, nota: str = "") -> Path:
    """Velas japonesas con MA(7), MA(25), MA(99) y volumen.

    `df` debe traer las columnas de agregar_indicadores (las medias se calculan
    sobre todo el historial, así la MA(99) es válida desde la primera vela).
    `marcas`: [{"fecha", "precio", "tipo": "compra"|"venta"}].
    """
    d = df.iloc[-dias:]
    x = np.arange(len(d))
    sube = (d["Close"] >= d["Open"]).to_numpy()
    colores = np.where(sube, C["sube"], C["baja"])

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.3, 1], "hspace": 0.04}
    )
    fig.patch.set_facecolor(C["superficie"])
    _ejes_limpios(ax)
    _ejes_limpios(axv)

    ax.vlines(x, d["Low"], d["High"], colors=colores, linewidth=0.8, zorder=2)
    cuerpo = (d["Close"] - d["Open"]).abs().to_numpy()
    minimo_cuerpo = float(d["Close"].mean()) * 0.0015
    _barras(ax, x, np.minimum(d["Open"], d["Close"]).to_numpy(), np.maximum(cuerpo, minimo_cuerpo), colores)

    leyenda = []
    for col, etiqueta, color in ESTILO_MA:
        if col in d and d[col].notna().any():
            ax.plot(x, d[col], color=color, linewidth=1.7, solid_capstyle="round", zorder=4)
            ultimo = d[col].dropna()
            texto = f"{etiqueta}  {_formato_precio(float(ultimo.iloc[-1]))}" if len(ultimo) else etiqueta
            leyenda.append(Line2D([0], [0], color=color, linewidth=2, label=texto))

    for marca in marcas or []:
        fecha = pd.Timestamp(marca["fecha"])
        if fecha < d.index[0] or fecha > d.index[-1]:
            continue
        pos = int(d.index.searchsorted(fecha))
        compra = marca.get("tipo", "compra") == "compra"
        ax.scatter([pos], [marca["precio"]], marker="^" if compra else "v", s=110, zorder=6,
                   color=C["sube"] if compra else C["baja"], edgecolors=C["superficie"], linewidths=1.5)
        ax.annotate("Compra" if compra else "Venta", (pos, marca["precio"]),
                    xytext=(0, -16 if compra else 12), textcoords="offset points",
                    ha="center", fontsize=8, color=C["tinta2"])
    if marcas:
        leyenda.append(Line2D([0], [0], marker="^", linestyle="", color=C["sube"], markersize=8, label="Compra"))

    cierre = float(d["Close"].iloc[-1])
    ax.scatter([x[-1]], [cierre], s=36, color=C["tinta"], edgecolors=C["superficie"], linewidths=1.5, zorder=7)
    ax.annotate(_formato_precio(cierre), (x[-1], cierre), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, fontweight="bold", color=C["tinta"])
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _formato_precio(v)))
    ax.set_xlim(-1, len(d) + 6)
    ax.legend(handles=leyenda, loc="upper left", frameon=False, fontsize=8.5, ncol=len(leyenda),
              labelcolor=C["tinta2"])

    cambio = cierre / float(df["Close"].iloc[-2]) - 1 if len(df) > 1 else 0.0
    titulo = f"{ticker}" + (f" · {nombre}" if nombre and nombre != ticker else "")
    ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold", pad=22)
    subtitulo = f"Cierre {_fecha_corta(d.index[-1])}: {_formato_precio(cierre)} ({cambio * 100:+.2f}%)"
    if nota:
        subtitulo += f"  ·  {nota}"
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9, color=C["tinta2"])

    _barras(axv, x, np.zeros(len(d)), d["Volume"].to_numpy(dtype=float), colores, alpha=0.45)
    axv.set_ylim(0, float(d["Volume"].max()) * 1.08 or 1)
    axv.yaxis.tick_right()
    axv.yaxis.set_major_formatter(FuncFormatter(_formato_volumen))
    axv.set_ylabel("Volumen", fontsize=8, color=C["tenue"])
    marcas_x = np.linspace(0, len(d) - 1, 7).astype(int)
    axv.set_xticks(marcas_x)
    axv.set_xticklabels([_fecha_corta(d.index[i]) for i in marcas_x])

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=100, facecolor=C["superficie"], bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafico_cartera(historial: list[dict], referencia: pd.Series | None, ruta: str | Path) -> Path | None:
    """Valor de la cartera simulada frente al S&P 500, ambos en base 100."""
    if len(historial) < 2:
        return None
    serie = pd.Series({pd.Timestamp(h["fecha"]): h["valor"] for h in historial}).sort_index()
    base = serie / serie.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(C["superficie"])
    _ejes_limpios(ax)
    lineas = [("Cartera simulada", base, C["serie1"])]
    if referencia is not None and len(referencia):
        ref = referencia.reindex(serie.index, method="ffill").dropna()
        if len(ref) >= 2:
            lineas.append(("S&P 500 (SPY)", ref / ref.iloc[0] * 100, C["serie2"]))
    for etiqueta, s, color in lineas:
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=2, label=etiqueta)
        ax.annotate(f"{etiqueta}: {s.iloc[-1]:.1f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=C["tinta2"])
    ax.axhline(100, color=C["eje"], linewidth=0.9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5, ncol=2,
              labelcolor=C["tinta2"])
    ax.set_title("Cartera simulada vs. S&P 500 (base 100)", loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.set_xlim(serie.index[0], serie.index[-1] + (serie.index[-1] - serie.index[0]) * 0.18)
    _fechas_espanol(ax)
    ruta = Path(ruta)
    fig.savefig(ruta, dpi=100, facecolor=C["superficie"], bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafico_aprendizaje(historial: list[dict], madurez_cfg: dict, ruta: str | Path) -> Path | None:
    """Evolución de AUC y lift de la red, sesión a sesión, frente al mínimo exigido."""
    if not historial:
        return None
    sesiones = [h["sesion"] for h in historial]
    paneles = [
        ("auc", "AUC en validación", madurez_cfg["auc_minimo"], 0.5),
        ("lift", "Lift del 5% mejor valorado", madurez_cfg["lift_minimo"], 1.0),
    ]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(C["superficie"])
    for ax, (clave, titulo, minimo, azar) in zip(ejes, paneles):
        _ejes_limpios(ax)
        valores = [h.get(clave) if h.get(clave) is not None else np.nan for h in historial]
        ax.axhline(azar, color=C["eje"], linewidth=0.9)
        ax.axhline(minimo, color=C["tenue"], linewidth=0.9)
        ax.text(sesiones[0], minimo, " mínimo para estar lista", va="bottom", fontsize=7.5, color=C["tenue"])
        ax.text(sesiones[0], azar, " azar", va="bottom", fontsize=7.5, color=C["tenue"])
        ax.plot(sesiones, valores, color=C["serie1"], linewidth=2, marker="o" if len(sesiones) < 30 else None,
                markersize=4, markeredgecolor=C["superficie"])
        ax.set_title(titulo, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Sesión de entrenamiento", fontsize=8)
    fig.tight_layout()
    ruta = Path(ruta)
    fig.savefig(ruta, dpi=100, facecolor=C["superficie"], bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafico_maestro(df: pd.DataFrame, ticker: str, operaciones: list[dict], probabilidades: pd.Series | None,
                    ruta: str | Path, dias: int = 250) -> Path:
    """Cómo enseña el maestro (compra en mínimos, vende en máximos) y qué ve la red."""
    d = df.iloc[-dias:]
    filas = 2 if probabilidades is not None else 1
    fig, ejes = plt.subplots(filas, 1, figsize=(11, 5.6 if filas == 2 else 4), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]} if filas == 2 else None, squeeze=False)
    fig.patch.set_facecolor(C["superficie"])
    ax = ejes[0][0]
    _ejes_limpios(ax)
    ax.plot(d.index, d["Close"], color=C["tinta2"], linewidth=1.4, label="Precio de cierre")
    for col, etiqueta, color in ESTILO_MA:
        if col in d:
            ax.plot(d.index, d[col], color=color, linewidth=1.2, alpha=0.8, label=etiqueta)
    visibles = [op for op in operaciones if op["fecha_compra"] >= d.index[0]]
    if visibles:
        ax.scatter([op["fecha_compra"] for op in visibles], [op["precio_compra"] for op in visibles], marker="^",
                   s=90, color=C["sube"], edgecolors=C["superficie"], linewidths=1.5, zorder=5,
                   label="Maestro compra (mínimo)")
        ax.scatter([op["fecha_venta"] for op in visibles], [op["precio_venta"] for op in visibles], marker="v",
                   s=90, color=C["baja"], edgecolors=C["superficie"], linewidths=1.5, zorder=5,
                   label="Maestro vende (máximo)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, ncol=6,
              labelcolor=C["tinta2"], handletextpad=0.4, columnspacing=1.2)
    ax.set_title(f"{ticker}: operaciones ideales del maestro", loc="left", fontsize=12, fontweight="bold", pad=26)
    if filas == 2:
        axp = ejes[1][0]
        _ejes_limpios(axp)
        p = probabilidades.reindex(d.index)
        axp.plot(p.index, p.to_numpy(), color=C["serie1"], linewidth=1.6)
        axp.fill_between(p.index, 0, p.to_numpy(), color=C["serie1"], alpha=0.1)
        axp.set_ylabel("Prob. red", fontsize=8, color=C["tenue"])
        axp.set_ylim(0, max(0.05, float(np.nanmax(p.to_numpy())) * 1.15) if p.notna().any() else 1)
    _fechas_espanol(ejes[-1][0])
    ruta = Path(ruta)
    fig.savefig(ruta, dpi=100, facecolor=C["superficie"], bbox_inches="tight")
    plt.close(fig)
    return ruta
