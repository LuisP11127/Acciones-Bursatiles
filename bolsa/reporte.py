"""Reportes en Markdown (se ven directamente en GitHub y en GitHub Pages)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cartera import formato_dinero, sector_es

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
AVISO = ("> ⚠️ Herramienta educativa y experimental. **No es asesoría financiera.** Los datos pueden tener "
         "errores o retrasos; investiga por tu cuenta antes de invertir.")
NAVEGACION = ("[🏠 Panel](README.md) · [🚀 Oportunidades](oportunidades.md) · [💼 Seguimiento](seguimiento.md) · "
              "[🧠 Red neuronal](red_neuronal.md) · [📄 Datos CSV](datos/escaneo.csv)")


# --------------------------------------------------------------------------
# Utilidades de formato
# --------------------------------------------------------------------------

def celda(texto) -> str:
    return str(texto).replace("|", "\\|").replace("\n", " ")


def pct(valor, decimales: int = 1, signo: bool = True) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    redondeado = round(valor * 100, decimales) + 0.0  # evita "-0.0%"
    return f"{redondeado:{'+' if signo else ''}.{decimales}f}%"


def num(valor, decimales: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.{decimales}f}"


def fecha_larga(fecha) -> str:
    ts = pd.Timestamp(fecha)
    return f"{ts.day} {MESES[ts.month - 1]} {ts.year}"


def barra(progreso: float, ancho: int = 20) -> str:
    llenos = int(round(max(0.0, min(1.0, progreso)) * ancho))
    return f"`{'█' * llenos}{'░' * (ancho - llenos)}` {progreso * 100:.0f}%"


def emoji_sentimiento(valor: float | None) -> str:
    if valor is None:
        return "⚪"
    if valor >= 0.15:
        return "🟢"
    if valor <= -0.15:
        return "🔴"
    return "⚪"


def emoji_etiqueta(etiqueta: str | None) -> str:
    """Emoji del sentimiento agregado (coincide con la etiqueta de texto)."""
    if etiqueta in ("Muy positivo", "Positivo"):
        return "🟢"
    if etiqueta in ("Muy negativo", "Negativo"):
        return "🔴"
    return "⚪"


def emoji_tendencia(tendencia: str) -> str:
    return {"Alcista fuerte": "⏫", "Alcista": "🔼", "Lateral": "⏸️", "Bajista": "🔽"}.get(tendencia, "")


def ancla(ticker: str) -> str:
    return ticker.lower().replace("-", "")


def _escribir(ruta: Path, lineas: list[str]) -> None:
    ruta.write_text("\n".join(lineas).rstrip() + "\n", encoding="utf-8")


def _encabezado(titulo: str, ctx: dict) -> list[str]:
    lineas = [f"# {titulo}", ""]
    meta = f"**Datos al cierre del {fecha_larga(ctx['fecha_datos'])}** · generado {ctx['generado']}"
    lineas += [meta, "", NAVEGACION, ""]
    if ctx.get("sintetico"):
        lineas += ["> 🧪 **Modo sin conexión: datos SINTÉTICOS** generados para probar el sistema. "
                   "No representan el mercado real.", ""]
    return lineas


def _lista_noticias(noticias: dict | None, maximo: int = 6) -> list[str]:
    if not noticias or not noticias.get("noticias"):
        return ["_Sin noticias recientes._"]
    lineas = []
    for n in noticias["noticias"][:maximo]:
        fecha = f"{n.fecha.day} {MESES[n.fecha.month - 1]}" if n.fecha else "s/f"
        titulo = celda(n.titulo)
        enlace = f"[{titulo}]({n.enlace})" if n.enlace else titulo
        lineas.append(f"- {emoji_sentimiento(n.sentimiento)} {fecha} · {celda(n.fuente)} — {enlace} "
                      f"`{n.sentimiento:+.2f}`")
    return lineas


def _componentes(registro: dict) -> str:
    nombres = (("tecnico", "Técnico"), ("momentum", "Momentum"), ("noticias", "Noticias"),
               ("analistas", "Analistas"), ("red_neuronal", "Red neuronal"))
    return " · ".join(f"{etiqueta} **{registro['componentes'][clave]:.0f}**"
                      for clave, etiqueta in nombres if registro["componentes"].get(clave) is not None)


# --------------------------------------------------------------------------
# Panel principal
# --------------------------------------------------------------------------

def escribir_panel(ruta: Path, ctx: dict) -> None:
    lineas = _encabezado("📈 Escáner de acciones de EE.UU.", ctx)
    lineas += [f"Se analizaron **{ctx['n_analizadas']}** acciones de EE.UU.; "
               f"**{ctx['n_fase2']}** pasaron al análisis de noticias y analistas.", "", AVISO, ""]

    lineas += ["## 🌎 Mercado", "", "| Índice | Precio | Día | 1 mes | 3 meses | Tendencia (MA 7/25/99) |",
               "|---|---:|---:|---:|---:|---|"]
    for idx in ctx["indices"]:
        lineas.append(f"| **{idx['ticker']}** {celda(idx.get('nombre', ''))} | {formato_dinero(idx['precio'])} | "
                      f"{pct(idx['cambio_1d'], 2)} | "
                      f"{pct(idx['ret_21'])} | {pct(idx['ret_63'])} | "
                      f"{emoji_tendencia(idx['tendencia'])} {idx['tendencia']} |")
    if ctx["indices"] and ctx["indices"][0].get("grafico"):
        lineas += ["", f"![{ctx['indices'][0]['ticker']}]({ctx['indices'][0]['grafico']})"]

    lineas += ["", "## 🚀 Acciones con mayor potencial de subida", "",
               "| # | Acción | Sector | Precio | Puntaje | Señal | Potencial | Noticias | Tendencia |",
               "|---:|---|---|---:|---:|---|---:|---|---|"]
    for i, r in enumerate(ctx["top"], 1):
        noticias = r.get("noticias") or {}
        lineas.append(
            f"| {i} | [**{r['ticker']}**](oportunidades.md#{ancla(r['ticker'])}) {celda(r['nombre'])} | "
            f"{sector_es(r.get('sector'))} | {formato_dinero(r['tecnico']['precio'])} | **{r['puntaje']:.0f}** | "
            f"{r['recomendacion']} | {pct(r['potencial'])} | "
            f"{emoji_etiqueta(noticias.get('etiqueta'))} {noticias.get('etiqueta', '—')} | "
            f"{emoji_tendencia(r['tecnico']['tendencia'])} {r['tecnico']['tendencia']} |")
    if not ctx["top"]:
        lineas.append("| — | Sin candidatos hoy | | | | | | | |")
    lineas += ["", "Gráficos, señales y noticias de cada una en **[Oportunidades](oportunidades.md)**.", ""]

    red = ctx["red"]
    madurez = red["madurez"]
    lineas += ["## 🧠 Red neuronal", "",
               f"Estado: **{madurez['nivel']}** — madurez {barra(madurez['progreso'])} · "
               f"{red['meta']['sesiones']} sesiones de entrenamiento · "
               f"{red['meta']['ejemplos_vistos']:,} ejemplos vistos", ""]
    if madurez["lista"] and len(red["predicciones"]):
        lineas += ["La red ya aprendió lo suficiente. Sus acciones favoritas hoy:", "",
                   "| # | Acción | Probabilidad | × media |", "|---:|---|---:|---:|"]
        for i, p in enumerate(red["predicciones"][:5], 1):
            lineas.append(f"| {i} | **{p['ticker']}** {celda(p['nombre'])} | {pct(p['probabilidad'], 1, False)} | "
                          f"{num(p['lift'], 1)}× |")
    else:
        lineas.append("Todavía está aprendiendo: sus predicciones se registran y evalúan cada día, pero aún no "
                      "influyen en el puntaje. Detalles en **[Red neuronal](red_neuronal.md)**.")

    resumen = ctx["cartera"]["resumen"]
    lineas += ["", "## 💼 Seguimiento", "",
               f"Cartera simulada: **{formato_dinero(resumen['valor'])}** ({pct(resumen['rendimiento'])} desde el inicio) · "
               f"{resumen['posiciones']} posiciones abiertas · {resumen['operaciones_cerradas']} operaciones cerradas"
               + (f" · {pct(resumen['tasa_acierto'], 0, False)} ganadoras" if resumen["tasa_acierto"] is not None else ""),
               ""]
    eventos = ctx["cartera"]["eventos"]
    if eventos:
        lineas.append("**Movimientos de hoy:**")
        for e in eventos:
            if e["tipo"] == "compra":
                lineas.append(f"- 🟢 Compra **{e['ticker']}**: {e['acciones']} acciones a {formato_dinero(e['precio_compra'])} "
                              f"(puntaje {e['puntaje_compra']:.0f})")
            else:
                lineas.append(f"- 🔴 Venta **{e['ticker']}** a {formato_dinero(e['precio_venta'])}: "
                              f"{pct(e['resultado_pct'])} — {e['motivo_venta']}")
        lineas.append("")
    if ctx["manuales"]:
        lineas.append(f"Tus acciones: {len(ctx['manuales'])} en seguimiento.")
    lineas += ["Razones de cada compra y por qué es importante en **[Seguimiento](seguimiento.md)**.", ""]

    if ctx["rendimiento_pasado"]:
        lineas += ["## 📅 ¿Cómo les fue a las recomendaciones anteriores?", "",
                   "| Fecha | Top 5 de ese día | Rend. medio desde entonces | S&P 500 en el mismo periodo |",
                   "|---|---|---:|---:|"]
        for fila in ctx["rendimiento_pasado"]:
            lineas.append(f"| {fila['fecha']} | {', '.join(fila['tickers'])} | {pct(fila['rendimiento'])} | "
                          f"{pct(fila['mercado'])} |")
        lineas.append("")

    lineas += ["## ⚙️ Cómo se calcula el puntaje", "",
               "1. **Técnico**: posición del precio frente a las **MA(7), MA(25) y MA(99)**, cruces entre medias, "
               "retrocesos a la MA(25) y cercanía al máximo de 52 semanas.",
               "2. **Momentum**: RSI, MACD, volumen y fuerza relativa frente al S&P 500.",
               "3. **Noticias**: sentimiento de los titulares recientes (los más nuevos pesan más).",
               "4. **Analistas**: precio objetivo medio y recomendación de consenso.",
               "5. **Red neuronal**: solo cuando supera sus criterios de madurez.",
               "", "Este reporte se actualiza solo con GitHub Actions después de cada cierre del mercado."]
    _escribir(ruta, lineas)


# --------------------------------------------------------------------------
# Oportunidades
# --------------------------------------------------------------------------

def _bloque_red(registro: dict, red_lista: bool) -> list[str]:
    if registro.get("prob_red") is None:
        return []
    texto = (f"**🧠 Red neuronal:** {pct(registro['prob_red'], 1, False)} de probabilidad de buen punto de compra "
             f"({num(registro.get('lift_red'), 1)}× la media)")
    if not red_lista:
        texto += " — _experimental, aún no cuenta en el puntaje_"
    return [texto, ""]


def escribir_oportunidades(ruta: Path, ctx: dict) -> None:
    lineas = _encabezado("🚀 Acciones con mayor potencial de subida", ctx)
    lineas += ["Ordenadas por puntaje total (0 a 100). Cada gráfico muestra velas diarias con las medias móviles "
               "**MA(7)** (azul), **MA(25)** (naranja) y **MA(99)** (violeta), y el volumen debajo.", "", AVISO, ""]
    red_lista = ctx["red"]["madurez"]["lista"]
    for i, r in enumerate(ctx["top"], 1):
        t = r["tecnico"]
        lineas += [f'<a id="{ancla(r["ticker"])}"></a>', "",
                   f"## {i}. {r['ticker']} · {celda(r['nombre'])}", "",
                   f"**Puntaje {r['puntaje']:.0f}/100 · {r['recomendacion']}** · {sector_es(r.get('sector'))} · "
                   f"Potencial estimado **{pct(r['potencial'])}** ({r['base_potencial']})", ""]
        if r.get("grafico"):
            lineas += [f"![Gráfico de {r['ticker']}]({r['grafico']})", ""]
        lineas += ["| Precio | Día | 1 mes | 3 meses | RSI | MA(7) | MA(25) | MA(99) | Máx. 52 sem. |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                   f"| {formato_dinero(t['precio'])} | {pct(t['cambio_1d'], 2)} | {pct(t['ret_21'])} | {pct(t['ret_63'])} | "
                   f"{t['rsi']:.0f} | {formato_dinero(t['ma7'])} | {formato_dinero(t['ma25'])} | "
                   f"{formato_dinero(t['ma99'])} | {formato_dinero(t['max_52s'])} ({pct(t['dist_max_52s'])}) |", "",
                   f"Componentes: {_componentes(r)} · Tendencia: {emoji_tendencia(t['tendencia'])} {t['tendencia']}", ""]
        lineas += ["**✅ Por qué tiene potencial**", ""]
        lineas += [f"- {s}" for s in t["senales_positivas"]] or ["- _Sin señales técnicas destacadas._"]
        if t["senales_negativas"]:
            lineas += ["", "**⚠️ Riesgos**", ""] + [f"- {s}" for s in t["senales_negativas"]]
        noticias = r.get("noticias") or {}
        lineas += ["", f"**📰 Noticias recientes** — sentimiento: {emoji_etiqueta(noticias.get('etiqueta'))} "
                       f"{noticias.get('etiqueta', '—')}"
                       + (f" ({noticias['sentimiento']:+.2f})" if noticias.get("sentimiento") is not None else ""), ""]
        lineas += _lista_noticias(noticias)
        lineas.append("")
        a = r.get("analistas")
        if a:
            texto = (f"**🏦 Analistas:** precio objetivo medio {formato_dinero(a['objetivo'])} ({pct(a['potencial'])}), "
                     f"{a['num_analistas']} opiniones")
            if a.get("recomendacion"):
                texto += f", consenso {a['recomendacion']:.1f} (1 = compra fuerte, 5 = venta)"
            lineas += [texto, ""]
        lineas += _bloque_red(r, red_lista)
        lineas += ["---", ""]
    if not ctx["top"]:
        lineas += ["_Hoy ninguna acción pasó los filtros._"]
    _escribir(ruta, lineas)


# --------------------------------------------------------------------------
# Seguimiento
# --------------------------------------------------------------------------

def _bloque_posicion(p: dict) -> list[str]:
    reg = p.get("registro")
    pnl = p.get("pnl")
    lineas = [f'<a id="seg-{ancla(p["ticker"])}"></a>', "",
              f"### {p['ticker']} · {celda(p.get('nombre') or p['ticker'])}", ""]
    if p.get("grafico"):
        lineas += [f"![Seguimiento de {p['ticker']}]({p['grafico']})", ""]
    compra = f"**Comprada el {p['fecha_compra'] or 'fecha no indicada'} a {formato_dinero(p['precio_compra'])}**"
    if p.get("acciones"):
        compra += f" · {num(p['acciones'], 0 if float(p['acciones']).is_integer() else 4)} acciones"
    if p.get("puntaje_compra") is not None:
        compra += f" · puntaje al comprar {p['puntaje_compra']:.0f}"
    lineas += [compra, ""]
    resultado = f"**{pct(pnl)}**"
    if p.get("resultado_usd") is not None:
        resultado += f" ({formato_dinero(p['resultado_usd'])})"
    tendencia = reg["tecnico"]["tendencia"] if reg else "—"
    puntaje_hoy = f"{reg['puntaje']:.0f}" if reg else "—"
    lineas += ["| Precio actual | Resultado | Días | Stop-loss | Objetivo | Tendencia | Puntaje hoy |",
               "|---:|---:|---:|---:|---:|---|---:|",
               f"| {formato_dinero(p.get('precio_actual'))} | {resultado} | {p.get('dias', '—')} | "
               f"{formato_dinero(p.get('stop_loss'))} | {formato_dinero(p.get('objetivo'))} | {tendencia} | "
               f"{puntaje_hoy} |", ""]
    etiqueta_razones = "📝 Razones de la compra" + (" (análisis automático)" if p.get("razones_auto") else "")
    lineas += [f"**{etiqueta_razones}**", ""] + [f"- {celda(r)}" for r in p.get("razones") or ["_Sin razones registradas._"]]
    etiqueta_imp = "⭐ Por qué es importante" + (" (análisis automático)" if p.get("importancia_auto") else "")
    lineas += ["", f"**{etiqueta_imp}**", ""] + [f"- {celda(i)}" for i in p.get("importancia") or ["_Sin datos._"]]
    lineas += ["", f"**📊 Tesis de inversión:** {p.get('tesis', 'Sin datos')}", ""]
    if p.get("alertas"):
        lineas += ["**🔔 Alertas**", ""] + [f"- {a}" for a in p["alertas"]] + [""]
    if reg:
        lineas += ["**📰 Noticias recientes**", ""] + _lista_noticias(reg.get("noticias"), 4) + [""]
    bitacora = p.get("bitacora") or []
    if bitacora:
        lineas += ["<details><summary>📒 Bitácora de seguimiento</summary>", "",
                   "| Fecha | Precio | Resultado | Nota |", "|---|---:|---:|---|"]
        for b in reversed(bitacora[-10:]):
            lineas.append(f"| {b['fecha']} | {formato_dinero(b['precio'])} | {pct(b['pnl'])} | {celda(b['nota'])} |")
        lineas += ["", "</details>", ""]
    return lineas + ["---", ""]


def escribir_seguimiento(ruta: Path, ctx: dict) -> None:
    c = ctx["cartera"]
    resumen = c["resumen"]
    lineas = _encabezado("💼 Seguimiento de acciones", ctx)
    lineas += ["Cada acción comprada guarda **por qué se compró** y **por qué es importante**, y cada día se revisa "
               "si la tesis sigue en pie frente a las MA(7), MA(25), MA(99) y las noticias.", "",
               "## Cartera simulada", "",
               f"Capital inicial {formato_dinero(c['capital_inicial'])}. Compra automáticamente las mejores "
               f"oportunidades (puntaje ≥ {ctx['cfg']['cartera']['puntaje_minimo_compra']}) y vende por stop-loss, "
               "objetivo de ganancia o ruptura de tendencia.", "",
               "| Valor | Rendimiento | Efectivo | Invertido | Posiciones | Cerradas | % ganadoras |",
               "|---:|---:|---:|---:|---:|---:|---:|",
               f"| **{formato_dinero(resumen['valor'])}** | {pct(resumen['rendimiento'])} | {formato_dinero(resumen['efectivo'])} | "
               f"{formato_dinero(resumen['invertido'])} | {resumen['posiciones']} | {resumen['operaciones_cerradas']} | "
               f"{pct(resumen['tasa_acierto'], 0, False)} |", ""]
    if c.get("grafico"):
        lineas += [f"![Cartera vs S&P 500]({c['grafico']})", ""]
    if c["posiciones"]:
        lineas += ["| Acción | Compra | Precio compra | Actual | Resultado | Días | Tesis |",
                   "|---|---|---:|---:|---:|---:|---|"]
        for p in c["posiciones"]:
            lineas.append(f"| [**{p['ticker']}**](#seg-{ancla(p['ticker'])}) | {p['fecha_compra']} | "
                          f"{formato_dinero(p['precio_compra'])} | {formato_dinero(p['precio_actual'])} | "
                          f"{pct(p['pnl'])} | {p['dias']} | {p['tesis'].split(':')[0]} |")
        lineas.append("")
        for p in c["posiciones"]:
            lineas += _bloque_posicion(p)
    else:
        lineas += ["_La cartera simulada no tiene posiciones abiertas._", ""]

    lineas += ["## 👤 Tus acciones", ""]
    if ctx["manuales"]:
        lineas.append("Declaradas en `config/mis_acciones.yaml`.")
        lineas.append("")
        for p in ctx["manuales"]:
            lineas += _bloque_posicion(p)
    else:
        lineas += ["Agrega tus compras reales en `config/mis_acciones.yaml` (ticker, fecha, precio, razones e "
                   "importancia) y aparecerán aquí con su seguimiento diario.", ""]

    lineas += ["## 📚 Operaciones cerradas", ""]
    if c["cerradas"]:
        lineas += ["| Acción | Compra | Venta | Resultado | Días | Motivo de venta | Razón principal de la compra |",
                   "|---|---|---|---:|---:|---|---|"]
        for op in reversed(c["cerradas"][-40:]):
            razon = (op.get("razones") or ["—"])[1 if len(op.get("razones") or []) > 1 else 0]
            lineas.append(f"| **{op['ticker']}** | {op['fecha_compra']} a {formato_dinero(op['precio_compra'])} | "
                          f"{op['fecha_venta']} a {formato_dinero(op['precio_venta'])} | {pct(op['resultado_pct'])} "
                          f"({formato_dinero(op['resultado_usd'])}) | {op['dias']} | {celda(op['motivo_venta'])} | "
                          f"{celda(razon)} |")
    else:
        lineas.append("_Todavía no hay operaciones cerradas._")
    _escribir(ruta, lineas)


# --------------------------------------------------------------------------
# Red neuronal
# --------------------------------------------------------------------------

def escribir_red(ruta: Path, ctx: dict) -> None:
    red = ctx["red"]
    meta, madurez, rcfg = red["meta"], red["madurez"], ctx["cfg"]["red_neuronal"]
    lineas = _encabezado("🧠 Red neuronal", ctx)
    lineas += [f"**Estado: {madurez['nivel']}** — madurez {barra(madurez['progreso'])}", ""]
    if madurez["lista"]:
        lineas += ["✅ La red superó todos los criterios: sus predicciones ya cuentan en el puntaje de cada acción "
                   f"(peso {ctx['cfg']['pesos']['red_neuronal']}).", ""]
    else:
        lineas += ["⏳ La red sigue aprendiendo. Cada día entrena, predice y se evalúa; cuando cumpla todos los "
                   "criterios de abajo, sus recomendaciones se activarán solas.", ""]

    lineas += ["## Cómo aprende", "",
               f"- **El maestro compra en el mínimo y vende en el máximo.** Mirando el pasado, marca como *buen punto de "
               f"compra* cada día en que, comprando al cierre, el precio subió al menos **{pct(rcfg['ganancia_minima'], 0, False)}** "
               f"en los siguientes **{rcfg['horizonte']} días** sin caer antes más de **{pct(rcfg['caida_maxima'], 0, False)}**.",
               "- **La red solo ve lo que se sabía ese día**: distancia a las MA(7), MA(25), MA(99), pendientes de las "
               "medias, retornos, RSI, MACD, volatilidad, volumen, posición en el rango de 52 semanas, velas y el "
               "estado del S&P 500 (31 variables). Con miles de ejemplos aprende a reconocer esos momentos.",
               f"- **Arquitectura:** perceptrón multicapa {len(meta['caracteristicas'])} → "
               f"{' → '.join(str(c) for c in meta['capas'])} → 1"
               + (f" ({red['parametros']:,} parámetros)" if red.get("parametros") else "")
               + ", Leaky ReLU, optimizador Adam, escrita en NumPy.",
               "- **Aprendizaje lento y continuo:** cada ejecución de GitHub Actions entrena una sesión corta y "
               "guarda los pesos; los sábados hace una sesión larga. Nunca empieza de cero.",
               "- **Sin trampas:** la validación usa los meses más recientes (separados del entrenamiento) y además "
               f"cada día se registran sus {rcfg['top_predicciones']} favoritas, que se comprueban "
               f"{rcfg['horizonte']} días después con lo que pasó realmente.", ""]

    lineas += ["## Criterios para estar lista", "", "| Criterio | Actual | Requerido | Progreso | |",
               "|---|---:|---:|---|---|"]
    for f in madurez["criterios"]:
        actual = f["actual"]
        actual_txt = "—" if actual is None else (f"{actual:,.0f}" if float(actual).is_integer() else f"{actual:.3f}")
        lineas.append(f"| {f['criterio']} | {actual_txt} | {f['requerido']} | {barra(f['progreso'], 10)} | "
                      f"{'✅' if f['cumple'] else '⏳'} |")
    lineas += ["", f"Sesiones: **{meta['sesiones']}** · épocas: **{meta['epocas_totales']}** · ejemplos vistos: "
                   f"**{meta['ejemplos_vistos']:,}** · creada: {meta['creada']} · último entrenamiento: "
                   f"{meta.get('ultimo_entrenamiento') or '—'}", ""]
    if red.get("grafico_aprendizaje"):
        lineas += ["## Evolución del aprendizaje", "", f"![Aprendizaje]({red['grafico_aprendizaje']})", ""]
    if red.get("grafico_maestro"):
        lineas += ["## El maestro en acción", "",
                   f"Triángulos verdes: donde el maestro compra (mínimo); rojos: donde vende (máximo). Abajo, la "
                   f"probabilidad que la red asigna cada día. Ejemplo: **{red['ticker_maestro']}**.", "",
                   f"![Maestro]({red['grafico_maestro']})", ""]

    historial = meta.get("historial") or []
    if historial:
        u = historial[-1]
        lineas += ["## Última sesión", "",
                   "| Sesión | Fecha | Épocas | Ejemplos | AUC | AUC referencia | Tasa base | Precisión top 5% | Lift | "
                   "Top 10 diario | Ganancia máx. top 5% | Ganancia máx. media |",
                   "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                   f"| {u['sesion']} | {u['fecha']} | {u['epocas']} | {u['ejemplos']:,} | {num(u.get('auc'), 3)} | "
                   f"{num(u.get('auc_referencia'), 3)} | "
                   f"{pct(u.get('tasa_base'), 1, False)} | {pct(u.get('precision_top'), 1, False)} | {num(u.get('lift'))} | "
                   f"{pct(u.get('precision_top10_diaria'), 1, False)} | {pct(u.get('ganancia_top'))} | "
                   f"{pct(u.get('ganancia_media'))} |", "",
                   "_AUC 0.5 = azar, 1.0 = perfecto. AUC referencia = la regla simple de elegir las acciones más "
                   "volátiles. Lift = cuántas veces más aciertos que eligiendo al azar._", ""]

    vivo = meta.get("vivo") or {}
    lineas += ["## Evaluación en vivo", ""]
    if vivo.get("evaluadas"):
        lineas += ["| Evaluadas | Aciertos | Precisión | Tasa base | Ventaja | Ganancia máx. media | Pendientes |",
                   "|---:|---:|---:|---:|---:|---:|---:|",
                   f"| {vivo['evaluadas']} | {vivo['aciertos']} | {pct(vivo.get('precision'), 1, False)} | "
                   f"{pct(vivo.get('tasa_base'), 1, False)} | {num(vivo.get('ventaja'))}× | "
                   f"{pct(vivo.get('ganancia_max_media'))} | {vivo.get('pendientes', 0)} |", ""]
        if red.get("ultimas_evaluadas"):
            lineas += ["| Fecha | Acción | Probabilidad | Precio | Ganancia máx. | Peor momento antes del máx. | Resultado |",
                       "|---|---|---:|---:|---:|---:|---|"]
            for p in red["ultimas_evaluadas"]:
                lineas.append(f"| {p['fecha']} | {p['ticker']} | {pct(p['probabilidad'], 1, False)} | "
                              f"{formato_dinero(p['precio'])} | {pct(p.get('ganancia_max'))} | {pct(p.get('caida'))} | "
                              f"{'✅ acierto' if p['resultado'] else '❌ fallo'} |")
            lineas.append("")
    else:
        lineas += [f"_Aún no hay predicciones evaluadas: cada predicción se comprueba {rcfg['horizonte']} días "
                   "hábiles después de hacerse._", ""]

    titulo = "## Recomendaciones de la red" if madurez["lista"] else "## Predicciones de hoy (experimentales)"
    lineas += [titulo, ""]
    if len(red["predicciones"]):
        if not madurez["lista"]:
            lineas += ["_Todavía no son fiables: se muestran para seguir su evolución._", ""]
        lineas += ["| # | Acción | Probabilidad | × media | Precio | Tendencia |", "|---:|---|---:|---:|---:|---|"]
        for i, p in enumerate(red["predicciones"], 1):
            lineas.append(f"| {i} | **{p['ticker']}** {celda(p['nombre'])} | {pct(p['probabilidad'], 1, False)} | "
                          f"{num(p['lift'], 1)}× | {formato_dinero(p['precio'])} | {p.get('tendencia', '—')} |")
    else:
        lineas.append("_La red todavía no tiene suficientes datos para predecir._")
    _escribir(ruta, lineas)
